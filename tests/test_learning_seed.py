from __future__ import annotations

import unittest
from unittest.mock import patch

from polymarket_bot.learning import PairTimingLearner
from polymarket_bot.learning_seed import (
    _extract_seed_trades,
    _iter_activity,
    seed_pair_learning_from_trader_history,
)
from polymarket_bot.models import Timeframe


class _FakeStorage:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []
        self.stats: dict[tuple[str, str, int], dict[str, object]] = {}

    def load_pair_learning_stats(self):
        return []

    def record_pair_learning_event(self, **kwargs):
        self.events.append(kwargs)

    def upsert_pair_learning_stat(self, **kwargs):
        key = (
            str(kwargs["timeframe"]),
            str(kwargs["entry_side"]),
            int(kwargs["sec_bucket"]),
        )
        self.stats[key] = kwargs


class LearningSeedTests(unittest.TestCase):
    def test_extract_seed_trades_uses_force_timeframe(self) -> None:
        events = [
            {
                "type": "TRADE",
                "side": "BUY",
                "conditionId": "0xabc",
                "timestamp": 1700000000,
                "price": 0.47,
                "outcomeIndex": 0,
                "transactionHash": "0x01",
                "asset": "a",
            },
            {
                "type": "TRADE",
                "side": "BUY",
                "conditionId": "0xabc",
                "timestamp": 1700000020,
                "price": 0.53,
                "outcomeIndex": 1,
                "transactionHash": "0x02",
                "asset": "b",
            },
        ]
        trades, result = _extract_seed_trades(
            events,
            forced_timeframe=Timeframe.FIVE_MIN,
        )
        self.assertEqual(result.trades_imported, 2)
        self.assertEqual(result.skipped_unknown_timeframe, 0)
        self.assertTrue(all(item.timeframe == Timeframe.FIVE_MIN for item in trades))

    def test_seed_history_pairs_opposite_trades_and_proxies_unmatched(self) -> None:
        fake_events = [
            {
                "type": "TRADE",
                "side": "BUY",
                "conditionId": "0xcond",
                "timestamp": 1700000000,
                "price": 0.40,
                "outcomeIndex": 0,
                "transactionHash": "0x11",
                "asset": "a",
            },
            {
                "type": "TRADE",
                "side": "BUY",
                "conditionId": "0xcond",
                "timestamp": 1700000030,
                "price": 0.58,
                "outcomeIndex": 1,
                "transactionHash": "0x12",
                "asset": "b",
            },
            {
                "type": "TRADE",
                "side": "BUY",
                "conditionId": "0xcond",
                "timestamp": 1700000300,
                "price": 0.35,
                "outcomeIndex": 0,
                "transactionHash": "0x13",
                "asset": "c",
            },
        ]
        storage = _FakeStorage()
        learner = PairTimingLearner(storage)  # type: ignore[arg-type]

        with patch(
            "polymarket_bot.learning_seed._iter_activity",
            return_value=(fake_events, 1),
        ):
            result = seed_pair_learning_from_trader_history(
                learner=learner,
                data_api_url="https://data-api.polymarket.com",
                user="0xtest",
                forced_timeframe=Timeframe.FIVE_MIN,
                pair_gap_seconds=120.0,
            )

        self.assertEqual(result.pages_fetched, 1)
        self.assertEqual(result.trades_imported, 3)
        self.assertEqual(result.seeded_pair_events, 2)
        self.assertEqual(result.seeded_proxy_events, 1)

        sources = {str(row.get("source")) for row in storage.events}
        self.assertIn("seed_history_pair", sources)
        self.assertIn("seed_history_proxy", sources)
        self.assertEqual(len(storage.events), 3)

    def test_iter_activity_stops_on_late_http_400(self) -> None:
        page0 = [{"type": "TRADE"}]
        with patch(
            "polymarket_bot.learning_seed.get_json",
            side_effect=[page0, RuntimeError("HTTP Error 400: Bad Request")],
        ):
            events, pages = _iter_activity(
                data_api_url="https://data-api.polymarket.com",
                user="0xtest",
                page_size=100,
                max_pages=20,
            )
        self.assertEqual(pages, 1)
        self.assertEqual(events, page0)


if __name__ == "__main__":
    unittest.main()

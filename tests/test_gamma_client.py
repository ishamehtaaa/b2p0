from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from polymarket_bot.clients_gamma import GammaClient


class FakeGammaClient(GammaClient):
    def __init__(self, items_by_tag: dict[int, list[dict]]) -> None:
        super().__init__(base_url="https://gamma-api.polymarket.com", timeout_seconds=1.0)
        self._items_by_tag = items_by_tag

    def fetch_markets(self, tag_id: int, limit: int = 2000) -> list[dict]:
        return list(self._items_by_tag.get(tag_id, []))


class GammaClientTests(unittest.TestCase):
    def test_uses_end_time_minus_tenor_for_effective_start(self) -> None:
        now = datetime.now(tz=timezone.utc)
        # Raw start is in the past, but contract window (end-5m) is in the future.
        # This must be filtered to avoid pre-start trading.
        item = {
            "id": "m-prestart-5m",
            "question": "Bitcoin Up or Down - February 17, 11:40PM-11:45PM ET",
            "startDate": (now - timedelta(hours=1)).isoformat(),
            "endDate": (now + timedelta(minutes=8)).isoformat(),
            "conditionId": "0x" + ("11" * 32),
            "clobTokenIds": '["tok-up","tok-down"]',
            "outcomes": ["Up", "Down"],
            "liquidityNum": "1000",
            "volume24hr": "100",
            "bestBid": "0.49",
            "bestAsk": "0.51",
            "spread": "0.02",
            "orderMinSize": "5",
            "slug": "btc-updown-5m-test",
        }
        client = FakeGammaClient({102892: [item]})
        markets = client.fetch_btc_markets([102892], limit_per_tag=5)
        self.assertEqual(markets, [])

    def test_includes_market_once_effective_window_has_started(self) -> None:
        now = datetime.now(tz=timezone.utc)
        item = {
            "id": "m-live-5m",
            "question": "Bitcoin Up or Down - February 17, 11:35PM-11:40PM ET",
            "startDate": (now - timedelta(hours=2)).isoformat(),
            "endDate": (now + timedelta(minutes=2)).isoformat(),
            "conditionId": "0x" + ("22" * 32),
            "clobTokenIds": '["tok-up","tok-down"]',
            "outcomes": ["Up", "Down"],
            "liquidityNum": "1000",
            "volume24hr": "100",
            "bestBid": "0.49",
            "bestAsk": "0.51",
            "spread": "0.02",
            "orderMinSize": "5",
            "slug": "btc-updown-5m-test-live",
        }
        client = FakeGammaClient({102892: [item]})
        markets = client.fetch_btc_markets([102892], limit_per_tag=5)
        self.assertEqual(len(markets), 1)
        market = markets[0]
        self.assertLessEqual(market.start_time, now)
        expected_start = market.end_time - timedelta(minutes=5)
        self.assertEqual(market.start_time, expected_start)

    def test_parses_neg_risk_flag(self) -> None:
        now = datetime.now(tz=timezone.utc)
        item = {
            "id": "m-neg-risk",
            "question": "Bitcoin Up or Down - test",
            "startDate": (now - timedelta(hours=1)).isoformat(),
            "endDate": (now + timedelta(minutes=4)).isoformat(),
            "conditionId": "0x" + ("33" * 32),
            "clobTokenIds": '["tok-up","tok-down"]',
            "outcomes": ["Up", "Down"],
            "liquidityNum": "1000",
            "volume24hr": "100",
            "bestBid": "0.49",
            "bestAsk": "0.51",
            "spread": "0.02",
            "orderMinSize": "5",
            "slug": "btc-neg-risk-test",
            "negRisk": True,
        }
        client = FakeGammaClient({102892: [item]})
        markets = client.fetch_btc_markets([102892], limit_per_tag=5)
        self.assertEqual(len(markets), 1)
        self.assertTrue(markets[0].is_neg_risk)

    def test_does_not_fallback_to_title_key(self) -> None:
        now = datetime.now(tz=timezone.utc)
        item = {
            "id": "m-title-only",
            "title": "Bitcoin Up or Down - test",
            "startDate": (now - timedelta(hours=1)).isoformat(),
            "endDate": (now + timedelta(minutes=4)).isoformat(),
            "conditionId": "0x" + ("11" * 32),
            "clobTokenIds": '["tok-up","tok-down"]',
            "outcomes": ["Up", "Down"],
            "slug": "btc-title-only-test",
        }
        client = FakeGammaClient({102892: [item]})
        markets = client.fetch_btc_markets([102892], limit_per_tag=5)
        self.assertEqual(markets, [])


if __name__ == "__main__":
    unittest.main()

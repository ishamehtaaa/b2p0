from __future__ import annotations

import json
import unittest
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from polymarket_bot.clients_spot import BtcSpotClient, BtcSpotStream


class SpotClientTests(unittest.TestCase):
    def test_get_price_reads_coinbase_amount(self) -> None:
        client = BtcSpotClient(url="https://example.com", timeout_seconds=1.0)
        with patch(
            "polymarket_bot.clients_spot.get_json",
            return_value={"data": {"amount": "101234.56"}},
        ):
            price = client.get_price()
        self.assertAlmostEqual(price, 101234.56, places=6)

    def test_get_price_does_not_fallback_to_price_key(self) -> None:
        client = BtcSpotClient(url="https://example.com", timeout_seconds=1.0)
        with patch(
            "polymarket_bot.clients_spot.get_json",
            return_value={"price": "101234.56"},
        ):
            with self.assertRaises(RuntimeError):
                client.get_price()

    def test_stream_parses_binance_trade_payload(self) -> None:
        ticks: list[tuple[float, float]] = []
        client = BtcSpotClient(url="https://example.com", timeout_seconds=1.0)
        stream = BtcSpotStream(
            ws_url="wss://example.com/ws",
            rest_client=client,
            on_price=lambda ts, px: ticks.append((ts, px)),
        )
        stream._on_message(None, json.dumps({"p": "100000.0", "E": 1700000000000}))
        self.assertEqual(len(ticks), 1)
        ts, px = ticks[0]
        self.assertAlmostEqual(px, 100000.0, places=6)
        self.assertGreater(ts, 0.0)

    def test_stream_ignores_non_trade_price_fields(self) -> None:
        ticks: list[tuple[float, float]] = []
        client = BtcSpotClient(url="https://example.com", timeout_seconds=1.0)
        stream = BtcSpotStream(
            ws_url="wss://example.com/ws",
            rest_client=client,
            on_price=lambda ts, px: ticks.append((ts, px)),
        )
        stream._on_message(None, json.dumps({"c": "100000.0", "E": 1700000000000}))
        self.assertEqual(ticks, [])


if __name__ == "__main__":
    unittest.main()

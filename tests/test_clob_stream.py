from __future__ import annotations

import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from polymarket_bot.clients_clob import ClobMarketStream


class ClobStreamTests(unittest.TestCase):
    def test_book_payload_updates_cache(self) -> None:
        stream = ClobMarketStream("wss://example.com/ws/market")
        stream.process_payload(
            {
                "event_type": "book",
                "asset_id": "tok-1",
                "bids": [{"price": "0.42", "size": "100"}, {"price": "0.41", "size": "50"}],
                "asks": [{"price": "0.58", "size": "90"}],
                "timestamp": "1234",
            }
        )
        books = stream.get_books(["tok-1"])
        self.assertIn("tok-1", books)
        book = books["tok-1"]
        self.assertEqual(book.timestamp_ms, 1234)
        self.assertAlmostEqual(book.bids[0].price, 0.42)
        self.assertAlmostEqual(book.asks[0].price, 0.58)

    def test_price_change_patches_levels(self) -> None:
        stream = ClobMarketStream("wss://example.com/ws/market")
        stream.process_payload(
            {
                "event_type": "book",
                "asset_id": "tok-2",
                "bids": [{"price": "0.40", "size": "20"}],
                "asks": [{"price": "0.60", "size": "25"}],
                "timestamp": "1000",
            }
        )
        stream.process_payload(
            {
                "event_type": "price_change",
                "timestamp": "2000",
                "price_changes": [
                    {"asset_id": "tok-2", "side": "BUY", "price": "0.41", "size": "15"},
                    {"asset_id": "tok-2", "side": "SELL", "price": "0.60", "size": "0"},
                ],
            }
        )

        book = stream.get_books(["tok-2"])["tok-2"]
        self.assertEqual(book.timestamp_ms, 2000)
        self.assertAlmostEqual(book.bids[0].price, 0.41)
        self.assertEqual(len(book.asks), 0)

    def test_camel_case_payload_is_ignored(self) -> None:
        stream = ClobMarketStream("wss://example.com/ws/market")
        stream.process_payload(
            {
                "eventType": "book",
                "assetId": "tok-ignored",
                "bids": [{"price": "0.42", "size": "10"}],
                "asks": [{"price": "0.58", "size": "10"}],
                "timestamp": "1234",
            }
        )
        books = stream.get_books(["tok-ignored"])
        self.assertEqual(books, {})


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from polymarket_bot.execution import PaperExecutor
from polymarket_bot.models import OrderBookLevel, OrderBookSnapshot, OrderIntent, Side, TimeInForce
from tests.helpers import test_config


class PaperExecutorTests(unittest.TestCase):
    def test_order_lifecycle_open_fill_cancel(self) -> None:
        cfg = test_config()
        executor = PaperExecutor(cfg)
        book = OrderBookSnapshot(
            token_id="t1",
            timestamp_ms=0,
            bids=[OrderBookLevel(price=0.49, size=100)],
            asks=[OrderBookLevel(price=0.51, size=100)],
        )
        books = {"t1": book}

        intent = OrderIntent(
            market_id="m1",
            token_id="t1",
            side=Side.BUY,
            price=0.50,
            size=10,
            tif=TimeInForce.GTC,
            post_only=True,
            engine="engine_pair_arb",
            expected_edge=0.01,
        )
        open_result = executor.place_order(intent, books)
        self.assertEqual(open_result.status, "open")

        # Move ask through quote -> should fill on sweep.
        books["t1"] = OrderBookSnapshot(
            token_id="t1",
            timestamp_ms=1,
            bids=[OrderBookLevel(price=0.49, size=100)],
            asks=[OrderBookLevel(price=0.50, size=100)],
        )
        fills = executor.sweep(books)
        self.assertEqual(len(fills), 1)
        _, fill = fills[0]
        self.assertEqual(fill.status, "filled")
        self.assertEqual(fill.filled_size, 10)

        # Place another open order and cancel it.
        open_result_2 = executor.place_order(intent, {"t1": book})
        self.assertEqual(open_result_2.status, "open")
        canceled = executor.cancel_order(open_result_2.order_id)
        self.assertTrue(canceled)

    def test_ioc_fills_at_best_book_price(self) -> None:
        cfg = test_config()
        executor = PaperExecutor(cfg)
        book = OrderBookSnapshot(
            token_id="t1",
            timestamp_ms=0,
            bids=[OrderBookLevel(price=0.48, size=100)],
            asks=[OrderBookLevel(price=0.52, size=100)],
        )
        books = {"t1": book}

        buy_intent = OrderIntent(
            market_id="m1",
            token_id="t1",
            side=Side.BUY,
            price=0.56,
            size=10,
            tif=TimeInForce.IOC,
            post_only=False,
            engine="engine_pair_arb",
            expected_edge=0.0,
        )
        buy_result = executor.place_order(buy_intent, books)
        self.assertEqual(buy_result.status, "filled")
        self.assertEqual(buy_result.filled_price, 0.52)

        sell_intent = OrderIntent(
            market_id="m1",
            token_id="t1",
            side=Side.SELL,
            price=0.40,
            size=10,
            tif=TimeInForce.IOC,
            post_only=False,
            engine="engine_pair_arb",
            expected_edge=0.0,
        )
        sell_result = executor.place_order(sell_intent, books)
        self.assertEqual(sell_result.status, "filled")
        self.assertEqual(sell_result.filled_price, 0.48)

    def test_ioc_rejects_when_not_marketable(self) -> None:
        cfg = test_config()
        executor = PaperExecutor(cfg)
        book = OrderBookSnapshot(
            token_id="t1",
            timestamp_ms=0,
            bids=[OrderBookLevel(price=0.48, size=100)],
            asks=[OrderBookLevel(price=0.52, size=100)],
        )
        books = {"t1": book}

        buy_intent = OrderIntent(
            market_id="m1",
            token_id="t1",
            side=Side.BUY,
            price=0.50,
            size=10,
            tif=TimeInForce.IOC,
            post_only=False,
            engine="engine_pair_arb",
            expected_edge=0.0,
        )
        buy_result = executor.place_order(buy_intent, books)
        self.assertEqual(buy_result.status, "rejected")

        sell_intent = OrderIntent(
            market_id="m1",
            token_id="t1",
            side=Side.SELL,
            price=0.50,
            size=10,
            tif=TimeInForce.IOC,
            post_only=False,
            engine="engine_pair_arb",
            expected_edge=0.0,
        )
        sell_result = executor.place_order(sell_intent, books)
        self.assertEqual(sell_result.status, "rejected")

    def test_settlement_helpers_available_in_paper_mode(self) -> None:
        cfg = test_config()
        executor = PaperExecutor(cfg)
        merge = executor.merge_pairs(
            market_id="m1",
            primary_token_id="up",
            secondary_token_id="down",
            size=7.5,
        )
        self.assertTrue(merge.success)
        self.assertEqual(merge.status, "merged")
        self.assertAlmostEqual(merge.merged_size, 7.5, places=6)

        redeem = executor.redeem_all()
        self.assertTrue(redeem.success)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

from polymarket_bot.execution import LiveExecutor
from polymarket_bot.models import OrderIntent, Side, TimeInForce
from tests.helpers import test_config


class LiveExecutorTests(unittest.TestCase):
    def test_signature_type_defaults_to_proxy_when_funder_differs(self) -> None:
        cfg = test_config(poly_signature_type=None)
        executor = LiveExecutor(cfg)
        self.assertEqual(executor._infer_signature_type("0x111", "0x222"), 1)

    def test_signature_type_defaults_to_eoa_when_funder_matches(self) -> None:
        cfg = test_config(poly_signature_type=None)
        executor = LiveExecutor(cfg)
        self.assertEqual(executor._infer_signature_type("0x111", "0x111"), 0)

    def test_signature_type_uses_explicit_override(self) -> None:
        cfg = test_config(poly_signature_type=2)
        executor = LiveExecutor(cfg)
        self.assertEqual(executor._infer_signature_type("0x111", "0x222"), 2)

    def test_api_creds_fallback_reads_env_triplet(self) -> None:
        env = {
            "POLY_API_KEY": "k",
            "POLY_API_SECRET": "s",
            "POLY_API_PASSPHRASE": "p",
        }
        with patch.dict("os.environ", env, clear=True):
            creds = LiveExecutor._api_creds_from_env()
        self.assertEqual(creds, {"key": "k", "secret": "s", "passphrase": "p"})

    def test_api_creds_fallback_requires_all_fields(self) -> None:
        env = {
            "POLY_API_KEY": "k",
            "POLY_API_SECRET": "s",
        }
        with patch.dict("os.environ", env, clear=True):
            creds = LiveExecutor._api_creds_from_env()
        self.assertIsNone(creds)

    def test_get_order_result_parses_order_state(self) -> None:
        cfg = test_config(poly_signature_type=2)
        executor = LiveExecutor(cfg)
        executor._bootstrap_done = True
        executor._preflight_done = True

        class _DummyClient:
            @staticmethod
            def get_order(order_id: str) -> dict[str, object]:
                return {
                    "id": order_id,
                    "status": "live",
                    "filledSize": "4",
                    "avgPrice": "0.27",
                    "fee": "0.01",
                }

        executor.client = _DummyClient()
        intent = OrderIntent(
            market_id="m1",
            token_id="token-up",
            side=Side.BUY,
            price=0.30,
            size=10.0,
            tif=TimeInForce.GTC,
            post_only=True,
            engine="engine_pair_arb",
            expected_edge=0.0,
        )
        result = executor.get_order_result("oid-1", intent)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.order_id, "oid-1")
        self.assertEqual(result.status, "live")
        self.assertAlmostEqual(result.filled_size, 4.0, places=6)
        self.assertAlmostEqual(result.filled_price, 0.27, places=6)
        self.assertAlmostEqual(result.fee_paid, 0.01, places=6)

    def test_get_order_result_does_not_infer_fill_from_matched_status_without_size(self) -> None:
        cfg = test_config(poly_signature_type=2)
        executor = LiveExecutor(cfg)
        executor._bootstrap_done = True
        executor._preflight_done = True

        class _DummyClient:
            @staticmethod
            def get_order(order_id: str) -> dict[str, object]:
                return {
                    "id": order_id,
                    "status": "MATCHED",
                    "price": "0.90",
                    "size_matched": "0",
                }

        executor.client = _DummyClient()
        intent = OrderIntent(
            market_id="m1",
            token_id="token-up",
            side=Side.BUY,
            price=0.90,
            size=10.0,
            tif=TimeInForce.GTC,
            post_only=True,
            engine="engine_pair_arb",
            expected_edge=0.0,
        )
        result = executor.get_order_result("oid-phantom", intent)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.status, "matched")
        self.assertAlmostEqual(result.filled_size, 0.0, places=9)
        self.assertAlmostEqual(result.filled_price, 0.0, places=9)

    def test_parse_fill_uses_explicit_fields_only(self) -> None:
        payload = {
            "status": "filled",
            "filledSize": "5",
            "avgPrice": "0.25",
            "fee": "0.01",
            "matchedSize": "999",  # ignored by strict parser
        }
        fill = LiveExecutor._parse_fill(payload, context="test")
        self.assertAlmostEqual(fill.filled_size, 5.0, places=9)
        self.assertAlmostEqual(fill.filled_price, 0.25, places=9)
        self.assertAlmostEqual(fill.fee_paid, 0.01, places=9)

    def test_parse_fill_does_not_infer_from_status(self) -> None:
        payload = {"status": "filled"}
        fill = LiveExecutor._parse_fill(payload, context="test")
        self.assertAlmostEqual(fill.filled_size, 0.0, places=9)
        self.assertAlmostEqual(fill.filled_price, 0.0, places=9)
        self.assertAlmostEqual(fill.fee_paid, 0.0, places=9)

    def test_get_collateral_balance_reads_balance_allowance(self) -> None:
        cfg = test_config(poly_signature_type=2)
        executor = LiveExecutor(cfg)
        executor._bootstrap_done = True
        executor._preflight_done = True
        executor._signature_type = 2

        calls: list[object] = []

        class _DummyClient:
            def get_balance_allowance(self, params: object) -> dict[str, object]:
                calls.append(params)
                return {"balance": "123.45", "allowances": {"exchange": "999"}}

        class _AssetType:
            COLLATERAL = "COLLATERAL"
            CONDITIONAL = "CONDITIONAL"

        class _BalanceAllowanceParams:
            def __init__(
                self,
                asset_type: str | None = None,
                token_id: str | None = None,
                signature_type: int = -1,
            ) -> None:
                self.asset_type = asset_type
                self.token_id = token_id
                self.signature_type = signature_type

        clob_types_mod = types.ModuleType("py_clob_client.clob_types")
        clob_types_mod.AssetType = _AssetType
        clob_types_mod.BalanceAllowanceParams = _BalanceAllowanceParams
        clob_pkg = types.ModuleType("py_clob_client")
        clob_pkg.clob_types = clob_types_mod

        executor.client = _DummyClient()
        with patch.dict(
            sys.modules,
            {
                "py_clob_client": clob_pkg,
                "py_clob_client.clob_types": clob_types_mod,
            },
        ):
            balance = executor.get_collateral_balance()

        self.assertAlmostEqual(balance, 123.45, places=6)
        self.assertEqual(len(calls), 1)
        sent = calls[0]
        self.assertEqual(getattr(sent, "asset_type", None), "COLLATERAL")
        self.assertEqual(getattr(sent, "token_id", "bad"), None)
        self.assertEqual(getattr(sent, "signature_type", None), 2)

    def test_get_token_balance_reads_balance_allowance(self) -> None:
        cfg = test_config(poly_signature_type=1)
        executor = LiveExecutor(cfg)
        executor._bootstrap_done = True
        executor._preflight_done = True
        executor._signature_type = 1

        calls: list[object] = []

        class _DummyClient:
            def get_balance_allowance(self, params: object) -> dict[str, object]:
                calls.append(params)
                return {"balance": "35000000"}

        class _AssetType:
            COLLATERAL = "COLLATERAL"
            CONDITIONAL = "CONDITIONAL"

        class _BalanceAllowanceParams:
            def __init__(
                self,
                asset_type: str | None = None,
                token_id: str | None = None,
                signature_type: int = -1,
            ) -> None:
                self.asset_type = asset_type
                self.token_id = token_id
                self.signature_type = signature_type

        clob_types_mod = types.ModuleType("py_clob_client.clob_types")
        clob_types_mod.AssetType = _AssetType
        clob_types_mod.BalanceAllowanceParams = _BalanceAllowanceParams
        clob_pkg = types.ModuleType("py_clob_client")
        clob_pkg.clob_types = clob_types_mod

        executor.client = _DummyClient()
        with patch.dict(
            sys.modules,
            {
                "py_clob_client": clob_pkg,
                "py_clob_client.clob_types": clob_types_mod,
            },
        ):
            balance = executor.get_token_balance("token-up")

        self.assertAlmostEqual(balance, 35.0, places=6)
        self.assertEqual(len(calls), 1)
        sent = calls[0]
        self.assertEqual(getattr(sent, "asset_type", None), "CONDITIONAL")
        self.assertEqual(getattr(sent, "token_id", None), "token-up")
        self.assertEqual(getattr(sent, "signature_type", None), 1)

    def test_parse_balance_allowance_scales_integer_units(self) -> None:
        scaled = LiveExecutor._parse_balance_allowance(
            {"balance": "483310435"},
            context="test_balance",
        )
        self.assertAlmostEqual(scaled, 483.310435, places=6)

        already_decimal = LiveExecutor._parse_balance_allowance(
            {"balance": "123.45"},
            context="test_balance",
        )
        self.assertAlmostEqual(already_decimal, 123.45, places=6)

    def test_get_balance_allowance_requires_balance_key(self) -> None:
        with self.assertRaises(RuntimeError):
            LiveExecutor._parse_balance_allowance(
                {"allowances": {"exchange": "0"}},
                context="test_balance",
            )

    def test_parse_maker_trade_fills_filters_wallet_orders(self) -> None:
        cfg = test_config(poly_signature_type=1)
        executor = LiveExecutor(cfg)
        executor._signer_address = "0xSigner"
        executor._funder_address = "0xFunder"
        payload = [
            {
                "id": "trade-1",
                "match_time": "1000",
                "maker_orders": [
                    {
                        "order_id": "oid-self",
                        "maker_address": "0xFunder",
                        "matched_amount": "5",
                        "price": "0.77",
                        "fee_rate_bps": "1000",
                        "asset_id": "token-up",
                        "side": "BUY",
                    },
                    {
                        "order_id": "oid-other",
                        "maker_address": "0xOther",
                        "matched_amount": "5",
                        "price": "0.77",
                        "fee_rate_bps": "1000",
                        "asset_id": "token-up",
                        "side": "BUY",
                    },
                ],
            }
        ]
        fills = executor._parse_maker_trade_fills(payload, after_ts=900)
        self.assertEqual(len(fills), 1)
        fill = fills[0]
        self.assertEqual(fill.trade_id, "trade-1")
        self.assertEqual(fill.order_id, "oid-self")
        self.assertEqual(fill.token_id, "token-up")
        self.assertEqual(fill.side, Side.BUY)
        self.assertAlmostEqual(fill.size, 5.0, places=6)
        self.assertAlmostEqual(fill.price, 0.77, places=6)
        self.assertEqual(fill.fee_rate_bps, 1000)
        self.assertEqual(fill.match_time, 1000)

    def test_get_recent_maker_trade_fills_calls_client(self) -> None:
        cfg = test_config(poly_signature_type=1)
        executor = LiveExecutor(cfg)
        executor._bootstrap_done = True
        executor._preflight_done = True
        executor._signer_address = "0xSigner"
        executor._funder_address = "0xFunder"

        class _DummyClient:
            def __init__(self) -> None:
                self.calls = 0

            def get_trades(self, _params: object, next_cursor: str = "MA==") -> list[dict]:
                self.calls += 1
                _ = next_cursor
                return [
                    {
                        "id": "trade-2",
                        "match_time": "1200",
                        "maker_orders": [
                            {
                                "order_id": "oid-self",
                                "maker_address": "0xFunder",
                                "matched_amount": "3",
                                "price": "0.25",
                                "fee_rate_bps": "1000",
                                "asset_id": "token-down",
                                "side": "SELL",
                            }
                        ],
                    }
                ]

        dummy = _DummyClient()
        executor.client = dummy
        fills = executor.get_recent_maker_trade_fills(after_ts=1100)
        self.assertEqual(dummy.calls, 1)
        self.assertEqual(len(fills), 1)
        self.assertEqual(fills[0].side, Side.SELL)
        self.assertEqual(fills[0].token_id, "token-down")

    def test_preflight_fails_fast_without_retry(self) -> None:
        cfg = test_config(poly_signature_type=2)
        executor = LiveExecutor(cfg)

        class _DummyClient:
            def __init__(self) -> None:
                self.calls = 0

            def get_api_keys(self) -> None:
                self.calls += 1
                raise RuntimeError("invalid signature")

        dummy_client = _DummyClient()

        def _bootstrap_stub() -> None:
            executor.client = dummy_client
            executor._signer_address = "0xsigner"
            executor._funder_address = "0xfunder"
            executor._signature_type = 2

        executor._bootstrap_client = _bootstrap_stub  # type: ignore[method-assign]

        with self.assertRaises(RuntimeError) as ctx:
            executor.preflight()
        self.assertIn("no retries", str(ctx.exception).lower())
        self.assertEqual(dummy_client.calls, 1)

    def test_place_order_invalid_signature_raises_immediately(self) -> None:
        cfg = test_config(poly_signature_type=2)
        executor = LiveExecutor(cfg)
        executor._bootstrap_done = True
        executor._preflight_done = True
        executor._signature_type = 2
        executor._signer_address = "0xsigner"
        executor._funder_address = "0xfunder"

        class _OrderArgs:
            def __init__(self, **kwargs: object) -> None:
                self.payload = kwargs

        class _OrderType:
            GTC = "GTC"
            IOC = "IOC"
            FOK = "FOK"

        clob_types_mod = types.ModuleType("py_clob_client.clob_types")
        clob_types_mod.OrderArgs = _OrderArgs
        clob_types_mod.OrderType = _OrderType
        clob_pkg = types.ModuleType("py_clob_client")
        clob_pkg.clob_types = clob_types_mod

        class _DummyClient:
            def __init__(self) -> None:
                self.create_calls = 0

            def create_order(self, _args: object) -> object:
                self.create_calls += 1
                raise RuntimeError("invalid signature")

        dummy_client = _DummyClient()
        executor.client = dummy_client

        intent = OrderIntent(
            market_id="m1",
            token_id="token-up",
            side=Side.BUY,
            price=0.30,
            size=10.0,
            tif=TimeInForce.GTC,
            post_only=True,
            engine="engine_pair_arb",
            expected_edge=0.0,
        )
        with patch.dict(
            sys.modules,
            {
                "py_clob_client": clob_pkg,
                "py_clob_client.clob_types": clob_types_mod,
            },
        ):
            with self.assertRaises(RuntimeError) as ctx:
                executor.place_order(intent, {})

        self.assertIn("no retries", str(ctx.exception).lower())
        self.assertEqual(dummy_client.create_calls, 1)

    def test_receipt_payload_is_strict_and_typed(self) -> None:
        receipt = {
            "status": "0x1",
            "transactionHash": bytes.fromhex("ab" * 32),
            "blockHash": "0x" + ("cd" * 32),
            "blockNumber": "123",
            "gasUsed": 21000,
            "effectiveGasPrice": "1000000000",
        }
        payload = LiveExecutor._receipt_payload(receipt)
        self.assertEqual(payload["status"], 1)
        self.assertEqual(payload["transactionHash"], "0x" + ("ab" * 32))
        self.assertEqual(payload["blockHash"], "0x" + ("cd" * 32))
        self.assertEqual(payload["blockNumber"], 123)
        self.assertEqual(payload["gasUsed"], 21000)
        self.assertEqual(payload["effectiveGasPrice"], 1_000_000_000)

    def test_pending_merge_stays_pending_when_receipt_unparseable(self) -> None:
        cfg = test_config(poly_signature_type=2)
        executor = LiveExecutor(cfg)
        executor._pending_merges["m1"] = {
            "tx_hash": "0xabc",
            "market_id": "m1",
            "condition_id": "0x" + ("11" * 32),
            "is_neg_risk": False,
            "route": "direct",
            "target": "0x" + ("22" * 20),
            "merged_size": 5.0,
        }
        executor._read_receipt = lambda _tx: {"blockNumber": 1}  # type: ignore[method-assign]

        result = executor._resolve_pending_merge("m1")
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.status, "pending")
        self.assertIn("receipt_parse_error", result.raw)
        self.assertIn("m1", executor._pending_merges)


if __name__ == "__main__":
    unittest.main()

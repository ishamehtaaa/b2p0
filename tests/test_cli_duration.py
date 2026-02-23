from __future__ import annotations

import unittest

from polymarket_bot.main import _apply_duration_profile, build_parser
from tests.helpers import test_config


class CliDurationTests(unittest.TestCase):
    def test_apply_duration_5m(self) -> None:
        cfg = test_config(
            enabled_tags=(102892, 102467, 102175),
            max_trade_markets_5m=3,
            max_trade_markets_15m=3,
            max_trade_markets_1h=3,
        )
        profiled = _apply_duration_profile(cfg, "5m")
        self.assertEqual(profiled.enabled_tags, (102892,))
        self.assertEqual(profiled.max_trade_markets_5m, 3)
        self.assertEqual(profiled.max_trade_markets_15m, 0)
        self.assertEqual(profiled.max_trade_markets_1h, 0)

    def test_apply_duration_5m_forces_min_cap(self) -> None:
        cfg = test_config(
            max_trade_markets_5m=0,
            max_trade_markets_15m=3,
            max_trade_markets_1h=3,
        )
        profiled = _apply_duration_profile(cfg, "5m")
        self.assertEqual(profiled.max_trade_markets_5m, 1)
        self.assertEqual(profiled.max_trade_markets_15m, 0)
        self.assertEqual(profiled.max_trade_markets_1h, 0)

    def test_apply_duration_15m(self) -> None:
        cfg = test_config(
            enabled_tags=(102892, 102467, 102175),
            max_trade_markets_5m=3,
            max_trade_markets_15m=3,
            max_trade_markets_1h=3,
        )
        profiled = _apply_duration_profile(cfg, "15m")
        self.assertEqual(profiled.enabled_tags, (102467,))
        self.assertEqual(profiled.max_trade_markets_5m, 0)
        self.assertEqual(profiled.max_trade_markets_15m, 3)
        self.assertEqual(profiled.max_trade_markets_1h, 0)

    def test_apply_duration_1h(self) -> None:
        cfg = test_config(
            enabled_tags=(102892, 102467, 102175),
            max_trade_markets_5m=3,
            max_trade_markets_15m=3,
            max_trade_markets_1h=3,
        )
        profiled = _apply_duration_profile(cfg, "1h")
        self.assertEqual(profiled.enabled_tags, (102175,))
        self.assertEqual(profiled.max_trade_markets_5m, 0)
        self.assertEqual(profiled.max_trade_markets_15m, 0)
        self.assertEqual(profiled.max_trade_markets_1h, 3)

    def test_parser_accepts_duration_for_run(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["run", "--duration", "5m"])
        self.assertEqual(args.command, "run")
        self.assertEqual(args.duration, "5m")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest
from unittest.mock import patch

from polymarket_bot.execution import LiveExecutor
from tests.helpers import test_config


class LiveExecutorTests(unittest.TestCase):
    def test_signature_type_defaults_to_proxy_when_funder_differs(self) -> None:
        cfg = test_config(poly_signature_type=None)
        executor = LiveExecutor(cfg)
        self.assertEqual(executor._infer_signature_type("0x111", "0x222"), 2)

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


if __name__ == "__main__":
    unittest.main()

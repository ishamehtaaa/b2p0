from __future__ import annotations

import json
import os
import ssl
from functools import lru_cache
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def _resolve_ca_bundle() -> tuple[str | None, str | None]:
    env_cafile = os.getenv("SSL_CERT_FILE")
    if env_cafile:
        return env_cafile, None

    env_capath = os.getenv("SSL_CERT_DIR")
    if env_capath:
        return None, env_capath

    try:
        import certifi  # type: ignore

        return certifi.where(), None
    except Exception:
        try:
            from pip._vendor import certifi as pip_certifi  # type: ignore

            return pip_certifi.where(), None
        except Exception:
            return None, None


@lru_cache(maxsize=1)
def _ssl_context() -> ssl.SSLContext:
    cafile, capath = _resolve_ca_bundle()
    if cafile:
        return ssl.create_default_context(cafile=cafile)
    if capath:
        return ssl.create_default_context(capath=capath)
    return ssl.create_default_context()


def websocket_sslopt() -> dict[str, object]:
    cafile, capath = _resolve_ca_bundle()
    sslopt: dict[str, object] = {
        "cert_reqs": ssl.CERT_REQUIRED,
        "check_hostname": True,
    }
    if cafile:
        sslopt["ca_certs"] = cafile
    if capath:
        sslopt["ca_cert_path"] = capath
    return sslopt


def get_json(url: str, params: dict[str, str] | None = None, timeout: float = 10.0):
    if params:
        query = urlencode(params)
        separator = "&" if "?" in url else "?"
        full_url = f"{url}{separator}{query}"
    else:
        full_url = url

    request = Request(full_url, headers={"User-Agent": "polymarket-bot/0.1"})
    with urlopen(request, timeout=timeout, context=_ssl_context()) as response:
        body = response.read().decode("utf-8")
    return json.loads(body)

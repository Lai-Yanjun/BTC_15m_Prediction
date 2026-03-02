from __future__ import annotations

import os
from typing import Any


def inject_proxy_env(
    *,
    proxy: str | None,
    http_proxy: str | None,
    https_proxy: str | None,
    no_proxy: str,
) -> dict[str, Any]:
    if proxy:
        os.environ["http_proxy"] = str(proxy)
        os.environ["https_proxy"] = str(proxy)
        os.environ["HTTP_PROXY"] = str(proxy)
        os.environ["HTTPS_PROXY"] = str(proxy)
    if http_proxy:
        os.environ["http_proxy"] = str(http_proxy)
        os.environ["HTTP_PROXY"] = str(http_proxy)
    if https_proxy:
        os.environ["https_proxy"] = str(https_proxy)
        os.environ["HTTPS_PROXY"] = str(https_proxy)
    os.environ["no_proxy"] = str(no_proxy)
    os.environ["NO_PROXY"] = str(no_proxy)
    return current_proxy_env()


def current_proxy_env() -> dict[str, Any]:
    return {
        "http_proxy": os.getenv("http_proxy"),
        "https_proxy": os.getenv("https_proxy"),
        "HTTP_PROXY": os.getenv("HTTP_PROXY"),
        "HTTPS_PROXY": os.getenv("HTTPS_PROXY"),
        "no_proxy": os.getenv("no_proxy"),
        "NO_PROXY": os.getenv("NO_PROXY"),
    }

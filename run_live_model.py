from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="15m stacking Polymarket live trader")
    p.add_argument("--config", default="trade_config.yaml", help="配置文件路径")
    p.add_argument("--run-once", action="store_true", help="只执行一轮信号与下单")
    p.add_argument("--shadow", action="store_true", help="强制不下单，仅记录日志")
    p.add_argument("--proxy", default=None, help="同时设置 HTTP/HTTPS 代理，例如 http://127.0.0.1:7890")
    p.add_argument("--http-proxy", default=None, help="仅设置 HTTP 代理")
    p.add_argument("--https-proxy", default=None, help="仅设置 HTTPS 代理")
    p.add_argument("--no-proxy", default="localhost,127.0.0.1", help="NO_PROXY，默认 localhost,127.0.0.1")
    return p


def _setup_proxy_env(*, proxy: str | None, http_proxy: str | None, https_proxy: str | None, no_proxy: str) -> None:
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


def main() -> int:
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root / "src"))

    from polymarket_trade.config import load_trade_config
    from polymarket_trade.model_artifact import ensure_model_artifact
    from polymarket_trade.runner import run_live_loop

    args = build_parser().parse_args()
    _setup_proxy_env(
        proxy=args.proxy,
        http_proxy=args.http_proxy,
        https_proxy=args.https_proxy,
        no_proxy=str(args.no_proxy),
    )
    cfg = load_trade_config(args.config)
    ensure_model_artifact(root, cfg)
    if args.shadow:
        from dataclasses import replace

        cfg = replace(cfg, live_enabled=False)
    run_live_loop(cfg, run_once=bool(args.run_once))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


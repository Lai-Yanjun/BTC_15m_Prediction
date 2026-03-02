from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Shadow runner (always no real orders)")
    p.add_argument("--config", default="trade_config.yaml", help="配置文件路径")
    p.add_argument("--run-once", action="store_true", help="只执行一轮信号与日志")
    p.add_argument("--proxy", default=None, help="同时设置 HTTP/HTTPS 代理，例如 http://127.0.0.1:7897")
    p.add_argument("--http-proxy", default=None, help="仅设置 HTTP 代理")
    p.add_argument("--https-proxy", default=None, help="仅设置 HTTPS 代理")
    p.add_argument("--no-proxy", default="localhost,127.0.0.1", help="NO_PROXY，默认 localhost,127.0.0.1")
    return p


def main() -> int:
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root / "src"))

    from polymarket_trade.config import load_trade_config
    from polymarket_trade.model_artifact import ensure_model_artifact
    from polymarket_trade.proxy_env import inject_proxy_env
    from polymarket_trade.runner import run_live_loop

    args = build_parser().parse_args()
    inject_proxy_env(
        proxy=args.proxy,
        http_proxy=args.http_proxy,
        https_proxy=args.https_proxy,
        no_proxy=str(args.no_proxy),
    )
    cfg = load_trade_config(args.config)
    ensure_model_artifact(root, cfg)

    # Shadow 模式强制关闭真实下单，避免误操作。
    cfg_shadow = replace(cfg, live_enabled=False)
    run_live_loop(cfg_shadow, run_once=bool(args.run_once))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

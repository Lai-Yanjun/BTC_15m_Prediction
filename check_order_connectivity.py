from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import requests


def _contains_geo_block(msg: str) -> bool:
    txt = (msg or "").lower()
    return ("403" in txt and "geo" in txt) or ("geoblock" in txt) or ("geo_block_403" in txt)


def _fetch_text(url: str, timeout_sec: float) -> str:
    with requests.Session() as sess:
        sess.trust_env = True
        resp = sess.get(url, timeout=float(timeout_sec), headers={"user-agent": "order-connectivity-check/1.0"})
    resp.raise_for_status()
    return (resp.text or "").strip()


def _fetch_json(url: str, timeout_sec: float) -> Any:
    with requests.Session() as sess:
        sess.trust_env = True
        resp = sess.get(
            url,
            timeout=float(timeout_sec),
            headers={"accept": "application/json", "user-agent": "order-connectivity-check/1.0"},
        )
    resp.raise_for_status()
    return resp.json()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Check proxy egress and Polymarket order connectivity")
    p.add_argument("--config", default="trade_config.yaml", help="配置文件路径")
    p.add_argument("--proxy", default=None, help="同时设置 HTTP/HTTPS 代理，例如 http://127.0.0.1:7897")
    p.add_argument("--http-proxy", default=None, help="仅设置 HTTP 代理")
    p.add_argument("--https-proxy", default=None, help="仅设置 HTTPS 代理")
    p.add_argument("--no-proxy", default="localhost,127.0.0.1", help="NO_PROXY，默认 localhost,127.0.0.1")
    p.add_argument("--timeout-sec", type=float, default=12.0, help="HTTP 请求超时秒数")
    p.add_argument("--test-order", action="store_true", help="执行真实小额下单+撤单")
    p.add_argument("--order-usdc", type=float, default=1.0, help="测试下单金额（USDC）")
    p.add_argument("--token-id", default="", help="测试 token id，留空时优先用 trade.token_up")
    p.add_argument("--limit-price", type=float, default=0.5, help="测试限价")
    p.add_argument("--cancel-after-sec", type=int, default=2, help="下单后多少秒撤单")
    return p


def main() -> int:
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root / "src"))

    from polymarket_trade.config import load_trade_config
    from polymarket_trade.executor import PolymarketExecutor
    from polymarket_trade.proxy_env import current_proxy_env, inject_proxy_env

    args = build_parser().parse_args()
    inject_proxy_env(
        proxy=args.proxy,
        http_proxy=args.http_proxy,
        https_proxy=args.https_proxy,
        no_proxy=str(args.no_proxy),
    )

    report: dict[str, Any] = {
        "ok": True,
        "proxy_env": current_proxy_env(),
        "egress_ip": None,
        "data_api": None,
        "clob": None,
        "order_test": None,
        "error_code": None,
    }
    final_rc = 0

    try:
        report["egress_ip"] = _fetch_text("https://ifconfig.me/ip", float(args.timeout_sec))
    except Exception as e:
        report["ok"] = False
        report["egress_ip"] = {"ok": False, "error": f"{type(e).__name__}:{e}"}
        final_rc = 1

    try:
        data = _fetch_json("https://data-api.polymarket.com/markets?limit=1", float(args.timeout_sec))
        report["data_api"] = {"ok": True, "type": type(data).__name__}
    except Exception as e:
        msg = f"{type(e).__name__}:{e}"
        report["ok"] = False
        report["data_api"] = {"ok": False, "error": msg}
        if _contains_geo_block(msg):
            report["error_code"] = "GEO_BLOCK_403"
            final_rc = 2
        elif final_rc == 0:
            final_rc = 1

    cfg = load_trade_config(args.config)
    token_id = str(args.token_id or cfg.token_up or "").strip()
    if not token_id and cfg.auto_update_15m_market:
        # auto_update 场景下配置常为空，这里给出明确提示，避免误判为网络问题。
        report["clob"] = {"ok": False, "error": "MISSING_TOKEN_ID"}
        if report["ok"]:
            report["ok"] = False
            if final_rc == 0:
                final_rc = 1
    else:
        try:
            executor = PolymarketExecutor(cfg)
            book = executor.client.get_order_book(token_id)
            report["clob"] = {
                "ok": True,
                "token_id": token_id,
                "tick_size": (book.get("tick_size") if isinstance(book, dict) else None),
                "min_order_size": (book.get("min_order_size") if isinstance(book, dict) else None),
            }
        except Exception as e:
            msg = f"{type(e).__name__}:{e}"
            report["ok"] = False
            report["clob"] = {"ok": False, "token_id": token_id, "error": msg}
            if _contains_geo_block(msg):
                report["error_code"] = "GEO_BLOCK_403"
                final_rc = 2
            elif final_rc == 0:
                final_rc = 1

        if args.test_order and report["clob"] and report["clob"].get("ok"):
            try:
                result = executor.buy_token_post_only(
                    token_id=token_id,
                    usdc=float(args.order_usdc),
                    limit_price=float(args.limit_price),
                    post_only=True,
                    cancel_after_sec=max(0, int(args.cancel_after_sec)),
                )
                report["order_test"] = {
                    "ok": bool(result.ok),
                    "order_id": result.order_id,
                    "detail": result.detail,
                }
                if not result.ok:
                    report["ok"] = False
                    if _contains_geo_block(result.detail):
                        report["error_code"] = "GEO_BLOCK_403"
                        final_rc = 2
                    elif final_rc == 0:
                        final_rc = 1
            except Exception as e:
                msg = f"{type(e).__name__}:{e}"
                report["ok"] = False
                report["order_test"] = {"ok": False, "error": msg}
                if _contains_geo_block(msg):
                    report["error_code"] = "GEO_BLOCK_403"
                    final_rc = 2
                elif final_rc == 0:
                    final_rc = 1

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return final_rc


if __name__ == "__main__":
    raise SystemExit(main())

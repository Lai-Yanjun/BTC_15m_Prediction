from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import requests
import yaml


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


def _parse_listish(v) -> list:
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        txt = v.strip()
        if txt.startswith("[") and txt.endswith("]"):
            try:
                arr = json.loads(txt)
                return arr if isinstance(arr, list) else []
            except Exception:
                return []
    return []


def _seconds_per_bar(timeframe: str) -> int:
    tf = (timeframe or "").strip().lower()
    if tf.endswith("m"):
        return int(tf[:-1]) * 60
    raise ValueError(f"Unsupported timeframe: {timeframe}")


def _load_trade_config_min(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError("trade_config.yaml 必须是 YAML mapping")
    market = raw.get("market", {}) or {}
    return {
        "timeframe": str(market.get("timeframe", "15m")),
        "market_slug_prefix": str(market.get("market_slug_prefix", "btc-updown-15m")),
        "gamma_api_base": str(market.get("gamma_api_base", "https://gamma-api.polymarket.com")),
    }


def _fetch_gamma_market(
    *,
    url: str,
    timeout_sec: float,
    retries: int,
    backoff_base_sec: float,
    trust_env_proxy: bool,
) -> tuple[dict, list[dict]]:
    attempt_logs: list[dict] = []
    last_err: Exception | None = None
    for i in range(max(1, int(retries))):
        start = time.perf_counter()
        try:
            with requests.Session() as sess:
                sess.trust_env = bool(trust_env_proxy)
                resp = sess.get(
                    url,
                    timeout=float(timeout_sec),
                    headers={"accept": "application/json", "user-agent": "gamma-connectivity-check/1.0"},
                )
            elapsed_ms = round((time.perf_counter() - start) * 1000.0, 2)
            attempt_logs.append(
                {
                    "attempt": i + 1,
                    "elapsed_ms": elapsed_ms,
                    "status_code": int(resp.status_code),
                }
            )
            if resp.status_code != 200:
                raise RuntimeError(f"GAMMA_HTTP_{resp.status_code}")
            payload = resp.json()
            if not isinstance(payload, dict):
                raise RuntimeError("GAMMA_INVALID_JSON")
            return payload, attempt_logs
        except Exception as e:
            elapsed_ms = round((time.perf_counter() - start) * 1000.0, 2)
            last_err = e
            attempt_logs.append(
                {
                    "attempt": i + 1,
                    "elapsed_ms": elapsed_ms,
                    "error": f"{type(e).__name__}:{e}",
                }
            )
            if i < retries - 1:
                time.sleep(float(backoff_base_sec) * (2**i))
    raise RuntimeError(f"GAMMA_REQUEST_FAILED:{type(last_err).__name__}:{last_err} attempts={attempt_logs}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Test gamma-api connectivity for BTC 15m market mapping.")
    ap.add_argument("--config", default="trade_config.yaml", help="Path to trade config.")
    ap.add_argument("--slot-start-ts", type=int, default=0, help="Market slot start timestamp in UTC seconds.")
    ap.add_argument("--timeout-sec", type=float, default=float(os.getenv("GAMMA_TIMEOUT_SEC", "12")))
    ap.add_argument("--retries", type=int, default=int(os.getenv("GAMMA_RETRIES", "3")))
    ap.add_argument("--backoff-base-sec", type=float, default=float(os.getenv("GAMMA_BACKOFF_BASE_SEC", "0.25")))
    ap.add_argument(
        "--trust-env-proxy",
        type=str,
        default=("true" if _env_bool("GAMMA_TRUST_ENV_PROXY", True) else "false"),
        help="Use true/false. true means honor HTTP(S)_PROXY from env.",
    )
    args = ap.parse_args()

    cfg = _load_trade_config_min(args.config)
    bar_sec = _seconds_per_bar(cfg["timeframe"])
    now_ts = int(datetime.now(timezone.utc).timestamp())
    slot_start_ts = int(args.slot_start_ts) if int(args.slot_start_ts) > 0 else (now_ts // bar_sec) * bar_sec
    slug = f"{cfg['market_slug_prefix']}-{slot_start_ts}"
    url = f"{cfg['gamma_api_base'].rstrip('/')}/markets/slug/{slug}"
    trust_env_proxy = str(args.trust_env_proxy).strip().lower() in {"1", "true", "yes", "on"}

    print(
        json.dumps(
            {
                "checking": "gamma_api",
                "url": url,
                "slot_start_ts": slot_start_ts,
                "slot_start_utc": datetime.fromtimestamp(slot_start_ts, tz=timezone.utc).isoformat(),
                "timeout_sec": float(args.timeout_sec),
                "retries": int(args.retries),
                "backoff_base_sec": float(args.backoff_base_sec),
                "trust_env_proxy": trust_env_proxy,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    try:
        payload, attempt_logs = _fetch_gamma_market(
            url=url,
            timeout_sec=float(args.timeout_sec),
            retries=int(args.retries),
            backoff_base_sec=float(args.backoff_base_sec),
            trust_env_proxy=trust_env_proxy,
        )

        outcomes = [str(x) for x in _parse_listish(payload.get("outcomes"))]
        token_ids = [str(x) for x in _parse_listish(payload.get("clobTokenIds"))]
        up_token = ""
        down_token = ""
        for i, outcome in enumerate(outcomes):
            if i >= len(token_ids):
                break
            if outcome.lower() == "up":
                up_token = token_ids[i]
            elif outcome.lower() == "down":
                down_token = token_ids[i]
        condition_id = str(payload.get("conditionId") or payload.get("condition_id") or "")

        result = {
            "ok": bool(up_token and down_token and condition_id),
            "attempts": attempt_logs,
            "slug": slug,
            "question": str(payload.get("question") or ""),
            "condition_id": condition_id,
            "token_up": up_token,
            "token_down": down_token,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["ok"] else 2
    except Exception as e:
        print(
            json.dumps(
                {
                    "ok": False,
                    "slug": slug,
                    "error": f"{type(e).__name__}:{e}",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())

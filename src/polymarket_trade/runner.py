from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import requests

from polymarket_trade.config import TradeConfig
from polymarket_trade.executor import PolymarketExecutor
from polymarket_trade.signal_model import Stacking15mSignal

REDEEM_POLL_SECONDS = 900  # 固定 15 分钟轮询一次 redeem/余额


@dataclass
class RiskState:
    orders_in_hour: deque[float]
    usdc_today: float
    day_key: str


def _now_day_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _update_risk_state(st: RiskState) -> None:
    now = time.time()
    while st.orders_in_hour and now - st.orders_in_hour[0] > 3600:
        st.orders_in_hour.popleft()
    day_now = _now_day_utc()
    if day_now != st.day_key:
        st.day_key = day_now
        st.usdc_today = 0.0


def _seconds_per_bar(timeframe: str) -> int:
    if timeframe.endswith("m"):
        return int(timeframe[:-1]) * 60
    raise ValueError(f"Unsupported timeframe: {timeframe}")


def _parse_expiry_utc(s: str) -> datetime | None:
    txt = (s or "").strip()
    if not txt:
        return None
    if txt.endswith("Z"):
        txt = txt[:-1] + "+00:00"
    dt = datetime.fromisoformat(txt)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _calc_order_usdc(cfg: TradeConfig, equity_usdc: float) -> float:
    if not cfg.dynamic_sizing:
        return float(cfg.order_usdc)
    base = max(0.0, float(equity_usdc)) * max(0.0, float(cfg.equity_ratio))
    base = max(float(cfg.min_order_usdc), min(float(cfg.max_order_usdc), float(base)))
    return float(base)


def _run_claim_command(command: str, timeout_sec: int) -> dict:
    if not command.strip():
        return {"ok": False, "reason": "EMPTY_CLAIM_COMMAND"}
    try:
        completed = subprocess.run(
            command,
            shell=True,
            check=False,
            timeout=max(1, int(timeout_sec)),
            capture_output=True,
            text=True,
        )
        stdout = (completed.stdout or "").strip()
        stderr = (completed.stderr or "").strip()
        return {
            "ok": completed.returncode == 0,
            "returncode": int(completed.returncode),
            "stdout": stdout[-1000:],
            "stderr": stderr[-1000:],
        }
    except Exception as e:
        return {"ok": False, "reason": f"CLAIM_CMD_ERROR:{type(e).__name__}", "detail": repr(e)}


def _build_claim_command(command: str, condition_id: str) -> str:
    base = (command or "").strip()
    cid = (condition_id or "").strip()
    if not base:
        return ""
    if not cid:
        return base
    if "{condition_id}" in base:
        return base.replace("{condition_id}", cid)
    # 默认给结算脚本补上 condition id，避免配置里 condition_id 为空时自动 redeem 失败
    return f"{base} --condition-id {shlex.quote(cid)}"


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


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


def _gamma_get_json(url: str) -> dict:
    timeout_sec = float(os.getenv("GAMMA_TIMEOUT_SEC", "12"))
    retries = max(1, int(os.getenv("GAMMA_RETRIES", "3")))
    backoff_base_sec = float(os.getenv("GAMMA_BACKOFF_BASE_SEC", "0.25"))
    trust_env_proxy = _env_bool("GAMMA_TRUST_ENV_PROXY", True)

    last_err: Exception | None = None
    for i in range(retries):
        try:
            with requests.Session() as sess:
                sess.trust_env = trust_env_proxy
                r = sess.get(
                    url,
                    timeout=timeout_sec,
                    headers={"accept": "application/json", "user-agent": "btc-15m-trader/1.0"},
                )
            if r.status_code != 200:
                raise RuntimeError(f"GAMMA_HTTP_{r.status_code}")
            out = r.json()
            if not isinstance(out, dict):
                raise RuntimeError("GAMMA_INVALID_JSON")
            return out
        except Exception as e:
            last_err = e
            if i < retries - 1:
                time.sleep(backoff_base_sec * (2**i))
    raise RuntimeError(f"GAMMA_REQUEST_FAILED:{type(last_err).__name__}:{last_err}")


def _fetch_market_by_slot(cfg: TradeConfig, slot_start_ts: int) -> dict:
    slug = f"{cfg.market_slug_prefix}-{int(slot_start_ts)}"
    url = f"{cfg.gamma_api_base.rstrip('/')}/markets/slug/{slug}"
    m = _gamma_get_json(url)
    outcomes = [str(x) for x in _parse_listish(m.get("outcomes"))]
    token_ids = [str(x) for x in _parse_listish(m.get("clobTokenIds"))]
    if len(outcomes) != len(token_ids) or len(outcomes) < 2:
        raise RuntimeError("INVALID_OUTCOME_TOKEN_MAPPING")
    token_up = ""
    token_down = ""
    for i, outcome in enumerate(outcomes):
        o = outcome.lower()
        if o == "up":
            token_up = token_ids[i]
        elif o == "down":
            token_down = token_ids[i]
    if not token_up or not token_down:
        raise RuntimeError("UP_DOWN_TOKENS_NOT_FOUND")
    condition_id = str(m.get("conditionId") or m.get("condition_id") or "").strip()
    expiry_utc = str(m.get("endDate") or m.get("endDateIso") or "").strip()
    if not condition_id:
        raise RuntimeError("MISSING_CONDITION_ID")
    return {
        "slug": slug,
        "question": str(m.get("question") or ""),
        "condition_id": condition_id,
        "token_up": token_up,
        "token_down": token_down,
        "expiry_utc": expiry_utc,
    }


def run_live_loop(cfg: TradeConfig, *, run_once: bool = False) -> None:
    if (not cfg.auto_update_15m_market) and (not cfg.token_up or not cfg.token_down):
        raise ValueError("trade_config.yaml 里必须配置 token_up / token_down")

    signal = Stacking15mSignal(Path.cwd(), cfg)
    executor = PolymarketExecutor(cfg)
    log_path = Path(cfg.output_jsonl)
    st = RiskState(orders_in_hour=deque(), usdc_today=0.0, day_key=_now_day_utc())
    expiry_utc = _parse_expiry_utc(cfg.market_expiry_utc)
    equity_usdc = float(cfg.initial_equity_usdc)
    settlement_done = False
    settlement_last_error: str | None = None
    next_claim_poll_ts = 0.0
    last_claim_result: dict | None = None
    current_condition_id = cfg.condition_id
    current_token_up = cfg.token_up
    current_token_down = cfg.token_down
    current_market_slug = ""
    current_market_question = ""
    current_market_error = None
    last_market_slot_start_ts: int | None = None
    if cfg.live_enabled:
        try:
            equity_usdc = float(executor.get_collateral_balance_usdc())
        except Exception as e:
            settlement_last_error = f"INIT_BALANCE_ERROR:{type(e).__name__}"

    last_trigger_close_ts: int | None = None
    bar_sec = _seconds_per_bar(cfg.timeframe)
    while True:
        now_ts = time.time()
        now_dt = datetime.now(timezone.utc)
        expiry_passed = bool(expiry_utc and now_dt.timestamp() >= (expiry_utc.timestamp() + float(cfg.settlement_grace_sec)))
        trigger_close_ts = (int(now_ts) // bar_sec) * bar_sec

        if (not run_once) and (not expiry_passed):
            if last_trigger_close_ts is None:
                # 首次启动时，跳过“刚过去的收盘点”，避免对历史 K 线立即下单。
                last_trigger_close_ts = trigger_close_ts
            if trigger_close_ts <= last_trigger_close_ts:
                next_close_ts = last_trigger_close_ts + bar_sec
                time.sleep(min(float(cfg.poll_seconds), max(0.2, next_close_ts - now_ts)))
                continue

        market_slot_start_ts = trigger_close_ts
        if cfg.auto_update_15m_market and (last_market_slot_start_ts != market_slot_start_ts):
            try:
                md = _fetch_market_by_slot(cfg, market_slot_start_ts)
                current_market_slug = str(md["slug"])
                current_market_question = str(md["question"])
                current_condition_id = str(md["condition_id"])
                current_token_up = str(md["token_up"])
                current_token_down = str(md["token_down"])
                expiry_utc = _parse_expiry_utc(str(md["expiry_utc"]))
                current_market_error = None
            except Exception as e:
                current_market_error = f"AUTO_MARKET_ERROR:{type(e).__name__}:{e}"
            last_market_slot_start_ts = market_slot_start_ts

        _update_risk_state(st)

        if expiry_passed:
            action = "HOLD"
            token_id = ""
            side_prob = 0.0
            risk_block = "MARKET_EXPIRED_WAIT_SETTLEMENT"
            exec_result = None
            up_prob = None
            down_prob = None
            claim_polled = False

            if cfg.live_enabled and not settlement_done:
                if now_ts >= next_claim_poll_ts:
                    claim_polled = True
                    next_claim_poll_ts = now_ts + float(REDEEM_POLL_SECONDS)
                    if cfg.auto_claim_enabled:
                        claim_command = _build_claim_command(cfg.claim_command, current_condition_id)
                        last_claim_result = _run_claim_command(claim_command, cfg.claim_timeout_sec)
                        if not last_claim_result.get("ok", False):
                            settlement_last_error = str(last_claim_result.get("reason", "CLAIM_COMMAND_FAILED"))
                        else:
                            settlement_last_error = None
                    try:
                        bal = float(executor.get_collateral_balance_usdc())
                        changed = abs(bal - equity_usdc) > 1e-8
                        equity_usdc = bal
                        settlement_done = changed
                    except Exception as e:
                        settlement_last_error = f"SETTLEMENT_BALANCE_ERROR:{type(e).__name__}"
        else:
            sig = signal.predict_for_close_ts(trigger_close_ts)
            up_prob = float(sig.up_prob)
            down_prob = float(sig.down_prob)
            claim_polled = False

            action = "HOLD"
            token_id = ""
            side_prob = 0.0
            if current_market_error:
                risk_block = current_market_error
            elif up_prob >= cfg.upper_threshold:
                action = "BUY_UP"
                token_id = current_token_up
                side_prob = up_prob
            elif up_prob <= cfg.lower_threshold:
                action = "BUY_DOWN"
                token_id = current_token_down
                side_prob = down_prob

            order_usdc = _calc_order_usdc(cfg, equity_usdc)

            risk_block = None if not current_market_error else current_market_error
            if action != "HOLD":
                if len(st.orders_in_hour) >= cfg.max_orders_per_hour:
                    risk_block = "MAX_ORDERS_PER_HOUR"
                elif st.usdc_today + order_usdc > cfg.max_usdc_per_day:
                    risk_block = "MAX_USDC_PER_DAY"

            exec_result = None
            if action != "HOLD" and risk_block is None and cfg.live_enabled:
                # 保留买入边际：默认最高 0.51，置信度 > 0.60 时最高 0.52。
                prob_cap = 0.52 if side_prob > 0.60 else 0.51
                limit_px = min(cfg.max_price, max(cfg.min_price, side_prob), prob_cap)
                exec_result = executor.buy_token_post_only(
                    token_id=token_id,
                    usdc=order_usdc,
                    limit_price=limit_px,
                    post_only=cfg.post_only,
                    cancel_after_sec=cfg.cancel_after_sec,
                )
                if exec_result.ok:
                    st.orders_in_hour.append(time.time())
                    st.usdc_today += order_usdc

        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "candle_ts": None if expiry_passed else str(sig.candle_ts),
            "up_prob": None if up_prob is None else round(up_prob, 6),
            "down_prob": None if down_prob is None else round(down_prob, 6),
            "upper_threshold": cfg.upper_threshold,
            "lower_threshold": cfg.lower_threshold,
            "action": action,
            "token_id": token_id,
            "risk_block": risk_block,
            "live_enabled": cfg.live_enabled,
            "order_usdc": _calc_order_usdc(cfg, equity_usdc),
            "equity_usdc": round(float(equity_usdc), 6),
            "equity_ratio": cfg.equity_ratio,
            "dynamic_sizing": cfg.dynamic_sizing,
            "market_expiry_utc": cfg.market_expiry_utc,
            "market_slug": current_market_slug,
            "market_question": current_market_question,
            "condition_id": current_condition_id,
            "auto_update_15m_market": cfg.auto_update_15m_market,
            "auto_market_error": current_market_error,
            "settlement_done": settlement_done,
            "settlement_last_error": settlement_last_error,
            "claim_poll_seconds": REDEEM_POLL_SECONDS,
            "claim_polled": claim_polled,
            "last_claim_result": last_claim_result,
            "exec": None
            if exec_result is None
            else {
                "ok": exec_result.ok,
                "order_id": exec_result.order_id,
                "detail": exec_result.detail,
            },
        }
        _append_jsonl(log_path, record)
        print(record)
        last_trigger_close_ts = trigger_close_ts

        if run_once:
            return
        if expiry_passed:
            time.sleep(min(1.0, float(cfg.poll_seconds)))
        else:
            time.sleep(0.2)


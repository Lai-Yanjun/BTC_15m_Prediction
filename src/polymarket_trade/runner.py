from __future__ import annotations

import json
import subprocess
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from polymarket_trade.config import TradeConfig
from polymarket_trade.executor import PolymarketExecutor
from polymarket_trade.signal_model import Stacking15mSignal


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


def run_live_loop(cfg: TradeConfig, *, run_once: bool = False) -> None:
    if not cfg.token_up or not cfg.token_down:
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
        next_close_ts = (int(now_ts) // bar_sec + 1) * bar_sec
        trigger_ts = next_close_ts - 1  # 收盘前 1 秒触发

        if (not run_once) and (not expiry_passed):
            if now_ts < trigger_ts:
                time.sleep(min(float(cfg.poll_seconds), max(0.2, trigger_ts - now_ts)))
                continue
            if last_trigger_close_ts == next_close_ts:
                time.sleep(0.2)
                continue

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
                    next_claim_poll_ts = now_ts + float(max(60, cfg.claim_poll_seconds))
                    if cfg.auto_claim_enabled:
                        last_claim_result = _run_claim_command(cfg.claim_command, cfg.claim_timeout_sec)
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
            sig = signal.predict_latest()
            up_prob = float(sig.up_prob)
            down_prob = float(sig.down_prob)
            claim_polled = False

            action = "HOLD"
            token_id = ""
            side_prob = 0.0
            if up_prob >= cfg.upper_threshold:
                action = "BUY_UP"
                token_id = cfg.token_up
                side_prob = up_prob
            elif up_prob <= cfg.lower_threshold:
                action = "BUY_DOWN"
                token_id = cfg.token_down
                side_prob = down_prob

            order_usdc = _calc_order_usdc(cfg, equity_usdc)

            risk_block = None
            if action != "HOLD":
                if len(st.orders_in_hour) >= cfg.max_orders_per_hour:
                    risk_block = "MAX_ORDERS_PER_HOUR"
                elif st.usdc_today + order_usdc > cfg.max_usdc_per_day:
                    risk_block = "MAX_USDC_PER_DAY"

            exec_result = None
            if action != "HOLD" and risk_block is None and cfg.live_enabled:
                limit_px = min(cfg.max_price, max(cfg.min_price, side_prob))
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
            "settlement_done": settlement_done,
            "settlement_last_error": settlement_last_error,
            "claim_poll_seconds": cfg.claim_poll_seconds,
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
        last_trigger_close_ts = next_close_ts

        if run_once:
            return
        if expiry_passed:
            time.sleep(min(1.0, float(cfg.poll_seconds)))
        else:
            time.sleep(0.2)


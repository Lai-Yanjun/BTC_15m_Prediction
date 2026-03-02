from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TradeConfig:
    symbol: str
    timeframe: str
    lookback_bars: int
    condition_id: str
    neg_risk: bool
    upper_threshold: float
    lower_threshold: float
    order_usdc: float
    dynamic_sizing: bool
    equity_ratio: float
    min_order_usdc: float
    max_order_usdc: float
    initial_equity_usdc: float
    poll_seconds: int
    output_jsonl: str

    token_up: str
    token_down: str

    live_enabled: bool
    auto_train_if_missing: bool
    max_orders_per_hour: int
    max_usdc_per_day: float
    slippage: float
    min_price: float
    max_price: float
    post_only: bool
    cancel_after_sec: int
    market_expiry_utc: str
    settlement_grace_sec: int
    auto_claim_enabled: bool
    claim_poll_seconds: int
    claim_command: str
    claim_timeout_sec: int

    model_opt_json: str
    model_dir: str


def _get(d: dict[str, Any], key: str, default: Any = None) -> Any:
    return d[key] if key in d else default


def load_trade_config(path: str | Path) -> TradeConfig:
    p = Path(path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("trade_config.yaml 必须是 YAML mapping")

    market = raw.get("market", {}) or {}
    signal = raw.get("signal", {}) or {}
    trade = raw.get("trade", {}) or {}
    risk = raw.get("risk", {}) or {}
    paths = raw.get("paths", {}) or {}

    return TradeConfig(
        symbol=str(_get(market, "symbol", "BTC/USDT")),
        timeframe=str(_get(market, "timeframe", "15m")),
        lookback_bars=int(_get(market, "lookback_bars", 1200)),
        condition_id=str(_get(market, "condition_id", "")),
        neg_risk=bool(_get(market, "neg_risk", False)),
        upper_threshold=float(_get(signal, "upper_threshold", 0.60)),
        lower_threshold=float(_get(signal, "lower_threshold", 0.40)),
        order_usdc=float(_get(trade, "order_usdc", 5.0)),
        dynamic_sizing=bool(_get(trade, "dynamic_sizing", True)),
        equity_ratio=float(_get(trade, "equity_ratio", 0.085)),
        min_order_usdc=float(_get(trade, "min_order_usdc", 2.0)),
        max_order_usdc=float(_get(trade, "max_order_usdc", 100.0)),
        initial_equity_usdc=float(_get(trade, "initial_equity_usdc", 100.0)),
        poll_seconds=int(_get(trade, "poll_seconds", 30)),
        output_jsonl=str(_get(paths, "output_jsonl", "reports/live_trade_log.jsonl")),
        token_up=str(_get(trade, "token_up", "")),
        token_down=str(_get(trade, "token_down", "")),
        live_enabled=bool(_get(trade, "live_enabled", True)),
        auto_train_if_missing=bool(_get(trade, "auto_train_if_missing", True)),
        max_orders_per_hour=int(_get(risk, "max_orders_per_hour", 20)),
        max_usdc_per_day=float(_get(risk, "max_usdc_per_day", 200)),
        slippage=float(_get(risk, "slippage", 0.03)),
        min_price=float(_get(risk, "min_price", 0.05)),
        max_price=float(_get(risk, "max_price", 0.95)),
        post_only=bool(_get(trade, "post_only", True)),
        cancel_after_sec=int(_get(trade, "cancel_after_sec", 5)),
        market_expiry_utc=str(_get(market, "expiry_utc", "")),
        settlement_grace_sec=int(_get(trade, "settlement_grace_sec", 7200)),
        auto_claim_enabled=bool(_get(trade, "auto_claim_enabled", False)),
        claim_poll_seconds=int(_get(trade, "claim_poll_seconds", 900)),
        claim_command=str(_get(trade, "claim_command", "python run_settlement.py --config trade_config.yaml")),
        claim_timeout_sec=int(_get(trade, "claim_timeout_sec", 300)),
        model_opt_json=str(_get(paths, "model_opt_json", "outputs/opt_15m_details.json")),
        model_dir=str(_get(paths, "model_dir", "models/fixed_15m_stacking")),
    )


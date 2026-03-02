from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class PredictaConfig:
    atr_length: int = 14
    st_factor: float = 3.0
    st_period: int = 10
    min_confluence: int = 5
    min_volume_ratio: float = 0.8
    adx_threshold: int = 25
    adx_len: int = 14
    percentile_window: int = 100
    stoch_len: int = 14
    stoch_smooth: int = 3
    rsi_len: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    vol_sma_len: int = 20
    avg_body_len: int = 5


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def _atr(df: pd.DataFrame, length: int) -> pd.Series:
    tr = _true_range(df)
    return tr.ewm(alpha=1 / length, adjust=False).mean()


def _dmi_adx(df: pd.DataFrame, length: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    high = df["high"]
    low = df["low"]
    prev_high = high.shift(1)
    prev_low = low.shift(1)

    up_move = high - prev_high
    down_move = prev_low - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)
    atr_val = _atr(df, length).replace(0, np.nan)

    plus_di = 100 * plus_dm.ewm(alpha=1 / length, adjust=False).mean() / atr_val
    minus_di = 100 * minus_dm.ewm(alpha=1 / length, adjust=False).mean() / atr_val

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / length, adjust=False).mean()
    return plus_di.fillna(0), minus_di.fillna(0), adx.fillna(0)


def _stoch_k(df: pd.DataFrame, length: int = 14) -> pd.Series:
    lowest = df["low"].rolling(length, min_periods=1).min()
    highest = df["high"].rolling(length, min_periods=1).max()
    denom = (highest - lowest).replace(0, np.nan)
    k = 100 * (df["close"] - lowest) / denom
    return k.fillna(50.0)


def _macd(close: pd.Series, fast: int, slow: int, signal: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    macd_line = _ema(close, fast) - _ema(close, slow)
    signal_line = _ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def _rolling_percentrank(series: pd.Series, window: int) -> pd.Series:
    def _pct(x: np.ndarray) -> float:
        valid = x[~np.isnan(x)]
        if valid.size == 0:
            return np.nan
        return 100 * (np.sum(valid <= valid[-1]) / valid.size)

    return series.rolling(window=window, min_periods=1).apply(_pct, raw=True)


def _custom_supertrend(
    df: pd.DataFrame,
    st_factor: float,
    st_period: int,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    trend_atr = _atr(df, st_period).to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()

    hl2 = (high + low) / 2.0
    upper_raw = hl2 + st_factor * trend_atr
    lower_raw = hl2 - st_factor * trend_atr

    n = len(df)
    upper_band = np.full(n, np.nan)
    lower_band = np.full(n, np.nan)
    trend_direction = np.ones(n, dtype=int)

    for i in range(n):
        prev_upper = upper_band[i - 1] if i > 0 and not np.isnan(upper_band[i - 1]) else upper_raw[i]
        prev_lower = lower_band[i - 1] if i > 0 and not np.isnan(lower_band[i - 1]) else lower_raw[i]
        prev_direction = trend_direction[i - 1] if i > 0 else 1
        close_prev = close[i - 1] if i > 0 else close[i]

        lower_band[i] = max(lower_raw[i], prev_lower) if close_prev > prev_lower else lower_raw[i]
        upper_band[i] = min(upper_raw[i], prev_upper) if close_prev < prev_upper else upper_raw[i]

        if prev_direction == -1:
            trend_direction[i] = 1 if close[i] < lower_band[i] else -1
        else:
            trend_direction[i] = -1 if close[i] > upper_band[i] else 1

    trend_line = np.where(trend_direction == -1, lower_band, upper_band)
    is_uptrend = trend_direction == -1
    is_downtrend = trend_direction == 1
    trend_changed = pd.Series(trend_direction, index=df.index).diff().fillna(0).ne(0)

    return (
        pd.Series(trend_line, index=df.index),
        pd.Series(is_uptrend, index=df.index),
        pd.Series(is_downtrend, index=df.index),
        trend_changed,
    )


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"OHLCV 缺少字段: {missing}")

    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out = out.sort_values("timestamp").drop_duplicates("timestamp")
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=required[1:])
    out = out[(out["high"] >= out["low"]) & (out["volume"] >= 0)]
    out = out.reset_index(drop=True)
    return out


def download_binance_spot_ohlcv(
    symbol: str = "BTC/USDT",
    timeframe: str = "5m",
    start_utc: str = "2024-01-01",
    max_bars: int = 300_000,
    out_csv: Optional[str] = None,
) -> pd.DataFrame:
    import ccxt

    exchange = ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "spot"}})
    since = exchange.parse8601(f"{start_utc}T00:00:00Z")
    step_ms = exchange.parse_timeframe(timeframe) * 1000
    limit = 1000
    rows: list[list[float]] = []

    while True:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < limit or len(rows) >= max_bars:
            break
        since = batch[-1][0] + step_ms

    data = rows[:max_bars]
    df = pd.DataFrame(data, columns=["timestamp_ms", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    df = df.drop(columns=["timestamp_ms"])
    df = clean_ohlcv(df)

    if out_csv:
        p = Path(out_csv)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False)
    return df


def compute_predicta_v4(df: pd.DataFrame, cfg: PredictaConfig | None = None) -> pd.DataFrame:
    cfg = cfg or PredictaConfig()
    data = clean_ohlcv(df).copy()

    close = data["close"]
    open_ = data["open"]
    high = data["high"]
    low = data["low"]
    volume = data["volume"]

    trend_line, is_uptrend, is_downtrend, trend_changed = _custom_supertrend(data, cfg.st_factor, cfg.st_period)

    candle_range = (high - low).replace(0, np.nan)
    buy_volume = np.where(candle_range > 0, volume * (close - low) / candle_range, volume * 0.5)
    sell_volume = np.where(candle_range > 0, volume * (high - close) / candle_range, volume * 0.5)
    volume_delta = pd.Series(buy_volume - sell_volume, index=data.index).fillna(0.0)
    delta_ema = _ema(volume_delta, 10)
    delta_momentum = volume_delta > delta_ema
    delta_bullish = volume_delta > 0
    delta_bearish = volume_delta < 0

    atr = _atr(data, cfg.atr_length)
    ema8 = _ema(close, 8)
    ema21 = _ema(close, 21)
    ema50 = _ema(close, 50)

    rsi = _rsi(close, cfg.rsi_len)
    rsi_above_50 = rsi > 50
    rsi_below_50 = rsi < 50

    macd_line, signal_line, macd_hist = _macd(close, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal)

    stoch_k = _stoch_k(data, cfg.stoch_len)
    stoch_d = stoch_k.rolling(cfg.stoch_smooth, min_periods=1).mean()

    vol_sma = volume.rolling(cfg.vol_sma_len, min_periods=1).mean()
    vol_ratio = (volume / vol_sma.replace(0, np.nan)).fillna(1.0)
    avg_body = (close - open_).abs().rolling(cfg.avg_body_len, min_periods=1).mean()

    di_plus, di_minus, adx_value = _dmi_adx(data, cfg.adx_len)

    atr_percentile = _rolling_percentrank(atr, cfg.percentile_window)
    vol_regime = np.select(
        [atr_percentile > 75, atr_percentile < 25],
        ["HIGH", "LOW"],
        default="MEDIUM",
    )
    vol_multiplier = np.select(
        [atr_percentile > 75, atr_percentile < 25],
        [0.85, 1.15],
        default=1.0,
    )
    dynamic_threshold = np.select(
        [adx_value > 30, adx_value > 25, adx_value > 20],
        [55, 60, 65],
        default=70,
    )

    macd_score_long = np.select(
        [(macd_line > signal_line) & (macd_hist > 0), macd_line > signal_line, macd_hist > 0],
        [100, 70, 50],
        default=20,
    )
    macd_score_short = np.select(
        [(macd_line < signal_line) & (macd_hist < 0), macd_line < signal_line, macd_hist < 0],
        [100, 70, 50],
        default=20,
    )

    rsi_score_long = np.select(
        [rsi < 30, rsi < 40, rsi < 50, rsi < 60],
        [100, 85, 70, 50],
        default=25,
    )
    rsi_score_short = np.select(
        [rsi > 70, rsi > 60, rsi > 50, rsi > 40],
        [100, 85, 70, 50],
        default=25,
    )

    stoch_score_long = np.select(
        [(stoch_k > stoch_d) & (stoch_k < 20), (stoch_k > stoch_d) & (stoch_k < 50), stoch_k > stoch_d],
        [100, 85, 65],
        default=25,
    )
    stoch_score_short = np.select(
        [(stoch_k < stoch_d) & (stoch_k > 80), (stoch_k < stoch_d) & (stoch_k > 50), stoch_k < stoch_d],
        [100, 85, 65],
        default=25,
    )

    vol_score = np.select(
        [vol_ratio > 2.0, vol_ratio > 1.5, vol_ratio > 1.0, vol_ratio > 0.8],
        [100, 80, 60, 45],
        default=25,
    )

    delta_score_long = np.select(
        [(volume_delta > 0) & delta_momentum, volume_delta > 0, volume_delta > -delta_ema.abs()],
        [100, 75, 40],
        default=20,
    )
    delta_score_short = np.select(
        [(volume_delta < 0) & (~delta_momentum), volume_delta < 0, volume_delta < delta_ema.abs()],
        [100, 75, 40],
        default=20,
    )

    adx_score = np.select(
        [adx_value > 35, adx_value > 30, adx_value > 25, adx_value > 20],
        [100, 85, 70, 50],
        default=30,
    )

    trend_score_long = np.select(
        [is_uptrend & (ema8 > ema21) & (ema21 > ema50), is_uptrend & (ema8 > ema21), is_uptrend],
        [100, 80, 60],
        default=0,
    )
    trend_score_short = np.select(
        [is_downtrend & (ema8 < ema21) & (ema21 < ema50), is_downtrend & (ema8 < ema21), is_downtrend],
        [100, 80, 60],
        default=0,
    )

    long_score_raw = (
        trend_score_long * 0.23
        + macd_score_long * 0.18
        + delta_score_long * 0.15
        + rsi_score_long * 0.12
        + stoch_score_long * 0.12
        + adx_score * 0.10
        + vol_score * 0.10
    )
    short_score_raw = (
        trend_score_short * 0.23
        + macd_score_short * 0.18
        + delta_score_short * 0.15
        + rsi_score_short * 0.12
        + stoch_score_short * 0.12
        + adx_score * 0.10
        + vol_score * 0.10
    )

    long_score = np.clip(np.round(long_score_raw * vol_multiplier), 0, 100)
    short_score = np.clip(np.round(short_score_raw * vol_multiplier), 0, 100)
    total_raw = long_score + short_score
    long_pct = np.where(total_raw > 0, np.round(long_score / total_raw * 100), 50.0)
    short_pct = 100 - long_pct

    confluence_long = (
        is_uptrend.astype(int)
        + (ema8 > ema21).astype(int)
        + (macd_line > signal_line).astype(int)
        + (stoch_k > stoch_d).astype(int)
        + (vol_ratio >= cfg.min_volume_ratio).astype(int)
        + (adx_value > cfg.adx_threshold).astype(int)
        + rsi_above_50.astype(int)
        + delta_bullish.astype(int)
    )
    confluence_short = (
        is_downtrend.astype(int)
        + (ema8 < ema21).astype(int)
        + (macd_line < signal_line).astype(int)
        + (stoch_k < stoch_d).astype(int)
        + (vol_ratio >= cfg.min_volume_ratio).astype(int)
        + (adx_value > cfg.adx_threshold).astype(int)
        + rsi_below_50.astype(int)
        + delta_bearish.astype(int)
    )

    volume_ok = vol_ratio >= cfg.min_volume_ratio
    long_perfect = (
        is_uptrend
        & (long_pct >= dynamic_threshold)
        & (confluence_long >= cfg.min_confluence)
        & volume_ok
        & rsi_above_50
        & delta_bullish
    )
    short_perfect = (
        is_downtrend
        & (short_pct >= dynamic_threshold)
        & (confluence_short >= cfg.min_confluence)
        & volume_ok
        & rsi_below_50
        & delta_bearish
    )

    bull_signal = (ema8 > ema21) & (ema8.shift(1) <= ema21.shift(1)) & is_uptrend & delta_bullish
    bear_signal = (ema8 < ema21) & (ema8.shift(1) >= ema21.shift(1)) & is_downtrend & delta_bearish

    out = data.copy()
    out["atr"] = atr
    out["trend_line"] = trend_line
    out["is_uptrend"] = is_uptrend
    out["is_downtrend"] = is_downtrend
    out["trend_changed"] = trend_changed
    out["volume_delta"] = volume_delta
    out["delta_ema"] = delta_ema
    out["delta_momentum"] = delta_momentum
    out["rsi"] = rsi
    out["macd_line"] = macd_line
    out["signal_line"] = signal_line
    out["macd_hist"] = macd_hist
    out["stoch_k"] = stoch_k
    out["stoch_d"] = stoch_d
    out["vol_ratio"] = vol_ratio
    out["avg_body"] = avg_body
    out["di_plus"] = di_plus
    out["di_minus"] = di_minus
    out["adx_value"] = adx_value
    out["atr_percentile"] = atr_percentile
    out["vol_regime"] = vol_regime
    out["vol_multiplier"] = vol_multiplier
    out["dynamic_threshold"] = dynamic_threshold
    out["long_score"] = long_score
    out["short_score"] = short_score
    out["long_pct"] = long_pct
    out["short_pct"] = short_pct
    out["confluence_long"] = confluence_long
    out["confluence_short"] = confluence_short
    out["long_perfect"] = long_perfect
    out["short_perfect"] = short_perfect
    out["bull_signal"] = bull_signal
    out["bear_signal"] = bear_signal

    out["next_close"] = out["close"].shift(-1)
    out["y_true"] = (out["next_close"] > out["close"]).astype(int)
    out["long_prob"] = out["long_pct"] / 100.0
    out["pred_dir"] = (out["long_prob"] >= 0.5).astype(int)
    out["is_flat_next"] = (out["next_close"] == out["close"]).astype(int)

    warmup = max(cfg.atr_length, cfg.st_period, 50, cfg.percentile_window)
    out["is_valid"] = False
    out.loc[warmup : len(out) - 2, "is_valid"] = True

    return out


def split_train_valid(df: pd.DataFrame, train_ratio: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio 必须在 (0,1) 区间。")
    n = len(df)
    cut = int(n * train_ratio)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()

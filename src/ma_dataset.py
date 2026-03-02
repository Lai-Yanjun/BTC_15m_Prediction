from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from predicta_v4 import clean_ohlcv, download_binance_spot_ohlcv


@dataclass
class SplitConfig:
    test_days: int = 30
    valid_days: int = 30
    start_utc: str = "2024-01-01"


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)


def _atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()


def _stoch(df: pd.DataFrame, length: int = 14, smooth: int = 3) -> tuple[pd.Series, pd.Series]:
    lo = df["low"].rolling(length, min_periods=1).min()
    hi = df["high"].rolling(length, min_periods=1).max()
    k = 100 * (df["close"] - lo) / (hi - lo).replace(0, np.nan)
    k = k.fillna(50.0)
    d = k.rolling(smooth, min_periods=1).mean()
    return k, d


def _macd(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = _ema(close, 12)
    ema_slow = _ema(close, 26)
    macd_line = ema_fast - ema_slow
    signal = _ema(macd_line, 9)
    hist = macd_line - signal
    return macd_line, signal, hist


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    data = clean_ohlcv(df).copy()

    close = data["close"]
    open_ = data["open"]
    high = data["high"]
    low = data["low"]
    vol = data["volume"]

    # Momentum and return features.
    for lag in (1, 2, 3, 5, 10, 20):
        data[f"ret_{lag}"] = close.pct_change(lag)
        data[f"logret_{lag}"] = np.log(close / close.shift(lag))

    # Candle geometry.
    body = close - open_
    rng = (high - low).replace(0, np.nan)
    data["body_pct"] = body / rng
    data["upper_wick_pct"] = (high - np.maximum(close, open_)) / rng
    data["lower_wick_pct"] = (np.minimum(close, open_) - low) / rng
    data["close_loc_in_range"] = (close - low) / rng
    data["range_pct"] = rng / close.replace(0, np.nan)

    # Trend and distance.
    ema8 = _ema(close, 8)
    ema21 = _ema(close, 21)
    ema50 = _ema(close, 50)
    data["ema8_vs_21"] = (ema8 - ema21) / close.replace(0, np.nan)
    data["ema21_vs_50"] = (ema21 - ema50) / close.replace(0, np.nan)
    data["close_vs_ema8"] = (close - ema8) / close.replace(0, np.nan)
    data["close_vs_ema21"] = (close - ema21) / close.replace(0, np.nan)
    data["ema8_slope_3"] = ema8.pct_change(3)
    data["ema21_slope_5"] = ema21.pct_change(5)

    # Volatility.
    atr14 = _atr(data, 14)
    data["atr_pct"] = atr14 / close.replace(0, np.nan)
    for window in (5, 10, 20):
        data[f"realized_vol_{window}"] = close.pct_change().rolling(window, min_periods=1).std()

    ma20 = close.rolling(20, min_periods=1).mean()
    std20 = close.rolling(20, min_periods=1).std()
    data["bb_width"] = (2 * std20) / ma20.replace(0, np.nan)

    # Volume and pressure proxies.
    vol_ma5 = vol.rolling(5, min_periods=1).mean()
    vol_ma20 = vol.rolling(20, min_periods=1).mean()
    data["vol_ratio_5"] = vol / vol_ma5.replace(0, np.nan)
    data["vol_ratio_20"] = vol / vol_ma20.replace(0, np.nan)
    data["vol_zscore_20"] = (vol - vol_ma20) / vol.rolling(20, min_periods=1).std().replace(0, np.nan)
    data["signed_volume"] = np.sign(body).fillna(0) * vol

    buy_proxy = np.where(rng > 0, vol * (close - low) / rng, vol * 0.5)
    sell_proxy = np.where(rng > 0, vol * (high - close) / rng, vol * 0.5)
    data["volume_delta_proxy"] = buy_proxy - sell_proxy
    data["volume_delta_proxy_norm"] = data["volume_delta_proxy"] / vol.replace(0, np.nan)

    # Oscillators.
    data["rsi14"] = _rsi(close, 14)
    stoch_k, stoch_d = _stoch(data, 14, 3)
    data["stoch_k"] = stoch_k
    data["stoch_d"] = stoch_d
    macd_line, macd_signal, macd_hist = _macd(close)
    data["macd_line"] = macd_line
    data["macd_signal"] = macd_signal
    data["macd_hist"] = macd_hist

    # Regime proxies.
    atr_rank = atr14.rolling(100, min_periods=1).rank(pct=True)
    data["vol_regime_code"] = np.where(atr_rank > 0.75, 2, np.where(atr_rank < 0.25, 0, 1))
    data["trend_regime_code"] = np.where(ema8 > ema21, 1, 0)
    data["confluence_count"] = (
        (ema8 > ema21).astype(int)
        + (data["macd_line"] > data["macd_signal"]).astype(int)
        + (data["rsi14"] > 50).astype(int)
        + (data["stoch_k"] > data["stoch_d"]).astype(int)
        + (data["volume_delta_proxy"] > 0).astype(int)
    )

    # Time features.
    ts = pd.to_datetime(data["timestamp"], utc=True)
    hour = ts.dt.hour.astype(float)
    dow = ts.dt.dayofweek.astype(float)
    data["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    data["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    data["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    data["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    # Label: next-close direction.
    data["next_close"] = close.shift(-1)
    data["y_true"] = (data["next_close"] > close).astype(int)
    data["is_flat_next"] = (data["next_close"] == close).astype(int)

    # Warmup region is removed to avoid unstable rolling metrics.
    warmup = 120
    data["is_valid"] = False
    data.loc[warmup : len(data) - 2, "is_valid"] = True

    return data


def _split_by_time(df: pd.DataFrame, cfg: SplitConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = df.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
    end_ts = d["timestamp"].max()
    test_start = end_ts - pd.Timedelta(days=cfg.test_days)
    valid_start = test_start - pd.Timedelta(days=cfg.valid_days)

    train = d[d["timestamp"] < valid_start].copy()
    valid = d[(d["timestamp"] >= valid_start) & (d["timestamp"] < test_start)].copy()
    test = d[d["timestamp"] >= test_start].copy()
    return train, valid, test


def prepare_timeframe_dataset(
    timeframe: str,
    data_dir: Path,
    cfg: SplitConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or SplitConfig()
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / f"btc_usdt_{timeframe}.csv"

    if csv_path.exists():
        raw = pd.read_csv(csv_path)
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    else:
        raw = download_binance_spot_ohlcv(
            symbol="BTC/USDT",
            timeframe=timeframe,
            start_utc=cfg.start_utc,
            out_csv=str(csv_path),
        )

    feat = build_features(raw)
    feat = feat[feat["is_valid"] & feat["next_close"].notna()].copy()

    train, valid, test = _split_by_time(feat, cfg)

    non_feature_cols = {
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "next_close",
        "y_true",
        "is_flat_next",
        "is_valid",
    }
    feature_cols = [c for c in feat.columns if c not in non_feature_cols]

    # Replace inf and fill with train medians to avoid leakage.
    for frame in (train, valid, test):
        frame[feature_cols] = frame[feature_cols].replace([np.inf, -np.inf], np.nan)
    medians = train[feature_cols].median()
    train[feature_cols] = train[feature_cols].fillna(medians)
    valid[feature_cols] = valid[feature_cols].fillna(medians)
    test[feature_cols] = test[feature_cols].fillna(medians)

    return {
        "raw": raw,
        "features": feat,
        "train": train,
        "valid": valid,
        "test": test,
        "feature_cols": feature_cols,
        "split_cfg": cfg,
    }

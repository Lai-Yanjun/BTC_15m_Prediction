from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd
from joblib import dump, load

from ma_dataset import build_features
from models import fit_stacking_meta, predict_stacking, train_catboost, train_lightgbm, train_logistic
from polymarket_trade.config import TradeConfig


@dataclass
class SignalOutput:
    candle_ts: pd.Timestamp
    up_prob: float
    down_prob: float


class Stacking15mSignal:
    def __init__(self, root: Path, cfg: TradeConfig) -> None:
        self.root = root
        self.cfg = cfg
        self.exchange = ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "spot"}})
        self.feature_cols: list[str] = []
        self.medians: pd.Series | None = None

        self.model_log = None
        self.model_lgb = None
        self.model_cat = None
        self.model_meta = None
        self.stack_subset: list[str] = ["p_logistic", "p_lightgbm"]
        self.model_dir = self.root / self.cfg.model_dir

        self._load_or_train_fixed()

    def _fetch_recent_ohlcv(self) -> pd.DataFrame:
        rows = self.exchange.fetch_ohlcv(self.cfg.symbol, timeframe=self.cfg.timeframe, limit=self.cfg.lookback_bars)
        df = pd.DataFrame(rows, columns=["timestamp_ms", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
        return df.drop(columns=["timestamp_ms"])

    def _load_opt_details(self) -> dict:
        p = self.root / self.cfg.model_opt_json
        if not p.exists():
            raise FileNotFoundError(f"找不到优化结果文件: {p}")
        return json.loads(p.read_text(encoding="utf-8"))

    def _load_or_train_fixed(self) -> None:
        meta_path = self.model_dir / "meta.json"
        log_path = self.model_dir / "logistic.joblib"
        lgb_path = self.model_dir / "lightgbm.joblib"
        cat_path = self.model_dir / "catboost.joblib"
        stack_path = self.model_dir / "stacking_meta.joblib"
        if all(p.exists() for p in (meta_path, log_path, lgb_path, cat_path, stack_path)):
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.feature_cols = list(meta["feature_cols"])
            self.stack_subset = list(meta["stack_subset"])
            self.medians = pd.Series(meta["medians"])
            self.model_log = load(log_path)
            self.model_lgb = load(lgb_path)
            self.model_cat = load(cat_path)
            self.model_meta = load(stack_path)
            return

        if not self.cfg.auto_train_if_missing:
            raise RuntimeError(f"固定模型不存在: {self.model_dir}，请先训练并导出后再运行。")

        self._train_and_save_fixed()

    def _train_and_save_fixed(self) -> None:
        opt = self._load_opt_details()
        best_base = opt["best_base_params"]
        best_stack = opt["best_stacking"]
        self.stack_subset = list(best_stack.get("subset", ["p_logistic", "p_lightgbm"]))

        raw = self._fetch_recent_ohlcv()
        feat = build_features(raw)
        feat = feat[feat["is_valid"] & feat["next_close"].notna()].copy().sort_values("timestamp").reset_index(drop=True)

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
        self.feature_cols = [c for c in feat.columns if c not in non_feature_cols]

        n = len(feat)
        train_end = int(n * 0.60)
        valid_end = int(n * 0.70)
        train_df = feat.iloc[:train_end].copy()
        valid_df = feat.iloc[train_end:valid_end].copy()

        self.medians = train_df[self.feature_cols].replace([np.inf, -np.inf], np.nan).median()
        for d in (train_df, valid_df):
            d[self.feature_cols] = d[self.feature_cols].replace([np.inf, -np.inf], np.nan).fillna(self.medians)

        xtr, ytr = train_df[self.feature_cols], train_df["y_true"].astype(int)
        xva, yva = valid_df[self.feature_cols], valid_df["y_true"].astype(int)

        self.model_log = train_logistic(xtr, ytr, params=best_base["logistic"]["params"])
        self.model_lgb = train_lightgbm(xtr, ytr, xva, yva, params=best_base["lightgbm"]["params"])
        self.model_cat = train_catboost(xtr, ytr, xva, yva, params=best_base["catboost"]["params"])

        vp = pd.DataFrame(
            {
                "p_logistic": self.model_log.predict_proba(xva)[:, 1],
                "p_lightgbm": self.model_lgb.predict_proba(xva)[:, 1],
                "p_catboost": self.model_cat.predict_proba(xva)[:, 1],
            },
            index=valid_df.index,
        )
        self.model_meta = fit_stacking_meta(vp[self.stack_subset], yva, params=best_stack["meta_params"])

        self.model_dir.mkdir(parents=True, exist_ok=True)
        dump(self.model_log, self.model_dir / "logistic.joblib")
        dump(self.model_lgb, self.model_dir / "lightgbm.joblib")
        dump(self.model_cat, self.model_dir / "catboost.joblib")
        dump(self.model_meta, self.model_dir / "stacking_meta.joblib")
        meta = {
            "feature_cols": self.feature_cols,
            "stack_subset": self.stack_subset,
            "medians": self.medians.to_dict(),
        }
        (self.model_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def predict_latest(self) -> SignalOutput:
        raw = self._fetch_recent_ohlcv()
        feat = build_features(raw)
        row = feat.iloc[-1:].copy()
        row[self.feature_cols] = row[self.feature_cols].replace([np.inf, -np.inf], np.nan).fillna(self.medians)
        x = row[self.feature_cols]

        prob_df = pd.DataFrame(
            {
                "p_logistic": self.model_log.predict_proba(x)[:, 1],
                "p_lightgbm": self.model_lgb.predict_proba(x)[:, 1],
                "p_catboost": self.model_cat.predict_proba(x)[:, 1],
            }
        )
        up_prob = float(predict_stacking(self.model_meta, prob_df[self.stack_subset])[0])
        up_prob = float(np.clip(up_prob, 1e-6, 1 - 1e-6))
        ts = pd.to_datetime(row["timestamp"].iloc[0], utc=True)
        return SignalOutput(candle_ts=ts, up_prob=up_prob, down_prob=1.0 - up_prob)


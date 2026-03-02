from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
)


@dataclass
class CalibrationBundle:
    isotonic: IsotonicRegression
    platt: LogisticRegression


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return np.nan, np.nan
    p = successes / n
    denom = 1 + (z**2 / n)
    center = (p + (z**2) / (2 * n)) / denom
    margin = (z / denom) * np.sqrt((p * (1 - p) / n) + (z**2 / (4 * n**2)))
    return center - margin, center + margin


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    unique = np.unique(y_true)
    if len(unique) < 2:
        return np.nan
    return roc_auc_score(y_true, y_prob)


def direction_metrics(df: pd.DataFrame, y_col: str = "y_true", prob_col: str = "long_prob", pred_col: str = "pred_dir") -> dict:
    data = df[[y_col, prob_col, pred_col]].dropna()
    y_true = data[y_col].astype(int).to_numpy()
    y_prob = data[prob_col].astype(float).clip(1e-6, 1 - 1e-6).to_numpy()
    y_pred = data[pred_col].astype(int).to_numpy()

    n = len(data)
    wins = int((y_true == y_pred).sum())
    acc = accuracy_score(y_true, y_pred)
    low, high = wilson_ci(wins, n)
    p_value = float(binomtest(k=wins, n=n, p=0.5, alternative="two-sided").pvalue) if n > 0 else np.nan

    return {
        "n": n,
        "accuracy": acc,
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred) if len(np.unique(y_pred)) > 1 else np.nan,
        "auc": _safe_auc(y_true, y_prob),
        "accuracy_ci_low": low,
        "accuracy_ci_high": high,
        "binom_test_pvalue_vs_50": p_value,
        "wins": wins,
        "baseline_50_accuracy": 0.5,
    }


def calibration_table(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float).clip(0, 1)

    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(y_prob, bins, right=True)
    idx = np.clip(idx, 1, n_bins)

    rows = []
    for b in range(1, n_bins + 1):
        mask = idx == b
        if not np.any(mask):
            rows.append(
                {
                    "bin": b,
                    "bin_left": bins[b - 1],
                    "bin_right": bins[b],
                    "count": 0,
                    "pred_mean": np.nan,
                    "true_rate": np.nan,
                    "abs_gap": np.nan,
                }
            )
            continue
        p_mean = float(np.mean(y_prob[mask]))
        t_rate = float(np.mean(y_true[mask]))
        rows.append(
            {
                "bin": b,
                "bin_left": bins[b - 1],
                "bin_right": bins[b],
                "count": int(mask.sum()),
                "pred_mean": p_mean,
                "true_rate": t_rate,
                "abs_gap": abs(p_mean - t_rate),
            }
        )
    return pd.DataFrame(rows)


def calibration_metrics(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float).clip(1e-6, 1 - 1e-6)

    table = calibration_table(y_true, y_prob, n_bins=n_bins)
    nonempty = table["count"] > 0
    if nonempty.any():
        ece = float((table.loc[nonempty, "abs_gap"] * table.loc[nonempty, "count"]).sum() / table.loc[nonempty, "count"].sum())
        mce = float(table.loc[nonempty, "abs_gap"].max())
    else:
        ece, mce = np.nan, np.nan

    return {
        "ece": ece,
        "mce": mce,
        "brier": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "calibration_table": table,
    }


def bootstrap_ci(
    values: pd.DataFrame,
    metric_fn: Callable[[pd.DataFrame], float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return np.nan, np.nan

    sampled = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        sampled.append(metric_fn(values.iloc[idx]))
    alpha = (1 - ci) / 2
    return float(np.quantile(sampled, alpha)), float(np.quantile(sampled, 1 - alpha))


def evaluate_by_group(
    df: pd.DataFrame,
    group_col: str,
    y_col: str = "y_true",
    prob_col: str = "long_prob",
    pred_col: str = "pred_dir",
) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(group_col):
        m = direction_metrics(g, y_col=y_col, prob_col=prob_col, pred_col=pred_col)
        rows.append({"group": key, **m})
    return pd.DataFrame(rows).sort_values("group")


def fit_calibrators(train_df: pd.DataFrame, y_col: str = "y_true", prob_col: str = "long_prob") -> CalibrationBundle:
    data = train_df[[y_col, prob_col]].dropna()
    x = data[prob_col].astype(float).to_numpy()
    y = data[y_col].astype(int).to_numpy()

    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(x, y)

    platt = LogisticRegression(solver="lbfgs")
    platt.fit(x.reshape(-1, 1), y)

    return CalibrationBundle(isotonic=isotonic, platt=platt)


def apply_calibration(
    df: pd.DataFrame,
    bundle: CalibrationBundle,
    prob_col: str = "long_prob",
    out_iso_col: str = "long_prob_iso",
    out_platt_col: str = "long_prob_platt",
) -> pd.DataFrame:
    out = df.copy()
    p = out[prob_col].astype(float).fillna(0.5).clip(0, 1).to_numpy()
    out[out_iso_col] = bundle.isotonic.predict(p)
    out[out_platt_col] = bundle.platt.predict_proba(p.reshape(-1, 1))[:, 1]
    return out


def summarize_all(
    df: pd.DataFrame,
    y_col: str = "y_true",
    prob_col: str = "long_prob",
    pred_col: Optional[str] = None,
) -> dict:
    pred_col = pred_col or "pred_dir"
    d = df.copy()
    if pred_col not in d.columns:
        d[pred_col] = (d[prob_col] >= 0.5).astype(int)
    d = d[[y_col, prob_col, pred_col]].dropna()

    d_metrics = direction_metrics(d, y_col=y_col, prob_col=prob_col, pred_col=pred_col)
    c_metrics = calibration_metrics(d[y_col].to_numpy(), d[prob_col].to_numpy())

    return {
        "direction": d_metrics,
        "calibration": {k: v for k, v in c_metrics.items() if k != "calibration_table"},
        "calibration_table": c_metrics["calibration_table"],
    }

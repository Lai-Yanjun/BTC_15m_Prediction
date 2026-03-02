from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from evaluation import direction_metrics
from ma_dataset import SplitConfig, prepare_timeframe_dataset
from models import fit_stacking_meta, predict_prob_table, predict_stacking, soft_vote, train_all_models


def _best_threshold(y_true: pd.Series, prob: np.ndarray, grid: Iterable[float] | None = None) -> float:
    grid = grid or np.arange(0.40, 0.61, 0.01)
    y = y_true.astype(int).to_numpy()
    best_t, best_acc = 0.5, -1.0
    for t in grid:
        acc = float(((prob >= t).astype(int) == y).mean())
        if acc > best_acc:
            best_acc = acc
            best_t = float(t)
    return best_t


def _eval(y_true: pd.Series, prob: np.ndarray, threshold: float) -> dict:
    d = pd.DataFrame({"y_true": y_true.astype(int), "long_prob": np.clip(prob, 1e-6, 1 - 1e-6)})
    d["pred_dir"] = (d["long_prob"] >= threshold).astype(int)
    out = direction_metrics(d, y_col="y_true", prob_col="long_prob", pred_col="pred_dir")
    out["threshold"] = threshold
    return out


def _rolling_windows(
    df: pd.DataFrame,
    train_days: int = 180,
    valid_days: int = 30,
    test_days: int = 30,
    max_windows: int = 8,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    ts = pd.to_datetime(df["timestamp"], utc=True)
    start_all = ts.min()
    end_all = ts.max()

    window_total = pd.Timedelta(days=train_days + valid_days + test_days)
    step = pd.Timedelta(days=test_days)
    cursor_end = end_all
    windows = []

    while cursor_end - window_total >= start_all and len(windows) < max_windows:
        test_end = cursor_end
        test_start = test_end - pd.Timedelta(days=test_days)
        valid_start = test_start - pd.Timedelta(days=valid_days)
        train_start = valid_start - pd.Timedelta(days=train_days)
        windows.append((train_start, valid_start, test_start, test_end))
        cursor_end = cursor_end - step

    windows.reverse()
    return windows


def _run_timeframe(
    timeframe: str,
    root: Path,
    train_days: int,
    valid_days: int,
    test_days: int,
    max_windows: int,
) -> dict:
    ds = prepare_timeframe_dataset(timeframe, root / "data", cfg=SplitConfig(start_utc="2024-01-01"))
    feat = ds["features"].copy()
    feat = feat[feat["is_valid"] & feat["next_close"].notna()].copy()
    feature_cols = ds["feature_cols"]

    windows = _rolling_windows(
        feat,
        train_days=train_days,
        valid_days=valid_days,
        test_days=test_days,
        max_windows=max_windows,
    )

    rows = []
    for i, (train_start, valid_start, test_start, test_end) in enumerate(windows, start=1):
        train = feat[(feat["timestamp"] >= train_start) & (feat["timestamp"] < valid_start)].copy()
        valid = feat[(feat["timestamp"] >= valid_start) & (feat["timestamp"] < test_start)].copy()
        test = feat[(feat["timestamp"] >= test_start) & (feat["timestamp"] < test_end)].copy()
        if len(train) < 2000 or len(valid) < 300 or len(test) < 300:
            continue

        trained = train_all_models(train, valid, feature_cols, y_col="y_true")
        valid_probs = predict_prob_table(trained, valid, feature_cols)
        test_probs = predict_prob_table(trained, test, feature_cols)

        # Baseline single model.
        p_log_valid = valid_probs["p_logistic"].to_numpy()
        p_log_test = test_probs["p_logistic"].to_numpy()
        t_log = _best_threshold(valid["y_true"], p_log_valid)
        m_log = _eval(test["y_true"], p_log_test, t_log)

        # Best ensemble variants.
        p_vote_valid = soft_vote(valid_probs)
        p_vote_test = soft_vote(test_probs)
        t_vote = _best_threshold(valid["y_true"], p_vote_valid)
        m_vote = _eval(test["y_true"], p_vote_test, t_vote)

        meta = fit_stacking_meta(valid_probs, valid["y_true"])
        p_stack_valid = predict_stacking(meta, valid_probs)
        p_stack_test = predict_stacking(meta, test_probs)
        t_stack = _best_threshold(valid["y_true"], p_stack_valid)
        m_stack = _eval(test["y_true"], p_stack_test, t_stack)

        rows.extend(
            [
                {
                    "window_idx": i,
                    "timeframe": timeframe,
                    "train_start": train_start.isoformat(),
                    "test_start": test_start.isoformat(),
                    "test_end": test_end.isoformat(),
                    "model": "logistic",
                    "n_test": m_log["n"],
                    "accuracy": m_log["accuracy"],
                    "auc": m_log["auc"],
                    "p_value_vs_50": m_log["binom_test_pvalue_vs_50"],
                },
                {
                    "window_idx": i,
                    "timeframe": timeframe,
                    "train_start": train_start.isoformat(),
                    "test_start": test_start.isoformat(),
                    "test_end": test_end.isoformat(),
                    "model": "ensemble_soft_vote",
                    "n_test": m_vote["n"],
                    "accuracy": m_vote["accuracy"],
                    "auc": m_vote["auc"],
                    "p_value_vs_50": m_vote["binom_test_pvalue_vs_50"],
                },
                {
                    "window_idx": i,
                    "timeframe": timeframe,
                    "train_start": train_start.isoformat(),
                    "test_start": test_start.isoformat(),
                    "test_end": test_end.isoformat(),
                    "model": "ensemble_stacking",
                    "n_test": m_stack["n"],
                    "accuracy": m_stack["accuracy"],
                    "auc": m_stack["auc"],
                    "p_value_vs_50": m_stack["binom_test_pvalue_vs_50"],
                },
            ]
        )

    out_df = pd.DataFrame(rows)
    return {
        "timeframe": timeframe,
        "rows": out_df,
    }


def _summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["timeframe", "model"])
    return (
        g.agg(
            windows=("window_idx", "nunique"),
            mean_acc=("accuracy", "mean"),
            min_acc=("accuracy", "min"),
            max_acc=("accuracy", "max"),
            win_rate_over_50=("accuracy", lambda x: float(np.mean(x > 0.5))),
            sig_rate_p_lt_005=("p_value_vs_50", lambda x: float(np.mean(x < 0.05))),
            mean_auc=("auc", "mean"),
        )
        .reset_index()
        .sort_values(["timeframe", "mean_acc"], ascending=[True, False])
    )


def _build_report(summary: pd.DataFrame, detail: pd.DataFrame) -> str:
    lines = [
        "# 滚动窗口稳定性报告",
        "",
        "口径：每个窗口采用 `train=180天 / valid=30天 / test=30天`，滚动步长30天；标签为 `close[t+1] > close[t]`。",
        "",
        "## 汇总",
        "| timeframe | model | windows | mean_acc | min_acc | max_acc | win_rate_over_50 | sig_rate_p_lt_005 | mean_auc |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['timeframe']} | {r['model']} | {int(r['windows'])} | {r['mean_acc']:.4f} | {r['min_acc']:.4f} | {r['max_acc']:.4f} | {r['win_rate_over_50']:.2f} | {r['sig_rate_p_lt_005']:.2f} | {r['mean_auc']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## 明细（每窗口）",
            "| idx | timeframe | model | test_start | test_end | n_test | acc | auc | p_value_vs_50 |",
            "|---:|---|---|---|---|---:|---:|---:|---:|",
        ]
    )
    detail = detail.sort_values(["timeframe", "window_idx", "model"])
    for _, r in detail.iterrows():
        lines.append(
            f"| {int(r['window_idx'])} | {r['timeframe']} | {r['model']} | {str(r['test_start'])[:10]} | {str(r['test_end'])[:10]} | {int(r['n_test'])} | {r['accuracy']:.4f} | {r['auc']:.4f} | {r['p_value_vs_50']:.4g} |"
        )
    return "\n".join(lines)


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    outputs = root / "outputs"
    reports = root / "reports"
    outputs.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for tf in ("5m", "15m"):
        result = _run_timeframe(tf, root, train_days=180, valid_days=30, test_days=30, max_windows=8)
        if not result["rows"].empty:
            all_rows.append(result["rows"])

    if not all_rows:
        raise RuntimeError("滚动窗口样本不足，未生成结果。")

    detail = pd.concat(all_rows, ignore_index=True)
    summary = _summary(detail)
    detail.to_csv(outputs / "rolling_stability_detail.csv", index=False)
    summary.to_csv(outputs / "rolling_stability_summary.csv", index=False)

    with open(outputs / "rolling_stability_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

    report_text = _build_report(summary, detail)
    report_path = reports / "rolling_stability_report.md"
    report_path.write_text(report_text, encoding="utf-8")
    print(f"完成: {report_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation import direction_metrics
from ma_dataset import SplitConfig, prepare_timeframe_dataset
from models import fit_stacking_meta, predict_prob_table, predict_stacking, soft_vote, train_all_models


def _best_threshold(y_true: pd.Series, prob: np.ndarray) -> float:
    grid = np.arange(0.40, 0.61, 0.01)
    y = y_true.astype(int).to_numpy()
    best_t, best_acc = 0.5, -1.0
    for t in grid:
        pred = (prob >= t).astype(int)
        acc = float((pred == y).mean())
        if acc > best_acc:
            best_acc = acc
            best_t = float(t)
    return best_t


def _eval(y_true: pd.Series, prob: np.ndarray, threshold: float) -> dict:
    d = pd.DataFrame(
        {
            "y_true": y_true.astype(int).to_numpy(),
            "long_prob": np.clip(prob, 1e-6, 1 - 1e-6),
        }
    )
    d["pred_dir"] = (d["long_prob"] >= threshold).astype(int)
    m = direction_metrics(d, y_col="y_true", prob_col="long_prob", pred_col="pred_dir")
    m["threshold"] = threshold
    return m


def _prepare_fixed_model_timeframe(timeframe: str, root: Path, force_model: str | None = None) -> dict:
    ds = prepare_timeframe_dataset(timeframe, root / "data", cfg=SplitConfig(start_utc="2024-01-01"))
    feat = ds["features"].copy()
    feat = feat[feat["is_valid"] & feat["next_close"].notna()].copy().sort_values("timestamp").reset_index(drop=True)
    feature_cols = ds["feature_cols"]

    n = len(feat)
    train_end = int(n * 0.60)
    valid_end = int(n * 0.70)
    train_df = feat.iloc[:train_end].copy()
    valid_df = feat.iloc[train_end:valid_end].copy()
    test_pool = feat.iloc[valid_end:].copy()

    trained = train_all_models(train_df, valid_df, feature_cols, y_col="y_true")
    valid_probs = predict_prob_table(trained, valid_df, feature_cols)
    pool_probs = predict_prob_table(trained, test_pool, feature_cols)

    candidates: dict[str, dict] = {}

    # Single models.
    for name, col in (("logistic", "p_logistic"), ("lightgbm", "p_lightgbm"), ("catboost", "p_catboost")):
        if col not in valid_probs.columns:
            continue
        t = _best_threshold(valid_df["y_true"], valid_probs[col].to_numpy())
        m = _eval(valid_df["y_true"], valid_probs[col].to_numpy(), t)
        candidates[name] = {"threshold": t, "valid_accuracy": m["accuracy"]}

    # Soft vote.
    p_vote_valid = soft_vote(valid_probs)
    t_vote = _best_threshold(valid_df["y_true"], p_vote_valid)
    m_vote = _eval(valid_df["y_true"], p_vote_valid, t_vote)
    candidates["ensemble_soft_vote"] = {"threshold": t_vote, "valid_accuracy": m_vote["accuracy"]}

    # Stacking.
    meta = fit_stacking_meta(valid_probs, valid_df["y_true"])
    p_stack_valid = predict_stacking(meta, valid_probs)
    t_stack = _best_threshold(valid_df["y_true"], p_stack_valid)
    m_stack = _eval(valid_df["y_true"], p_stack_valid, t_stack)
    candidates["ensemble_stacking"] = {"threshold": t_stack, "valid_accuracy": m_stack["accuracy"]}

    # Choose one deployed fixed model by best valid accuracy, or force one.
    if force_model is not None:
        if force_model not in candidates:
            raise ValueError(f"force_model={force_model} 不可用，候选: {list(candidates.keys())}")
        deploy_name = force_model
        deploy_cfg = candidates[force_model]
    else:
        deploy_name, deploy_cfg = max(candidates.items(), key=lambda kv: kv[1]["valid_accuracy"])
    threshold = float(deploy_cfg["threshold"])

    if deploy_name == "logistic":
        deploy_prob = pool_probs["p_logistic"].to_numpy()
    elif deploy_name == "lightgbm":
        deploy_prob = pool_probs["p_lightgbm"].to_numpy()
    elif deploy_name == "catboost":
        deploy_prob = pool_probs["p_catboost"].to_numpy()
    elif deploy_name == "ensemble_soft_vote":
        deploy_prob = soft_vote(pool_probs)
    else:
        deploy_prob = predict_stacking(meta, pool_probs)

    out = test_pool[["timestamp", "y_true"]].copy()
    out["prob"] = np.clip(deploy_prob, 1e-6, 1 - 1e-6)
    out["pred"] = (out["prob"] >= threshold).astype(int)
    out["month"] = pd.to_datetime(out["timestamp"], utc=True).dt.strftime("%Y-%m")

    rows = []
    for month, g in out.groupby("month"):
        metrics = _eval(g["y_true"], g["prob"].to_numpy(), threshold)
        rows.append(
            {
                "timeframe": timeframe,
                "month": month,
                "model_fixed": deploy_name,
                "threshold": threshold,
                "n": int(metrics["n"]),
                "accuracy": float(metrics["accuracy"]),
                "delta_vs_50": float(metrics["accuracy"] - 0.5),
                "auc": float(metrics["auc"]),
                "p_value_vs_50": float(metrics["binom_test_pvalue_vs_50"]),
            }
        )

    monthly = pd.DataFrame(rows).sort_values("month").reset_index(drop=True)
    return {
        "timeframe": timeframe,
        "deployed_model": deploy_name,
        "threshold": threshold,
        "valid_best_accuracy": float(deploy_cfg["valid_accuracy"]),
        "monthly": monthly,
    }


def _plot_delta(df: pd.DataFrame, out_png: Path) -> None:
    plt.figure(figsize=(10, 5))
    for tf in sorted(df["timeframe"].unique()):
        d = df[df["timeframe"] == tf].copy()
        plt.plot(d["month"], d["delta_vs_50"], marker="o", linewidth=1.8, label=f"{tf}")
    plt.axhline(0.0, linestyle="--", linewidth=1.2)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Accuracy - 0.50")
    plt.title("Fixed Model Monthly Drift (Distance from 50%)")
    plt.grid(alpha=0.3)
    plt.legend()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=140)
    plt.close()


def _make_report(df: pd.DataFrame, meta: list[dict], png_path: Path) -> str:
    lines = [
        "# 固定模型月度漂移报告",
        "",
        "实验口径：先固定训练一次模型（train=前60%，valid=后10%用于选模型与阈值），再在余下30%数据按自然月测试。",
        "",
        f"- 偏差定义：`delta_vs_50 = accuracy - 0.50`",
        f"- 折线图：`{png_path.as_posix()}`",
        "",
        "## 固定部署模型",
    ]
    for m in meta:
        lines.append(
            f"- `{m['timeframe']}`: model=`{m['deployed_model']}`, threshold=`{m['threshold']:.2f}`, valid_best_accuracy=`{m['valid_best_accuracy']:.4f}`"
        )

    lines.extend(
        [
            "",
            "## 月度结果",
            "| timeframe | month | n | accuracy | delta_vs_50 | auc | p_value_vs_50 |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, r in df.iterrows():
        lines.append(
            f"| {r['timeframe']} | {r['month']} | {int(r['n'])} | {r['accuracy']:.4f} | {r['delta_vs_50']:+.4f} | {r['auc']:.4f} | {r['p_value_vs_50']:.4g} |"
        )

    s = (
        df.groupby("timeframe")
        .agg(
            months=("month", "count"),
            mean_acc=("accuracy", "mean"),
            mean_delta=("delta_vs_50", "mean"),
            min_delta=("delta_vs_50", "min"),
            max_delta=("delta_vs_50", "max"),
            negative_months=("delta_vs_50", lambda x: int(np.sum(x < 0))),
        )
        .reset_index()
    )
    lines.extend(
        [
            "",
            "## 汇总",
            "| timeframe | months | mean_acc | mean_delta | min_delta | max_delta | negative_months |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, r in s.iterrows():
        lines.append(
            f"| {r['timeframe']} | {int(r['months'])} | {r['mean_acc']:.4f} | {r['mean_delta']:+.4f} | {r['min_delta']:+.4f} | {r['max_delta']:+.4f} | {int(r['negative_months'])} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="固定模型月度漂移检测")
    parser.add_argument(
        "--force-model",
        choices=["logistic", "lightgbm", "catboost", "ensemble_soft_vote", "ensemble_stacking"],
        default=None,
        help="强制使用指定模型，不自动选择",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    outputs = root / "outputs"
    reports = root / "reports"
    outputs.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    all_monthly = []
    meta = []
    for tf in ("5m", "15m"):
        r = _prepare_fixed_model_timeframe(tf, root, force_model=args.force_model)
        all_monthly.append(r["monthly"])
        meta.append(
            {
                "timeframe": tf,
                "deployed_model": r["deployed_model"],
                "threshold": r["threshold"],
                "valid_best_accuracy": r["valid_best_accuracy"],
            }
        )

    monthly_df = pd.concat(all_monthly, ignore_index=True)
    suffix = f"_{args.force_model}" if args.force_model else ""
    monthly_df.to_csv(outputs / f"fixed_model_monthly_drift{suffix}.csv", index=False)

    chart_path = outputs / f"fixed_model_monthly_drift_delta{suffix}.png"
    _plot_delta(monthly_df, chart_path)

    with open(outputs / f"fixed_model_monthly_meta{suffix}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    report = _make_report(monthly_df, meta, chart_path)
    report_path = reports / f"fixed_model_monthly_drift_report{suffix}.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"完成: {report_path}")


if __name__ == "__main__":
    main()

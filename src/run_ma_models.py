from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from evaluation import direction_metrics
from ma_dataset import SplitConfig, prepare_timeframe_dataset
from models import fit_stacking_meta, predict_prob_table, predict_stacking, soft_vote, train_all_models


def best_threshold_from_valid(y_true: pd.Series, prob: np.ndarray, grid: Iterable[float] | None = None) -> tuple[float, float]:
    grid = grid or np.arange(0.40, 0.61, 0.01)
    y = y_true.astype(int).to_numpy()
    best_t = 0.5
    best_acc = -1.0
    for t in grid:
        pred = (prob >= t).astype(int)
        acc = float((pred == y).mean())
        if acc > best_acc:
            best_acc = acc
            best_t = float(t)
    return best_t, best_acc


def evaluate_single(y_true: pd.Series, prob: np.ndarray, threshold: float) -> dict:
    df = pd.DataFrame(
        {
            "y_true": y_true.astype(int).to_numpy(),
            "long_prob": np.asarray(prob).clip(1e-6, 1 - 1e-6),
        }
    )
    df["pred_dir"] = (df["long_prob"] >= threshold).astype(int)
    out = direction_metrics(df, y_col="y_true", prob_col="long_prob", pred_col="pred_dir")
    out["threshold"] = threshold
    return out


def run_for_timeframe(timeframe: str, root: Path, split_cfg: SplitConfig) -> dict:
    data_dir = root / "data"
    outputs_dir = root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    ds = prepare_timeframe_dataset(timeframe=timeframe, data_dir=data_dir, cfg=split_cfg)
    train_df, valid_df, test_df = ds["train"], ds["valid"], ds["test"]
    feature_cols = ds["feature_cols"]

    trained = train_all_models(train_df, valid_df, feature_cols, y_col="y_true")

    valid_probs = predict_prob_table(trained, valid_df, feature_cols)
    test_probs = predict_prob_table(trained, test_df, feature_cols)

    results: dict[str, dict] = {}
    thresholds: dict[str, float] = {}
    for model_name, prob_col in (
        ("logistic", "p_logistic"),
        ("lightgbm", "p_lightgbm"),
        ("catboost", "p_catboost"),
    ):
        if prob_col not in valid_probs.columns or prob_col not in test_probs.columns:
            continue
        t_best, valid_acc = best_threshold_from_valid(valid_df["y_true"], valid_probs[prob_col].to_numpy())
        res = evaluate_single(test_df["y_true"], test_probs[prob_col].to_numpy(), t_best)
        res["valid_best_accuracy"] = valid_acc
        results[model_name] = res
        thresholds[model_name] = t_best

    # Soft-voting ensemble.
    p_vote_valid = soft_vote(valid_probs)
    p_vote_test = soft_vote(test_probs)
    t_vote, valid_acc_vote = best_threshold_from_valid(valid_df["y_true"], p_vote_valid)
    vote_res = evaluate_single(test_df["y_true"], p_vote_test, t_vote)
    vote_res["valid_best_accuracy"] = valid_acc_vote
    results["ensemble_soft_vote"] = vote_res
    thresholds["ensemble_soft_vote"] = t_vote

    # Stacking ensemble.
    meta = fit_stacking_meta(valid_probs, valid_df["y_true"])
    p_stack_valid = predict_stacking(meta, valid_probs)
    p_stack_test = predict_stacking(meta, test_probs)
    t_stack, valid_acc_stack = best_threshold_from_valid(valid_df["y_true"], p_stack_valid)
    stack_res = evaluate_single(test_df["y_true"], p_stack_test, t_stack)
    stack_res["valid_best_accuracy"] = valid_acc_stack
    results["ensemble_stacking"] = stack_res
    thresholds["ensemble_stacking"] = t_stack

    summary = {
        "timeframe": timeframe,
        "n_train": int(len(train_df)),
        "n_valid": int(len(valid_df)),
        "n_test": int(len(test_df)),
        "features": len(feature_cols),
        "model_availability": trained.availability,
        "thresholds": thresholds,
        "results": results,
    }

    with open(outputs_dir / f"next_candle_models_{timeframe}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def _to_report(all_summary: list[dict]) -> str:
    lines = [
        "# 下一根K线方向三模型对比报告",
        "",
        "## 任务定义",
        "- 标签：`y = 1(close[t+1] > close[t])`，否则为 0。",
        "- 切分：`train`=更早历史，`valid`=测试前1个月，`test`=最近1个月。",
        "- 判定：`Accuracy > 0.5` 且 `binom_test_pvalue_vs_50 < 0.05` 视为显著高于随机。",
        "",
    ]
    for s in all_summary:
        lines.append(f"## {s['timeframe']}")
        lines.append(f"- 样本数：train={s['n_train']}，valid={s['n_valid']}，test={s['n_test']}，特征数={s['features']}")
        lines.append(f"- 可用模型：{s['model_availability']}")
        lines.append("")
        lines.append("| 模型 | Accuracy | AUC | p-value(vs50%) | Threshold |")
        lines.append("|---|---:|---:|---:|---:|")
        for model_name, m in s["results"].items():
            lines.append(
                f"| {model_name} | {m['accuracy']:.4f} | {m['auc']:.4f} | {m['binom_test_pvalue_vs_50']:.4g} | {m['threshold']:.2f} |"
            )
        lines.append("")
        best = max(s["results"].items(), key=lambda kv: kv[1]["accuracy"])
        lines.append(f"- 最佳模型：`{best[0]}`，Accuracy=`{best[1]['accuracy']:.4f}`，AUC=`{best[1]['auc']:.4f}`")
        lines.append("")

    lines.extend(
        [
            "## 总结口径",
            "- 若最佳模型 Accuracy 仅略高于 0.5 且 p-value 不显著，视为缺乏稳定方向 edge。",
            "- 若 soft-vote/stacking 持续优于单模型，后续可做滚动重训并纳入集成。",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    split_cfg = SplitConfig(test_days=30, valid_days=30, start_utc="2024-01-01")
    summaries = [run_for_timeframe("5m", root, split_cfg), run_for_timeframe("15m", root, split_cfg)]

    report_text = _to_report(summaries)
    report_path = reports_dir / "next_candle_direction_model_compare.md"
    report_path.write_text(report_text, encoding="utf-8")
    print(f"完成: {report_path}")


if __name__ == "__main__":
    main()

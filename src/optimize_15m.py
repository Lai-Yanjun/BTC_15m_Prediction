from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from evaluation import direction_metrics
from ma_dataset import SplitConfig, prepare_timeframe_dataset
from models import (
    fit_stacking_meta,
    predict_stacking,
    train_catboost,
    train_lightgbm,
    train_logistic,
)


def best_threshold(y_true: pd.Series, prob: np.ndarray, grid: Iterable[float] | None = None) -> tuple[float, float]:
    grid = grid or np.arange(0.40, 0.61, 0.01)
    y = y_true.astype(int).to_numpy()
    best_t, best_acc = 0.5, -1.0
    for t in grid:
        pred = (prob >= t).astype(int)
        acc = float((pred == y).mean())
        if acc > best_acc:
            best_acc = acc
            best_t = float(t)
    return best_t, best_acc


def evaluate(y_true: pd.Series, prob: np.ndarray, threshold: float) -> dict:
    d = pd.DataFrame({"y_true": y_true.astype(int), "long_prob": np.clip(prob, 1e-6, 1 - 1e-6)})
    d["pred_dir"] = (d["long_prob"] >= threshold).astype(int)
    out = direction_metrics(d, y_col="y_true", prob_col="long_prob", pred_col="pred_dir")
    out["threshold"] = threshold
    return out


def get_prob(model: object, x: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(x)[:, 1]


def losing_streak_stats(correct: np.ndarray) -> dict:
    lengths = []
    c = 0
    for v in correct:
        if v:
            if c > 0:
                lengths.append(c)
                c = 0
        else:
            c += 1
    if c > 0:
        lengths.append(c)
    if not lengths:
        return {"max_streak": 0, "p95_streak": 0.0, "freq_ge_8": 0.0, "count_streaks": 0}
    arr = np.asarray(lengths)
    return {
        "max_streak": int(arr.max()),
        "p95_streak": float(np.quantile(arr, 0.95)),
        "freq_ge_8": float(np.mean(arr >= 8)),
        "count_streaks": int(len(arr)),
    }


def rolling_windows(df: pd.DataFrame, train_days: int = 180, valid_days: int = 30, test_days: int = 30, max_windows: int = 6):
    ts = pd.to_datetime(df["timestamp"], utc=True)
    start_all, end_all = ts.min(), ts.max()
    window_total = pd.Timedelta(days=train_days + valid_days + test_days)
    step = pd.Timedelta(days=test_days)
    cursor_end = end_all
    out = []
    while cursor_end - window_total >= start_all and len(out) < max_windows:
        test_end = cursor_end
        test_start = test_end - pd.Timedelta(days=test_days)
        valid_start = test_start - pd.Timedelta(days=valid_days)
        train_start = valid_start - pd.Timedelta(days=train_days)
        out.append((train_start, valid_start, test_start, test_end))
        cursor_end = cursor_end - step
    out.reverse()
    return out


def run() -> None:
    root = Path(__file__).resolve().parent.parent
    outputs = root / "outputs"
    reports = root / "reports"
    outputs.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    stage_t0 = time.perf_counter()
    timers: dict[str, float] = {}

    # Stage 1: baseline + setup
    ds = prepare_timeframe_dataset("15m", root / "data", cfg=SplitConfig(test_days=30, valid_days=30, start_utc="2024-01-01"))
    train_df, valid_df, test_df = ds["train"], ds["valid"], ds["test"]
    feat_all = ds["features"][ds["features"]["is_valid"] & ds["features"]["next_close"].notna()].copy()
    cols = ds["feature_cols"]
    x_train, y_train = train_df[cols], train_df["y_true"].astype(int)
    x_valid, y_valid = valid_df[cols], valid_df["y_true"].astype(int)
    x_test, y_test = test_df[cols], test_df["y_true"].astype(int)

    base_default_log = train_logistic(x_train, y_train)
    base_default_lgb = train_lightgbm(x_train, y_train, x_valid, y_valid)
    base_default_cat = train_catboost(x_train, y_train, x_valid, y_valid)

    pvl = get_prob(base_default_log, x_valid)
    ptl = get_prob(base_default_log, x_test)
    tvl, _ = best_threshold(y_valid, pvl)
    baseline_log = evaluate(y_test, ptl, tvl)

    pvlgb = get_prob(base_default_lgb, x_valid)
    ptlgb = get_prob(base_default_lgb, x_test)
    tvlgb, _ = best_threshold(y_valid, pvlgb)
    baseline_lgb = evaluate(y_test, ptlgb, tvlgb)

    pvc = get_prob(base_default_cat, x_valid)
    ptc = get_prob(base_default_cat, x_test)
    tvc, _ = best_threshold(y_valid, pvc)
    baseline_cat = evaluate(y_test, ptc, tvc)

    baseline_probs_valid = pd.DataFrame({"p_logistic": pvl, "p_lightgbm": pvlgb, "p_catboost": pvc}, index=valid_df.index)
    baseline_probs_test = pd.DataFrame({"p_logistic": ptl, "p_lightgbm": ptlgb, "p_catboost": ptc}, index=test_df.index)
    baseline_meta = fit_stacking_meta(baseline_probs_valid, y_valid)
    pv_stack = predict_stacking(baseline_meta, baseline_probs_valid)
    pt_stack = predict_stacking(baseline_meta, baseline_probs_test)
    tvs, _ = best_threshold(y_valid, pv_stack)
    baseline_stack = evaluate(y_test, pt_stack, tvs)

    timers["stage1_baseline"] = time.perf_counter() - stage_t0

    # Stage 2: base model tuning
    t2 = time.perf_counter()
    logistic_grid = [{"C": c, "max_iter": 400} for c in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]]
    lgb_grid = []
    for leaves in [15, 31, 63]:
        for mcs in [40, 80]:
            for lr in [0.02, 0.03]:
                lgb_grid.append(
                    {
                        "num_leaves": leaves,
                        "min_child_samples": mcs,
                        "learning_rate": lr,
                        "n_estimators": 1000,
                        "subsample": 0.9,
                        "colsample_bytree": 0.9,
                    }
                )
    cat_grid = []
    for depth in [4, 6, 8]:
        for l2 in [3.0, 6.0]:
            for lr in [0.02, 0.03]:
                cat_grid.append({"depth": depth, "l2_leaf_reg": l2, "learning_rate": lr, "iterations": 1000, "verbose": False})

    search_rows = []

    best_log = {"valid_acc": -1.0, "params": None, "threshold": 0.5}
    for p in logistic_grid:
        m = train_logistic(x_train, y_train, params=p)
        pv = get_prob(m, x_valid)
        th, va = best_threshold(y_valid, pv)
        search_rows.append({"model": "logistic", "params": p, "valid_acc": va, "threshold": th})
        if va > best_log["valid_acc"]:
            best_log = {"valid_acc": va, "params": p, "threshold": th}

    best_lgb = {"valid_acc": -1.0, "params": None, "threshold": 0.5}
    for p in lgb_grid:
        m = train_lightgbm(x_train, y_train, x_valid, y_valid, params=p)
        pv = get_prob(m, x_valid)
        th, va = best_threshold(y_valid, pv)
        search_rows.append({"model": "lightgbm", "params": p, "valid_acc": va, "threshold": th})
        if va > best_lgb["valid_acc"]:
            best_lgb = {"valid_acc": va, "params": p, "threshold": th}

    best_cat = {"valid_acc": -1.0, "params": None, "threshold": 0.5}
    for p in cat_grid:
        m = train_catboost(x_train, y_train, x_valid, y_valid, params=p)
        pv = get_prob(m, x_valid)
        th, va = best_threshold(y_valid, pv)
        search_rows.append({"model": "catboost", "params": p, "valid_acc": va, "threshold": th})
        if va > best_cat["valid_acc"]:
            best_cat = {"valid_acc": va, "params": p, "threshold": th}

    timers["stage2_base_tuning"] = time.perf_counter() - t2

    # Train best base models and evaluate test
    best_m_log = train_logistic(x_train, y_train, params=best_log["params"])
    best_m_lgb = train_lightgbm(x_train, y_train, x_valid, y_valid, params=best_lgb["params"])
    best_m_cat = train_catboost(x_train, y_train, x_valid, y_valid, params=best_cat["params"])

    pv_log, pt_log = get_prob(best_m_log, x_valid), get_prob(best_m_log, x_test)
    pv_lgb, pt_lgb = get_prob(best_m_lgb, x_valid), get_prob(best_m_lgb, x_test)
    pv_cat, pt_cat = get_prob(best_m_cat, x_valid), get_prob(best_m_cat, x_test)

    tuned_base_test = {
        "logistic": evaluate(y_test, pt_log, float(best_log["threshold"])),
        "lightgbm": evaluate(y_test, pt_lgb, float(best_lgb["threshold"])),
        "catboost": evaluate(y_test, pt_cat, float(best_cat["threshold"])),
    }

    # Stage 3: stacking tuning
    t3 = time.perf_counter()
    valid_prob_df = pd.DataFrame({"p_logistic": pv_log, "p_lightgbm": pv_lgb, "p_catboost": pv_cat}, index=valid_df.index)
    test_prob_df = pd.DataFrame({"p_logistic": pt_log, "p_lightgbm": pt_lgb, "p_catboost": pt_cat}, index=test_df.index)

    subsets = [
        ["p_logistic", "p_lightgbm", "p_catboost"],
        ["p_lightgbm", "p_catboost"],
        ["p_logistic", "p_catboost"],
        ["p_logistic", "p_lightgbm"],
    ]
    meta_grid = [{"C": c, "max_iter": 400} for c in [0.2, 0.5, 1.0, 2.0, 5.0]]

    best_stack = {"valid_acc": -1.0, "meta_params": None, "subset": None, "threshold": 0.5}
    stack_rows = []
    for subset in subsets:
        vp = valid_prob_df[subset].copy()
        tp = test_prob_df[subset].copy()
        for mp in meta_grid:
            meta = fit_stacking_meta(vp, y_valid, params=mp)
            pvv = predict_stacking(meta, vp)
            th, va = best_threshold(y_valid, pvv)
            stack_rows.append({"subset": subset, "meta_params": mp, "valid_acc": va, "threshold": th})
            if va > best_stack["valid_acc"]:
                best_stack = {"valid_acc": va, "meta_params": mp, "subset": subset, "threshold": th}

    # final stack eval on test
    final_meta = fit_stacking_meta(valid_prob_df[best_stack["subset"]], y_valid, params=best_stack["meta_params"])
    p_test_stack = predict_stacking(final_meta, test_prob_df[best_stack["subset"]])
    tuned_stack_test = evaluate(y_test, p_test_stack, float(best_stack["threshold"]))
    timers["stage3_stack_tuning"] = time.perf_counter() - t3

    # Stage 4: rolling stability for tuned stack
    t4 = time.perf_counter()
    rw_rows = []
    for idx, (tr_start, va_start, te_start, te_end) in enumerate(rolling_windows(feat_all, max_windows=6), start=1):
        tr = feat_all[(feat_all["timestamp"] >= tr_start) & (feat_all["timestamp"] < va_start)].copy()
        va = feat_all[(feat_all["timestamp"] >= va_start) & (feat_all["timestamp"] < te_start)].copy()
        te = feat_all[(feat_all["timestamp"] >= te_start) & (feat_all["timestamp"] < te_end)].copy()
        if min(len(tr), len(va), len(te)) < 300:
            continue
        xtr, ytr = tr[cols], tr["y_true"].astype(int)
        xva, yva = va[cols], va["y_true"].astype(int)
        xte, yte = te[cols], te["y_true"].astype(int)

        ml = train_logistic(xtr, ytr, params=best_log["params"])
        mgb = train_lightgbm(xtr, ytr, xva, yva, params=best_lgb["params"])
        mcb = train_catboost(xtr, ytr, xva, yva, params=best_cat["params"])
        vpdf = pd.DataFrame(
            {
                "p_logistic": get_prob(ml, xva),
                "p_lightgbm": get_prob(mgb, xva),
                "p_catboost": get_prob(mcb, xva),
            }
        )
        tpdf = pd.DataFrame(
            {
                "p_logistic": get_prob(ml, xte),
                "p_lightgbm": get_prob(mgb, xte),
                "p_catboost": get_prob(mcb, xte),
            }
        )

        meta = fit_stacking_meta(vpdf[best_stack["subset"]], yva, params=best_stack["meta_params"])
        pva = predict_stacking(meta, vpdf[best_stack["subset"]])
        th, _ = best_threshold(yva, pva)
        pte = predict_stacking(meta, tpdf[best_stack["subset"]])
        met = evaluate(yte, pte, th)
        rw_rows.append(
            {
                "window_idx": idx,
                "test_start": str(te_start)[:10],
                "test_end": str(te_end)[:10],
                "n_test": met["n"],
                "accuracy": met["accuracy"],
                "auc": met["auc"],
                "p_value_vs_50": met["binom_test_pvalue_vs_50"],
            }
        )
    rolling_df = pd.DataFrame(rw_rows)
    rolling_summary = {
        "windows": int(len(rolling_df)),
        "mean_acc": float(rolling_df["accuracy"].mean()) if not rolling_df.empty else np.nan,
        "win_rate_over_50": float((rolling_df["accuracy"] > 0.5).mean()) if not rolling_df.empty else np.nan,
        "sig_rate_p_lt_005": float((rolling_df["p_value_vs_50"] < 0.05).mean()) if not rolling_df.empty else np.nan,
        "mean_auc": float(rolling_df["auc"].mean()) if not rolling_df.empty else np.nan,
    }
    timers["stage4_rolling_validate"] = time.perf_counter() - t4

    # Stage 5: streak risk compare baseline stack vs tuned stack on fixed split
    t5 = time.perf_counter()
    # baseline stack risk
    baseline_correct = ((pt_stack >= tvs).astype(int) == y_test.to_numpy())
    baseline_streak = losing_streak_stats(baseline_correct)
    # tuned stack risk
    tuned_correct = ((p_test_stack >= float(best_stack["threshold"])).astype(int) == y_test.to_numpy())
    tuned_streak = losing_streak_stats(tuned_correct)
    timers["stage5_streak_risk"] = time.perf_counter() - t5

    total_elapsed = time.perf_counter() - stage_t0
    timers["total_elapsed_hours"] = total_elapsed / 3600.0

    result = {
        "dataset": {"n_train": int(len(train_df)), "n_valid": int(len(valid_df)), "n_test": int(len(test_df)), "n_features": len(cols)},
        "baseline_test": {
            "logistic": baseline_log,
            "lightgbm": baseline_lgb,
            "catboost": baseline_cat,
            "ensemble_stacking": baseline_stack,
        },
        "best_base_params": {"logistic": best_log, "lightgbm": best_lgb, "catboost": best_cat},
        "tuned_base_test": tuned_base_test,
        "best_stacking": best_stack,
        "tuned_stacking_test": tuned_stack_test,
        "rolling_summary_tuned_stack": rolling_summary,
        "rolling_detail_tuned_stack": rolling_df.to_dict(orient="records"),
        "streak_risk_compare": {"baseline_stacking": baseline_streak, "tuned_stacking": tuned_streak},
        "timers": timers,
        "search_counts": {
            "logistic": len(logistic_grid),
            "lightgbm": len(lgb_grid),
            "catboost": len(cat_grid),
            "stacking_meta_trials": len(subsets) * len(meta_grid),
        },
    }

    with open(outputs / "opt_15m_details.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    rolling_df.to_csv(outputs / "opt_15m_rolling_detail.csv", index=False)
    pd.DataFrame(search_rows).to_json(outputs / "opt_15m_base_search.json", orient="records", force_ascii=False, indent=2)
    pd.DataFrame(stack_rows).to_json(outputs / "opt_15m_stack_search.json", orient="records", force_ascii=False, indent=2)

    report_lines = [
        "# 15m 三基模型+Stacking 优化报告",
        "",
        "## 数据与预算",
        f"- 样本：train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)}, features={len(cols)}",
        f"- 搜索规模：logistic={len(logistic_grid)}，lightgbm={len(lgb_grid)}，catboost={len(cat_grid)}，stacking={len(subsets) * len(meta_grid)}",
        f"- 总耗时（小时）：{timers['total_elapsed_hours']:.3f}",
        "",
        "## 基线 vs 优化后（test）",
        f"- baseline_stacking: acc={baseline_stack['accuracy']:.4f}, auc={baseline_stack['auc']:.4f}, p={baseline_stack['binom_test_pvalue_vs_50']:.4g}",
        f"- tuned_stacking: acc={tuned_stack_test['accuracy']:.4f}, auc={tuned_stack_test['auc']:.4f}, p={tuned_stack_test['binom_test_pvalue_vs_50']:.4g}",
        "",
        "## 最优参数",
        f"- logistic: {best_log}",
        f"- lightgbm: {best_lgb}",
        f"- catboost: {best_cat}",
        f"- stacking: {best_stack}",
        "",
        "## 滚动稳定性（tuned stacking）",
        f"- windows={rolling_summary['windows']}, mean_acc={rolling_summary['mean_acc']:.4f}, win_rate_over_50={rolling_summary['win_rate_over_50']:.2f}, sig_rate={rolling_summary['sig_rate_p_lt_005']:.2f}",
        "",
        "## 连败风险（fixed split）",
        f"- baseline_stacking: {baseline_streak}",
        f"- tuned_stacking: {tuned_streak}",
        "",
        "## 阶段耗时（秒）",
        f"- stage1_baseline={timers['stage1_baseline']:.1f}",
        f"- stage2_base_tuning={timers['stage2_base_tuning']:.1f}",
        f"- stage3_stack_tuning={timers['stage3_stack_tuning']:.1f}",
        f"- stage4_rolling_validate={timers['stage4_rolling_validate']:.1f}",
        f"- stage5_streak_risk={timers['stage5_streak_risk']:.1f}",
    ]
    report_path = reports / "opt_15m_base_plus_stacking.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"完成: {report_path}")


if __name__ == "__main__":
    run()

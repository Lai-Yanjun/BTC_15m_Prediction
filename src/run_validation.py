from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation import (
    apply_calibration,
    calibration_metrics,
    evaluate_by_group,
    fit_calibrators,
    summarize_all,
)
from predicta_v4 import PredictaConfig, compute_predicta_v4, download_binance_spot_ohlcv, split_train_valid


def _plot_reliability(table: pd.DataFrame, title: str, out_path: Path) -> None:
    t = table[table["count"] > 0].copy()
    if t.empty:
        return

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, label="Perfect calibration")
    plt.plot(t["pred_mean"], t["true_rate"], marker="o", linewidth=2, label="Model")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def _evaluate_timeframe(
    timeframe: str,
    data_dir: Path,
    outputs_dir: Path,
    reports_dir: Path,
    start_utc: str = "2024-01-01",
) -> dict:
    csv_path = data_dir / f"btc_usdt_{timeframe}.csv"
    if csv_path.exists():
        raw = pd.read_csv(csv_path)
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    else:
        raw = download_binance_spot_ohlcv(
            symbol="BTC/USDT",
            timeframe=timeframe,
            start_utc=start_utc,
            out_csv=str(csv_path),
        )

    cfg = PredictaConfig()
    feat = compute_predicta_v4(raw, cfg=cfg)
    valid = feat[feat["is_valid"]].copy()

    train, test = split_train_valid(valid, train_ratio=0.7)
    train = train[train["next_close"].notna()].copy()
    test = test[test["next_close"].notna()].copy()

    baseline = summarize_all(test)
    by_regime = evaluate_by_group(test, "vol_regime")

    bundles = fit_calibrators(train)
    test_calibrated = apply_calibration(test, bundles)
    test_calibrated["pred_dir_iso"] = (test_calibrated["long_prob_iso"] >= 0.5).astype(int)
    test_calibrated["pred_dir_platt"] = (test_calibrated["long_prob_platt"] >= 0.5).astype(int)

    iso_summary = summarize_all(test_calibrated, prob_col="long_prob_iso", pred_col="pred_dir_iso")
    platt_summary = summarize_all(test_calibrated, prob_col="long_prob_platt", pred_col="pred_dir_platt")

    b0 = calibration_metrics(test["y_true"].to_numpy(), test["long_prob"].to_numpy())
    bi = calibration_metrics(test_calibrated["y_true"].to_numpy(), test_calibrated["long_prob_iso"].to_numpy())
    bp = calibration_metrics(test_calibrated["y_true"].to_numpy(), test_calibrated["long_prob_platt"].to_numpy())

    _plot_reliability(
        b0["calibration_table"],
        f"Predicta V4 Reliability ({timeframe}) - Raw",
        outputs_dir / f"reliability_{timeframe}_raw.png",
    )
    _plot_reliability(
        bi["calibration_table"],
        f"Predicta V4 Reliability ({timeframe}) - Isotonic",
        outputs_dir / f"reliability_{timeframe}_iso.png",
    )
    _plot_reliability(
        bp["calibration_table"],
        f"Predicta V4 Reliability ({timeframe}) - Platt",
        outputs_dir / f"reliability_{timeframe}_platt.png",
    )

    test.to_csv(outputs_dir / f"predicta_features_{timeframe}_test.csv", index=False)
    by_regime.to_csv(outputs_dir / f"direction_by_regime_{timeframe}.csv", index=False)

    summary = {
        "timeframe": timeframe,
        "n_raw": int(len(raw)),
        "n_valid": int(len(valid)),
        "n_test": int(len(test)),
        "raw": {
            "direction": baseline["direction"],
            "calibration": baseline["calibration"],
        },
        "isotonic": {
            "direction": iso_summary["direction"],
            "calibration": iso_summary["calibration"],
        },
        "platt": {
            "direction": platt_summary["direction"],
            "calibration": platt_summary["calibration"],
        },
    }

    with open(outputs_dir / f"summary_{timeframe}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def _markdown_report(all_summary: list[dict]) -> str:
    lines = [
        "# Predicta V4 概率验证报告（BTC 现货 5m/15m）",
        "",
        "## 方法概览",
        "- 标签定义：`close[t+1] > close[t]` 为 1，否则为 0。",
        "- 时序切分：前 70% 用于校准训练，后 30% 用于验证。",
        "- 指标：方向（Accuracy/Balanced Accuracy/MCC/AUC）+ 校准（ECE/Brier/LogLoss）。",
        "- 后校准：Isotonic 与 Platt（Logistic）。",
        "",
        "## 结果摘要",
    ]
    for s in all_summary:
        raw = s["raw"]
        iso = s["isotonic"]
        platt = s["platt"]
        lines.extend(
            [
                f"### {s['timeframe']}",
                f"- 测试样本数：`{s['n_test']}`",
                f"- 原始方向准确率：`{raw['direction']['accuracy']:.4f}`，AUC：`{raw['direction']['auc']:.4f}`",
                f"- 原始校准：ECE=`{raw['calibration']['ece']:.4f}`，Brier=`{raw['calibration']['brier']:.4f}`，LogLoss=`{raw['calibration']['log_loss']:.4f}`",
                f"- Isotonic 校准后：ECE=`{iso['calibration']['ece']:.4f}`，Brier=`{iso['calibration']['brier']:.4f}`",
                f"- Platt 校准后：ECE=`{platt['calibration']['ece']:.4f}`，Brier=`{platt['calibration']['brier']:.4f}`",
                "",
            ]
        )
    lines.extend(
        [
            "## 判读建议",
            "- 若原始 ECE 明显偏高而后校准显著下降，说明 `longPct` 更像排序分数而非天然校准概率。",
            "- 若原始和后校准差异都很小，且 reliability 曲线接近对角线，可将其视为近似概率输入仓位管理。",
            "- 建议结合 `vol_regime` 分层结果，仅在稳定有效状态下使用该信号。",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data"
    outputs_dir = root / "outputs"
    reports_dir = root / "reports"
    data_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for tf in ("5m", "15m"):
        summaries.append(_evaluate_timeframe(tf, data_dir, outputs_dir, reports_dir))

    report_text = _markdown_report(summaries)
    report_path = reports_dir / "predicta_v4_btc_5m_15m.md"
    report_path.write_text(report_text, encoding="utf-8")

    print(f"验证完成，报告已写入: {report_path}")


if __name__ == "__main__":
    main()

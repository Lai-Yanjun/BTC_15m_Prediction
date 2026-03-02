# Predicta V4 概率验证报告（BTC 现货 5m/15m）

## 方法概览
- 标签定义：`close[t+1] > close[t]` 为 1，否则为 0。
- 时序切分：前 70% 用于校准训练，后 30% 用于验证。
- 指标：方向（Accuracy/Balanced Accuracy/MCC/AUC）+ 校准（ECE/Brier/LogLoss）。
- 后校准：Isotonic 与 Platt（Logistic）。

## 结果摘要
### 5m
- 测试样本数：`68299`
- 原始方向准确率：`0.4859`，AUC：`0.4830`
- 原始校准：ECE=`0.1268`，Brier=`0.2721`，LogLoss=`0.7410`
- Isotonic 校准后：ECE=`0.0040`，Brier=`0.2500`
- Platt 校准后：ECE=`0.0038`，Brier=`0.2498`

### 15m
- 测试样本数：`22747`
- 原始方向准确率：`0.4735`，AUC：`0.4675`
- 原始校准：ECE=`0.1322`，Brier=`0.2738`，LogLoss=`0.7445`
- Isotonic 校准后：ECE=`0.0020`，Brier=`0.2500`
- Platt 校准后：ECE=`0.0023`，Brier=`0.2491`

## 判读建议
- 若原始 ECE 明显偏高而后校准显著下降，说明 `longPct` 更像排序分数而非天然校准概率。
- 若原始和后校准差异都很小，且 reliability 曲线接近对角线，可将其视为近似概率输入仓位管理。
- 建议结合 `vol_regime` 分层结果，仅在稳定有效状态下使用该信号。
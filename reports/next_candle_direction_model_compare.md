# 下一根K线方向三模型对比报告

## 任务定义
- 标签：`y = 1(close[t+1] > close[t])`，否则为 0。
- 切分：`train`=更早历史，`valid`=测试前1个月，`test`=最近1个月。
- 判定：`Accuracy > 0.5` 且 `binom_test_pvalue_vs_50 < 0.05` 视为显著高于随机。

## 5m
- 样本数：train=210362，valid=8640，test=8641，特征数=47
- 可用模型：{'logistic': True, 'lightgbm': True, 'catboost': True}

| 模型 | Accuracy | AUC | p-value(vs50%) | Threshold |
|---|---:|---:|---:|---:|
| logistic | 0.5174 | 0.5270 | 0.001248 | 0.50 |
| lightgbm | 0.5120 | 0.5290 | 0.02668 | 0.54 |
| catboost | 0.5175 | 0.5282 | 0.001158 | 0.51 |
| ensemble_soft_vote | 0.5204 | 0.5307 | 0.0001524 | 0.51 |
| ensemble_stacking | 0.5203 | 0.5306 | 0.0001661 | 0.50 |

- 最佳模型：`ensemble_soft_vote`，Accuracy=`0.5204`，AUC=`0.5307`

## 15m
- 样本数：train=70040，valid=2880，test=2881，特征数=47
- 可用模型：{'logistic': True, 'lightgbm': True, 'catboost': True}

| 模型 | Accuracy | AUC | p-value(vs50%) | Threshold |
|---|---:|---:|---:|---:|
| logistic | 0.5234 | 0.5388 | 0.01253 | 0.51 |
| lightgbm | 0.5234 | 0.5272 | 0.01253 | 0.53 |
| catboost | 0.5279 | 0.5427 | 0.002868 | 0.52 |
| ensemble_soft_vote | 0.5245 | 0.5376 | 0.009088 | 0.51 |
| ensemble_stacking | 0.5300 | 0.5415 | 0.001349 | 0.50 |

- 最佳模型：`ensemble_stacking`，Accuracy=`0.5300`，AUC=`0.5415`

## 总结口径
- 若最佳模型 Accuracy 仅略高于 0.5 且 p-value 不显著，视为缺乏稳定方向 edge。
- 若 soft-vote/stacking 持续优于单模型，后续可做滚动重训并纳入集成。
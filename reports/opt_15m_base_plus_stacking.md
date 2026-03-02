# 15m 三基模型+Stacking 优化报告

## 数据与预算
- 样本：train=70040, valid=2880, test=2881, features=47
- 搜索规模：logistic=8，lightgbm=12，catboost=12，stacking=20
- 总耗时（小时）：0.181

## 基线 vs 优化后（test）
- baseline_stacking: acc=0.5300, auc=0.5415, p=0.001349
- tuned_stacking: acc=0.5318, auc=0.5407, p=0.0006943

## 最优参数
- logistic: {'valid_acc': 0.5506944444444445, 'params': {'C': 0.1, 'max_iter': 400}, 'threshold': 0.5100000000000001}
- lightgbm: {'valid_acc': 0.5434027777777778, 'params': {'num_leaves': 15, 'min_child_samples': 80, 'learning_rate': 0.02, 'n_estimators': 1000, 'subsample': 0.9, 'colsample_bytree': 0.9}, 'threshold': 0.5100000000000001}
- catboost: {'valid_acc': 0.5503472222222222, 'params': {'depth': 6, 'l2_leaf_reg': 6.0, 'learning_rate': 0.02, 'iterations': 1000, 'verbose': False}, 'threshold': 0.5100000000000001}
- stacking: {'valid_acc': 0.5496527777777778, 'meta_params': {'C': 5.0, 'max_iter': 400}, 'subset': ['p_logistic', 'p_lightgbm'], 'threshold': 0.5000000000000001}

## 滚动稳定性（tuned stacking）
- windows=6, mean_acc=0.5212, win_rate_over_50=1.00, sig_rate=0.67

## 连败风险（fixed split）
- baseline_stacking: {'max_streak': 8, 'p95_streak': 4.0, 'freq_ge_8': 0.004092769440654843, 'count_streaks': 733}
- tuned_stacking: {'max_streak': 8, 'p95_streak': 4.0, 'freq_ge_8': 0.0013605442176870747, 'count_streaks': 735}

## 阶段耗时（秒）
- stage1_baseline=27.0
- stage2_base_tuning=443.5
- stage3_stack_tuning=0.3
- stage4_rolling_validate=139.0
- stage5_streak_risk=0.0
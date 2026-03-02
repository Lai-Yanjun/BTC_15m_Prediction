from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover - optional dependency
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except Exception:  # pragma: no cover - optional dependency
    CatBoostClassifier = None


@dataclass
class TrainedModels:
    logistic: Optional[Pipeline]
    lightgbm: Optional[object]
    catboost: Optional[object]
    availability: dict[str, bool]


def _merge_params(defaults: dict, overrides: Optional[dict]) -> dict:
    if not overrides:
        return defaults
    out = defaults.copy()
    out.update(overrides)
    return out


def train_logistic(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    params: Optional[dict] = None,
) -> Pipeline:
    clf_defaults = {
        "C": 0.5,
        "max_iter": 400,
        "solver": "lbfgs",
        "n_jobs": None,
        "random_state": 42,
    }
    clf_params = _merge_params(clf_defaults, params)
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(**clf_params),
            ),
        ]
    )
    model.fit(x_train, y_train)
    return model


def train_lightgbm(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    params: Optional[dict] = None,
) -> Optional[object]:
    if LGBMClassifier is None:
        return None
    defaults = {
        "objective": "binary",
        "n_estimators": 1200,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 0.2,
        "min_child_samples": 60,
        "random_state": 42,
        "n_jobs": -1,
    }
    model = LGBMClassifier(**_merge_params(defaults, params))
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_valid, y_valid)],
        eval_metric="binary_logloss",
        callbacks=[],
    )
    return model


def train_catboost(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    params: Optional[dict] = None,
) -> Optional[object]:
    if CatBoostClassifier is None:
        return None
    defaults = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "iterations": 1200,
        "learning_rate": 0.03,
        "depth": 6,
        "l2_leaf_reg": 6.0,
        "random_seed": 42,
        "verbose": False,
    }
    model = CatBoostClassifier(**_merge_params(defaults, params))
    model.fit(x_train, y_train, eval_set=(x_valid, y_valid), verbose=False)
    return model


def train_all_models(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: list[str],
    y_col: str = "y_true",
    logistic_params: Optional[dict] = None,
    lightgbm_params: Optional[dict] = None,
    catboost_params: Optional[dict] = None,
) -> TrainedModels:
    x_train = train_df[feature_cols]
    y_train = train_df[y_col].astype(int)
    x_valid = valid_df[feature_cols]
    y_valid = valid_df[y_col].astype(int)

    logistic = train_logistic(x_train, y_train, params=logistic_params)
    lightgbm = train_lightgbm(x_train, y_train, x_valid, y_valid, params=lightgbm_params)
    catboost = train_catboost(x_train, y_train, x_valid, y_valid, params=catboost_params)

    availability = {
        "logistic": logistic is not None,
        "lightgbm": lightgbm is not None,
        "catboost": catboost is not None,
    }
    return TrainedModels(
        logistic=logistic,
        lightgbm=lightgbm,
        catboost=catboost,
        availability=availability,
    )


def _get_prob(model: object, x: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(x)[:, 1]


def predict_prob_table(
    models: TrainedModels,
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    x = df[feature_cols]
    out = pd.DataFrame(index=df.index)
    out["p_logistic"] = _get_prob(models.logistic, x)
    if models.lightgbm is not None:
        out["p_lightgbm"] = _get_prob(models.lightgbm, x)
    if models.catboost is not None:
        out["p_catboost"] = _get_prob(models.catboost, x)
    return out


def soft_vote(prob_df: pd.DataFrame) -> np.ndarray:
    cols = [c for c in prob_df.columns if c.startswith("p_")]
    return prob_df[cols].mean(axis=1).to_numpy()


def fit_stacking_meta(valid_prob_df: pd.DataFrame, y_valid: pd.Series, params: Optional[dict] = None) -> LogisticRegression:
    cols = [c for c in valid_prob_df.columns if c.startswith("p_")]
    x_meta = valid_prob_df[cols].to_numpy()
    y_meta = y_valid.astype(int).to_numpy()
    defaults = {"C": 1.0, "max_iter": 300, "solver": "lbfgs", "random_state": 42}
    meta = LogisticRegression(**_merge_params(defaults, params))
    meta.fit(x_meta, y_meta)
    return meta


def predict_stacking(meta_model: LogisticRegression, test_prob_df: pd.DataFrame) -> np.ndarray:
    cols = [c for c in test_prob_df.columns if c.startswith("p_")]
    return meta_model.predict_proba(test_prob_df[cols].to_numpy())[:, 1]

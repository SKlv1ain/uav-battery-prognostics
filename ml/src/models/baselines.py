"""Baseline regressors used in both pipelines."""

from __future__ import annotations

from sklearn.linear_model import LinearRegression


def build_linear_regression() -> LinearRegression:
    return LinearRegression()


def build_xgboost(
    *,
    n_estimators: int = 300,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    random_state: int = 42,
    n_jobs: int = -1,
):
    from xgboost import XGBRegressor

    return XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
        n_jobs=n_jobs,
    )

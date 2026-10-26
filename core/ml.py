"""Leak-free feature engineering and model evaluation for price prediction.

Key rule: every feature on day t uses only information available at the close
of day t, and the target is the return from t to t+1. Models are always
compared with a naive baseline ("tomorrow = today"), because predicting raw
prices makes almost any model look accurate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from . import indicators as ind

try:
    import xgboost as xgb
except ImportError:  # optional
    xgb = None


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    ret = close.pct_change()
    feats = pd.DataFrame(index=df.index)
    for lag in (1, 2, 3, 5, 10):
        feats[f"ret_lag{lag}"] = ret.shift(lag - 1)  # lag1 = today's return
    feats["ret_5d"] = close.pct_change(5)
    feats["ret_20d"] = close.pct_change(20)
    feats["vol_10d"] = ret.rolling(10).std()
    feats["vol_20d"] = ret.rolling(20).std()
    feats["px_sma10"] = close / ind.sma(close, 10) - 1
    feats["px_sma50"] = close / ind.sma(close, 50) - 1
    feats["rsi14"] = ind.rsi(close, 14) / 100
    m = ind.macd(close)
    feats["macd_hist"] = m["Histogram"] / close
    feats["range"] = (df["High"] - df["Low"]) / close
    vol = df["Volume"].replace(0, np.nan)
    feats["volume_z"] = (vol - vol.rolling(20).mean()) / vol.rolling(20).std()
    return feats.replace([np.inf, -np.inf], np.nan)


def make_dataset(df: pd.DataFrame):
    """X (features at t), y (return t -> t+1), plus the close at t for
    converting predicted returns back to prices. The final row has no target
    and is returned separately for the next-day forecast."""
    feats = build_features(df)
    target = df["Close"].pct_change().shift(-1).rename("target")
    data = pd.concat([feats, target, df["Close"].rename("close_t")], axis=1)
    latest = data.drop(columns=["target"]).dropna().iloc[[-1]]
    data = data.dropna()
    return data[feats.columns], data["target"], data["close_t"], latest


def model_zoo() -> dict:
    models = {
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Ridge Regression": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "Decision Tree": DecisionTreeRegressor(max_depth=4, min_samples_leaf=30, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=5, min_samples_leaf=20,
                                               random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, max_depth=2,
                                                       learning_rate=0.03, subsample=0.8, random_state=42),
        "K-Nearest Neighbors": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=25)),
        "Support Vector Regression": make_pipeline(StandardScaler(), SVR(C=0.1, epsilon=0.005)),
    }
    if xgb is not None:
        models["XGBoost"] = xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.03,
                                             subsample=0.8, colsample_bytree=0.8, random_state=42)
    return models


def evaluate(df: pd.DataFrame, model_names: list[str], test_size: float = 0.2) -> dict:
    """Chronological train/test split. Returns metrics, test predictions and
    next-day forecasts for each model, plus the naive baseline."""
    X, y, close_t, latest = make_dataset(df)
    split = int(len(X) * (1 - test_size))
    X_tr, X_te, y_tr, y_te = X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]
    close_te = close_t.iloc[split:]
    actual_price = close_te * (1 + y_te)  # close at t+1
    actual_price.index = y_te.index

    zoo = model_zoo()
    rows, preds, forecasts = [], {}, {}

    def score(name, pred_ret):
        pred_price = close_te.values * (1 + pred_ret)
        direction_hit = np.mean(np.sign(pred_ret) == np.sign(y_te.values)) if name != "Naive (no change)" else np.nan
        rows.append({
            "Model": name,
            "MAE ($)": mean_absolute_error(actual_price, pred_price),
            "RMSE ($)": float(np.sqrt(mean_squared_error(actual_price, pred_price))),
            "Return MAE (%)": mean_absolute_error(y_te, pred_ret) * 100,
            "Direction accuracy": direction_hit,
        })
        preds[name] = pd.Series(pred_price, index=y_te.index)

    score("Naive (no change)", np.zeros(len(y_te)))
    last_close = float(latest["close_t"].iloc[0])
    for name in model_names:
        model = zoo[name]
        model.fit(X_tr, y_tr)
        score(name, model.predict(X_te))
        # refit on everything for the next-day forecast
        model.fit(X, y)
        r = float(model.predict(latest[X.columns])[0])
        forecasts[name] = {"return": r, "price": last_close * (1 + r)}

    metrics = pd.DataFrame(rows).set_index("Model")
    naive_mae = metrics.loc["Naive (no change)", "MAE ($)"]
    metrics["vs naive"] = metrics["MAE ($)"] / naive_mae - 1
    up_share = float((y_te > 0).mean())
    return {
        "metrics": metrics, "predictions": preds, "actual": actual_price,
        "forecasts": forecasts, "last_close": last_close, "last_date": latest.index[-1],
        "n_train": len(X_tr), "n_test": len(X_te), "up_share": up_share,
        "feature_names": list(X.columns),
    }


def feature_importance(df: pd.DataFrame) -> pd.Series:
    X, y, _, _ = make_dataset(df)
    rf = RandomForestRegressor(n_estimators=200, max_depth=5, min_samples_leaf=20, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    return pd.Series(rf.feature_importances_, index=X.columns).sort_values()

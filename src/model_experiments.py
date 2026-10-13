"""
src/model_experiments.py
========================
Deliverable D2 (extended) - systematic model / encoder / scaler experiment.

Runs every combination of

    13 model families  x  2 categorical encoders  x  3 numeric scalers
    (= 78 configurations, plus CatBoost with its native categorical handling)

on the *same* chronological validation fortnight (18-31 Oct 2025) with the same
leakage-safe feature matrix from `src.features`, and writes a single leaderboard
`reports/model_grid.csv` (raw and anomaly-trimmed metrics + train time).

Encoders : onehot  -> sklearn OneHotEncoder on `zone_id`
           label   -> sklearn OrdinalEncoder on `zone_id`
Scalers  : standard -> StandardScaler
           minmax   -> MinMaxScaler
           none     -> no scaling

Usage:  python src/model_experiments.py
"""

from __future__ import annotations
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                              HistGradientBoostingRegressor, GradientBoostingRegressor)
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.train import build_split, feature_columns, load_masters, VAL_START, VAL_END, RS
from src.features import target_cap

warnings.filterwarnings("ignore")
ANOMALY_TRIPS = 150
CAT = "zone_id"


def estimator(name):
    """A fresh estimator for every grid cell (modest budgets keep the grid fast)."""
    if name == "Ridge":
        return Ridge(alpha=5.0)
    if name == "Lasso":
        return Lasso(alpha=0.1, max_iter=5000)
    if name == "ElasticNet":
        return ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000)
    if name == "AdaBoost":
        return AdaBoostRegressor(n_estimators=120, learning_rate=0.05, random_state=RS)
    if name == "MLP":
        return MLPRegressor(hidden_layer_sizes=(96, 48), max_iter=60, early_stopping=True,
                            random_state=RS)
    if name == "KNN":
        return KNeighborsRegressor(n_neighbors=12, weights="distance", n_jobs=-1)
    if name == "DecisionTree":
        return DecisionTreeRegressor(max_depth=14, min_samples_leaf=4, random_state=RS)
    if name == "RandomForest":
        return RandomForestRegressor(n_estimators=100, min_samples_leaf=3, max_features="sqrt",
                                     random_state=RS, n_jobs=-1)
    if name == "ExtraTrees":
        return ExtraTreesRegressor(n_estimators=100, min_samples_leaf=3, max_features="sqrt",
                                   random_state=RS, n_jobs=-1)
    if name == "HistGradientBoosting":
        return HistGradientBoostingRegressor(max_iter=300, max_depth=8, learning_rate=0.05,
                                             random_state=RS)
    if name == "GradientBoosting":
        return GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.05,
                                         subsample=0.9, random_state=RS)
    if name == "LightGBM":
        return lgb.LGBMRegressor(n_estimators=300, num_leaves=63, learning_rate=0.05,
                                 subsample=0.9, colsample_bytree=0.9, random_state=RS,
                                 n_jobs=-1, verbose=-1)
    if name == "XGBoost":
        return xgb.XGBRegressor(n_estimators=300, max_depth=8, learning_rate=0.05, subsample=0.9,
                                colsample_bytree=0.9, random_state=RS, n_jobs=-1, tree_method="hist")
    raise ValueError(name)


MODELS = ["Ridge", "Lasso", "ElasticNet", "KNN", "DecisionTree", "AdaBoost", "RandomForest",
          "ExtraTrees", "GradientBoosting", "HistGradientBoosting", "LightGBM", "XGBoost", "MLP"]

# 13 model families x 2 categorical encoders x 3 numeric scalers = 78 configs,
# plus CatBoost with its native categorical handling (the reference row).
ENCODERS = ["onehot", "label"]
SCALERS = ["standard", "minmax", "none"]


def clean_matrix(df, feats):
    return df[feats].astype(float).replace([np.inf, -np.inf], np.nan)


def encode(Xtr, Xv, encoder, scaler):
    """Fit encoder + scaler on train only, return numpy matrices."""
    if encoder == "onehot":
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        A = enc.fit_transform(Xtr[[CAT]])
        B = enc.transform(Xv[[CAT]])
        Xtr = np.hstack([Xtr.drop(columns=[CAT]).to_numpy(float), A])
        Xv = np.hstack([Xv.drop(columns=[CAT]).to_numpy(float), B])
    else:  # label / ordinal
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        Xtr = Xtr.copy(); Xv = Xv.copy()
        Xtr[CAT] = enc.fit_transform(Xtr[[CAT]])
        Xv[CAT] = enc.transform(Xv[[CAT]])
        Xtr = Xtr.to_numpy(float); Xv = Xv.to_numpy(float)
    if scaler == "standard":
        sc = StandardScaler()
    elif scaler == "minmax":
        sc = MinMaxScaler()
    else:
        sc = None
    if sc is not None:
        Xtr = sc.fit_transform(np.nan_to_num(Xtr, nan=-1.0))
        Xv = sc.transform(np.nan_to_num(Xv, nan=-1.0))
    else:
        Xtr = np.nan_to_num(Xtr, nan=-1.0); Xv = np.nan_to_num(Xv, nan=-1.0)
    return Xtr, Xv


def main():
    os.makedirs("reports", exist_ok=True)
    tr_raw, _ = load_masters()
    hist, val = build_split(tr_raw, tr_raw, VAL_START)
    hist = hist[hist["trips"].notna()]
    val = val[(val["pickup_datetime"] >= VAL_START) & (val["pickup_datetime"] < VAL_END) &
              val["trips"].notna()]
    feats = feature_columns(hist)
    cap = target_cap(hist)
    ytr = np.clip(hist["trips"].values, 0, cap)
    yval = val["trips"].values
    rob = yval <= ANOMALY_TRIPS
    Xtr0, Xval0 = clean_matrix(hist, feats), clean_matrix(val, feats)
    print(f"grid on {len(feats)} features | train {Xtr0.shape} | val {Xval0.shape} | "
          f"cap {cap:.0f} | anomalies {int((~rob).sum())}", flush=True)

    rows = []

    def record(model, enc, sc, pred, secs):
        rows.append({"model": model, "encoder": enc, "scaler": sc,
                     "RMSE": round(float(np.sqrt(mean_squared_error(yval, pred))), 4),
                     "MAE": round(float(mean_absolute_error(yval, pred)), 4),
                     "R2": round(float(r2_score(yval, pred)), 4),
                     "robust_RMSE": round(float(np.sqrt(mean_squared_error(yval[rob], pred[rob]))), 4),
                     "robust_R2": round(float(r2_score(yval[rob], pred[rob])), 4),
                     "train_time_s": round(secs, 2)})
        r = rows[-1]
        print(f"  {model:20s} {enc:6s} {sc:8s} RMSE {r['RMSE']:6.3f} R2 {r['R2']:.4f} "
              f"robust_R2 {r['robust_R2']:.4f} ({r['train_time_s']:.1f}s)", flush=True)

    # -------- 10 models x 2 encoders x 3 scalers --------
    for model in MODELS:
        for enc in ENCODERS:
            for sc in SCALERS:
                Xtr, Xv = encode(Xtr0, Xval0, enc, sc)
                est = estimator(model)
                t0 = time.time()
                est.fit(Xtr, ytr)
                pred = np.clip(est.predict(Xv), 0, None)
                record(model, enc, sc, pred, time.time() - t0)

    # -------- CatBoost with native categorical handling (reference row) --------
    Xtr = Xtr0.copy(); Xv = Xval0.copy()
    Xtr[CAT] = Xtr[CAT].astype(int).astype(str); Xv[CAT] = Xv[CAT].astype(int).astype(str)
    t0 = time.time()
    cb = CatBoostRegressor(iterations=500, depth=7, learning_rate=0.05, random_seed=RS,
                           verbose=0, thread_count=-1)
    cb.fit(Xtr, ytr, cat_features=[CAT])
    record("CatBoost", "native", "none", np.clip(cb.predict(Xv), 0, None), time.time() - t0)

    grid = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    grid.to_csv("reports/model_grid.csv", index=False)

    # compact per-encoder / per-scaler view of the winner
    best = grid.iloc[0]
    summary = (grid.groupby(["encoder", "scaler"])["robust_R2"].mean().reset_index()
               .pivot(index="encoder", columns="scaler", values="robust_R2"))
    summary.to_csv("reports/model_grid_summary.csv")
    print(f"\nBEST GRID CELL: {best['model']} | {best['encoder']} | {best['scaler']} "
          f"| RMSE {best['RMSE']} R2 {best['R2']} robust_R2 {best['robust_R2']}", flush=True)
    print("\nmean robust_R2 by encoder x scaler:\n", summary.round(4).to_string(), flush=True)
    print("\nTop 10 configurations:\n",
          grid.head(10).to_string(index=False), flush=True)
    return grid


if __name__ == "__main__":
    main()

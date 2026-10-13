"""
src/predict.py
==============
Generate the submission file for the 4,032 test zone-hours using the trained
model. The feature matrix was produced by the same leakage-safe pipeline used
for training (src/features.py via src/train.py).
"""

from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SUB_PATH = "submission/team_demandflow_submission.csv"
TEMPLATE = "data/raw/submission_template.csv"


def generate_submission():
    art = joblib.load("models/final_model.joblib")
    model, feats = art["model"], art["features"]
    test = pd.read_csv("data/processed/test_features.csv")
    print(f"model: {art.get('model_name')} | test features: {test.shape}")

    pred = np.clip(model.predict(test[feats]), 0, None)
    sub = pd.DataFrame({"row_id": test["row_id"], "predicted_trips": np.round(pred, 2)})

    # validity (G4): exact columns, 4,032 unique ids in template order, no blanks/negatives
    tmpl = pd.read_csv(TEMPLATE)
    assert list(sub.columns) == ["row_id", "predicted_trips"], "wrong columns"
    assert len(sub) == 4032, f"expected 4032 rows, got {len(sub)}"
    assert sub["row_id"].is_unique, "duplicate row_id"
    assert sub["row_id"].tolist() == tmpl["row_id"].tolist(), "row_id order differs from template"
    assert sub["predicted_trips"].notna().all(), "blank prediction"
    assert (sub["predicted_trips"] >= 0).all(), "negative prediction"

    os.makedirs("submission", exist_ok=True)
    sub.to_csv(SUB_PATH, index=False)
    print(f"wrote {SUB_PATH}: {len(sub)} rows, mean {sub.predicted_trips.mean():.2f}, "
          f"max {sub.predicted_trips.max():.1f}")
    print("submission validity checks: PASS")


if __name__ == "__main__":
    generate_submission()

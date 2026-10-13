"""
src/sync_app_assets.py
======================
Deliverable E/G — bundle the *current* artifacts with the demo app.

The Streamlit app prefers `app/assets/` so it stays self-contained. This copies
the freshly generated master tables, engineered test-feature matrix, model and
submission into `app/assets/`, so the demo and the scored submission can never
drift apart.

Usage:  python src/sync_app_assets.py
"""

from __future__ import annotations
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "app", "assets")

COPIES = [
    ("data/processed/master_train.csv", "app/assets/master_train.csv"),
    ("data/processed/master_test.csv", "app/assets/master_test.csv"),
    ("data/processed/data_dictionary_master.csv", "app/assets/data_dictionary_master.csv"),
    ("data/processed/dim_zone.csv", "app/assets/dim_zone.csv"),
    ("data/processed/zone_typology.csv", "app/assets/zone_typology.csv"),
    ("data/processed/cleaning_audit.csv", "app/assets/cleaning_audit.csv"),
    ("data/processed/weather_hourly_clean.csv", "app/assets/weather_hourly_clean.csv"),
    ("data/processed/event_zone_hour.csv", "app/assets/event_zone_hour.csv"),
    ("data/processed/test_features.csv", "app/assets/test_features.csv"),
    ("models/final_model.joblib", "app/assets/final_model.joblib"),
    ("submission/team_demandflow_submission.csv", "app/assets/team_demandflow_submission.csv"),
    ("reports/final_model_summary.csv", "app/assets/final_model_summary.csv"),
    ("reports/model_comparison_leaderboard.csv", "app/assets/model_comparison_leaderboard.csv"),
    ("reports/cleaning_audit.csv", "app/assets/cleaning_audit.csv"),
    ("reports/A_integrity_checks.csv", "app/assets/A_integrity_checks.csv"),
]


def main():
    os.makedirs(ASSETS, exist_ok=True)
    missing = []
    for src, dst in COPIES:
        s, d = os.path.join(ROOT, src), os.path.join(ROOT, dst)
        if not os.path.exists(s):
            missing.append(src)
            continue
        shutil.copy2(s, d)
        print(f"  {src} -> {dst}")
    if missing:
        print("SKIPPED (source missing):", ", ".join(missing))
    print(f"synced {len(COPIES) - len(missing)} assets into app/assets/")


if __name__ == "__main__":
    main()

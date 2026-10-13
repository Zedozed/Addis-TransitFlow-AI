# 🚕 Addis DemandFlow — Ride Demand Intelligence for Addis Ababa

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Model](https://img.shields.io/badge/%E2%85%A0%C2%B2-0.9045-success)
![RMSE](https://img.shields.io/badge/RMSE-8.35-blue)
![Zones](https://img.shields.io/badge/Zone--hours-4032-informational)

> **Team `team_demandflow`** · Hourly ride-demand forecasting + operational dispatch
> decision-support for the 12 canonical zones of Addis Ababa.

---

## ✨ Why DemandFlow

Buses, ride-hailing fleets and dispatchers in Addis Ababa do not know how many drivers a
zone needs *next hour*. Demand spikes around **afternoon rainstorms**, **football matches**,
**concerts** and **holidays**, while supply chases it late. **Addis DemandFlow** closes that
gap with a two-part product:

1. **A forecast engine** — pairs the 14-day official test fortnight (1–14 Nov 2025) with a
   live what-if simulator that answers *"how would demand shift if it starts raining now?"*.
2. **A Streamlit operations platform** — 10 analytics pages (executive dashboard, demand,
   zone, weather, events, forecasting, model performance, data quality, data architecture,
   about) built directly on the pipeline's cleaned and joined artifacts.

## 🚀 Headline results (every number from this repo's artifacts)

| Metric | **DemandFlow blend** | Seasonal-naive baseline | Gain |
| :-- | --: | --: | --: |
| **R² (3,914 held-out zone-hours)** | **0.9045** | 0.8252 | **+0.079** |
| **RMSE** | **8.352** | 11.299 | **−26.1%** |
| **MAE** | **5.738** | 7.360 | **−22.0%** |

- **Model of record:** 50/50 blend of `CatBoost(800, depth 6)` + `MLP(96, 48)` over **57
  leakage-safe features**, chosen from a **13-model × 2-encoder × 3-scaler grid (78
  configurations + CatBoost native)** on a strictly chronological holdout (18–31 Oct 2025).
- **Robust by design:** beats the seasonal-naive baseline on all **4 rolling-origin folds**
  (RMSE 8.93 ± 0.66 vs 11.78).
- **Signal pays:** ablation R² climbs calendar+trend **0.8722** → +weather **0.8820** →
  +events **0.8959** → both **0.9037**.
- **Leakage audit:** feeding forecast-time-unknown columns (`active_drivers`,
  `avg_wait_min`, `avg_fare_birr`) inflates R² to 0.986 — exactly why they are excluded.
- **Data repair, not data removal:** 125 impossible `trips` rows (a ×8 export-scaling bug)
  were repaired as `trips/8` rather than dropped, keeping every zone-hour in the target.

---

## 📊 The pipeline in one look

```text
85,460 raw trip rows ─┐
 7,538 weather hours ─┼─► cleaning → 12-zone dimension · EAT clock proof · joins
165 calendar events  ─┘             → 84,614 master zone-hours · 57 features
                                    → 79-config model search → CatBoost+MLP blend
                                    → 4,032-row scored submission (R² 0.904)
                                    → Streamlit platform · report (md/docx/pdf) · decks
```

Key artifacts, kept reproducible end-to-end:

| Artifact | Where | What it proves / provides |
| :-- | :-- | :-- |
| Scored prediction | `submission/team_demandflow_submission.csv` | 4,032 rows · `[row_id, predicted_trips]` · validity asserted |
| Clean master tables | `data/processed/master_train.csv` (84,614) / `master_test.csv` (4,032) | one row = one zone + one hour |
| Integrity checks | `reports/A_integrity_checks.csv` | 12 automated PASS/FAIL assertions |
| 12 visualizations | `figures/fig01…fig12` + `figure_captions.md` | demand anatomy, rain dose-response, event uplift, importance |
| Model leaderboard | `reports/model_comparison_leaderboard.csv` · `model_grid.csv` | 16 ranked families + the 79-cell experiment grid |
| Full model report | `reports/model_report.md` / `.docx` / `.pdf` | models, lags, ablation, baselines, rolling origin |
| Demo platform | `app/app.py` | the 10-page operations dashboard |
| Trained model | `models/final_model.joblib` | fitted model + exact feature list |

## 🛠️ Quick start

```bash
# 1. Environment
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Full pipeline, in this order
python src/cleaning.py            # raw → master tables + 12 integrity checks
python src/report_a.py            # writes reports/A_cleaning_and_integration.md
python src/analysis.py            # writes reports/B_analysis_report.md
python src/train.py               # leaderboard, tuning, ablation, rolling origin, final model
python src/model_experiments.py   # 13 models × 2 encoders × 3 scalers grid
python src/predict.py             # writes submission/team_demandflow_submission.csv
python src/visualize.py           # figures/fig01…fig12 + captions
python src/make_slides.py         # presentation/team_demandflow_slides.{pptx,pdf,md}
python src/make_report.py         # reports/model_report.{md,docx,pdf} + full deck
python src/sync_app_assets.py     # bundles artifacts into app/assets/

# 3. Launch the platform
streamlit run app/app.py          # opens at http://localhost:8501
```

Prefer notebooks? Run `notebooks/01_cleaning_and_integration.ipynb` →
`02_analysis_report.ipynb` → `03_visualizations.ipynb` →
`04_modeling_and_evaluation.ipynb` top-to-bottom.

---

## 📁 Repository layout

```text
addis-demandflow/
├── README.md                                  # this file
├── requirements.txt                           # pinned package versions
├── submission/                                # scored prediction
│   └── team_demandflow_submission.csv         # 4,032 rows [row_id, predicted_trips]
├── data/
│   ├── raw/                                   # immutable original exports
│   └── processed/                             # master tables, dictionary, features
├── notebooks/                                 # runnable drivers (A–D)
├── src/                                       # reusable pipeline code (A–G)
├── models/final_model.joblib                  # fitted model + feature list
├── figures/                                   # 12 PNGs + captions
├── reports/                                   # A/B/D reports, grids, audits, model report
├── app/                                       # Streamlit operations platform (demo)
│   ├── app.py
│   └── assets/                                # bundled artifacts for the demo
└── presentation/                              # 5-slide deck + full report deck
    ├── team_demandflow_slides.pptx / .pdf / .md
    └── team_demandflow_full_report.pptx
```

## ✅ Verified checklist

- [x] **Submission** — exact columns, template order, no blanks/negatives/duplicates
      (asserted in `src/predict.py`).
- [x] **12 visualizations** with exact required names + captions.
- [x] **12 automated integrity checks** all PASS (`reports/A_integrity_checks.csv`).
- [x] **Chronological validation only** — the test fortnight is never scored during model
      selection; `random_state=42` everywhere; `data/raw/` never edited.
- [x] **One-command demo** — `streamlit run app/app.py`.

---

## 👤 Developer & contact

Built and maintained by **Zedingil Zelalem** — feel free to connect:

| | |
| :-- | :-- |
| 💼 **Portfolio** (includes GitHub & social links) | [zedingil.vercel.app](https://zedingil.vercel.app) |
| 🌐 **Ethio Bacos Tech Hub** | [ethio-bacos-tech-hub.vercel.app](https://ethio-bacos-tech-hub.vercel.app) |

Want to connect on GitHub? Visit the portfolio above — the GitHub profile is linked there,
or open an issue in this repository and I'll get back to you.

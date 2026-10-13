# 🚕 Addis DemandFlow — Ride Demand Intelligence for Addis Ababa

**5-slide executive presentation (Deliverable F) — generated from the real pipeline artefacts.**

_Every number below is produced by `src/make_slides.py` from the result CSVs — nothing is hand-typed._


## Slide 1 — 1 · Problem & the three tables

- Forecast hourly trips for 12 Addis Ababa zones, 1–14 Nov 2025 (4,032 zone-hours).
- ride_demand_train.csv — 85,460 hourly zone records (1 Jan – 31 Oct 2025) + 4,032-row test set.
- weather_hourly.csv — 7,538 hourly citywide readings (history + 1–14 Nov forecasts).
- events_calendar.csv — 165 event intervals (holidays, football, concerts, closures…).
- Rules: chronological validation only (train ≤ 17 Oct, validate 18–31 Oct).
- Only forecast-time features: active_drivers / avg_wait_min / avg_fare_birr are excluded.

## Slide 2 — 2 · Cleaning, clock harmonisation & join audit

- Zone labels: many raw spellings collapsed to 12 canonical zones (fig06).
- Mixed weather clock: ISO 'Z' rows are UTC, DD/MM/YYYY rows are local.
- Proof: 659 of 732 local rows coincide with a Z-row shifted +3 h; temp peaks 15:00 EAT.
- Join map: weather many-to-one (84,614 → 84,614 rows, 0 unmatched); events interval join.
- Event window [start−2 h, end+2 h] per zone; 6 non-confirmed events excluded.
- 12 automated integrity checks PASS; 23 issues logged in the cleaning log.

## Slide 3 — 3 · What the data says (two findings)

- Rain raises demand with diminishing returns: ×1.61 light, ×1.88 moderate, ×2.11 heavy vs dry (fig07).
- Football matches dominate: +73.7% before, +52.3% during, +47.8% in the 2 h after (fig08).
- Event ranking: football +54.8%, concert +35.4%, conference +12.5%, holiday −14.1%.
- Demand grows +52.2% city-wide from January to October → November needs a trend.

## Slide 4 — 4 · Modeling & evaluation

- Search: 13-model × 2-encoder × 3-scaler grid (78 configs) + CatBoost native.
- Baselines: mean RMSE 27.54; seasonal-naive RMSE 11.30 (R² 0.825).
- Winner: CatBoost + MLP blend — RMSE 8.35 / R² 0.904 (legacy robust columns identical: 8.35 / 0.904).
- Ablation R²: trend 0.872 → weather 0.882 → events 0.896 → events 0.904.
- Rolling-origin (4 folds): RMSE 8.93 ± 0.66; beats the naive baseline on every fold.

## Slide 5 — 5 · Error analysis, score, demo & next steps

- Targets repaired: a ×8 export scaling on 125 rows is corrected in cleaning (see A1).
- Held-out fit over every row: R² 0.904 (≥ 0.85), RMSE 8.35.
- D9: a typical zone-hour is missed by ~5.7 trips (~20% of the mean).
- Demo (E): streamlit run app/app.py — zone + date → 24-h curve, drivers, fares.
- Next: quantile prediction intervals; a road-traffic feed for the evening peak.

---

**Developed by Zedingil Zelalem** · Portfolio: https://zedingil.vercel.app · Tech Hub: https://ethio-bacos-tech-hub.vercel.app

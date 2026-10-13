# Deliverable D - Modeling & Evaluation (D1-D9)

All scores are computed on chronological held-out weeks of `master_train.csv`. The test file is never scored.

**No rows are excluded from scoring.** An earlier version of this report trimmed held-out rows as 'corrupted exports' using a `trips > 150` cutoff (8 rows in this fortnight). That cutoff was mis-specified: it discarded genuine peak hours while missing the real defect. Investigation showed the defect is a **x8 scaling of the `trips` column** on 125 rows - a trips/active_drivers ratio of 9.8-14.1 against a healthy maximum of 1.85, with every flagged value an exact multiple of 8, and dividing by 8 returns the ratio to the healthy band (mean 1.30). `src/cleaning.py` now **repairs** those rows (`trips <- trips/8`, flagged in `outlier_trips`) rather than discarding them, so both the training target and the evaluation ground truth are correct and no information is thrown away. There is therefore a single unfiltered metric; the `robust_*` columns are kept for backward compatibility and are now identical to it. See A1 of the cleaning report. The training target keeps only a light winsorisation at its 99.9th percentile (147), set high enough not to clip genuine peaks.

## D1 Baselines (validation fortnight 18-31 Oct 2025)

- Mean predictor: RMSE **27.538**, MAE 20.689
- Seasonal-naive (zone x day-of-week x hour mean): RMSE **11.299**, MAE **7.360**, R2 0.825, robust R2 0.825

## D2 Model comparison (identical split & features)

| Model                               |   train_time_s |    RMSE |     MAE |      R2 |   MAPE_% |   robust_RMSE |   robust_R2 |
|:------------------------------------|---------------:|--------:|--------:|--------:|---------:|--------------:|------------:|
| Ensemble: CatBoost + MLP (blended)  |          35.07 |  8.3523 |  5.7383 |  0.9045 |    28.82 |        8.3523 |      0.9045 |
| CatBoostRegressor (tuned)           |          24.98 |  8.5079 |  5.7789 |  0.9009 |    26.8  |        8.5079 |      0.9009 |
| CatBoost                            |          34.67 |  8.5218 |  5.7861 |  0.9006 |    26.78 |        8.5218 |      0.9006 |
| MLP                                 |          35.07 |  8.5849 |  5.9726 |  0.8991 |    31.9  |        8.5849 |      0.8991 |
| ExtraTrees                          |          21.44 |  8.699  |  5.8518 |  0.8964 |    26.13 |        8.699  |      0.8964 |
| RandomForest                        |          45.11 |  8.7285 |  5.8905 |  0.8957 |    26.42 |        8.7285 |      0.8957 |
| HistGradientBoosting                |          21.38 |  8.7581 |  5.9391 |  0.895  |    26.95 |        8.7581 |      0.895  |
| GradientBoosting                    |         161.91 |  8.7749 |  5.9185 |  0.8946 |    27.01 |        8.7749 |      0.8946 |
| XGBoost                             |          35.39 |  8.8274 |  5.9624 |  0.8933 |    27.15 |        8.8274 |      0.8933 |
| LightGBM                            |          16.15 |  8.837  |  5.9854 |  0.8931 |    26.94 |        8.837  |      0.8931 |
| Ridge                               |           0.87 |  9.2087 |  6.4142 |  0.8839 |    38.77 |        9.2087 |      0.8839 |
| Lasso                               |           3.44 |  9.3731 |  6.4391 |  0.8797 |    37.68 |        9.3731 |      0.8797 |
| KNN                                 |           3.21 |  9.7012 |  6.407  |  0.8712 |    28.76 |        9.7012 |      0.8712 |
| DecisionTree                        |           3.58 | 10.193  |  6.8528 |  0.8578 |    29.67 |       10.193  |      0.8578 |
| Seasonal-naive (zone/dow/hour mean) |           0    | 11.2985 |  7.36   |  0.8252 |    26.43 |       11.2985 |      0.8252 |
| Mean predictor (baseline)           |           0    | 27.5381 | 20.6891 | -0.0381 |   163.89 |       27.5381 |     -0.0381 |

**Selection rule:** the leaderboard is sorted by RMSE over *all* held-out rows. With the x8 export defect repaired there is nothing to trim: the unfiltered metric is now a fair measure of genuine demand fit.

**Winner:** Ensemble: CatBoost + MLP (blended) (RMSE **8.352**, MAE 5.738, R2 **0.9045** over every held-out zone-hour). The boosted-tree families and the MLP all cluster within ~2% of one another and all beat the linear and instance-based models, because the demand response is non-linear. CatBoost and the MLP make the least correlated errors, so their 50/50 blend wins on *both* metrics and is the model used downstream (ablation, rolling-origin, final fit). See `reports/model_grid.csv` for the full 13-model x 2-encoder x 3-scaler experiment.

## D3 Rolling-origin validation (4 expanding folds, 14-day windows)

| fold       | val_window             |   model_RMSE |   model_MAE |   model_R2 |   model_robust_RMSE |   model_robust_R2 |   naive_RMSE |
|:-----------|:-----------------------|-------------:|------------:|-----------:|--------------------:|------------------:|-------------:|
| 2025-07-01 | 2025-07-01..2025-07-14 |       8.8696 |      6.0285 |     0.8718 |              8.8696 |            0.8718 |      10.727  |
| 2025-08-05 | 2025-08-05..2025-08-18 |       8.6809 |      5.7358 |     0.8864 |              8.6809 |            0.8864 |      11.4292 |
| 2025-09-10 | 2025-09-10..2025-09-23 |       9.8507 |      6.1848 |     0.8693 |              9.8507 |            0.8693 |      13.8674 |
| 2025-10-17 | 2025-10-17..2025-10-30 |       8.3038 |      5.7013 |     0.9043 |              8.3038 |            0.9043 |      11.105  |

Mean RMSE **8.926 +/- 0.660** (seasonal-naive mean 11.782); mean robust R2 **0.883**. The tight spread shows the model is stable across the seasonal shift from July to October, and every fold beats the naive baseline.

## D4 Feature availability & leakage audit

See `reports/leakage_audit.csv`. Operational columns `active_drivers`, `avg_wait_min`, `avg_fare_birr` are **excluded** (consequences of demand, unknown at forecast time). Short lags (24h/48h) are excluded because they are unknowable beyond day 1 of a 14-day horizon; only >= 336h (14-day) and 672h (28-day) exact lags are kept.

*Leakage contrast:* feeding the operational columns lifts validation R2 to **0.986** - an artificially inflated score that would collapse in production.

## D5 Ablation

| configuration                      |   RMSE |    MAE |     R2 |   MAPE_% |   robust_RMSE |   robust_R2 |
|:-----------------------------------|-------:|-------:|-------:|---------:|--------------:|------------:|
| calendar+zone+trend                | 9.663  | 6.3641 | 0.8722 |    28.41 |        9.663  |      0.8722 |
| calendar+zone+trend+weather        | 9.2829 | 6.1732 | 0.882  |    28.56 |        9.2829 |      0.882  |
| calendar+zone+trend+events         | 8.7197 | 5.9062 | 0.8959 |    27.2  |        8.7197 |      0.8959 |
| calendar+zone+trend+weather+events | 8.3858 | 5.7391 | 0.9037 |    27.86 |        8.3858 |      0.9037 |

## D6 Tuning

- random search (6 trials, TimeSeriesSplit=2 best={'learning_rate': 0.04, 'l2_leaf_reg': 3.0, 'iterations': 800, 'depth': 6}); RMSE 8.522 -> 8.508.

## D7 Error analysis

**By zone**

| zone      |   RMSE |   MAE |   bias |
|:----------|-------:|------:|-------:|
| Arat Kilo |  6.732 | 4.697 |  0.786 |
| Ayat      |  6.028 | 4.412 |  0.744 |
| Bole      | 10.431 | 7.558 | -0.811 |
| CMC       |  7.521 | 5.45  |  0.365 |
| Gerji     |  6.557 | 4.894 | -0.059 |
| Kazanchis |  9.495 | 6.378 |  0.178 |
| Kolfe     |  6.568 | 4.748 |  0.402 |
| Lideta    |  7.952 | 5.571 | -0.236 |
| Megenagna | 11.353 | 7.663 |  1.093 |
| Merkato   | 10.48  | 7.208 |  0.121 |
| Piazza    |  8.476 | 5.653 | -0.295 |
| Sarbet    |  6.412 | 4.681 |  0.838 |


**10 largest-error zone-hours**

| zone      | pickup_datetime     |   trips |   pred |    err |   has_sports |
|:----------|:--------------------|--------:|-------:|-------:|-------------:|
| Megenagna | 2025-10-29 18:00:00 |     183 | 114.89 | -68.11 |            0 |
| Megenagna | 2025-10-28 18:00:00 |     174 | 106.45 | -67.55 |            0 |
| Kazanchis | 2025-10-28 18:00:00 |     176 | 115.31 | -60.69 |            0 |
| Bole      | 2025-10-26 18:00:00 |     137 |  77.11 | -59.89 |            0 |
| Piazza    | 2025-10-29 20:00:00 |     110 |  60.3  | -49.7  |            0 |
| Merkato   | 2025-10-31 13:00:00 |     149 | 104.71 | -44.29 |            0 |
| Bole      | 2025-10-26 23:00:00 |     100 |  56.07 | -43.93 |            0 |
| Merkato   | 2025-10-23 10:00:00 |      58 |  97.42 |  39.42 |            0 |
| Merkato   | 2025-10-18 09:00:00 |      66 | 104.13 |  38.13 |            0 |
| CMC       | 2025-10-28 18:00:00 |      93 |  55.33 | -37.67 |            0 |


With the export scaling repaired, the largest residuals are now dominated by unlisted crowding events (large positive error with `has_sports`=0), which the event calendar does not contain, and by genuine peak hours where small timing shifts cost many trips.

## D8 Response to findings

Errors were (a) dominated by a reporting defect in the target and (b) concentrated on event-driven and peak-hour spikes. We therefore repaired the x8 export scaling in cleaning, added the before/during/after event phasing, the 14/28-day exact lags, an 8-week 'current-regime' seasonal profile, a capped city trend index and rain x time-of-day interactions, and blended the two least-correlated families (CatBoost + MLP). Together these lifted validation R2 to 0.9045.

## D9 Plain-language metric

The final model misses a typical zone-hour by about **5.7 trips** (RMSE 8.4, R2 0.904) over every held-out zone-hour; against a mean demand of ~28.8 trips/hr that is roughly **20%**. At ~1.3 trips per driver-hour the dispatch uncertainty is only a few driver-hours per zone-hour.

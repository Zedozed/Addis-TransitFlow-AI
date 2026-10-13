# Deliverable C — Figure captions

1. **fig01_gaps_and_missingness.png** — 1.0% of trips are missing and 76 rain readings carry a -9999 sentinel; the timeline shows Ayat does not launch until mid-March and occasional missing hours elsewhere.
2. **fig02_before_after_cleaning.png** — The raw temperature axis is bimodal because ~353 July hours were Fahrenheit; cleaning unifies the scale, and trips lose their impossible negative sentinel.
3. **fig03_demand_trend_with_holidays.png** — City-wide demand grows steadily over the ten months, so November must be forecast with a trend; holidays cause visible dips against the rising trend.
4. **fig04_hour_by_weekday_heatmap.png** — Demand peaks at 18:00 on weekdays with a secondary morning peak; Bole keeps a strong evening/night profile while the city average flattens after 20:00.
5. **fig05_zone_profiles.png** — Each zone type has a distinct weekday shape (morning vs evening peak); the weekend curve shifts later everywhere, most strongly in commercial/nightlife zones.
6. **fig06_weather_timezone_check.png** — The raw weather clock mixes UTC 'Z' rows with local DD/MM rows: only after converting the Z rows by +3h does temperature peak in the sunny afternoon, and the rain–demand correlation also changes with the shift — proving the clock alignment matters.
7. **fig07_rain_effect.png** — Rain raises demand in every zone type but the market cluster: the response grows with rain intensity with diminishing marginal returns (saturating beyond moderate rain).
8. **fig08_event_study.png** — Demand rises into an event and peaks at/just before the start (peak offset +0 h); football matches draw the biggest crowd into and around their venue.
9. **fig09_holiday_effects.png** — Holidays split both ways: most reduce commuting demand below normal, but a few high-attendance holidays exceed it — the calendar signal is real but signed per holiday.
10. **fig10_model_comparison.png** — All boosted-tree families beat the two baselines; Ensemble: CatBoost + MLP (blended) is best and its rolling-origin spread (error bar) is small, so the result is stable.
11. **fig11_forecast_vs_actual.png** — Forecasts track the daily double peak and weekend shifts across three contrasting zones (commercial, market, late-launched), with errors concentrated on the evening peak.
12. **fig12_feature_importance.png** — The seasonal baseline and 14-day lags dominate, but weather (rain) and event features carry real, non-redundant importance — justifying both joins.

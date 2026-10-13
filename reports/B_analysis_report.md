# Deliverable B — Data Analysis Report

All figures below are computed from the cleaned, joined `master_train.csv` (84,614 zone-hours, 1 Jan – 31 Oct 2025).

## B1.1 Volume by zone

| zone      |   total_trips |   mean_per_hour |   hours |   share_% |
|:----------|--------------:|----------------:|--------:|----------:|
| Merkato   |      285653   |            40.5 |    7188 |      12.1 |
| Bole      |      277516   |            39.2 |    7196 |      11.7 |
| Megenagna |      270077   |            38.2 |    7196 |      11.4 |
| Kazanchis |      224386   |            31.7 |    7194 |       9.5 |
| Piazza    |      199789   |            28.2 |    7199 |       8.5 |
| Lideta    |      198219   |            28   |    7202 |       8.4 |
| CMC       |      182392   |            25.7 |    7196 |       7.7 |
| Gerji     |      170611   |            24.2 |    7199 |       7.2 |
| Kolfe     |      162357   |            23   |    7196 |       6.9 |
| Sarbet    |      153014   |            21.7 |    7189 |       6.5 |
| Arat Kilo |      144568   |            20.5 |    7201 |       6.1 |
| Ayat      |       95482.2 |            17.8 |    5458 |       4   |

- Highest-demand zone: **Merkato** (12.1% of the city total).
- Ayat is the late-launching zone: it only appears from 2025-03-15 and has 5458 of a possible ~7,296 hours — its low total is an availability artefact, not weak demand (its mean/hour is 17.8).

## B1.2 Hour-of-day profile by zone type

KMeans (k=4) on each zone's normalised weekday hourly curve groups the 12 zones as:

| zone      | zone_type                                        |
|:----------|:-------------------------------------------------|
| Arat Kilo | Evening peak / commercial-nightlife (peak 18:00) |
| Ayat      | Early-morning peak (peak 07:00)                  |
| Bole      | Early-morning peak (peak 06:00)                  |
| CMC       | Early-morning peak (peak 07:00)                  |
| Gerji     | Early-morning peak (peak 07:00)                  |
| Kazanchis | Evening peak / commercial-nightlife (peak 18:00) |
| Kolfe     | Early-morning peak (peak 07:00)                  |
| Lideta    | Evening peak / commercial-nightlife (peak 18:00) |
| Megenagna | Evening peak / commercial-nightlife (peak 18:00) |
| Merkato   | Daytime / market (peak 13:00)                    |
| Piazza    | Evening peak / commercial-nightlife (peak 18:00) |
| Sarbet    | Early-morning peak (peak 07:00)                  |

Peak / quiet hour per group (weekday):

- **Daytime / market (peak 13:00)**: peak 13:00 (89.3 trips/hr), quiet 00:00 (4.1).
- **Early-morning peak (peak 06:00)**: peak 06:00 (58.3 trips/hr), quiet 02:00 (10.7).
- **Early-morning peak (peak 07:00)**: peak 07:00 (46.7 trips/hr), quiet 02:00 (3.7).
- **Evening peak / commercial-nightlife (peak 18:00)**: peak 18:00 (79.7 trips/hr), quiet 02:00 (5.0).

## B1.3 Weekday vs weekend

| zone      |      0 |      1 |   weekend_weekday_ratio |
|:----------|-------:|-------:|------------------------:|
| Bole      | 35.907 | 47.55  |                   1.324 |
| Gerji     | 23.639 | 25.586 |                   1.082 |
| Sarbet    | 21.182 | 22.895 |                   1.081 |
| CMC       | 25.19  | 27.109 |                   1.076 |
| Kolfe     | 22.506 | 24.097 |                   1.071 |
| Ayat      | 17.567 | 18.476 |                   1.052 |
| Merkato   | 41.945 | 36.921 |                   0.88  |
| Lideta    | 29.523 | 24.223 |                   0.82  |
| Megenagna | 40.378 | 32.86  |                   0.814 |
| Kazanchis | 37.03  | 18.461 |                   0.499 |
| Piazza    | 33.08  | 16.048 |                   0.485 |
| Arat Kilo | 23.977 | 11.568 |                   0.482 |

- Busiest-on-weekends zone: **Bole** (ratio 1.32).
- Its day-of-week shape (Mon→Sun, trips/hr): 34.7, 34.4, 34.1, 35.2, 41.1, 49.7, 45.4.

## B1.4 Trend, January→October

- First four weeks: **39,452 trips/week avg**; last four weeks: **60,058 trips/week avg** → **+52.2%** over the period (515 trips/week added on trend).
- Implication: demand is growing, so a model that ignores the trend (or a seasonal-naive baseline) will under-forecast November. We therefore keep a trend feature (`days_since_start`); the 14-day exact lags carry the recent level.

## B2.1 Weather clock check

- Raw weather mixes **ISO `Z` (UTC)** rows with **`DD/MM/YYYY` local rows**; 659 of the 732 local rows already coincide with a `Z` row shifted +3 h, which proves they are the same instants on the local EAT clock. We convert `Z`→+3 h and keep the local rows as-is (see fig06).
- On the corrected clock the daily temperature peaks at **15:00 EAT**, the physically expected afternoon maximum (a mis-aligned clock would peak late at night).
- Rain–demand correlation by hour shift 0–5: 0h +0.680, 1h +0.674, 2h +0.623, 3h +0.566, 4h +0.510, 5h +0.444. The pattern changes with the shift, confirming alignment matters.

## B2.2 Rain effect by zone type

| zone_type                                        |   matched_ratio_rainy_vs_dry |   n_slots |
|:-------------------------------------------------|-----------------------------:|----------:|
| Daytime / market (peak 13:00)                    |                        0.891 |        95 |
| Early-morning peak (peak 06:00)                  |                        1.269 |        95 |
| Early-morning peak (peak 07:00)                  |                        1.26  |       473 |
| Evening peak / commercial-nightlife (peak 18:00) |                        1.284 |       475 |

Rain raises demand in 3 of 4 zone types (riders switch from walking to cars); the largest uplift is Evening peak / commercial-nightlife (peak 18:00) (×1.28) and the only exception is Daytime / market (peak 13:00) (×0.89), where a wet market day suppresses trips.

## B2.3 Rain dose-response

| rain_class   |   demand_ratio_vs_none |   mean_trips |
|:-------------|-----------------------:|-------------:|
| none         |                  1     |        26.7  |
| light        |                  1.606 |        42.89 |
| moderate     |                  1.881 |        50.22 |
| heavy        |                  2.113 |        56.43 |

The response increases with rain but with diminishing marginal returns: light rain ×1.61, moderate ×1.88, heavy ×2.11 vs dry. The largest step is dry→light; beyond moderate the extra demand per mm flattens (saturation).

## B3.1 Public holidays

| date       | dow       |   trips |   normal_trips |   index |
|:-----------|:----------|--------:|---------------:|--------:|
| 2025-01-07 | Tuesday   |    4805 |           6471 |   0.743 |
| 2025-04-18 | Friday    |    6093 |           7823 |   0.779 |
| 2025-06-06 | Friday    |    6832 |           8751 |   0.781 |
| 2025-09-04 | Thursday  |    7444 |           9171 |   0.812 |
| 2025-09-11 | Thursday  |    7783 |           9349 |   0.832 |
| 2025-05-28 | Wednesday |    6774 |           8019 |   0.845 |
| 2025-05-01 | Thursday  |    6611 |           7736 |   0.855 |
| 2025-03-31 | Monday    |    6275 |           7235 |   0.867 |
| 2025-01-08 | Wednesday |    6006 |           6571 |   0.914 |
| 2025-06-07 | Saturday  |    7082 |           7457 |   0.95  |
| 2025-04-17 | Thursday  |    7247 |           7621 |   0.951 |
| 2025-06-05 | Thursday  |    8028 |           8381 |   0.958 |
| 2025-09-27 | Saturday  |    8306 |           8648 |   0.96  |
| 2025-09-05 | Friday    |    9418 |           9746 |   0.966 |
| 2025-04-20 | Sunday    |    6059 |           6257 |   0.968 |
| 2025-05-04 | Sunday    |    6301 |           6470 |   0.974 |
| 2025-09-03 | Wednesday |    9051 |           9196 |   0.984 |
| 2025-05-05 | Monday    |    6241 |           6337 |   0.985 |
| 2025-01-20 | Monday    |    6363 |           6374 |   0.998 |
| 2025-01-19 | Sunday    |    5292 |           5226 |   1.013 |
| 2025-01-18 | Saturday  |    6211 |           5995 |   1.036 |
| 2025-03-01 | Saturday  |    6517 |           6262 |   1.041 |
| 2025-01-06 | Monday    |    6721 |           6390 |   1.052 |
| 2025-04-19 | Saturday  |    7548 |           7147 |   1.056 |
| 2025-03-02 | Sunday    |    5736 |           5362 |   1.07  |
| 2025-04-30 | Wednesday |    8298 |           7729 |   1.074 |
| 2025-04-01 | Tuesday   |    7793 |           7220 |   1.079 |
| 2025-03-03 | Monday    |    7509 |           6858 |   1.095 |
| 2025-05-29 | Thursday  |    8939 |           8140 |   1.098 |
| 2025-09-12 | Friday    |   10867 |           9686 |   1.122 |
| 2025-09-10 | Wednesday |   10627 |           9356 |   1.136 |
| 2025-03-30 | Sunday    |    6533 |           5740 |   1.138 |
| 2025-05-02 | Friday    |    9411 |           8145 |   1.155 |
| 2025-05-06 | Tuesday   |    7696 |           6485 |   1.187 |
| 2025-09-28 | Sunday    |    8964 |           7508 |   1.194 |
| 2025-09-26 | Friday    |   11751 |           9764 |   1.204 |
| 2025-05-27 | Tuesday   |    8677 |           6658 |   1.303 |
| 2025-04-21 | Monday    |    7745 |           5834 |   1.328 |

- Not every holiday reduces demand: the lowest index is 0.74, the highest 1.33.
## B3.2 Football match window study

| window       |   match_zone_mean |   baseline_mean |   uplift_% |
|:-------------|------------------:|----------------:|-----------:|
| before (-2h) |             69.79 |           40.19 |       73.7 |
| during       |             52.41 |           34.41 |       52.3 |
| after (+2h)  |             44.61 |           30.18 |       47.8 |

The **before (-2h)** window shows the largest uplift (+73.7%), i.e. demand peaks as crowds travel to the venue; the in-match and post-match hours still sit well above the non-event baseline.

## B3.3 Event type ranking

| event_type     |   n_zone_hours |   uplift_% |
|:---------------|---------------:|-----------:|
| football_match |            270 |       54.8 |
| concert        |            215 |       35.4 |
| conference     |            454 |       12.5 |
| exhibition     |            578 |       11.8 |
| public_holiday |           4247 |      -14.1 |
| road_closure   |            451 |      -21.3 |

Football matches and concerts dominate; conferences and exhibitions add only a small uplift, and road closures / public holidays reduce demand on net.

## B3.4 Cancelled & unlisted events

- **(a) Cancelled events.** 6 of 165 calendar rows are not `confirmed`. Their affected zone-hours show a demand ratio of **×0.946 (n=6)** against the zone×dow×hour baseline - statistically indistinguishable from an ordinary hour, so cancelled rows are correctly excluded from the event features.

- **(b) Unlisted spikes.** Of the 20 largest above-trend city days, 5 have no calendar entry at all.
- The three largest **zone-days** with no listed event:
  - **Merkato 2025-05-10**: 1376 trips, **×2.08** its 7-day trend (peak zone-hour 12:00 = 148 trips)
  - **Arat Kilo 2025-05-15**: 534 trips, **×1.67** its 7-day trend (peak zone-hour 08:00 = 60 trips)
  - **Piazza 2025-05-15**: 700 trips, **×1.63** its 7-day trend (peak zone-hour 18:00 = 74 trips)
  These are above-trend demands with no calendar entry — likely unlisted gatherings, a market day, or a weather-driven shift; a small part of demand will therefore always be unforecastable (see the error analysis in Deliverable D).

## B4.1 Operational variables vs demand

|                |   corr_with_trips |
|:---------------|------------------:|
| trips          |             1     |
| active_drivers |             0.987 |
| avg_wait_min   |             0.534 |
| avg_fare_birr  |             0.006 |

- `active_drivers` correlates tightly with trips because supply follows demand — the causation runs demand→supply, not the reverse.
- `avg_wait_min` and `avg_fare_birr` are likewise *consequences* of demand/surge, not causes. None can be a forecast input (D4/Rule 6); all exist only in the trip history.

## B4.2 Gaps and outages

| zone      | min                 | max                 |   size |
|:----------|:--------------------|:--------------------|-------:|
| Arat Kilo | 2025-01-01 00:00:00 | 2025-10-31 23:00:00 |   7201 |
| Ayat      | 2025-03-15 00:00:00 | 2025-10-31 23:00:00 |   5458 |
| Bole      | 2025-01-01 00:00:00 | 2025-10-31 23:00:00 |   7196 |
| CMC       | 2025-01-01 00:00:00 | 2025-10-31 23:00:00 |   7196 |
| Gerji     | 2025-01-01 00:00:00 | 2025-10-31 23:00:00 |   7199 |
| Kazanchis | 2025-01-01 00:00:00 | 2025-10-31 23:00:00 |   7194 |
| Kolfe     | 2025-01-01 00:00:00 | 2025-10-31 23:00:00 |   7196 |
| Lideta    | 2025-01-01 00:00:00 | 2025-10-31 23:00:00 |   7202 |
| Megenagna | 2025-01-01 00:00:00 | 2025-10-31 23:00:00 |   7196 |
| Merkato   | 2025-01-01 00:00:00 | 2025-10-31 23:00:00 |   7188 |
| Piazza    | 2025-01-01 00:00:00 | 2025-10-31 23:00:00 |   7199 |
| Sarbet    | 2025-01-01 00:00:00 | 2025-10-31 23:00:00 |   7189 |
- **Late-launching zone**: Ayat starts later than the others (see min date) — its missing early hours are not an outage.
- **Genuine outage**: any calendar period absent for *all* zones at once is a platform outage; isolated missing hours in individual zones are random dropouts.
- Treatment: we keep a row for every zone-hour we must forecast, and let lag/baseline features carry the level; no row is deleted from the test grid.

## B4.3 Pay-period effect

- Mean demand on pay-period days (1–3 and 28–31) vs the rest: **×1.050** (after controlling for hour-of-day). The effect is small, so `is_payday` is kept as a feature but is not expected to dominate.

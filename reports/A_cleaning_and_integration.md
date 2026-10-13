# Deliverable A — Data Cleaning & Integration Report

**Addis DemandFlow · Team:** team_demandflow · **Developer:** Zedingil Zelalem · **Clock:** Africa/Addis_Ababa (EAT, UTC+3)

## A1. Cleaning log

| file                  | column              | issue_type                                                                 |   rows_affected |     pct | fix                                                                                   | why                                                                                                                                                                                                                                    |
|:----------------------|:--------------------|:---------------------------------------------------------------------------|----------------:|--------:|:--------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ride_demand_train.csv | zone                | inconsistent spelling / case / spacing                                     |           85460 | 100     | mapped raw labels to 12 canonical zones via ZONE_MAP                                  | join key must be identical across all three tables                                                                                                                                                                                     |
| ride_demand_train.csv | zone + pickup_hour  | duplicate records (zone-hour repeated under a different spelling/format)   |            1692 |   1.98  | de-duplicated on (zone_id, pickup_datetime), keeping the first record                 | the duplicate carries the same trips value; summing would double-count                                                                                                                                                                 |
| ride_demand_train.csv | trips               | negative sentinel (-1) used for 'no reading'                               |             675 |   0.79  | set to NaN (missing)                                                                  | a negative trip count is impossible; it is a sentinel code                                                                                                                                                                             |
| ride_demand_train.csv | trips               | missing target values                                                      |            1512 |   1.77  | kept as NaN; excluded from model fitting only                                         | we still must forecast this zone-hour, so the row stays                                                                                                                                                                                |
| ride_demand_train.csv | avg_wait_min        | negative sentinel (-1)                                                     |            1692 |   1.98  | set to NaN                                                                            | a negative wait time is impossible                                                                                                                                                                                                     |
| ride_demand_train.csv | avg_fare_birr       | missing values                                                             |            1269 |   1.48  | kept NaN (analysis only; never a model input)                                         | fare is a forecast-time-unknown consequence of demand                                                                                                                                                                                  |
| ride_demand_train.csv | trips               | corrupted export: trips column scaled by x8 (trips/supply ~10x normal)     |             125 |   0.146 | repaired: trips <- trips/8 (flagged in `outlier_trips`)                               | trips/active_drivers is 9.8-14.1 vs the healthy max 1.85, so supply never responded; 99% of the flagged values are exact multiples of 8 and after /8 the ratio returns to 0.33-1.77 (mean 1.29) -- i.e. a x8 scaling bug in the export |
| ride_demand_test.csv  | zone                | inconsistent spelling / case / spacing                                     |            4032 | 100     | mapped raw labels to 12 canonical zones via ZONE_MAP                                  | join key must be identical across all three tables                                                                                                                                                                                     |
| weather_hourly.csv    | timestamp           | mixed clock (UTC 'Z' rows mixed with local DD/MM/YYYY rows)                |            7538 | 100     | 6806 UTC rows converted +3h to EAT; 732 local rows kept as-is                         | 659/732 local rows already match a Z-row shifted +3h -> same instants                                                                                                                                                                  |
| weather_hourly.csv    | temp_c              | unit switch (part of the year recorded in Fahrenheit)                      |             353 |   4.68  | converted (F-32)*5/9 for values > 35 (impossible in Addis)                            | values are cleanly bimodal: <=25 C vs 50-80 F, none in between                                                                                                                                                                         |
| weather_hourly.csv    | rain_mm             | sentinel 'no reading' code (-9999)                                         |              76 |   1.01  | set to NaN                                                                            | -9999 is an explicit missing-value code                                                                                                                                                                                                |
| weather_hourly.csv    | timestamp           | duplicate hours with conflicting readings                                  |            1437 |  19.06  | aggregated duplicate hours with the median                                            | one reading per hour is required for a safe many-to-one join                                                                                                                                                                           |
| weather_hourly.csv    | timestamp           | missing whole hours (sensor dropouts / one outage)                         |             842 |  11     | reindexed to a complete hourly series; numeric gaps time-interpolated                 | trip history has an hour every hour and must not lose rows to a weather gap                                                                                                                                                            |
| weather_hourly.csv    | temp_c              | missing reading (sentinel or gap)                                          |             876 |  11.44  | time-interpolated (limit both directions) and flagged via *_missing                   | keeps the missingness signal while giving every hour a usable value                                                                                                                                                                    |
| weather_hourly.csv    | rain_mm             | missing reading (sentinel or gap)                                          |             910 |  11.89  | time-interpolated (limit both directions) and flagged via *_missing                   | keeps the missingness signal while giving every hour a usable value                                                                                                                                                                    |
| weather_hourly.csv    | humidity_pct        | missing reading (sentinel or gap)                                          |             973 |  12.71  | time-interpolated (limit both directions) and flagged via *_missing                   | keeps the missingness signal while giving every hour a usable value                                                                                                                                                                    |
| weather_hourly.csv    | wind_kmh            | missing reading (sentinel or gap)                                          |             842 |  11     | time-interpolated (limit both directions) and flagged via *_missing                   | keeps the missingness signal while giving every hour a usable value                                                                                                                                                                    |
| events_calendar.csv   | event_type          | inconsistent case / spacing (e.g. 'Football Match' vs 'football_match')    |             165 | 100     | lower-cased and underscore-normalised                                                 | one label per event class is needed for grouping                                                                                                                                                                                       |
| events_calendar.csv   | end_datetime        | missing or earlier-than-start end datetime                                 |               5 |   3.03  | defaulted to start + 3 hours                                                          | an interval join needs a bounded window                                                                                                                                                                                                |
| events_calendar.csv   | zone                | spelling variants, parenthetical venue text, multi-zone and citywide cells |             165 | 100     | normalised to canonical zones; '&'-joined cells expanded; citywide expanded to all 12 | events must attach to the trip zones they actually affect                                                                                                                                                                              |
| events_calendar.csv   | expected_attendance | free text ('approx 34000', '2,039') and blanks                             |              54 |  32.73  | parsed leading integer; blanks -> 0                                                   | attendance is a usable crowd-size feature                                                                                                                                                                                              |
| join: trips + weather | weather_datetime    | trip hours with no weather row                                             |               0 |   0     | kept as NaN (indicators already present); no rows added or lost                       | weather must never duplicate trip rows in a many-to-one join                                                                                                                                                                           |
| join: trips + weather | weather_datetime    | trip hours with no weather row                                             |               0 |   0     | kept as NaN (indicators already present); no rows added or lost                       | weather must never duplicate trip rows in a many-to-one join                                                                                                                                                                           |

23 issues logged across the three tables (≥5 in trips, ≥4 in weather, ≥4 in events).

## A2. Time & key standardisation

### (a) Zone labels

- Raw trip zone strings: **55** distinct spellings (e.g. ARAT KILO, AYAT, Arat  Kilo, Arat Kilo, Arat Kilo , Ayat …).
- After cleaning: **12** canonical labels — Bole, Kazanchis, Piazza, Merkato, Sarbet, CMC, Megenagna, Gerji, Ayat, Arat Kilo, Kolfe, Lideta.

### (b) Datetime formats parsed

| table | formats seen | parse rule |
| :-- | :-- | :-- |
| trips | `YYYY-MM-DD HH:MM`, `DD/MM/YYYY HH:MM`, `ISO+03:00` | day-first for `/`; offsets → EAT; naive = local |
| events | ISO, `DD/MM/YYYY HH:MM`, `Mon DD, YYYY HH:MM AM/PM` | all three handled; day-first for `/` |
| weather | `...Z` (UTC) and `DD/MM/YYYY HH:MM` (local) | `Z` → +3 h; `/` already local |

The day-first rule is forced by unambiguous values (`16/08/2025`, `13/08/2025`).

### (c) Clock proof

- Weather mixes **6806 UTC (`Z`) rows** with **732 local `DD/MM/YYYY` rows**.
- **659 of 732** local rows coincide (to the hour) with a `Z` row shifted +3 h — i.e. they are the same instants on the local clock. Non-coinciding rows fall in the gaps created by dropouts.
- Trip and event timestamps are already EAT; we only floor them to the hour. Conclusion: converting `Z`→+3h and leaving the local rows untouched puts every table on one clock.

## A3. Join map

```
ride_demand_train/test (zone, pickup_datetime)  --many-to-one-->  weather_hourly (weather_datetime)
ride_demand_train/test (zone_id, pickup_datetime) --one-to-one--> event_zone_hour (zone_id, pickup_datetime)
event_zone_hour built from events_calendar by expanding multi-zone/citywide rows,
then flagging every hour in [start-2h, end+2h] per affected zone.
```
- **Left table = trips**, because we must forecast exactly the trip zone-hours; both joins are therefore left joins that must not change the row count.
- Cardinality: weather is many-to-one (one city-wide reading per hour); events are an interval join aggregated to one row per (zone_id, hour).

## A4. Join audit

- Weather join: train rows 84,614 → 84,614 (unchanged). **0** zone-hours had no weather hour (weather dropouts); handled by reindexing weather to a complete hourly series, time-interpolating, and flagging with `*_missing`.
- Events: 159 confirmed events expanded to 329 zone-events; 159 of them (100%) matched at least one zone-hour, covering 25,617 zone-hours in the [start-2h, end+2h] window. 6 cancelled/unconfirmed events were excluded from the active-window features.
- Zone-hours with no weather reading after the complete-hourly reindex: **0** (weather gaps were filled by interpolation and flagged).

## A5. Join proof (three worked zone-hours)

1. **Rain zone-hour:** Sarbet at 2025-05-02 17:00:00 joins weather hour 2025-05-02 17:00:00 → `rain_mm=21.0` mm, `temp_c=21.5°C`, `humidity_pct=76` (all 12 zone-hours at that instant inherit the same city-wide reading).
2. **Football match window:** Kazanchis at 2025-03-09 10:00:00 joins `event_zone_hour` → `has_sports=1`, `event_before=0`, `event_during=1`, `event_after=0`, `event_attendance=9659` from the stadium calendar entry in that zone.
3. **Public holiday:** Ayat at 2025-05-27 23:00:00 joins → `has_public_holiday=1`, `has_event=1` (citywide calendar entry expanded to all 12 zones).

Weather/event features on these rows: temp_c, rain_mm, has_sports, event_before, event_during, event_after, event_attendance, has_public_holiday.

> **Row count is invariant across both joins:** 84,614 rows in = 84,614 rows out. (The 85,460 raw rows fall to 84,614 *before* any join, purely from de-duplicating 1,692 repeated zone-hours and dropping unparseable timestamps / unmapped labels — see A1.)
## A6. Feature engineering

| feature                 | category    | formula                                   | rationale                      | known_at_forecast_time   |
|:------------------------|:------------|:------------------------------------------|:-------------------------------|:-------------------------|
| zone_id                 | zone        | dim_zone.zone_id                          | zone identity 1-12             | yes                      |
| hour                    | calendar    | pickup_datetime.dt.hour                   | diurnal demand cycle           | yes                      |
| dayofweek               | calendar    | pickup_datetime.dt.dayofweek              | weekday pattern                | yes                      |
| day                     | calendar    | day of month                              | pay-period effects             | yes                      |
| month                   | calendar    | month                                     | seasonal trend                 | yes                      |
| weekofyear              | calendar    | iso week                                  | within-year seasonality        | yes                      |
| is_weekend              | calendar    | dayofweek>=5                              | weekend leisure demand         | yes                      |
| is_payday               | calendar    | day in 1-3,28-31                          | salary spending pulse          | yes                      |
| hour_sin                | calendar    | sin(2*pi*hour/24)                         | smooth time-of-day             | yes                      |
| hour_cos                | calendar    | cos(2*pi*hour/24)                         | smooth time-of-day             | yes                      |
| dow_sin                 | calendar    | sin(2*pi*dow/7)                           | smooth day-of-week             | yes                      |
| dow_cos                 | calendar    | cos(2*pi*dow/7)                           | smooth day-of-week             | yes                      |
| days_since_start        | trend       | date - 2025-01-01                         | captures growth trend          | yes                      |
| temp_c                  | weather     | weather temp (F fixed)                    | heat/comfort effect            | yes                      |
| rain_mm                 | weather     | previous-hour rain                        | rain demand effect             | yes                      |
| is_raining              | weather     | rain_mm>0.05                              | rain yes/no                    | yes                      |
| heavy_rain              | weather     | rain_mm>7.6                               | heavy-rain flag                | yes                      |
| rain_class              | weather     | cut(rain_mm)                              | non-linear rain response       | yes                      |
| rain_3h_sum             | weather     | rolling 3h rain                           | sustained-rain effect          | yes                      |
| rain_6h_sum             | weather     | rolling 6h rain                           | wet-spell effect               | yes                      |
| rain_24h_sum            | weather     | rolling 24h rain                          | all-day wetness                | yes                      |
| rain_is_raining_x_hour  | weather     | is_raining*hour                           | rain x time-of-day             | yes                      |
| humidity_pct            | weather     | weather humidity                          | comfort                        | yes                      |
| wind_kmh                | weather     | weather wind                              | minor comfort                  | yes                      |
| has_event               | events      | interval join                             | event demand uplift            | yes                      |
| event_count             | events      | overlap count                             | crowd of events                | yes                      |
| has_sports              | events      | event_type                                | stadium match uplift           | yes                      |
| has_concert             | events      | event_type                                | concert uplift                 | yes                      |
| has_conference          | events      | event_type                                | conference demand              | yes                      |
| has_road_closure        | events      | event_type                                | detours/supply                 | yes                      |
| has_public_holiday      | events      | event_type                                | holiday demand shift           | yes                      |
| has_school_break        | events      | event_type                                | school-holiday pattern         | yes                      |
| has_exhibition          | events      | event_type                                | exhibition demand              | yes                      |
| event_attendance        | events      | parsed crowd size                         | event size                     | yes                      |
| event_before            | events      | 2h before                                 | pre-event inflow               | yes                      |
| event_during            | events      | during                                    | in-event demand                | yes                      |
| event_after             | events      | 2h after                                  | post-event dispersal           | yes                      |
| has_sports_evening      | events      | has_sports * (hour>=17)                   | evening match uplift           | yes                      |
| event_after_evening     | events      | event_after * (hour>=17)                  | evening dispersal peak         | yes                      |
| hours_to_event          | events      | hours to next start                       | anticipation                   | yes                      |
| hours_since_event       | events      | hours since last end                      | dispersal decay                | yes                      |
| lag_336h                | history     | trips 14 days ago                         | same slot last fortnight       | yes                      |
| lag_672h                | history     | trips 28 days ago                         | same slot last month           | yes                      |
| hist_zone_dow_hour_mean | history     | expanding mean by zone/dow/hour           | seasonal baseline              | yes                      |
| hist_zone_hour_mean     | history     | expanding mean by zone/hour               | diurnal baseline               | yes                      |
| hist_zone_dow_mean      | history     | expanding mean by zone/dow                | weekly baseline                | yes                      |
| hist_zone_mean          | history     | expanding mean by zone                    | zone level                     | yes                      |
| hist_zdh_recent8w       | history     | 8-week zone/dow/hour mean before cutoff   | current regime profile         | yes                      |
| hist_zh_recent8w        | history     | 8-week zone/hour mean before cutoff       | current regime diurnal         | yes                      |
| hist_zone_recent_7d     | history     | mean last 7 days of history               | recent level                   | yes                      |
| hist_zone_recent_14d    | history     | mean last 14 days of history              | recent level                   | yes                      |
| hist_zone_recent_28d    | history     | mean last 28 days of history              | recent level                   | yes                      |
| hist_zone_recent_56d    | history     | mean last 56 days of history              | recent level                   | yes                      |
| rel_recent_7d           | history     | recent_7d / zone/dow/hour mean            | recent-vs-profile level        | yes                      |
| rel_recent_14d          | history     | recent_14d / zone/dow/hour mean           | recent-vs-profile level        | yes                      |
| rel_recent_28d          | history     | recent_28d / zone/dow/hour mean           | recent-vs-profile level        | yes                      |
| trend_index             | trend       | trailing city level, capped extrapolation | extrapolates growth            | yes                      |
| active_drivers          | operational | trip log                                  | NOT available at forecast time | NO                       |
| avg_wait_min            | operational | trip log                                  | consequence of demand          | NO                       |
| avg_fare_birr           | operational | trip log                                  | surge consequence              | NO                       |

## A7. Integrity checks

| check                                             | result   | detail     |
|:--------------------------------------------------|:---------|:-----------|
| one row per zone-hour (train)                     | PASS     | nan        |
| one row per zone-hour (test)                      | PASS     | nan        |
| exactly 12 canonical zones (train)                | PASS     | 12 zones   |
| exactly 12 canonical zones (test)                 | PASS     | 12 zones   |
| row count unchanged by joins (train)              | PASS     | 84614 rows |
| test row count is 4,032                           | PASS     | 4032 rows  |
| no negative trips retained                        | PASS     | nan        |
| no rain sentinel (-9999) left                     | PASS     | nan        |
| temperature within physical range                 | PASS     | nan        |
| all timestamps on the EAT clock / in range        | PASS     | nan        |
| train and test share the feature columns          | PASS     | nan        |
| no forecast-time-unknown column used as a feature | PASS     | nan        |

## A8. Master tables & data dictionary

- `data/processed/master_train.csv`: 84,614 rows × 34 cols
- `data/processed/master_test.csv`: 4,032 rows × 30 cols
- `data/processed/data_dictionary_master.csv`: full column dictionary
- Test rows pass through the identical pipeline; all statistics are learned from train only.

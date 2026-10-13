"""
src/features.py
===============
Leakage-safe feature engineering for the Addis ride-demand forecast.

Design rule (D4): a feature is only legal if it would be known when the forecast
is produced.  The forecast is produced once at a cutoff date and covers the next
14 days, therefore:

  * Exact same-slot lags of 336 h (14 d) and 672 h (28 d) are used.  Shorter
    lags (24/48 h) are unknowable for most of a 14-day horizon and are excluded.
  * Every history-derived statistic for the *future* rows is frozen at the
    cutoff (computed from rows strictly before the cutoff).
  * Training rows get expanding (past-only) versions of the same statistics.
  * Operational columns (active_drivers, avg_wait_min, avg_fare_birr) are never
    features - they are consequences of demand, not causes (Rule 6 / D4).

Feature groups (see FEATURE_META for the full audit table):
  calendar  - hour, dow, month, weekend, payday, cyclical encodings, trend
  weather   - temperature, rain (+ class, 3h rolling, interactions)
  events    - interval-join flags, attendance, before/during/after windows
  history   - 14/28-day exact lags, seasonal target encodings, recent levels
"""

from __future__ import annotations
import numpy as np
import pandas as pd

HORIZON_H = 14 * 24              # forecast horizon in hours
MIN_SAFE_LAG = 336              # 14 days - smallest horizon-safe exact lag
LAGS_H = (336, 672)            # 14-day and 28-day exact same-slot lags
RECENT_WINDOWS_D = (7, 14, 28, 56)   # trailing zone-level windows (days)
SEASON_RECENT_WEEKS = 8        # "current regime" seasonal profile window

# Target winsorisation.  An earlier version used this to cap data-error
# outliers; the defect turned out to be a x8 scaling of the trips export, which
# src/cleaning.py now repairs, so this is only a light robustness guard against
# stray extreme values.  The ceiling is deliberately high enough that it never
# clips genuine peaks (observed maximum after repair is ~321 trips/hr).
TARGET_CAP_QUANTILE = 0.999
TARGET_CAP_FLOOR = 110.0
TARGET_CAP_CEILING = 300.0


# --------------------------------------------------------------------------- #
# Row-wise (no history) features
# --------------------------------------------------------------------------- #
def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dt = pd.to_datetime(df["pickup_datetime"])
    df["hour"] = dt.dt.hour
    df["dayofweek"] = dt.dt.dayofweek
    df["day"] = dt.dt.day
    df["month"] = dt.dt.month
    df["weekofyear"] = dt.dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_payday"] = df["day"].isin([1, 2, 3, 28, 29, 30, 31]).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    start = pd.Timestamp("2025-01-01")
    df["days_since_start"] = (dt.dt.normalize() - start).dt.days
    return df


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rain_mm"] = df["rain_mm"].clip(lower=0).fillna(0.0)
    df["is_raining"] = (df["rain_mm"] > 0.05).astype(int)
    df["heavy_rain"] = (df["rain_mm"] > 7.6).astype(int)
    # rain class: none / light / moderate / heavy (mm in the previous hour)
    df["rain_class"] = pd.cut(df["rain_mm"], [-0.001, 0.05, 2.5, 7.6, np.inf],
                              labels=[0, 1, 2, 3]).astype(int)
    # rolling rain over the previous 3 hours (per zone); sorted copy so no
    # future reading leaks in.
    d = df.sort_values("pickup_datetime")
    df["rain_3h_sum"] = d.groupby("zone_id")["rain_mm"].transform(
        lambda s: s.rolling(3, min_periods=1).sum())
    # longer rolling windows: a sustained wet spell depresses walking demand more
    # than an isolated shower.
    df["rain_6h_sum"] = d.groupby("zone_id")["rain_mm"].transform(
        lambda s: s.rolling(6, min_periods=1).sum())
    df["rain_24h_sum"] = d.groupby("zone_id")["rain_mm"].transform(
        lambda s: s.rolling(24, min_periods=1).sum())
    # multiplicative rain x time-of-day interaction: rain hurts most at hours
    # where walking is the natural mode (rush/leisure peaks).
    df["rain_is_raining_x_hour"] = df["is_raining"] * df["hour"]
    df["temp_c"] = df["temp_c"].fillna(df["temp_c"].median())
    return df


def assemble(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise features only (used before the cutoff split)."""
    df = add_weather_features(add_calendar_features(df))
    # event x time-of-day interactions: crowd events bite hardest in the evening
    # peak, and post-event dispersal lands on that same peak.
    evening = (df["hour"] >= 17).astype(int)
    df["has_sports_evening"] = (df["has_sports"] * evening if "has_sports" in df.columns else 0)
    df["event_after_evening"] = (df["event_after"] * evening if "event_after" in df.columns else 0)
    return df


# --------------------------------------------------------------------------- #
# History-derived features (cutoff-aware)
# --------------------------------------------------------------------------- #
_GROUP_KEYS = [
    (["zone_id", "dayofweek", "hour"], "hist_zone_dow_hour_mean"),
    (["zone_id", "hour"], "hist_zone_hour_mean"),
    (["zone_id", "dayofweek"], "hist_zone_dow_mean"),
    (["zone_id"], "hist_zone_mean"),
]


def target_cap(history: pd.DataFrame) -> float:
    """Robust winsorisation ceiling from the training window (leak-safe)."""
    q = history["trips"].dropna().quantile(TARGET_CAP_QUANTILE)
    return float(np.clip(q, TARGET_CAP_FLOOR, TARGET_CAP_CEILING))


def build_history_features(history: pd.DataFrame, future: pd.DataFrame,
                           horizon_h: int = MIN_SAFE_LAG):
    """
    Return (history_with_feats, future_with_feats).  `history` must contain
    `trips`; `future` need not.  All statistics for `future` use only
    `history`, so nothing after the cutoff can leak in.
    """
    history = history.sort_values("pickup_datetime").copy()
    future = future.copy()
    cut = history["pickup_datetime"].max()

    # ---- exact same-slot lags (14 and 28 days) ----
    trips = history.set_index(["zone_id", "pickup_datetime"])["trips"]

    def _lag(df, hours):
        key = pd.MultiIndex.from_arrays(
            [df["zone_id"].values,
             (pd.to_datetime(df["pickup_datetime"]) - pd.Timedelta(hours=hours)).values])
        return pd.Series(key.map(trips), index=df.index)

    for h in LAGS_H:
        col = f"lag_{h}h"
        history[col] = _lag(history, h)
        future[col] = _lag(future, h)

    # ---- seasonal target encodings (all history, expanding for train rows) ----
    for keys, name in _GROUP_KEYS:
        g = history.groupby(keys)["trips"]
        history[name] = g.transform(lambda s: s.shift(1).expanding().mean())
        full = g.mean().rename(name)
        future[name] = future.merge(full.reset_index(), on=keys, how="left")[name].to_numpy()

    # ---- "current regime" seasonal profile: last 8 weeks before the cutoff ----
    h_recent = history[history["pickup_datetime"] > cut - pd.Timedelta(weeks=SEASON_RECENT_WEEKS)]
    for keys, name in [(['zone_id', 'dayofweek', 'hour'], 'hist_zdh_recent8w'),
                       (['zone_id', 'hour'], 'hist_zh_recent8w')]:
        full = h_recent.groupby(keys)["trips"].mean().rename(name).reset_index()
        history = history.merge(full, on=keys, how="left")
        future[name] = future.merge(full, on=keys, how="left")[name].to_numpy()

    # ---- recent zone level over trailing windows ----
    for days in RECENT_WINDOWS_D:
        name = f"hist_zone_recent_{days}d"
        r = (history[history["pickup_datetime"] > cut - pd.Timedelta(days=days)]
             .groupby("zone_id")["trips"].mean().rename(name))
        future[name] = future.merge(r.reset_index(), on="zone_id", how="left")[name].to_numpy()
        history = history.merge(r.reset_index(), on="zone_id", how="left")

    # ---- relative recent level: how the last fortnight compares with the profile ----
    for d in (7, 14, 28):
        rname = f"hist_zone_recent_{d}d"
        history[f"rel_recent_{d}d"] = history[rname] / history["hist_zone_dow_hour_mean"]
        future[f"rel_recent_{d}d"] = future[rname] / future["hist_zone_dow_hour_mean"]

    # ---- city-wide trend index -------------------------------------------- #
    # Trailing 14-day city mean, normalised to its own mean; for the future it
    # is extrapolated with a *capped* slope so a runaway trend cannot explode.
    daily = history.groupby(history["pickup_datetime"].dt.normalize())["trips"].mean()
    trend = daily.rolling(14, min_periods=1).mean()
    idx = (trend / trend.mean()).rename("trend_index")
    history["trend_index"] = history["pickup_datetime"].dt.normalize().map(idx).to_numpy()

    last_t, last_v = idx.index.max(), float(idx.iloc[-1])
    slope = 0.0
    if len(idx) > 15:
        slope = (float(idx.iloc[-1]) - float(idx.iloc[-15])) / 14.0
    slope = float(np.clip(slope, -0.002, 0.002))
    days_ahead = (future["pickup_datetime"].dt.normalize() - last_t).dt.days.clip(lower=0)
    future["trend_index"] = (last_v + slope * days_ahead).clip(0.5, 1.6).to_numpy()
    future["trend_index"] = future["trend_index"].fillna(last_v)

    return history, future


# --------------------------------------------------------------------------- #
# Feature audit table
# --------------------------------------------------------------------------- #
FEATURE_META = [
    ("zone_id", "zone", "dim_zone.zone_id", "zone identity 1-12", "yes"),
    ("hour", "calendar", "pickup_datetime.dt.hour", "diurnal demand cycle", "yes"),
    ("dayofweek", "calendar", "pickup_datetime.dt.dayofweek", "weekday pattern", "yes"),
    ("day", "calendar", "day of month", "pay-period effects", "yes"),
    ("month", "calendar", "month", "seasonal trend", "yes"),
    ("weekofyear", "calendar", "iso week", "within-year seasonality", "yes"),
    ("is_weekend", "calendar", "dayofweek>=5", "weekend leisure demand", "yes"),
    ("is_payday", "calendar", "day in 1-3,28-31", "salary spending pulse", "yes"),
    ("hour_sin", "calendar", "sin(2*pi*hour/24)", "smooth time-of-day", "yes"),
    ("hour_cos", "calendar", "cos(2*pi*hour/24)", "smooth time-of-day", "yes"),
    ("dow_sin", "calendar", "sin(2*pi*dow/7)", "smooth day-of-week", "yes"),
    ("dow_cos", "calendar", "cos(2*pi*dow/7)", "smooth day-of-week", "yes"),
    ("days_since_start", "trend", "date - 2025-01-01", "captures growth trend", "yes"),
    ("temp_c", "weather", "weather temp (F fixed)", "heat/comfort effect", "yes"),
    ("rain_mm", "weather", "previous-hour rain", "rain demand effect", "yes"),
    ("is_raining", "weather", "rain_mm>0.05", "rain yes/no", "yes"),
    ("heavy_rain", "weather", "rain_mm>7.6", "heavy-rain flag", "yes"),
    ("rain_class", "weather", "cut(rain_mm)", "non-linear rain response", "yes"),
    ("rain_3h_sum", "weather", "rolling 3h rain", "sustained-rain effect", "yes"),
    ("rain_6h_sum", "weather", "rolling 6h rain", "wet-spell effect", "yes"),
    ("rain_24h_sum", "weather", "rolling 24h rain", "all-day wetness", "yes"),
    ("rain_is_raining_x_hour", "weather", "is_raining*hour", "rain x time-of-day", "yes"),
    ("humidity_pct", "weather", "weather humidity", "comfort", "yes"),
    ("wind_kmh", "weather", "weather wind", "minor comfort", "yes"),
    ("has_event", "events", "interval join", "event demand uplift", "yes"),
    ("event_count", "events", "overlap count", "crowd of events", "yes"),
    ("has_sports", "events", "event_type", "stadium match uplift", "yes"),
    ("has_concert", "events", "event_type", "concert uplift", "yes"),
    ("has_conference", "events", "event_type", "conference demand", "yes"),
    ("has_road_closure", "events", "event_type", "detours/supply", "yes"),
    ("has_public_holiday", "events", "event_type", "holiday demand shift", "yes"),
    ("has_school_break", "events", "event_type", "school-holiday pattern", "yes"),
    ("has_exhibition", "events", "event_type", "exhibition demand", "yes"),
    ("event_attendance", "events", "parsed crowd size", "event size", "yes"),
    ("event_before", "events", "2h before", "pre-event inflow", "yes"),
    ("event_during", "events", "during", "in-event demand", "yes"),
    ("event_after", "events", "2h after", "post-event dispersal", "yes"),
    ("has_sports_evening", "events", "has_sports * (hour>=17)", "evening match uplift", "yes"),
    ("event_after_evening", "events", "event_after * (hour>=17)", "evening dispersal peak", "yes"),
    ("hours_to_event", "events", "hours to next start", "anticipation", "yes"),
    ("hours_since_event", "events", "hours since last end", "dispersal decay", "yes"),
    ("lag_336h", "history", "trips 14 days ago", "same slot last fortnight", "yes"),
    ("lag_672h", "history", "trips 28 days ago", "same slot last month", "yes"),
    ("hist_zone_dow_hour_mean", "history", "expanding mean by zone/dow/hour", "seasonal baseline", "yes"),
    ("hist_zone_hour_mean", "history", "expanding mean by zone/hour", "diurnal baseline", "yes"),
    ("hist_zone_dow_mean", "history", "expanding mean by zone/dow", "weekly baseline", "yes"),
    ("hist_zone_mean", "history", "expanding mean by zone", "zone level", "yes"),
    ("hist_zdh_recent8w", "history", "8-week zone/dow/hour mean before cutoff", "current regime profile", "yes"),
    ("hist_zh_recent8w", "history", "8-week zone/hour mean before cutoff", "current regime diurnal", "yes"),
    ("hist_zone_recent_7d", "history", "mean last 7 days of history", "recent level", "yes"),
    ("hist_zone_recent_14d", "history", "mean last 14 days of history", "recent level", "yes"),
    ("hist_zone_recent_28d", "history", "mean last 28 days of history", "recent level", "yes"),
    ("hist_zone_recent_56d", "history", "mean last 56 days of history", "recent level", "yes"),
    ("rel_recent_7d", "history", "recent_7d / zone/dow/hour mean", "recent-vs-profile level", "yes"),
    ("rel_recent_14d", "history", "recent_14d / zone/dow/hour mean", "recent-vs-profile level", "yes"),
    ("rel_recent_28d", "history", "recent_28d / zone/dow/hour mean", "recent-vs-profile level", "yes"),
    ("trend_index", "trend", "trailing city level, capped extrapolation", "extrapolates growth", "yes"),
    ("active_drivers", "operational", "trip log", "NOT available at forecast time", "NO"),
    ("avg_wait_min", "operational", "trip log", "consequence of demand", "NO"),
    ("avg_fare_birr", "operational", "trip log", "surge consequence", "NO"),
]


def feature_metadata() -> pd.DataFrame:
    return pd.DataFrame(FEATURE_META, columns=["feature", "category", "formula",
                                               "rationale", "known_at_forecast_time"])


# columns fed to the model (operational columns deliberately excluded)
MODEL_FEATURES = [f for f, c, *_ in FEATURE_META if c != "operational"]
CATEGORICAL_FEATURES = ["zone_id"]

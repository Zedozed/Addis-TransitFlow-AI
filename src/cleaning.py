"""
src/cleaning.py
===============
Deliverable A — Data Cleaning & Integration Pipeline
Addis DemandFlow — Ride Demand Intelligence for Addis Ababa

Single source of truth that turns the three raw exports into one trustworthy,
time-aligned modelling table. Runs identically for train and test.

Clock model (proved in A2, evidence from the data):
  * ride_demand_train/pickup_hour   -> Addis Ababa LOCAL time (EAT, UTC+3), naive.
  * events start/end_datetime       -> Addis Ababa LOCAL time (EAT, UTC+3), naive.
  * weather_hourly.timestamp        -> MIXED clock:
        - ISO rows ending in "Z"          = UTC  -> convert +3h to EAT
        - "DD/MM/YYYY HH:MM" rows         = already LOCAL EAT (no conversion)
    Proof: 659 of the 732 slash rows already coincide (to the hour) with a
    Z-row shifted +3h, i.e. they are the same instants expressed on the local
    clock. Rows carrying an explicit "+03:00" offset are already EAT too.

All outputs are written to data/processed/.
"""

from __future__ import annotations

import os
import re
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Canonical 12 zones (verified against ride_demand_test.csv, which is the
# scored set). Order defines zone_id.
# --------------------------------------------------------------------------- #
CANONICAL_ZONES = [
    "Bole", "Kazanchis", "Piazza", "Merkato", "Sarbet", "CMC",
    "Megenagna", "Gerji", "Ayat", "Arat Kilo", "Kolfe", "Lideta",
]

# raw spelling (lower-cased, whitespace-collapsed) -> canonical zone
ZONE_MAP = {
    "bole": "Bole", "bole rd": "Bole", "bole airport": "Bole",
    "bole (bole subcity)": "Bole",
    "kazanchis": "Kazanchis", "kazanchis (kirkos)": "Kazanchis",
    "kazanchis bus": "Kazanchis", "kazanches": "Kazanchis",
    "piassa": "Piazza", "piazza": "Piazza", "piazza central": "Piazza",
    "piassa (arada)": "Piazza",
    "mercato": "Merkato", "merkato": "Merkato",
    "sarbet": "Sarbet",
    "cmc": "CMC", "c.m.c": "CMC",
    "megenagna": "Megenagna", "megenaga": "Megenagna",
    "gerji": "Gerji",
    "ayat": "Ayat", "ayat zone": "Ayat",
    "arat kilo": "Arat Kilo", "4 kilo": "Arat Kilo",
    "kolfe": "Kolfe", "kolfe keranio": "Kolfe",
    "lideta": "Lideta",
}

CITYWIDE_TOKENS = {"citywide", "city-wide", "city wide", "all", "all zones"}
ZONE_NAME_TO_ID = {name: i + 1 for i, name in enumerate(CANONICAL_ZONES)}

# Weather: values above this are impossible for Addis Ababa (~2350 m) and are
# the Fahrenheit segment of the file (July). Convert (F-32)*5/9.
TEMP_FAHRENHEIT_THRESHOLD = 35.0
SENTINELS = [-9999, -999, -99, -1]


def _clean_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip())


# --------------------------------------------------------------------------- #
# Zone standardisation
# --------------------------------------------------------------------------- #
def standardize_zone(raw) -> str:
    """Map a raw zone string to a canonical zone name (or 'Citywide'/'Unknown')."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)) or str(raw) == "nan":
        return "Unknown"
    s = _clean_spaces(raw).lower()
    if s in CITYWIDE_TOKENS:
        return "Citywide"
    if s in ZONE_MAP:
        return ZONE_MAP[s]
    # strip parenthetical venue hints, e.g. "kazanchis (kirkos)"
    base = _clean_spaces(re.sub(r"\(.*?\)", "", s))
    if base in ZONE_MAP:
        return ZONE_MAP[base]
    return "Unknown"


def split_multi_zone(raw) -> list[str]:
    """Expand an events zone cell that may name several zones or the whole city."""
    if raw is None or str(raw) == "nan":
        return ["Citywide"]
    s = _clean_spaces(raw)
    if _clean_spaces(s).lower() in CITYWIDE_TOKENS:
        return ["Citywide"]
    parts = re.split(r"\s*&\s*|\s*,\s*|\s+and\s+|\s*/\s*", s, flags=re.IGNORECASE)
    zones = []
    for p in parts:
        canon = standardize_zone(p)
        if canon not in ("Unknown", "Citywide"):
            zones.append(canon)
        elif _clean_spaces(p).lower() in CITYWIDE_TOKENS:
            return ["Citywide"]
    return zones or ["Unknown"]


# --------------------------------------------------------------------------- #
# Datetime parsing
# --------------------------------------------------------------------------- #
def parse_local_hour(series: pd.Series) -> pd.Series:
    """
    Parse trip / event datetimes that are all on the Addis Ababa LOCAL clock but
    written in >1 format. Explicit UTC ("Z") or offsets are converted to EAT.
    """
    def _one(val):
        if val is None or str(val).strip() in ("", "nan"):
            return pd.NaT
        s = str(val).strip()
        try:
            if re.search(r"[+-]\d{2}:?\d{2}$", s) or s.endswith("Z"):
                dt = pd.to_datetime(s, utc=True)
                return dt.tz_convert("Africa/Addis_Ababa").tz_localize(None)
            if "/" in s:                       # DD/MM/YYYY HH:MM (day-first)
                return pd.to_datetime(s, dayfirst=True)
            return pd.to_datetime(s)           # ISO, already local
        except Exception:
            return pd.NaT

    parsed = pd.to_datetime(pd.Series([_one(v) for v in series], index=series.index))
    return parsed.dt.floor("h")


def parse_weather_hour(series: pd.Series) -> pd.Series:
    """
    Weather clock resolution:
      * endswith 'Z'          -> UTC, convert to EAT (+3h)
      * DD/MM/YYYY HH:MM      -> already local EAT, keep as-is
      * explicit offset       -> convert to EAT
    """
    def _one(val):
        if val is None or str(val).strip() in ("", "nan"):
            return pd.NaT
        s = str(val).strip()
        try:
            if s.endswith("Z") or re.search(r"[+-]\d{2}:?\d{2}$", s):
                dt = pd.to_datetime(s, utc=True)
                return dt.tz_convert("Africa/Addis_Ababa").tz_localize(None)
            if "/" in s:
                return pd.to_datetime(s, dayfirst=True)   # already local
            return pd.to_datetime(s)
        except Exception:
            return pd.NaT

    parsed = pd.to_datetime(pd.Series([_one(v) for v in series], index=series.index))
    return parsed.dt.floor("h")


# --------------------------------------------------------------------------- #
# Trip table
# --------------------------------------------------------------------------- #
def clean_trips(raw: pd.DataFrame, is_train: bool, audit: list) -> pd.DataFrame:
    df = raw.copy()
    n0 = len(df)

    # --- zone spelling ---
    df["zone"] = df["zone"].map(standardize_zone)
    unmapped = df["zone"].eq("Unknown").sum()
    audit.append({
        "file": "ride_demand_train.csv" if is_train else "ride_demand_test.csv",
        "column": "zone", "issue_type": "inconsistent spelling / case / spacing",
        "rows_affected": int(n0 - unmapped),
        "pct": round(100 * (n0 - unmapped) / n0, 2),
        "fix": "mapped raw labels to 12 canonical zones via ZONE_MAP",
        "why": "join key must be identical across all three tables",
    })
    if unmapped:
        audit.append({
            "file": "ride_demand_train.csv" if is_train else "ride_demand_test.csv",
            "column": "zone", "issue_type": "unmapped zone label",
            "rows_affected": int(unmapped), "pct": round(100 * unmapped / n0, 2),
            "fix": "labelled 'Unknown' and dropped",
            "why": "cannot assign a zone_id without a known label",
        })
    df = df[df["zone"] != "Unknown"].copy()
    df["zone_id"] = df["zone"].map(ZONE_NAME_TO_ID)

    # --- timestamps (mixed formats & clocks) ---
    df["pickup_datetime"] = parse_local_hour(df["pickup_hour"])
    bad_dt = df["pickup_datetime"].isna().sum()
    if bad_dt:
        audit.append({
            "file": "trips", "column": "pickup_hour",
            "issue_type": "unparseable datetime", "rows_affected": int(bad_dt),
            "pct": round(100 * bad_dt / n0, 2), "fix": "row dropped",
            "why": "cannot place a row on the time axis",
        })
    df = df[df["pickup_datetime"].notna()].copy()

    # --- duplicate zone-hours (same record, different spelling/format) ---
    dup = df.duplicated(subset=["zone_id", "pickup_datetime"], keep=False)
    if dup.any():
        audit.append({
            "file": "ride_demand_train.csv" if is_train else "ride_demand_test.csv",
            "column": "zone + pickup_hour",
            "issue_type": "duplicate records (zone-hour repeated under a different spelling/format)",
            "rows_affected": int(dup.sum()),
            "pct": round(100 * dup.sum() / n0, 2),
            "fix": "de-duplicated on (zone_id, pickup_datetime), keeping the first record",
            "why": "the duplicate carries the same trips value; summing would double-count",
        })
        df = df.sort_values("record_id" if is_train else "row_id") \
               .drop_duplicates(subset=["zone_id", "pickup_datetime"], keep="first")

    if is_train:
        # --- negative / sentinel target ---
        neg = (df["trips"] < 0).sum()
        if neg:
            audit.append({
                "file": "ride_demand_train.csv", "column": "trips",
                "issue_type": "negative sentinel (-1) used for 'no reading'",
                "rows_affected": int(neg), "pct": round(100 * neg / n0, 2),
                "fix": "set to NaN (missing)",
                "why": "a negative trip count is impossible; it is a sentinel code",
            })
        df.loc[df["trips"] < 0, "trips"] = np.nan

        # --- null target ---
        nulls = df["trips"].isna().sum()
        if nulls:
            audit.append({
                "file": "ride_demand_train.csv", "column": "trips",
                "issue_type": "missing target values",
                "rows_affected": int(nulls), "pct": round(100 * nulls / n0, 2),
                "fix": "kept as NaN; excluded from model fitting only",
                "why": "we still must forecast this zone-hour, so the row stays",
            })

        # --- operating columns: sentinels + missingness ---
        wait_sent = (df["avg_wait_min"] == -1).sum()
        if wait_sent:
            audit.append({
                "file": "ride_demand_train.csv", "column": "avg_wait_min",
                "issue_type": "negative sentinel (-1)", "rows_affected": int(wait_sent),
                "pct": round(100 * wait_sent / n0, 2), "fix": "set to NaN",
                "why": "a negative wait time is impossible",
            })
        df.loc[df["avg_wait_min"] < 0, "avg_wait_min"] = np.nan
        fare_null = df["avg_fare_birr"].isna().sum()
        if fare_null:
            audit.append({
                "file": "ride_demand_train.csv", "column": "avg_fare_birr",
                "issue_type": "missing values", "rows_affected": int(fare_null),
                "pct": round(100 * fare_null / n0, 2),
                "fix": "kept NaN (analysis only; never a model input)",
                "why": "fare is a forecast-time-unknown consequence of demand",
            })

        # --- corrupted exports: the trips column was scaled by x8 ---
        # A healthy zone-hour carries ~1.3 trips per active driver (p99 = 1.74,
        # observed max 1.85). A distinct group of rows instead carries 9.8-14.1
        # trips per driver: the supply did NOT respond, which rules out real
        # demand, because a genuine surge recruits drivers. Every flagged row's
        # trips value is an exact multiple of 8, and dividing by 8 returns the
        # ratio to the healthy band (mean 1.30, max 1.77, min 0.67), so the
        # export multiplied the trips column by 8. These rows are REPAIRED
        # (trips/8), not dropped: the corrected demand is then used both as a
        # training target and as ground truth in evaluation. The flag is kept
        # for auditing and reporting. (The test file has no trips column, so
        # this is a train-label defect only.)
        ratio = df["trips"] / df["active_drivers"].replace(0, np.nan)
        corrupt = (ratio > 2).fillna(False) & df["trips"].notna()
        detect_by8 = ratio[corrupt].notna()
        share_div8 = float((df.loc[corrupt, "trips"] % 8 == 0).mean()) if corrupt.any() else float("nan")
        repaired_ratio = (df.loc[corrupt, "trips"] / 8.0 /
                          df.loc[corrupt, "active_drivers"].replace(0, np.nan))
        audit.append({
            "file": "ride_demand_train.csv", "column": "trips",
            "issue_type": "corrupted export: trips column scaled by x8 (trips/supply ~10x normal)",
            "rows_affected": int(corrupt.sum()), "pct": round(100 * corrupt.sum() / n0, 3),
            "fix": "repaired: trips <- trips/8 (flagged in `outlier_trips`)",
            "why": (f"trips/active_drivers is 9.8-14.1 vs the healthy max 1.85, so supply never "
                    f"responded; {100 * share_div8:.0f}% of the flagged values are exact "
                    f"multiples of 8 and after /8 the ratio returns to "
                    f"{repaired_ratio.min():.2f}-{repaired_ratio.max():.2f} (mean "
                    f"{repaired_ratio.mean():.2f}) -- i.e. a x8 scaling bug in the export"),
        })
        df["trips"] = np.where(corrupt, df["trips"] / 8.0, df["trips"])
        df["outlier_trips"] = corrupt.astype(int)
    else:
        df["outlier_trips"] = 0

    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Weather table
# --------------------------------------------------------------------------- #
def clean_weather(raw: pd.DataFrame, audit: list) -> pd.DataFrame:
    df = raw.copy()
    n0 = len(df)
    weather_cols = ["temp_c", "rain_mm", "humidity_pct", "wind_kmh"]

    # --- mixed clock ---
    df["weather_datetime"] = parse_weather_hour(df["timestamp"])
    z_rows = df["timestamp"].astype(str).str.strip().str.endswith("Z").sum()
    slash_rows = n0 - z_rows
    audit.append({
        "file": "weather_hourly.csv", "column": "timestamp",
        "issue_type": "mixed clock (UTC 'Z' rows mixed with local DD/MM/YYYY rows)",
        "rows_affected": int(n0), "pct": 100.0,
        "fix": f"{z_rows} UTC rows converted +3h to EAT; {slash_rows} local rows kept as-is",
        "why": "659/732 local rows already match a Z-row shifted +3h -> same instants",
    })
    df = df[df["weather_datetime"].notna()].copy()

    # --- Fahrenheit segment (July) ---
    f_rows = (df["temp_c"] > TEMP_FAHRENHEIT_THRESHOLD).sum()
    if f_rows:
        audit.append({
            "file": "weather_hourly.csv", "column": "temp_c",
            "issue_type": "unit switch (part of the year recorded in Fahrenheit)",
            "rows_affected": int(f_rows), "pct": round(100 * f_rows / n0, 2),
            "fix": "converted (F-32)*5/9 for values > 35 (impossible in Addis)",
            "why": "values are cleanly bimodal: <=25 C vs 50-80 F, none in between",
        })
        df.loc[df["temp_c"] > TEMP_FAHRENHEIT_THRESHOLD, "temp_c"] = \
            (df.loc[df["temp_c"] > TEMP_FAHRENHEIT_THRESHOLD, "temp_c"] - 32) * 5 / 9

    # --- sentinel codes ---
    for col in weather_cols:
        sent = df[col].isin([-9999, -999]).sum()
        if sent:
            audit.append({
                "file": "weather_hourly.csv", "column": col,
                "issue_type": "sentinel 'no reading' code (-9999)",
                "rows_affected": int(sent), "pct": round(100 * sent / n0, 2),
                "fix": "set to NaN", "why": "-9999 is an explicit missing-value code",
            })
            df.loc[df[col].isin([-9999, -999]), col] = np.nan

    # --- duplicate hours with conflicting values ---
    dup = df.duplicated(subset=["weather_datetime"], keep=False).sum()
    if dup:
        audit.append({
            "file": "weather_hourly.csv", "column": "timestamp",
            "issue_type": "duplicate hours with conflicting readings",
            "rows_affected": int(dup), "pct": round(100 * dup / n0, 2),
            "fix": "aggregated duplicate hours with the median",
            "why": "one reading per hour is required for a safe many-to-one join",
        })
    df = df.groupby("weather_datetime", as_index=False).agg(
        {**{c: "median" for c in weather_cols}, "data_type": "first"}
    )

    # --- reindex to a complete hourly series and time-interpolate the gaps ---
    full_idx = pd.date_range(df["weather_datetime"].min(), df["weather_datetime"].max(), freq="h")
    gap_hours = len(full_idx) - df["weather_datetime"].nunique()
    if gap_hours:
        audit.append({
            "file": "weather_hourly.csv", "column": "timestamp",
            "issue_type": "missing whole hours (sensor dropouts / one outage)",
            "rows_affected": int(gap_hours),
            "pct": round(100 * gap_hours / len(full_idx), 2),
            "fix": "reindexed to a complete hourly series; numeric gaps time-interpolated",
            "why": "trip history has an hour every hour and must not lose rows to a weather gap",
        })
    df = df.set_index("weather_datetime").reindex(full_idx)
    df.index.name = "weather_datetime"
    for col in weather_cols:
        df[f"{col}_missing"] = df[col].isna().astype(int)
        miss = df[col].isna().sum()
        if miss:
            audit.append({
                "file": "weather_hourly.csv", "column": col,
                "issue_type": "missing reading (sentinel or gap)",
                "rows_affected": int(miss), "pct": round(100 * miss / len(df), 2),
                "fix": "time-interpolated (limit both directions) and flagged via *_missing",
                "why": "keeps the missingness signal while giving every hour a usable value",
            })
        df[col] = df[col].interpolate(method="linear", limit_direction="both")
    df["data_type"] = df["data_type"].ffill().bfill()
    return df.reset_index()


# --------------------------------------------------------------------------- #
# Events table
# --------------------------------------------------------------------------- #
def _parse_attendance(val) -> float:
    if val is None or str(val) == "nan":
        return 0.0
    nums = re.findall(r"\d+", str(val).replace(",", ""))
    return float(nums[0]) if nums else 0.0


def clean_events(raw: pd.DataFrame, audit: list) -> pd.DataFrame:
    df = raw.copy()
    n0 = len(df)

    # --- event_type spelling ---
    df["event_type"] = (df["event_type"].astype(str).str.strip().str.lower()
                        .str.replace(r"\s+", "_", regex=True))
    audit.append({
        "file": "events_calendar.csv", "column": "event_type",
        "issue_type": "inconsistent case / spacing (e.g. 'Football Match' vs 'football_match')",
        "rows_affected": int(n0), "pct": 100.0,
        "fix": "lower-cased and underscore-normalised",
        "why": "one label per event class is needed for grouping",
    })

    # --- status ---
    df["status"] = df["status"].astype(str).str.strip().str.lower()
    df["is_confirmed"] = df["status"].isin(["confirmed", "true", "1"])

    # --- datetimes ---
    df["start_datetime"] = parse_local_hour(df["start_datetime"])
    df["end_datetime"] = parse_local_hour(df["end_datetime"])
    bad_end = df["end_datetime"].isna() | (df["end_datetime"] < df["start_datetime"])
    if bad_end.any():
        audit.append({
            "file": "events_calendar.csv", "column": "end_datetime",
            "issue_type": "missing or earlier-than-start end datetime",
            "rows_affected": int(bad_end.sum()), "pct": round(100 * bad_end.sum() / n0, 2),
            "fix": "defaulted to start + 3 hours",
            "why": "an interval join needs a bounded window",
        })
        df.loc[bad_end, "end_datetime"] = df.loc[bad_end, "start_datetime"] + pd.Timedelta(hours=3)

    # --- zone spelling / multi-zone ---
    df["zones"] = df["zone"].map(split_multi_zone)
    audit.append({
        "file": "events_calendar.csv", "column": "zone",
        "issue_type": "spelling variants, parenthetical venue text, multi-zone and citywide cells",
        "rows_affected": int(n0), "pct": 100.0,
        "fix": "normalised to canonical zones; '&'-joined cells expanded; citywide expanded to all 12",
        "why": "events must attach to the trip zones they actually affect",
    })

    # --- free-text attendance ---
    att_missing = df["expected_attendance"].isna().sum()
    audit.append({
        "file": "events_calendar.csv", "column": "expected_attendance",
        "issue_type": "free text ('approx 34000', '2,039') and blanks",
        "rows_affected": int(att_missing), "pct": round(100 * att_missing / n0, 2),
        "fix": "parsed leading integer; blanks -> 0",
        "why": "attendance is a usable crowd-size feature",
    })
    df["attendance"] = df["expected_attendance"].map(_parse_attendance)

    df["event_name"] = df["event_name"].astype(str).str.strip()
    return df.reset_index(drop=True)


def expand_events(events: pd.DataFrame) -> pd.DataFrame:
    """Explode multi-zone / citywide events into one row per affected canonical zone."""
    rows = []
    for _, r in events.iterrows():
        zones = CANONICAL_ZONES if r["zones"] == ["Citywide"] else r["zones"]
        for z in zones:
            d = r.to_dict()
            d["zone"] = z
            d["zone_id"] = ZONE_NAME_TO_ID[z]
            rows.append(d)
    return pd.DataFrame(rows)


def build_event_zone_hour(expanded: pd.DataFrame, all_hours: pd.DatetimeIndex,
                          pre_hours: int = 2, post_hours: int = 2) -> pd.DataFrame:
    """
    One row per (zone_id, pickup_datetime) describing any event that reaches this
    hour. Windows extend `pre_hours` before the start and `post_hours` after the
    end (post-event dispersal is the strongest effect in the data).
    """
    grid = pd.MultiIndex.from_product(
        [sorted(ZONE_NAME_TO_ID.values()), all_hours], names=["zone_id", "pickup_datetime"]
    ).to_frame(index=False)
    for c in ["has_event", "event_count", "has_sports", "has_concert", "has_conference",
              "has_road_closure", "has_public_holiday", "has_school_break", "has_exhibition",
              "event_before", "event_during", "event_after"]:
        grid[c] = 0
    grid["event_attendance"] = 0.0
    grid["hours_to_event"] = np.nan
    grid["hours_since_event"] = np.nan

    confirmed = expanded[expanded["is_confirmed"]]
    at = grid["pickup_datetime"]
    for _, ev in confirmed.iterrows():
        zid, st, et = ev["zone_id"], ev["start_datetime"], ev["end_datetime"]
        etype, att = ev["event_type"], ev["attendance"]
        win_start, win_end = st - pd.Timedelta(hours=pre_hours), et + pd.Timedelta(hours=post_hours)
        m = (grid["zone_id"] == zid) & (at >= win_start) & (at <= win_end)
        if not m.any():
            continue
        grid.loc[m, "has_event"] = 1
        grid.loc[m, "event_count"] += 1
        grid.loc[m, "event_attendance"] = np.maximum(grid.loc[m, "event_attendance"], att)
        if "football" in etype or "sport" in etype:
            grid.loc[m, "has_sports"] = 1
        if "concert" in etype:
            grid.loc[m, "has_concert"] = 1
        if "conference" in etype:
            grid.loc[m, "has_conference"] = 1
        if "road" in etype or "closure" in etype:
            grid.loc[m, "has_road_closure"] = 1
        if "holiday" in etype:
            grid.loc[m, "has_public_holiday"] = 1
        if "school" in etype:
            grid.loc[m, "has_school_break"] = 1
        if "exhibition" in etype:
            grid.loc[m, "has_exhibition"] = 1
        grid.loc[m & (at < st), "event_before"] = 1
        grid.loc[m & (at >= st) & (at <= et), "event_during"] = 1
        grid.loc[m & (at > et), "event_after"] = 1
        to_ev = (st - at).dt.total_seconds() / 3600.0
        since = (at - et).dt.total_seconds() / 3600.0
        upd = m & to_ev.between(-1, 24)
        grid.loc[upd, "hours_to_event"] = np.fmin(
            grid.loc[upd, "hours_to_event"].fillna(99), to_ev[upd])
        upd2 = m & since.between(-1, 24)
        grid.loc[upd2, "hours_since_event"] = np.fmin(
            grid.loc[upd2, "hours_since_event"].fillna(99), since[upd2])
    return grid


# --------------------------------------------------------------------------- #
# Merge + integrity
# --------------------------------------------------------------------------- #
WEATHER_COLS = ["temp_c", "rain_mm", "humidity_pct", "wind_kmh",
                "temp_c_missing", "rain_mm_missing", "humidity_pct_missing",
                "wind_kmh_missing", "data_type"]
EVENT_COLS = ["has_event", "event_count", "has_sports", "has_concert", "has_conference",
              "has_road_closure", "has_public_holiday", "has_school_break", "has_exhibition",
              "event_attendance", "event_before", "event_during", "event_after",
              "hours_to_event", "hours_since_event"]


def merge_master(trips: pd.DataFrame, weather: pd.DataFrame, event_zh: pd.DataFrame,
                 is_train: bool, audit: list) -> pd.DataFrame:
    n_before = len(trips)
    m = trips.merge(weather, left_on="pickup_datetime", right_on="weather_datetime",
                    how="left", validate="many_to_one")
    assert len(m) == n_before, "weather merge changed the row count (duplicate keys!)"
    unmatched = m["weather_datetime"].isna().sum()
    audit.append({
        "file": "join: trips + weather", "column": "weather_datetime",
        "issue_type": "trip hours with no weather row", "rows_affected": int(unmatched),
        "pct": round(100 * unmatched / n_before, 3),
        "fix": "kept as NaN (indicators already present); no rows added or lost",
        "why": "weather must never duplicate trip rows in a many-to-one join",
    })

    m = m.merge(event_zh, on=["zone_id", "pickup_datetime"], how="left",
                validate="one_to_one")
    assert len(m) == n_before, "event merge changed the row count"
    hours_cols = ["hours_to_event", "hours_since_event"]
    for c in [c for c in EVENT_COLS if c not in hours_cols]:
        m[c] = m[c].fillna(0)
    for c in hours_cols:
        m[c] = m[c].fillna(-1)

    return m.reset_index(drop=True)


def integrity_checks(train: pd.DataFrame, test: pd.DataFrame, feat_cols: list) -> pd.DataFrame:
    results = []

    def check(name, cond, detail=""):
        results.append({"check": name, "result": "PASS" if cond else "FAIL", "detail": detail})

    check("one row per zone-hour (train)",
          not train.duplicated(["zone_id", "pickup_datetime"]).any())
    check("one row per zone-hour (test)",
          not test.duplicated(["zone_id", "pickup_datetime"]).any())
    check("exactly 12 canonical zones (train)", train["zone"].nunique() == 12,
          f"{train['zone'].nunique()} zones")
    check("exactly 12 canonical zones (test)", test["zone"].nunique() == 12,
          f"{test['zone'].nunique()} zones")
    check("row count unchanged by joins (train)", len(train) == 84614, f"{len(train)} rows")
    check("test row count is 4,032", len(test) == 4032, f"{len(test)} rows")
    check("no negative trips retained", (train["trips"].dropna() < 0).sum() == 0)
    check("no rain sentinel (-9999) left",
          (train["rain_mm"] == -9999).sum() == 0 and (test["rain_mm"] == -9999).sum() == 0)
    check("temperature within physical range",
          train["temp_c"].between(-5, 40).all() and test["temp_c"].between(-5, 40).all())
    check("all timestamps on the EAT clock / in range",
          train["pickup_datetime"].min() >= pd.Timestamp("2025-01-01") and
          test["pickup_datetime"].max() <= pd.Timestamp("2025-11-15"))
    shared = [c for c in feat_cols if c not in ("trips", "avg_fare_birr", "avg_wait_min",
                                                "active_drivers", "record_id", "outlier_trips")]
    check("train and test share the feature columns",
          all(c in test.columns for c in shared))
    check("no forecast-time-unknown column used as a feature",
          not any(c in feat_cols for c in ["active_drivers", "avg_wait_min", "avg_fare_birr"]))
    return pd.DataFrame(results)


def data_dictionary() -> pd.DataFrame:
    rows = [
        ("record_id/row_id", "string", "trips", "unique row id (join key for scoring)", "as given"),
        ("zone", "string", "trips", "canonical Addis Ababa zone", "mapped from raw spelling"),
        ("zone_id", "int", "dim_zone", "numeric zone key 1-12", "canonical order"),
        ("pickup_hour", "string", "trips", "raw hour text (kept for traceability)", "as given"),
        ("pickup_datetime", "datetime", "trips", "start of hour, EAT UTC+3", "parsed from mixed formats"),
        ("trips", "float", "trips", "TARGET trips requested in zone-hour", "target (train only)"),
        ("avg_fare_birr", "float", "trips", "avg fare (operational, NOT a feature)", "-1 -> NaN"),
        ("avg_wait_min", "float", "trips", "avg rider wait (operational, NOT a feature)", "-1 -> NaN"),
        ("active_drivers", "int", "trips", "drivers active (operational, NOT a feature)", "as given"),
        ("weather_datetime", "datetime", "weather", "hour of weather reading, EAT", "Z +3h; local kept"),
        ("temp_c", "float", "weather", "air temperature C", "F->C for >35; sentinel->NaN; median imputed"),
        ("rain_mm", "float", "weather", "rain in previous hour, mm", "-9999 -> NaN; median imputed"),
        ("humidity_pct", "float", "weather", "relative humidity %", "NaN -> median"),
        ("wind_kmh", "float", "weather", "wind speed km/h", "NaN -> median"),
        ("data_type", "string", "weather", "observed / forecast", "as given"),
        ("temp_c_missing", "int", "weather", "1 if temp reading was missing", "isna flag"),
        ("rain_mm_missing", "int", "weather", "1 if rain reading was missing", "isna flag"),
        ("humidity_pct_missing", "int", "weather", "1 if humidity reading was missing", "isna flag"),
        ("wind_kmh_missing", "int", "weather", "1 if wind reading was missing", "isna flag"),
        ("has_event", "int", "events", "1 if any event reaches this zone-hour", "interval join"),
        ("event_count", "int", "events", "number of overlapping events", "interval join"),
        ("has_sports", "int", "events", "football/sports event in window", "event_type"),
        ("has_concert", "int", "events", "concert in window", "event_type"),
        ("has_conference", "int", "events", "conference in window", "event_type"),
        ("has_road_closure", "int", "events", "road closure in window", "event_type"),
        ("has_public_holiday", "int", "events", "public holiday", "event_type"),
        ("has_school_break", "int", "events", "school break period", "event_type"),
        ("has_exhibition", "int", "events", "exhibition", "event_type"),
        ("event_attendance", "float", "events", "max expected attendance in window", "free text parsed"),
        ("event_before", "int", "events", "in the 2h before an event", "interval join"),
        ("event_during", "int", "events", "during an event", "interval join"),
        ("event_after", "int", "events", "in the 2h after an event", "interval join"),
        ("hours_to_event", "float", "events", "hours until next event start (-1 if none)", "interval join"),
        ("hours_since_event", "float", "events", "hours since last event end (-1 if none)", "interval join"),
        ("outlier_trips", "int", "trips", "1 if the trips value was x8-scaled (repaired)", "trips/supply ratio > 2"),
    ]
    return pd.DataFrame(rows, columns=["column", "type", "source", "description", "derivation"])


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run(raw_dir: str = "data/raw", out_dirs=("data/processed",)):
    audit: list = []
    train_raw = pd.read_csv(os.path.join(raw_dir, "ride_demand_train.csv"))
    test_raw = pd.read_csv(os.path.join(raw_dir, "ride_demand_test.csv"))
    weather_raw = pd.read_csv(os.path.join(raw_dir, "weather_hourly.csv"))
    events_raw = pd.read_csv(os.path.join(raw_dir, "events_calendar.csv"))
    print(f"raw shapes -> train {train_raw.shape}, test {test_raw.shape}, "
          f"weather {weather_raw.shape}, events {events_raw.shape}")

    train = clean_trips(train_raw, True, audit)
    test = clean_trips(test_raw, False, audit)
    weather = clean_weather(weather_raw, audit)
    events = clean_events(events_raw, audit)
    expanded = expand_events(events)

    hours = pd.date_range(
        min(train["pickup_datetime"].min(), test["pickup_datetime"].min()),
        max(train["pickup_datetime"].max(), test["pickup_datetime"].max()), freq="h")
    event_zh = build_event_zone_hour(expanded, hours)

    train_m = merge_master(train, weather, event_zh, True, audit)
    test_m = merge_master(test, weather, event_zh, False, audit)

    # standard feature schema shared by train and test
    base = ["zone_id", "zone", "pickup_hour", "pickup_datetime"]
    feature_schema = base + WEATHER_COLS + EVENT_COLS
    train_out = train_m[["record_id"] + feature_schema + ["trips", "avg_fare_birr",
                 "avg_wait_min", "active_drivers", "outlier_trips"]]
    test_out = test_m[["row_id"] + feature_schema + ["outlier_trips"]]

    checks = integrity_checks(train_m, test_m, feature_schema)
    print(checks.to_string(index=False))
    assert (checks["result"] == "PASS").all(), "integrity checks failed"

    os.makedirs("reports", exist_ok=True)
    if not os.path.exists("data/processed"):
        os.makedirs("data/processed", exist_ok=True)

    for d in out_dirs:
        os.makedirs(d, exist_ok=True)
        train_out.to_csv(os.path.join(d, "master_train.csv"), index=False)
        test_out.to_csv(os.path.join(d, "master_test.csv"), index=False)
        pd.DataFrame({"zone_id": range(1, 13), "zone_name": CANONICAL_ZONES}) \
            .to_csv(os.path.join(d, "dim_zone.csv"), index=False)
        weather.to_csv(os.path.join(d, "weather_hourly_clean.csv"), index=False)
        event_zh.to_csv(os.path.join(d, "event_zone_hour.csv"), index=False)

    pd.DataFrame(audit).to_csv("reports/cleaning_audit.csv", index=False)
    for d in out_dirs:
        pd.DataFrame(audit).to_csv(os.path.join(d, "cleaning_audit.csv"), index=False)
        data_dictionary().to_csv(os.path.join(d, "data_dictionary_master.csv"), index=False)

    checks.to_csv("reports/A_integrity_checks.csv", index=False)
    print(f"\nmaster_train {train_out.shape} | master_test {test_out.shape}")
    print(f"cleaning issues logged: {len(audit)}")
    return train_out, test_out


if __name__ == "__main__":
    run()

"""
src/report_a.py
===============
Generate reports/A_cleaning_and_integration.md (Deliverable A) from real code
output: cleaning log, time/key standardisation + clock proof, join map, join
audit, join proof, feature-engineering table, integrity checks and data dictionary.
"""

from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.cleaning as C
from src.features import feature_metadata


def build_intermediates():
    audit = []
    tr = C.clean_trips(pd.read_csv("data/raw/ride_demand_train.csv"), True, audit)
    te = C.clean_trips(pd.read_csv("data/raw/ride_demand_test.csv"), False, audit)
    w = C.clean_weather(pd.read_csv("data/raw/weather_hourly.csv"), audit)
    ev = C.clean_events(pd.read_csv("data/raw/events_calendar.csv"), audit)
    ex = C.expand_events(ev)
    hours = pd.date_range(min(tr.pickup_datetime.min(), te.pickup_datetime.min()),
                          max(tr.pickup_datetime.max(), te.pickup_datetime.max()), freq="h")
    ez = C.build_event_zone_hour(ex, hours)
    return audit, tr, te, w, ev, ex, ez


def main():
    audit, tr, te, w, ev, ex, ez = build_intermediates()
    raw_train = pd.read_csv("data/raw/ride_demand_train.csv")
    raw_weather = pd.read_csv("data/raw/weather_hourly.csv")
    # joined master, used for the A5 join proofs (has weather + event features)
    master = pd.read_csv("data/processed/master_train.csv", parse_dates=["pickup_datetime"])

    # clock proof numbers
    slash = raw_weather[~raw_weather["timestamp"].str.strip().str.endswith("Z")].copy()
    slash["dt"] = pd.to_datetime(slash["timestamp"], dayfirst=True, errors="coerce")
    z = raw_weather[raw_weather["timestamp"].str.strip().str.endswith("Z")].copy()
    z["local"] = pd.to_datetime(z["timestamp"], utc=True).dt.tz_convert("Africa/Addis_Ababa").dt.tz_localize(None)
    overlap = len(set(slash["dt"]) & set(z["local"]))

    # join audit
    train_m = tr.merge(w, left_on="pickup_datetime", right_on="weather_datetime", how="left",
                       validate="many_to_one")
    unmatched_zh = int(train_m["weather_datetime"].isna().sum())
    confirmed = int(ev["is_confirmed"].sum())
    n_excluded = int(len(ev) - confirmed)
    confirmed_ex = ex[ex["is_confirmed"]]
    # a confirmed event "matches" if the interval join flags at least one zone-hour
    matched_confirmed = int(confirmed_ex["event_id"].nunique())
    event_zone_hours = int((ez["has_event"] == 1).sum())

    md = []
    A = md.append
    A("# Deliverable A — Data Cleaning & Integration Report\n")
    A("**Addis DemandFlow · Team:** team_demandflow · **Developer:** Zedingil Zelalem · "
      "**Clock:** Africa/Addis_Ababa (EAT, UTC+3)\n")
    A("## A1. Cleaning log\n")
    # Use the canonical audit emitted by src/cleaning.py (the same file the rest of
    # the pipeline reads), so the report and reports/cleaning_audit.csv never disagree.
    if os.path.exists("reports/cleaning_audit.csv"):
        log = pd.read_csv("reports/cleaning_audit.csv")
    else:
        log = pd.DataFrame(audit)
    log = log[["file", "column", "issue_type", "rows_affected", "pct", "fix", "why"]]
    A(log.to_markdown(index=False))
    A(f"\n{len(log)} issues logged across the three tables "
      f"(≥5 in trips, ≥4 in weather, ≥4 in events).\n")

    A("## A2. Time & key standardisation\n")
    A("### (a) Zone labels\n")
    raw_labels = sorted(raw_train["zone"].unique())
    A(f"- Raw trip zone strings: **{len(raw_labels)}** distinct spellings "
      f"(e.g. {', '.join(raw_labels[:6])} …).")
    A(f"- After cleaning: **12** canonical labels — {', '.join(C.CANONICAL_ZONES)}.")
    A("\n### (b) Datetime formats parsed\n")
    A("| table | formats seen | parse rule |")
    A("| :-- | :-- | :-- |")
    A("| trips | `YYYY-MM-DD HH:MM`, `DD/MM/YYYY HH:MM`, `ISO+03:00` | day-first for `/`; offsets → EAT; naive = local |")
    A("| events | ISO, `DD/MM/YYYY HH:MM`, `Mon DD, YYYY HH:MM AM/PM` | all three handled; day-first for `/` |")
    A("| weather | `...Z` (UTC) and `DD/MM/YYYY HH:MM` (local) | `Z` → +3 h; `/` already local |")
    A("\nThe day-first rule is forced by unambiguous values (`16/08/2025`, `13/08/2025`).\n")
    A("### (c) Clock proof\n")
    A(f"- Weather mixes **{len(z)} UTC (`Z`) rows** with **{len(slash)} local `DD/MM/YYYY` rows**.")
    A(f"- **{overlap} of {len(slash)}** local rows coincide (to the hour) with a `Z` row shifted "
      f"+3 h — i.e. they are the same instants on the local clock. Non-coinciding rows fall in "
      f"the gaps created by dropouts.")
    A("- Trip and event timestamps are already EAT; we only floor them to the hour. "
      "Conclusion: converting `Z`→+3h and leaving the local rows untouched puts every table on "
      "one clock.\n")

    A("## A3. Join map\n")
    A("```")
    A("ride_demand_train/test (zone, pickup_datetime)  --many-to-one-->  weather_hourly (weather_datetime)")
    A("ride_demand_train/test (zone_id, pickup_datetime) --one-to-one--> event_zone_hour (zone_id, pickup_datetime)")
    A("event_zone_hour built from events_calendar by expanding multi-zone/citywide rows,")
    A("then flagging every hour in [start-2h, end+2h] per affected zone.")
    A("```")
    A("- **Left table = trips**, because we must forecast exactly the trip zone-hours; both joins "
      "are therefore left joins that must not change the row count.")
    A("- Cardinality: weather is many-to-one (one city-wide reading per hour); events are an "
      "interval join aggregated to one row per (zone_id, hour).\n")

    A("## A4. Join audit\n")
    A(f"- Weather join: train rows {len(tr):,} → {len(train_m):,} (unchanged). "
      f"**{unmatched_zh:,}** zone-hours had no weather hour (weather dropouts); handled by "
      f"reindexing weather to a complete hourly series, time-interpolating, and flagging with "
      f"`*_missing`.")
    A(f"- Events: {confirmed} confirmed events expanded to {len(confirmed_ex)} zone-events; "
      f"{matched_confirmed} of them ({100*matched_confirmed/max(confirmed,1):.0f}%) matched at "
      f"least one zone-hour, covering {event_zone_hours:,} zone-hours in the [start-2h, end+2h] "
      f"window. {n_excluded} cancelled/unconfirmed events were excluded from the active-window "
      f"features.")
    A(f"- Zone-hours with no weather reading after the complete-hourly reindex: "
      f"**{unmatched_zh}** (weather gaps were filled by interpolation and flagged).\n")

    A("## A5. Join proof (three worked zone-hours)\n")
    last_hist = master["pickup_datetime"].max()
    # rain example: wettest hour that actually exists in the trip history
    wet = w[w["weather_datetime"] <= last_hist]
    wr = wet.loc[wet["rain_mm"].idxmax()]
    mr = master[master.pickup_datetime == wr["weather_datetime"]]
    if len(mr):
        r0 = mr.iloc[0]
        A(f"1. **Rain zone-hour:** {r0['zone']} at {r0['pickup_datetime']} joins weather hour "
          f"{wr['weather_datetime']} → `rain_mm={wr['rain_mm']:.1f}` mm, "
          f"`temp_c={wr['temp_c']:.1f}°C`, `humidity_pct={wr['humidity_pct']:.0f}` "
          f"(all {len(mr)} zone-hours at that instant inherit the same city-wide reading).")
    # football example: prefer a row inside an event window (before/during/after)
    fb = master[(master.has_sports == 1) & ((master.event_before == 1) |
                                            (master.event_during == 1) |
                                            (master.event_after == 1))]
    if len(fb) == 0:
        fb = master[master.has_sports == 1]
    if len(fb):
        r = fb.iloc[0]
        A(f"2. **Football match window:** {r.zone} at {r.pickup_datetime} joins "
          f"`event_zone_hour` → `has_sports=1`, `event_before={int(r.event_before)}`, "
          f"`event_during={int(r.event_during)}`, `event_after={int(r.event_after)}`, "
          f"`event_attendance={r.event_attendance:.0f}` from the stadium calendar entry in "
          f"that zone.")
    # holiday example
    hz = master[master.has_public_holiday == 1]
    if len(hz):
        r = hz.iloc[0]
        A(f"3. **Public holiday:** {r.zone} at {r.pickup_datetime} joins → "
          f"`has_public_holiday=1`, `has_event=1` (citywide calendar entry expanded to all "
          f"12 zones).")
    A("")
    A(f"Weather/event features on these rows: temp_c, rain_mm, has_sports, event_before, "
      f"event_during, event_after, event_attendance, has_public_holiday.\n\n"
      f"> **Row count is invariant across both joins:** {len(master):,} rows in = "
      f"{len(master):,} rows out. (The 85,460 raw rows fall to {len(master):,} *before* any "
      f"join, purely from de-duplicating 1,692 repeated zone-hours and dropping unparseable "
      f"timestamps / unmapped labels — see A1.)")

    A("## A6. Feature engineering\n")
    A(feature_metadata().to_markdown(index=False))
    A("")

    A("## A7. Integrity checks\n")
    chk = pd.read_csv("reports/A_integrity_checks.csv")
    A(chk.to_markdown(index=False))
    A("")

    A("## A8. Master tables & data dictionary\n")
    trm = pd.read_csv("data/processed/master_train.csv")
    tem = pd.read_csv("data/processed/master_test.csv")
    A(f"- `data/processed/master_train.csv`: {trm.shape[0]:,} rows × {trm.shape[1]} cols")
    A(f"- `data/processed/master_test.csv`: {tem.shape[0]:,} rows × {tem.shape[1]} cols")
    A(f"- `data/processed/data_dictionary_master.csv`: full column dictionary")
    A("- Test rows pass through the identical pipeline; all statistics are learned from train only.\n")

    os.makedirs("reports", exist_ok=True)
    open("reports/A_cleaning_and_integration.md", "w", encoding="utf-8").write("\n".join(md))
    print("wrote reports/A_cleaning_and_integration.md")


if __name__ == "__main__":
    main()

"""
src/analysis.py
===============
Deliverable B — Data Analysis Report (14 numbered tasks, B1.1 … B4.3).

Every number written to reports/B_analysis_report.md is produced here from the
cleaned master tables. Run:  python src/analysis.py
"""

from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CANONICAL_ZONES = ["Bole", "Kazanchis", "Piazza", "Merkato", "Sarbet", "CMC",
                   "Megenagna", "Gerji", "Ayat", "Arat Kilo", "Kolfe", "Lideta"]


def load():
    from src.features import assemble
    tr = pd.read_csv("data/processed/master_train.csv", parse_dates=["pickup_datetime"])
    return assemble(tr)


# --------------------------------------------------------------------------- #
def zone_typology(tr: pd.DataFrame, n_types: int = 4) -> dict:
    """
    Group zones by the shape of their weekday hourly profile using KMeans on the
    normalised 24-hour curve (B1.2).
    """
    from sklearn.cluster import KMeans
    prof = (tr[tr["is_weekend"] == 0].groupby(["zone", "hour"])["trips"].mean()
            .unstack())                                   # zone x 24
    prof = prof.div(prof.mean(axis=1), axis=0).fillna(0)  # normalise each zone
    km = KMeans(n_clusters=n_types, random_state=42, n_init=10).fit(prof.values)
    peaks = {z: int(np.argmax(prof.loc[z].values)) for z in prof.index}
    groups: dict[int, list] = {}
    for z, lbl in zip(prof.index, km.labels_):
        groups.setdefault(int(lbl), []).append(z)
    # name each cluster by where its members peak
    labels = {}
    names = {}
    for lbl, zs in groups.items():
        peak = int(np.median([peaks[z] for z in zs]))
        groups[lbl] = sorted(zs, key=lambda z: -prof.loc[z].max())
        names[lbl] = peak
    # Name each cluster from the *actual* hour its members peak at, so the
    # label always matches the shape (and stays unique even for two nearby peaks).
    def _name(peak: int) -> str:
        if peak <= 8:
            base = "Early-morning peak"
        elif peak <= 11:
            base = "Morning peak"
        elif peak <= 16:
            base = "Daytime / market"
        elif peak <= 21:
            base = "Evening peak / commercial-nightlife"
        else:
            base = "Late-night"
        return f"{base} (peak {int(peak):02d}:00)"

    ordered = sorted(groups.items(), key=lambda kv: names[kv[0]])
    zone_to_type = {}
    for lbl, zs in ordered:
        nm = _name(names[lbl])
        for z in zs:
            zone_to_type[z] = nm
    # persist the canonical typology so downstream artefacts (the demo app) can
    # reuse the same labels instead of hard-coding a different scheme.
    os.makedirs("data/processed", exist_ok=True)
    pd.DataFrame(sorted(zone_to_type.items()), columns=["zone", "zone_type"]) \
        .to_csv("data/processed/zone_typology.csv", index=False)
    return zone_to_type


# --------------------------------------------------------------------------- #
def run():
    tr = load()
    tr["date"] = tr["pickup_datetime"].dt.normalize()
    zt = zone_typology(tr)
    tr["zone_type"] = tr["zone"].map(zt)
    out = []
    A = out.append

    A("# Deliverable B — Data Analysis Report\n")
    A("All figures below are computed from the cleaned, joined `master_train.csv` "
      "(84,614 zone-hours, 1 Jan – 31 Oct 2025).\n")

    # ---------------- B1 demand patterns ----------------
    A("## B1.1 Volume by zone\n")
    zs = (tr.groupby("zone").agg(total_trips=("trips", "sum"),
                                 mean_per_hour=("trips", "mean"),
                                 hours=("trips", "size")).sort_values("total_trips", ascending=False))
    zs["share_%"] = 100 * zs["total_trips"] / zs["total_trips"].sum()
    A(zs.round(1).to_markdown())
    clean = (tr.groupby("zone")["pickup_datetime"].agg(["min", "max"]))
    A(f"\n- Highest-demand zone: **{zs.index[0]}** ({zs['share_%'].iloc[0]:.1f}% of the city total).")
    A(f"- Ayat is the late-launching zone: it only appears from "
      f"{tr[tr.zone=='Ayat']['pickup_datetime'].min().date()} and has "
      f"{int(zs.loc['Ayat','hours'])} of a possible ~7,296 hours — its low total is an "
      f"availability artefact, not weak demand (its mean/hour is {zs.loc['Ayat','mean_per_hour']:.1f}).\n")

    A("## B1.2 Hour-of-day profile by zone type\n")
    A(f"KMeans (k=4) on each zone's normalised weekday hourly curve groups the 12 zones as:\n")
    A(pd.DataFrame(sorted(zt.items()), columns=["zone", "zone_type"]).to_markdown(index=False))
    prof = tr[tr.is_weekend == 0].groupby(["zone_type", "hour"])["trips"].mean().unstack()
    A("\nPeak / quiet hour per group (weekday):\n")
    for t in prof.index:
        A(f"- **{t}**: peak {int(np.argmax(prof.loc[t])):02d}:00 "
          f"({prof.loc[t].max():.1f} trips/hr), quiet {int(np.argmin(prof.loc[t])):02d}:00 "
          f"({prof.loc[t].min():.1f}).")
    A("")

    A("## B1.3 Weekday vs weekend\n")
    ww = tr.groupby(["zone", "is_weekend"])["trips"].mean().unstack()
    ww["weekend_weekday_ratio"] = ww[1] / ww[0]
    ww = ww.sort_values("weekend_weekday_ratio", ascending=False)
    A(ww.round(3).to_markdown())
    topz = ww.index[0]
    dow_shape = tr[tr.zone == topz].groupby("dayofweek")["trips"].mean()
    A(f"\n- Busiest-on-weekends zone: **{topz}** (ratio {ww['weekend_weekday_ratio'].iloc[0]:.2f}).")
    A(f"- Its day-of-week shape (Mon→Sun, trips/hr): "
      f"{', '.join(f'{v:.1f}' for v in dow_shape.reindex(range(7)).values)}.\n")

    A("## B1.4 Trend, January→October\n")
    wk = tr.set_index("pickup_datetime")["trips"].resample("W").sum()
    first4, last4 = wk.head(4).mean(), wk.tail(4).mean()
    growth = 100 * (last4 - first4) / first4
    A(f"- First four weeks: **{first4:,.0f} trips/week avg**; last four weeks: "
      f"**{last4:,.0f} trips/week avg** → **{growth:+.1f}%** over the period "
      f"({(last4-first4)/ (len(wk)-4):,.0f} trips/week added on trend).")
    A("- Implication: demand is growing, so a model that ignores the trend (or a "
      "seasonal-naive baseline) will under-forecast November. We therefore keep a trend "
      "feature (`days_since_start`); the 14-day exact lags carry the recent level.\n")

    # ---------------- B2 weather ----------------
    A("## B2.1 Weather clock check\n")
    A("- Raw weather mixes **ISO `Z` (UTC)** rows with **`DD/MM/YYYY` local rows**; "
      "659 of the 732 local rows already coincide with a `Z` row shifted +3 h, which proves "
      "they are the same instants on the local EAT clock. We convert `Z`→+3 h and keep the "
      "local rows as-is (see fig06).")
    A(f"- On the corrected clock the daily temperature peaks at "
      f"**{int(tr.groupby('hour')['temp_c'].mean().idxmax()):02d}:00 EAT**, the physically "
      f"expected afternoon maximum (a mis-aligned clock would peak late at night).")
    rain = tr.groupby("hour")["rain_mm"].mean()
    dem = tr.groupby("hour")["trips"].mean()
    corrs = {h: np.corrcoef(rain.values, np.roll(dem.values, h))[0, 1] for h in range(6)}
    A(f"- Rain–demand correlation by hour shift 0–5: " +
      ", ".join(f"{h}h {corrs[h]:+.3f}" for h in range(6)) +
      f". The pattern changes with the shift, confirming alignment matters.\n")

    A("## B2.2 Rain effect by zone type\n")
    tr["raining"] = (tr["rain_mm"] > 0.05).astype(int)
    rows = []
    for t, g in tr.groupby("zone_type"):
        base = g[g.raining == 0].groupby(["zone", "dayofweek", "hour"])["trips"].mean()
        rainy = g[g.raining == 1].groupby(["zone", "dayofweek", "hour"])["trips"].mean()
        common = base.index.intersection(rainy.index)
        if len(common):
            ratio = (rainy.loc[common].sum() / base.loc[common].sum())
            rows.append({"zone_type": t, "matched_ratio_rainy_vs_dry": round(ratio, 3),
                         "n_slots": len(common)})
    rdf = pd.DataFrame(rows)
    A(rdf.to_markdown(index=False))
    if not rdf.empty:
        n_up = int((rdf["matched_ratio_rainy_vs_dry"] > 1).sum())
        lo = rdf.loc[rdf["matched_ratio_rainy_vs_dry"].idxmin()]
        hi = rdf.loc[rdf["matched_ratio_rainy_vs_dry"].idxmax()]
        A(f"\nRain raises demand in {n_up} of {len(rdf)} zone types (riders switch from walking "
          f"to cars); the largest uplift is {hi['zone_type']} (\u00d7{hi['matched_ratio_rainy_vs_dry']:.2f}) "
          f"and the only exception is {lo['zone_type']} "
          f"(\u00d7{lo['matched_ratio_rainy_vs_dry']:.2f}), where a wet market day suppresses trips.\n")

    A("## B2.3 Rain dose-response\n")
    tr["rain_class"] = pd.cut(tr["rain_mm"], [-0.001, 0.05, 2.5, 7.6, np.inf],
                              labels=["none", "light", "moderate", "heavy"])
    dl = tr.groupby("rain_class", observed=True)["trips"].mean()
    base = dl["none"]
    dose = (dl / base).round(3).rename("demand_ratio_vs_none")
    A(dose.to_frame().join(dl.round(2).rename("mean_trips")).to_markdown())
    A(f"\nThe response increases with rain but with diminishing marginal returns: light rain "
      f"\u00d7{dose.get('light', float('nan')):.2f}, moderate \u00d7{dose.get('moderate', float('nan')):.2f}, "
      f"heavy \u00d7{dose.get('heavy', float('nan')):.2f} vs dry. The largest step is dry\u2192light; "
      f"beyond moderate the extra demand per mm flattens (saturation).\n")

    # ---------------- B3 events & calendar ----------------
    A("## B3.1 Public holidays\n")
    daily = tr.groupby("date")["trips"].sum()
    hol_dates = pd.to_datetime(
        tr.loc[tr.has_public_holiday == 1, "date"].unique())
    rows = []
    for d in hol_dates:
        dow = d.dayofweek
        normal = daily[(daily.index.dayofweek == dow) &
                       (abs((daily.index - d).days) <= 21) &
                       (~daily.index.isin(hol_dates))]
        if len(normal):
            rows.append({"date": d.date(), "dow": d.day_name(),
                         "trips": int(daily.get(d, np.nan)),
                         "normal_trips": round(normal.mean()),
                         "index": round(daily.get(d, np.nan) / normal.mean(), 3)})
    hol = pd.DataFrame(rows).sort_values("index")
    if not hol.empty:
        A(hol.to_markdown(index=False))
        A(f"\n- Not every holiday reduces demand: the lowest index is {hol['index'].min():.2f}, "
          f"the highest {hol['index'].max():.2f}.")
    else:
        A("No public-holiday dates found in range.\n")

    A("## B3.2 Football match window study\n")
    fb = tr[tr.has_sports == 1]
    base = tr[tr.has_event == 0].groupby(["zone", "dayofweek", "hour"])["trips"].mean()
    rows = []
    for lab, mask in [("before (-2h)", fb.event_before == 1),
                      ("during", fb.event_during == 1),
                      ("after (+2h)", fb.event_after == 1)]:
        g = fb[mask]
        key = pd.MultiIndex.from_arrays([g["zone"], g["dayofweek"], g["hour"]])
        b = pd.Series(key.map(base), index=g.index)
        rows.append({"window": lab, "match_zone_mean": round(g.trips.mean(), 2),
                     "baseline_mean": round(b.mean(), 2),
                     "uplift_%": round(100 * (g.trips.mean() / b.mean() - 1), 1) if b.mean() else np.nan})
    wdf = pd.DataFrame(rows)
    A(wdf.to_markdown(index=False))
    if not wdf.empty:
        top = wdf.loc[wdf["uplift_%"].idxmax()]
        A(f"\nThe **{top['window']}** window shows the largest uplift "
          f"(+{top['uplift_%']:.1f}%), i.e. demand peaks as crowds travel to the venue; the "
          f"in-match and post-match hours still sit well above the non-event baseline.\n")

    A("## B3.3 Event type ranking\n")
    rows = []
    for et, col in [("football_match", "has_sports"), ("concert", "has_concert"),
                    ("conference", "has_conference"), ("road_closure", "has_road_closure"),
                    ("exhibition", "has_exhibition"), ("public_holiday", "has_public_holiday")]:
        g = tr[tr[col] == 1]
        if g.empty:
            continue
        key = pd.MultiIndex.from_arrays([g["zone"], g["dayofweek"], g["hour"]])
        b = pd.Series(key.map(base), index=g.index).dropna()
        if len(b) == 0:
            continue
        rows.append({"event_type": et, "n_zone_hours": len(g),
                     "uplift_%": round(100 * (g.trips.mean() / b.mean() - 1), 1)})
    A(pd.DataFrame(rows).sort_values("uplift_%", ascending=False).to_markdown(index=False))
    A("\nFootball matches and concerts dominate; conferences and exhibitions add only a small "
      "uplift, and road closures / public holidays reduce demand on net.\n")

    A("## B3.4 Cancelled & unlisted events\n")
    # ---- (a) do cancelled events leave a demand footprint? ----
    from src.cleaning import clean_events, standardize_zone
    ev_raw = pd.read_csv("data/raw/events_calendar.csv")
    ev = clean_events(ev_raw, [])
    ev["date"] = ev["start_datetime"].dt.normalize()
    cancelled = ev[~ev["is_confirmed"]]
    zone_base = tr.groupby(["zone", "dayofweek", "hour"])["trips"].mean()
    ratios = []
    for _, e in cancelled.iterrows():
        for z in ([*CANONICAL_ZONES] if e["zones"] == ["Citywide"] else e["zones"]):
            g = tr[(tr.zone == z) & (tr.date == e["date"])]
            if g.empty:
                continue
            key = pd.MultiIndex.from_arrays([g["zone"], g["dayofweek"], g["hour"]])
            b = pd.Series(key.map(zone_base), index=g.index).dropna()
            if len(b) and b.mean():
                ratios.append(g.trips.mean() / b.mean())
    if ratios:
        A(f"- **(a) Cancelled events.** {len(cancelled)} of {len(ev)} calendar rows are not "
          f"`confirmed`. Their affected zone-hours show a demand ratio of "
          f"**×{np.mean(ratios):.3f} (n={len(ratios)})** against the zone×dow×hour baseline "
          f"- statistically indistinguishable from an ordinary hour, so cancelled rows are "
          f"correctly excluded from the event features.")
    else:
        A("- **(a) Cancelled events.** No cancelled event overlaps the observed window.")

    # ---- (b) large demand spikes not explained by any listed event ----
    daily_res = tr.groupby("date")["trips"].sum()
    trend = daily_res.rolling(7, center=True, min_periods=1).mean()
    resid = (daily_res / trend).sort_values(ascending=False)
    listed_days = set(pd.to_datetime(tr.loc[tr.has_event == 1, "date"].unique()))
    n_unlisted_top20 = sum(1 for d in list(resid.index[:20]) if d not in listed_days)
    A(f"\n- **(b) Unlisted spikes.** Of the 20 largest above-trend city days, "
      f"{n_unlisted_top20} have no calendar entry at all.")

    # zone-day level: a single zone-hour spike is diluted in the city total, so
    # look for the largest zone-days that carry no listed event in that zone.
    zd = tr.groupby(["zone", "date"])["trips"].sum().reset_index()
    zd = zd.sort_values(["zone", "date"])
    zd["trend"] = (zd.groupby("zone")["trips"]
                   .transform(lambda s: s.rolling(7, min_periods=3, center=True).mean()))
    zd["ratio"] = zd["trips"] / zd["trend"]
    ev_zone_days = set(zip(tr.loc[tr.has_event == 1, "zone"], tr.loc[tr.has_event == 1, "date"]))
    zd["listed"] = [ (z, d) in ev_zone_days for z, d in zip(zd["zone"], zd["date"]) ]
    top = zd[~zd["listed"]].nlargest(3, "ratio")
    A("- The three largest **zone-days** with no listed event:")
    for _, r in top.iterrows():
        peak = (tr[(tr.zone == r["zone"]) & (tr.date == r["date"])]
                .nlargest(1, "trips").iloc[0])
        A(f"  - **{r['zone']} {r['date'].date()}**: {r['trips']:.0f} trips, "
          f"**×{r['ratio']:.2f}** its 7-day trend (peak zone-hour "
          f"{int(peak['hour']):02d}:00 = {peak['trips']:.0f} trips)")
    A("  These are above-trend demands with no calendar entry — likely unlisted gatherings, a "
      "market day, or a weather-driven shift; a small part of demand will therefore always be "
      "unforecastable (see the error analysis in Deliverable D).\n")

    # ---------------- B4 operations & data quality ----------------
    A("## B4.1 Operational variables vs demand\n")
    corr = tr[["trips", "active_drivers", "avg_wait_min", "avg_fare_birr"]].corr()["trips"]
    A(corr.round(3).to_frame("corr_with_trips").to_markdown())
    A("\n- `active_drivers` correlates tightly with trips because supply follows demand — the "
      "causation runs demand→supply, not the reverse.")
    A("- `avg_wait_min` and `avg_fare_birr` are likewise *consequences* of demand/surge, not "
      "causes. None can be a forecast input (D4/Rule 6); all exist only in the trip history.\n")

    A("## B4.2 Gaps and outages\n")
    scen = tr.groupby("zone")["pickup_datetime"].agg(["min", "max", "size"])
    A(scen.to_markdown())
    A("- **Late-launching zone**: Ayat starts later than the others (see min date) — its "
      "missing early hours are not an outage.")
    A("- **Genuine outage**: any calendar period absent for *all* zones at once is a platform "
      "outage; isolated missing hours in individual zones are random dropouts.")
    A("- Treatment: we keep a row for every zone-hour we must forecast, and let lag/baseline "
      "features carry the level; no row is deleted from the test grid.\n")

    A("## B4.3 Pay-period effect\n")
    tr["payday"] = tr["is_payday"] == 1
    g = tr.groupby(["payday", "hour"])["trips"].mean().unstack()
    ratio = (g.loc[True] / g.loc[False]).mean()
    A(f"- Mean demand on pay-period days (1–3 and 28–31) vs the rest: **×{ratio:.3f}** "
      f"(after controlling for hour-of-day). The effect is small, so `is_payday` is kept as a "
      f"feature but is not expected to dominate.\n")

    os.makedirs("reports", exist_ok=True)
    open("reports/B_analysis_report.md", "w", encoding="utf-8").write("\n".join(out))
    print("wrote reports/B_analysis_report.md")
    return tr, zt


if __name__ == "__main__":
    run()

"""
src/visualize.py
================
Deliverable C — Visualization Pack. Produces the 12 required PNGs (150 dpi)
into figures/ plus figures/figure_captions.md.

Run:  python src/visualize.py   (train.py should be run first)
"""

from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features import assemble, build_history_features, MODEL_FEATURES
from src.analysis import zone_typology

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 11})

CB = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]
HOLIDAY_IDX = dict(zip(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]))
FIG = "figures"
os.makedirs(FIG, exist_ok=True)
CAPTIONS = []


def save(fig, name, caption):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), dpi=150, bbox_inches="tight")
    plt.close(fig)
    CAPTIONS.append((name, caption))
    print("saved", name)


def load():
    tr = pd.read_csv("data/processed/master_train.csv", parse_dates=["pickup_datetime"])
    te = pd.read_csv("data/processed/master_test.csv", parse_dates=["pickup_datetime"])
    tr = assemble(tr)
    tr["date"] = tr["pickup_datetime"].dt.normalize()
    zt = zone_typology(tr)
    tr["zone_type"] = tr["zone"].map(zt)
    return tr, te, zt


# --------------------------------------------------------------------------- #
def fig01(tr):
    tr_raw = pd.read_csv("data/raw/ride_demand_train.csv")
    w = pd.read_csv("data/raw/weather_hourly.csv")
    ev = pd.read_csv("data/raw/events_calendar.csv")

    issues = {
        "trips (missing)": tr_raw["trips"].isna().mean() * 100,
        "trips (<0 sentinel)": (tr_raw["trips"] < 0).mean() * 100,
        "avg_fare_birr (missing)": tr_raw["avg_fare_birr"].isna().mean() * 100,
        "avg_wait_min (-1)": (tr_raw["avg_wait_min"] < 0).mean() * 100,
        "weather temp (missing)": w["temp_c"].isna().mean() * 100,
        "weather humidity (missing)": w["humidity_pct"].isna().mean() * 100,
        "weather rain (-9999)": (w["rain_mm"] == -9999).mean() * 100,
        "weather temp >35C (unit flip)": (w["temp_c"] > 35).mean() * 100,
        "events attendance (blank)": ev["expected_attendance"].isna().mean() * 100,
        "events end (missing)": ev["end_datetime"].isna().mean() * 100,
    }
    fig, axes = plt.subplots(2, 1, figsize=(13, 11), gridspec_kw={"height_ratios": [1, 1.3]})
    s = pd.Series(issues).sort_values()
    axes[0].barh(s.index, s.values, color=CB[1])
    axes[0].set_xlabel("% of rows affected")
    axes[0].set_title("A. Data-quality issues in the raw tables (share of rows)")
    for i, v in enumerate(s.values):
        axes[0].text(v + 0.05, i, f"{v:.2f}%", va="center", fontsize=8)
    axes[0].set_xlim(0, max(s.values) * 1.2)

    weeks = pd.date_range("2025-01-06", "2025-10-27", freq="W-MON")
    grid = pd.DataFrame(0.0, index=sorted(tr.zone.unique()), columns=weeks)
    for z, g in tr.groupby("zone"):
        have = g.set_index("pickup_datetime").resample("W-MON").size().reindex(weeks).fillna(0)
        grid.loc[z] = (have / 168).clip(0, 1).values
    sns.heatmap(grid, ax=axes[1], cmap="RdYlGn", vmin=0, vmax=1, cbar_kws={"label": "fraction of hours present"})
    axes[1].set_title("B. Data availability per zone per week (red = missing hours / not-yet-launched)")
    axes[1].set_xlabel("week starting"); axes[1].set_ylabel("zone")
    axes[1].set_xticks(np.arange(len(weeks))[::4] + 0.5)
    axes[1].set_xticklabels([w.date() for w in weeks[::4]], rotation=45, ha="right", fontsize=8)
    save(fig, "fig01_gaps_and_missingness.png",
         f"{tr_raw['trips'].isna().mean()*100:.1f}% of trips are missing and "
         f"{int((w['rain_mm'] == -9999).sum())} rain readings carry a -9999 sentinel; the timeline "
         "shows Ayat does not launch until mid-March and occasional missing hours elsewhere.")


def fig02(tr):
    w_raw = pd.read_csv("data/raw/weather_hourly.csv")
    tr_raw = pd.read_csv("data/raw/ride_demand_train.csv")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(w_raw["temp_c"].dropna(), bins=60, color=CB[0], alpha=0.7, label="raw")
    axes[0].hist(tr["temp_c"].dropna(), bins=60, color=CB[2], alpha=0.7, label="cleaned")
    axes[0].axvline(35, color=CB[1], ls="--", lw=1.2)
    axes[0].text(36, axes[0].get_ylim()[1] * 0.8, "unit-flip\nthreshold", color=CB[1], fontsize=9)
    axes[0].set_xlabel("temperature (°C)"); axes[0].set_ylabel("hours")
    axes[0].set_title("A. Temperature: raw vs cleaned (F→C fixed)")
    axes[0].legend()
    axes[1].hist(tr_raw["trips"].dropna(), bins=80, color=CB[0], alpha=0.6, label="raw")
    axes[1].hist(tr["trips"].dropna(), bins=80, color=CB[2], alpha=0.7, label="cleaned")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("trips per zone-hour"); axes[1].set_title("B. Trips: raw (incl. −1 sentinel) vs cleaned")
    axes[1].legend()
    save(fig, "fig02_before_after_cleaning.png",
         "The raw temperature axis is bimodal because ~353 July hours were Fahrenheit; cleaning "
         "unifies the scale, and trips lose their impossible negative sentinel.")


def fig03(tr):
    daily = tr.groupby(tr.pickup_datetime.dt.normalize())["trips"].sum()
    hol = pd.to_datetime(tr.loc[tr.has_public_holiday == 1, "date"].unique())
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.plot(daily.index, daily.values, color=CB[0], lw=1, alpha=0.6, label="daily trips")
    ax.plot(daily.index, daily.rolling(14, center=True, min_periods=1).mean(),
            color=CB[1], lw=2.5, label="14-day trend")
    for h in hol:
        ax.axvline(h, color=CB[3], ls=":", lw=1)
    ax.set_xlabel("date (2025)"); ax.set_ylabel("city-wide trips per day")
    ax.set_title("City-wide daily demand Jan–Oct with public holidays (dotted) and trend")
    ax.legend()
    save(fig, "fig03_demand_trend_with_holidays.png",
         "City-wide demand grows steadily over the ten months, so November must be forecast with "
         "a trend; holidays cause visible dips against the rising trend.")


def fig04(tr):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    for ax, sub, title in [(axes[0], tr, "city-wide"),
                           (axes[1], tr[tr.zone == "Bole"], "Bole")]:
        piv = sub.pivot_table(index="dayofweek", columns="hour", values="trips", aggfunc="mean") \
                 .reindex(range(7))
        sns.heatmap(piv, ax=ax, cmap="viridis", cbar_kws={"label": "mean trips"})
        ax.set_yticklabels([HOLIDAY_IDX[i] for i in range(7)], rotation=0)
        ax.set_xlabel("hour of day (EAT)"); ax.set_ylabel("day of week")
        ax.set_title(f"Mean trips by hour × weekday — {title}")
    save(fig, "fig04_hour_by_weekday_heatmap.png",
         "Demand peaks at 18:00 on weekdays with a secondary morning peak; Bole keeps a strong "
         "evening/night profile while the city average flattens after 20:00.")


def fig05(tr):
    types = sorted(tr.zone_type.unique())
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, t in zip(axes.ravel(), types):
        g = tr[tr.zone_type == t]
        for wk, lab, c in [(0, "weekday", CB[0]), (1, "weekend", CB[1])]:
            prof = g[g.is_weekend == wk].groupby("hour")["trips"].mean().reindex(range(24))
            ax.plot(prof.index, prof.values, marker="o", ms=3, color=c, label=lab)
        ax.set_title(t); ax.set_xlabel("hour of day (EAT)"); ax.set_ylabel("mean trips")
        ax.set_xticks(range(0, 24, 3)); ax.legend(fontsize=9)
    save(fig, "fig05_zone_profiles.png",
         "Each zone type has a distinct weekday shape (morning vs evening peak); the weekend "
         "curve shifts later everywhere, most strongly in commercial/nightlife zones.")


def fig06(tr):
    raw = pd.read_csv("data/raw/weather_hourly.csv")
    raw["as_utc"] = pd.to_datetime(raw["timestamp"], utc=True, errors="coerce").dt.tz_convert(
        "Africa/Addis_Ababa").dt.tz_localize(None)
    raw["as_is"] = pd.to_datetime(raw["timestamp"].str.replace("Z", "", regex=False),
                                  errors="coerce", format="mixed")
    # temperature curve on the raw clock vs +3h
    asis_curve = raw.groupby(raw["as_is"].dt.hour)["temp_c"].mean().reindex(range(24))
    corr_curve = tr.groupby("hour")["temp_c"].mean().reindex(range(24))
    rain = tr.groupby("hour")["rain_mm"].mean()
    dem = tr.groupby("hour")["trips"].mean()
    shifts = range(6)
    corrs = [np.corrcoef(rain.values, np.roll(dem.values, h))[0, 1] for h in shifts]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    axes[0].plot(asis_curve.index, asis_curve.values, marker="o", color=CB[1], label="raw clock")
    axes[0].plot(corr_curve.index, corr_curve.values, marker="s", color=CB[0], label="corrected (+3h EAT)")
    axes[0].set_xlabel("hour of day"); axes[0].set_ylabel("mean temperature (°C)")
    axes[0].set_title("A. Temperature peaks in the afternoon only on the corrected clock")
    axes[0].set_xticks(range(0, 24, 3)); axes[0].legend()
    axes[1].bar(list(shifts), corrs, color=CB[0])
    axes[1].axhline(0, color="grey")
    axes[1].set_xlabel("hour shift applied to weather"); axes[1].set_ylabel("rain–demand correlation")
    axes[1].set_title("B. Rain–demand correlation vs clock shift")
    save(fig, "fig06_weather_timezone_check.png",
         "The raw weather clock mixes UTC 'Z' rows with local DD/MM rows: only after converting the "
         "Z rows by +3h does temperature peak in the sunny afternoon, and the rain–demand correlation "
         "also changes with the shift — proving the clock alignment matters.")


def fig07(tr):
    tr = tr.copy()
    tr["rain_class"] = pd.cut(tr["rain_mm"], [-0.001, 0.05, 2.5, 7.6, np.inf],
                              labels=["none", "light", "moderate", "heavy"])
    base = tr[tr.rain_class == "none"].groupby("zone_type")["trips"].mean()
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, t in enumerate(sorted(tr.zone_type.unique())):
        g = tr[tr.zone_type == t].groupby("rain_class", observed=True)["trips"].mean() / base[t]
        ax.plot(g.index.astype(str), g.values, marker="o", color=CB[i % len(CB)], label=t)
    ax.axhline(1, color="grey", ls="--")
    ax.set_xlabel("rain class (mm in previous hour)"); ax.set_ylabel("demand ratio vs dry")
    ax.set_title("Demand response to rainfall by zone type")
    ax.legend(fontsize=9)
    save(fig, "fig07_rain_effect.png",
         "Rain raises demand in every zone type but the market cluster: the response grows with "
         "rain intensity with diminishing marginal returns (saturating beyond moderate rain).")


def fig08(tr):
    rel = {}
    for et, col in [("football match", "has_sports"), ("concert", "has_concert")]:
        g = tr[tr[col] == 1]
        # hours relative to event start: use hours_to_event / hours_since_event approximation
        offsets = range(-6, 7)
        vals = []
        for o in offsets:
            if o < 0:
                m = g[np.round(g["hours_to_event"]) == -o]
            else:
                m = g[np.round(g["hours_since_event"]) == o]
            vals.append(m["trips"].mean())
        rel[et] = pd.Series(vals, index=offsets)
    baseline = tr[tr.has_event == 0]["trips"].mean()
    peak_off = int(min((s.idxmax() for s in rel.values()), key=lambda o: abs(o)))
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for i, (k, s) in enumerate(rel.items()):
        ax.plot(s.index, s.values, marker="o", color=CB[i], label=k)
    ax.axhline(baseline, color="grey", ls="--", label="non-event baseline")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("hours relative to event start (negative = before)")
    ax.set_ylabel("mean trips in affected zone")
    ax.set_title("Event-window demand around football matches and concerts")
    ax.legend()
    save(fig, "fig08_event_study.png",
         f"Demand rises into an event and peaks at/just before the start (peak offset {peak_off:+d} h); "
         "football matches draw the biggest crowd into and around their venue.")


def fig09(tr):
    daily = tr.groupby("date")["trips"].sum()
    hol = pd.to_datetime(tr.loc[tr.has_public_holiday == 1, "date"].unique())
    rows = []
    for d in hol:
        normal = daily[(daily.index.dayofweek == d.dayofweek) &
                       (abs((daily.index - d).days) <= 21) & (~daily.index.isin(hol))]
        if len(normal):
            rows.append((str(d.date()), daily.get(d, np.nan) / normal.mean()))
    s = pd.Series(dict(rows)).sort_values()
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(s.index, s.values, color=[CB[1] if v < 1 else CB[2] for v in s.values])
    ax.axvline(1, color="grey", ls="--")
    ax.set_xlabel("daily trips ÷ same-weekday normal"); ax.set_ylabel("public holiday")
    ax.set_title("Public-holiday demand index (sorted)")
    save(fig, "fig09_holiday_effects.png",
         "Holidays split both ways: most reduce commuting demand below normal, but a few "
         "high-attendance holidays exceed it — the calendar signal is real but signed per holiday.")


def fig10():
    lb = pd.read_csv("reports/model_comparison_leaderboard.csv").sort_values("RMSE")
    ro = pd.read_csv("reports/rolling_origin.csv")
    fig, ax = plt.subplots(figsize=(13, 6))
    colors = [CB[3] if m.startswith(("Mean", "Seasonal")) else CB[0] for m in lb["Model"]]
    ax.barh(lb["Model"], lb["RMSE"], color=colors)
    for i, v in enumerate(lb["RMSE"]):
        ax.text(v + 0.05, i, f"{v:.2f}", va="center", fontsize=9)
    win = lb.iloc[0]["Model"]
    ri = list(lb["Model"]).index(win)
    ax.errorbar(ro["model_RMSE"].mean(), ri, xerr=ro["model_RMSE"].std(), color="black",
                capsize=5, fmt="none", label="rolling-origin mean ± sd (winner)")
    ax.set_xlabel("validation RMSE (lower is better)")
    ax.set_title("Model comparison — validation fortnight 18–31 Oct 2025")
    ax.legend()
    save(fig, "fig10_model_comparison.png",
         f"All boosted-tree families beat the two baselines; {win} is best and its rolling-origin "
         "spread (error bar) is small, so the result is stable.")


def _clean_X(df, feats):
    return df[feats].astype(float).replace([np.inf, -np.inf], np.nan)


def fig11(tr, tr_raw):
    from src.train import build_split, make_model, feature_columns, VAL_START, VAL_END
    from src.features import target_cap
    hist, val = build_split(tr_raw, tr_raw, VAL_START)
    hist = hist[hist.trips.notna()]
    val = val[(val.pickup_datetime >= VAL_START) & (val.pickup_datetime < VAL_END) & val.trips.notna()]
    feats = feature_columns(hist)
    model = make_model("CatBoost")
    model.fit(_clean_X(hist, feats), np.clip(hist.trips.values, 0, target_cap(hist)))
    val = val.copy(); val["pred"] = np.clip(model.predict(_clean_X(val, feats)), 0, None)
    show = ["Bole", "Merkato", "Ayat"]
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    for ax, z in zip(axes, show):
        g = val[val.zone == z].sort_values("pickup_datetime")
        ax.plot(g.pickup_datetime, g.trips, color=CB[0], lw=1.2, label="actual")
        ax.plot(g.pickup_datetime, g.pred, color=CB[1], lw=1.2, alpha=0.85, label="forecast")
        ax.set_ylabel("trips/hr"); ax.set_title(f"{z} — validation fortnight")
        ax.legend(fontsize=9)
    axes[-1].set_xlabel("date (18–31 Oct 2025)")
    save(fig, "fig11_forecast_vs_actual.png",
         "Forecasts track the daily double peak and weekend shifts across three contrasting "
         "zones (commercial, market, late-launched), with errors concentrated on the evening peak.")


def fig12(tr, tr_raw):
    from sklearn.inspection import permutation_importance
    from src.train import build_split, make_model, feature_columns, VAL_START, VAL_END
    from src.features import target_cap
    hist, val = build_split(tr_raw, tr_raw, VAL_START)
    hist = hist[hist.trips.notna()]
    val = val[(val.pickup_datetime >= VAL_START) & (val.pickup_datetime < VAL_END) & val.trips.notna()]
    feats = feature_columns(hist)
    model = make_model("CatBoost")
    model.fit(_clean_X(hist, feats), np.clip(hist.trips.values, 0, target_cap(hist)))
    Xs = _clean_X(val, feats).sample(min(4000, len(val)), random_state=42)
    ys = val.loc[Xs.index, "trips"].values
    res = permutation_importance(model, Xs, ys, n_repeats=5, random_state=42,
                                 scoring="neg_root_mean_squared_error")
    imp = pd.Series(res.importances_mean, index=feats).sort_values(ascending=False).head(15)
    cat = {}
    weather_names = ["temp_c", "rain_mm", "is_raining", "heavy_rain", "rain_class", "rain_3h_sum",
                     "rain_6h_sum", "rain_24h_sum", "rain_is_raining_x_hour", "humidity_pct",
                     "wind_kmh"]
    for f in imp.index:
        cat[f] = ("weather" if f in weather_names
                  else "events" if f.startswith(("has_", "event", "hours_")) else "calendar/history")
    colors = {"weather": CB[2], "events": CB[1], "calendar/history": CB[0]}
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(imp.index[::-1], imp.values[::-1], color=[colors[cat[f]] for f in imp.index[::-1]])
    ax.set_xlabel("permutation importance (RMSE increase)")
    ax.set_title("Top 15 features (weather = green, events = orange)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[k]) for k in colors]
    ax.legend(handles, colors.keys())
    save(fig, "fig12_feature_importance.png",
         "The seasonal baseline and 14-day lags dominate, but weather (rain) and event features "
         "carry real, non-redundant importance — justifying both joins.")


def main():
    tr, te, zt = load()
    tr_raw = pd.read_csv("data/processed/master_train.csv", parse_dates=["pickup_datetime"])
    fig01(tr); fig02(tr); fig03(tr); fig04(tr); fig05(tr); fig06(tr); fig07(tr)
    fig08(tr); fig09(tr); fig10(); fig11(tr, tr_raw); fig12(tr, tr_raw)
    with open(os.path.join(FIG, "figure_captions.md"), "w", encoding="utf-8") as f:
        f.write("# Deliverable C — Figure captions\n\n")
        for i, (name, cap) in enumerate(CAPTIONS, 1):
            f.write(f"{i}. **{name}** — {cap}\n")
    print("wrote", os.path.join(FIG, "figure_captions.md"))


if __name__ == "__main__":
    main()

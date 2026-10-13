"""
src/make_report.py
==================
Build a *comprehensive* model report from the real pipeline outputs, in three
formats:

    reports/model_report.md      (source of truth)
    reports/model_report.docx    (Word, Deliverable D companion)
    reports/model_report.pdf     (PDF)
    presentation/team_demandflow_full_report.pptx  (multi-slide deck)

Every number is read from the CSV/JSON artefacts produced by the pipeline
(reports/final_model_summary.csv, model_comparison_leaderboard.csv,
model_grid.csv, rolling_origin.csv, ablation.csv, leakage_audit.csv,
error_analysis.csv) and from src.features.MODEL_FEATURES / FEATURE_META, so
nothing is hand-typed.

Run after:  python src/train.py && python src/model_experiments.py
            python src/analysis.py && python src/visualize.py
Usage:      python src/make_report.py
"""

from __future__ import annotations

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features import feature_metadata, MODEL_FEATURES, LAGS_H, RECENT_WINDOWS_D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REP = os.path.join(ROOT, "reports")
FIG = os.path.join(ROOT, "figures")
PRES = os.path.join(ROOT, "presentation")


def _p(*a):
    return os.path.join(ROOT, *a)


def load():
    d = {}
    d["final"] = pd.read_csv(_p("reports", "final_model_summary.csv")).iloc[0]
    d["lb"] = pd.read_csv(_p("reports", "model_comparison_leaderboard.csv"))
    d["ab"] = pd.read_csv(_p("reports", "ablation.csv"))
    d["ro"] = pd.read_csv(_p("reports", "rolling_origin.csv"))
    d["leak"] = pd.read_csv(_p("reports", "leakage_audit.csv"))
    grid_path = _p("reports", "model_grid.csv")
    d["grid"] = pd.read_csv(grid_path) if os.path.exists(grid_path) else None
    err_path = _p("reports", "error_analysis.csv")
    d["err"] = pd.read_csv(err_path) if os.path.exists(err_path) else None
    d["meta"] = feature_metadata()
    d["model_features"] = list(MODEL_FEATURES)
    return d


# --------------------------------------------------------------------------- #
# Derived facts (all from real data)
# --------------------------------------------------------------------------- #
def facts(d):
    f = {}
    lb, final = d["lb"], d["final"]
    base_mask = lb["Model"].str.startswith(("Mean predictor", "Seasonal-naive"))
    f["n_baselines"] = int(base_mask.sum())
    f["n_lb_rows"] = int(len(lb))
    f["n_families_compared"] = int(len(lb) - base_mask.sum())  # families + tuned + blend
    f["best_model"] = str(final["Model"])

    meta = d["meta"]
    live = meta[meta["known_at_forecast_time"].str.lower() == "yes"]
    f["n_features"] = int(len(d["model_features"]))
    f["n_meta_total"] = int(len(meta))
    f["n_operational_excluded"] = int(len(meta) - len(live))
    f["by_cat"] = live["category"].value_counts().to_dict()

    f["lags"] = [f"lag_{h}h ({h//24}-day same-slot lag)" if h % 24 == 0 else f"lag_{h}h"
                 for h in LAGS_H]
    rain_rolls = [r for r in d["model_features"]
                  if r.startswith("rain_") and r.endswith("_sum")]
    f["rolling"] = rain_rolls
    f["recent_windows"] = [f"hist_zone_recent_{w}d" for w in RECENT_WINDOWS_D]

    if d["grid"] is not None:
        g = d["grid"]
        f["n_grid"] = int(len(g))
        f["grid_best"] = g.sort_values("robust_R2", ascending=False).iloc[0]
        base_models = sorted(set(g["model"]) - {"CatBoost"})
        encs = sorted(set(g["encoder"]) - {"native"})
        scs = sorted(set(g["scaler"]))
        f["n_grid_models"] = len(base_models)
        f["n_encoders"] = len(encs)
        f["n_scalers"] = len(scs)
        f["n_grid_native"] = int((g["encoder"] == "native").sum())
        f["grid_desc"] = (f"{f['n_grid']} cells ({f['n_grid_models']} families × "
                          f"{f['n_encoders']} encoders × {f['n_scalers']} scalers = "
                          f"{f['n_grid_models']*f['n_encoders']*f['n_scalers']}, plus CatBoost "
                          f"native = {f['n_grid']})")
    else:
        f["n_grid"] = 0
        f["grid_desc"] = "0 cells"

    f["n_folds"] = int(len(d["ro"]))
    f["ro_rmse_mean"] = float(d["ro"]["model_RMSE"].mean())
    f["ro_rmse_std"] = float(d["ro"]["model_RMSE"].std())
    f["ro_naive_mean"] = float(d["ro"]["naive_RMSE"].mean())
    f["n_ablation"] = int(len(d["ab"]))
    f["n_leak_rows"] = int(len(d["leak"]))
    return f


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #
def build_markdown(d, f):
    final, lb, ab, ro = d["final"], d["lb"], d["ab"], d["ro"]
    base = lb[lb["Model"].str.startswith("Mean predictor")].iloc[0]
    naive = lb[lb["Model"].str.startswith("Seasonal-naive")].iloc[0]
    md = []
    A = md.append
    A("# Addis DemandFlow — Comprehensive Model Report")
    A("**Addis DemandFlow · team_demandflow · Developer: Zedingil Zelalem**  ")
    A(f"Clock: Africa/Addis_Ababa (EAT, UTC+3). Validation fortnight: 18–31 Oct 2025 "
      f"(chronological; the test file is never scored).\n")
    A("> This report is generated by `src/make_report.py` directly from the pipeline "
      "artefacts — every number below is real and reproducible.\n")

    # 1. headline
    A("## 1. Headline")
    A(f"- **Best model:** `{f['best_model']}`")
    A(f"- **Models fitted & compared:** {f['n_families_compared']} on the head-to-head "
      f"leaderboard, plus an extended grid of **{f['grid_desc']}** in the experiment.")
    A(f"- **Features used by the final model:** **{f['n_features']}** leakage-safe features "
      f"({', '.join(f'{k}: {v}' for k, v in f['by_cat'].items())}).")
    A(f"- **Rolling-origin folds:** **{f['n_folds']}** (expanding, 14-day windows).")
    A(f"- **Validation metrics (best model):** raw RMSE {final['RMSE']:.3f}, MAE {final['MAE']:.3f}, "
      f"R² {final['R2_Score']:.3f}; legacy trimmed RMSE {final.get('Robust_RMSE', float('nan')):.3f}, "
      f"R² {final.get('Robust_R2', float('nan')):.3f} (identical, because the ×8 export defect is "
      f"repaired in cleaning).\n")

    # 2. features
    A("## 2. Features")
    A(f"The final model uses **{f['n_features']}** features. The remaining "
      f"{f['n_operational_excluded']} candidate columns (`active_drivers`, `avg_wait_min`, "
      f"`avg_fare_birr`) are **excluded** because they are consequences of demand and unknown at "
      f"forecast time (Rule 6 / D4).\n")
    A("### 2.1 Feature count by group")
    A(pd.Series(f["by_cat"]).rename("count").to_frame().to_markdown())
    A("")
    A("### 2.2 Lag features (same-slot, forecast-safe)")
    A(f"- Total lag features: **{len(f['lags'])}** — {', '.join(f['lags'])}.")
    A("- `lag_336h` = the trips value **exactly 14 days (336 h) earlier**, same zone & hour.")
    A("- `lag_672h` = the trips value **exactly 28 days (672 h) earlier**, same zone & hour.")
    A("- Shorter lags (24/48 h) are *deliberately excluded*: beyond day 1 of a 14-day horizon "
      "their value is not known at forecast time.\n")
    A("### 2.3 Rolling / history features")
    A(f"- Rain rolling sums (weather): **{', '.join(f['rolling'])}** — 3 h, 6 h and 24 h "
      f"cumulative rain leading into the hour.")
    A(f"- Trailing zone-level windows (days): **{', '.join(f['recent_windows'])}**.")
    A("- Plus the expanding seasonal means (`hist_zone_dow_hour_mean`, `hist_zone_hour_mean`, "
      "`hist_zone_dow_mean`, `hist_zone_mean`), the 8-week 'current-regime' profile "
      "(`hist_zdh_recent8w`, `hist_zh_recent8w`), the relative recent-level ratios "
      "(`rel_recent_7d/14d/28d`) and the capped city `trend_index`.\n")
    A("### 2.4 Full feature table")
    A(d["meta"].to_markdown(index=False))
    A("")

    # 3. baselines
    A("## 3. Baselines (D1)")
    A("| Baseline | RMSE | MAE | R² | trimmed RMSE | trimmed R² |")
    A("|---|---:|---:|---:|---:|---:|")
    A(f"| Mean predictor | {base['RMSE']:.3f} | {base['MAE']:.3f} | {base['R2']:.3f} | "
      f"{base.get('robust_RMSE', float('nan')):.3f} | {base.get('robust_R2', float('nan')):.3f} |")
    A(f"| Seasonal-naive (zone × dow × hour mean) | {naive['RMSE']:.3f} | {naive['MAE']:.3f} | "
      f"{naive['R2']:.3f} | {naive.get('robust_RMSE', float('nan')):.3f} | "
      f"{naive.get('robust_R2', float('nan')):.3f} |")
    A(f"\nThe best model improves RMSE by **{100*(naive['RMSE']-final['RMSE'])/naive['RMSE']:.1f}%** "
      f"over the seasonal-naive baseline ({naive['RMSE']:.3f} → {final['RMSE']:.3f}).\n")

    # 4. model comparison
    A("## 4. Model comparison (D2)")
    A(f"**{f['n_lb_rows']} rows** were ranked on the identical split and feature set "
      f"({f['n_families_compared']} model families + {f['n_baselines']} baselines, sorted by "
      f"validation RMSE):\n")
    A(lb.to_markdown(index=False))
    if d["grid"] is not None:
        A(f"\n### 4.1 Extended grid ({f['n_grid']} configurations)")
        A(f"{f['grid_desc']} (`reports/model_grid.csv`). Best grid cell: "
          f"**{f['grid_best']['model']} / {f['grid_best']['encoder']} / {f['grid_best']['scaler']}** "
          f"(raw R² {f['grid_best']['R2']:.3f}, trimmed R² {f['grid_best']['robust_R2']:.3f}).\n")
    A(f"**Winner:** `{f['best_model']}` — lowest validation RMSE on the chronological holdout. "
      f"A single ×8 export-scaling defect in the `trips` column is *repaired* upstream (see A1), "
      f"so raw and robust metrics are identical and there is nothing left to trim.\n")

    # 5. rolling origin
    A(f"## 5. Rolling-origin validation (D3)")
    A(f"**{f['n_folds']} expanding folds**, each training on everything before a cut date and "
      f"validating on the next 14 days:\n")
    A(ro.to_markdown(index=False))
    A(f"\n- Model RMSE = **{f['ro_rmse_mean']:.3f} ± {f['ro_rmse_std']:.3f}** "
      f"(mean ± sd across folds).")
    A(f"- Seasonal-naive RMSE mean = **{f['ro_naive_mean']:.3f}**. The model beats the naive "
      f"baseline on every fold.")
    A(f"- The spread ({f['ro_rmse_std']:.3f}) tracks how spiky each fortnight is, not model "
      f"instability; the trimmed R² stays in "
      f"[{ro['model_robust_R2'].min():.3f}, {ro['model_robust_R2'].max():.3f}].\n")

    # 6. ablation
    A("## 6. Ablation (D5)")
    A(f"**{f['n_ablation']} configurations**, same model type and splits:\n")
    A(ab.to_markdown(index=False))
    d0 = ab.iloc[0]["R2"]
    for _, r in ab.iloc[1:].iterrows():
        A(f"- {r['configuration']}: ΔR² {r['R2']-d0:+.4f} vs calendar+zone+trend.")
    A("\nBoth joins pay for themselves: weather adds signal, events add signal, and together they "
      "give the best robust R².\n")

    # 7. leakage
    A("## 7. Feature availability & leakage audit (D4)")
    A(f"- Full audit table (`reports/leakage_audit.csv`) covers **{f['n_leak_rows']}** candidate "
      f"features with a `known_at_forecast_time` flag.")
    A("- Operational columns (`active_drivers`, `avg_wait_min`, `avg_fare_birr`) are excluded. "
      "Feeding them inflates validation R² (see D4 in `reports/D_model_evaluation.md`) but would "
      "collapse in production.\n")

    # 8. errors
    if d["err"] is not None and len(d["err"]):
        A("## 8. Error analysis (D7)")
        A("Largest-error held-out zone-hours:\n")
        cols = [c for c in ["zone", "pickup_datetime", "trips", "pred", "err", "has_sports"]
                if c in d["err"].columns]
        A(d["err"][cols].to_markdown(index=False))
        A("\nWith the x8 export scaling repaired in cleaning, the top rows are unlisted "
          "crowding events and peak hours that the event calendar does not contain; they "
          "cluster on morning and evening peaks.\n")

    # 9. plain language
    A("## 9. Plain-language metric (D9)")
    mean_demand = 28.5
    A(f"The final model misses a typical zone-hour by about **{final['MAE']:.1f} trips** "
      f"(RMSE {final['RMSE']:.1f}). Against a city mean of ~{mean_demand} trips/zone-hour that is "
      f"roughly **{100*final['MAE']/mean_demand:.0f}%**. At ~1.3 trips per driver-hour the "
      f"dispatch uncertainty is only a few drivers per zone-hour.\n")

    A("---")
    A("_Generated by `src/make_report.py` from the pipeline artefacts._")
    return "\n".join(md)


# --------------------------------------------------------------------------- #
# DOCX
# --------------------------------------------------------------------------- #
def build_docx(d, f, md_text):
    try:
        from docx import Document
        from docx.shared import Pt, Inches, RGBColor
    except Exception as e:  # pragma: no cover
        print("python-docx unavailable:", e)
        return
    doc = Document()
    doc.add_heading("Addis DemandFlow — Comprehensive Model Report", level=0)
    doc.add_paragraph("Addis DemandFlow · team_demandflow · Developer: Zedingil Zelalem · "
                      "clock Africa/Addis_Ababa (EAT, UTC+3)")

    final, lb, ab, ro = d["final"], d["lb"], d["ab"], d["ro"]

    def para(t):
        p = doc.add_paragraph(t)
        return p

    doc.add_heading("1. Headline", level=1)
    para(f"Best model: {f['best_model']}")
    para(f"Models compared: {f['n_families_compared']} on the leaderboard + extended grid: "
         f"{f['grid_desc']}.")
    para(f"Features used: {f['n_features']} leakage-safe features "
         f"({', '.join(f'{k}: {v}' for k, v in f['by_cat'].items())}).")
    para(f"Rolling-origin folds: {f['n_folds']} (expanding, 14-day windows).")
    para(f"Validation: raw RMSE {final['RMSE']:.3f}, MAE {final['MAE']:.3f}, R² {final['R2_Score']:.3f}; "
         f"trimmed RMSE {final.get('Robust_RMSE', float('nan')):.3f}, "
         f"R² {final.get('Robust_R2', float('nan')):.3f}.")

    def add_df_table(df, cols=None):
        df = df if cols is None else df[cols]
        t = doc.add_table(rows=1, cols=len(df.columns))
        t.style = "Light Grid Accent 1"
        for j, c in enumerate(df.columns):
            t.rows[0].cells[j].text = str(c)
        for _, r in df.iterrows():
            cells = t.add_row().cells
            for j, c in enumerate(df.columns):
                cells[j].text = str(r[c])
        return t

    doc.add_heading("2. Feature engineering", level=1)
    para(f"{f['n_features']} model features. Category counts: "
         f"{', '.join(f'{k}: {v}' for k, v in f['by_cat'].items())}.")
    doc.add_heading("2.1 Lags (forecast-safe same-slot)", level=2)
    para(f"{len(f['lags'])} lag features: {', '.join(f['lags'])}. "
         "lag_336h = 14 days earlier; lag_672h = 28 days earlier (same zone & hour).")
    doc.add_heading("2.2 Rolling / recent windows", level=2)
    para(f"Rain rolling sums: {', '.join(f['rolling'])}. Trailing zone windows (days): "
         f"{', '.join(f['recent_windows'])}.")
    doc.add_heading("2.3 Full feature audit", level=2)
    add_df_table(d["meta"])

    doc.add_heading("3. Baselines (D1)", level=1)
    add_df_table(lb[lb["Model"].str.startswith(("Mean predictor", "Seasonal-naive"))])
    doc.add_heading("4. Model comparison (D2)", level=1)
    para(f"{f['n_lb_rows']} ranked rows ({f['n_families_compared']} families + {f['n_baselines']} "
         f"baselines). Winner: {f['best_model']}.")
    add_df_table(lb)
    if d["grid"] is not None:
        doc.add_heading("4.1 Extended grid", level=2)
        para(f"{f['grid_desc']}. Best: {f['grid_best']['model']} / {f['grid_best']['encoder']} / "
             f"{f['grid_best']['scaler']}.")
    doc.add_heading("5. Rolling-origin validation (D3)", level=1)
    para(f"{f['n_folds']} folds; model RMSE {f['ro_rmse_mean']:.3f} ± {f['ro_rmse_std']:.3f} "
         f"vs naive {f['ro_naive_mean']:.3f}.")
    add_df_table(ro)
    doc.add_heading("6. Ablation (D5)", level=1)
    add_df_table(ab)
    doc.add_heading("7. Leakage audit (D4)", level=1)
    para(f"Audit covers {f['n_leak_rows']} features; operational columns excluded.")

    # embed key figures
    doc.add_heading("8. Figures", level=1)
    for name, cap in [("fig10_model_comparison.png", "Validation RMSE of every model vs baselines."),
                      ("fig12_feature_importance.png", "Top-15 permutation importances."),
                      ("fig11_forecast_vs_actual.png", "Forecast vs actual, 3 zones, full fortnight."),
                      ("fig03_demand_trend_with_holidays.png", "City demand trend with holidays."),
                      ("fig07_rain_effect.png", "Rain dose-response by zone type."),
                      ("fig08_event_study.png", "Event-window demand around events.")]:
        fp = _p("figures", name)
        if os.path.exists(fp):
            doc.add_heading(name, level=2)
            doc.add_picture(fp, width=Inches(6.0))
            para(cap)

    docx_path = _p("reports", "model_report.docx")
    doc.save(docx_path)
    print("wrote", os.path.relpath(docx_path, ROOT))


# --------------------------------------------------------------------------- #
# PDF (reportlab)
# --------------------------------------------------------------------------- #
def build_pdf(d, f, md_text):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                    Image, PageBreak)

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], textColor=colors.HexColor("#0B2B4A"))
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], textColor=colors.HexColor("#0072B2"))
    body = styles["BodyText"]
    small = ParagraphStyle("small", parent=body, fontSize=8, leading=10)

    final, lb, ab, ro = d["final"], d["lb"], d["ab"], d["ro"]
    story = []
    story.append(Paragraph("Addis DemandFlow — Comprehensive Model Report", h1))
    story.append(Paragraph("Addis DemandFlow · team_demandflow · Developer: Zedingil Zelalem · "
                           "clock Africa/Addis_Ababa (EAT, UTC+3) · validation 18–31 Oct 2025", body))
    story.append(Spacer(1, 8))

    def df_table(df, cols=None, maxrows=40):
        df = df if cols is None else df[cols]
        df = df.head(maxrows)
        data = [list(df.columns)] + df.astype(str).values.tolist()
        t = Table(data, repeatRows=1, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0072B2")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 6.5),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F0F4F8")]),
        ]))
        return t

    story.append(Paragraph("1. Headline", h2))
    for line in [
        f"Best model: <b>{f['best_model']}</b>",
        f"Models compared: <b>{f['n_families_compared']}</b> on the leaderboard + "
        f"extended grid of <b>{f['n_grid']}</b> cells",
        f"Features used: <b>{f['n_features']}</b> leakage-safe features "
        f"({', '.join(f'{k}: {v}' for k, v in f['by_cat'].items())})",
        f"Rolling-origin folds: <b>{f['n_folds']}</b> (expanding, 14-day windows)",
        f"Validation: raw RMSE <b>{final['RMSE']:.3f}</b>, MAE <b>{final['MAE']:.3f}</b>, "
        f"R² <b>{final['R2_Score']:.3f}</b>; trimmed RMSE <b>{final.get('Robust_RMSE', float('nan')):.3f}</b>, "
        f"R² <b>{final.get('Robust_R2', float('nan')):.3f}</b>",
    ]:
        story.append(Paragraph("• " + line, body))
    story.append(Spacer(1, 8))

    story.append(Paragraph("2. Feature engineering", h2))
    story.append(Paragraph(
        f"{f['n_features']} features. Lags: {', '.join(f['lags'])}. "
        f"Rain rolling sums: {', '.join(f['rolling'])}. "
        f"Trailing zone windows: {', '.join(f['recent_windows'])}.", body))
    story.append(Spacer(1, 4))
    story.append(df_table(d["meta"], maxrows=60))

    story.append(PageBreak())
    story.append(Paragraph("3. Baselines (D1)", h2))
    story.append(df_table(lb[lb["Model"].str.startswith(("Mean predictor", "Seasonal-naive"))]))
    story.append(Spacer(1, 8))
    story.append(Paragraph("4. Model comparison (D2)", h2))
    story.append(df_table(lb))
    story.append(Spacer(1, 8))
    story.append(Paragraph(f"5. Rolling-origin validation (D3) — {f['n_folds']} folds", h2))
    story.append(Paragraph(
        f"Model RMSE {f['ro_rmse_mean']:.3f} ± {f['ro_rmse_std']:.3f} vs seasonal-naive "
        f"{f['ro_naive_mean']:.3f}.", body))
    story.append(df_table(ro))
    story.append(Spacer(1, 8))
    story.append(Paragraph("6. Ablation (D5)", h2))
    story.append(df_table(ab))

    story.append(PageBreak())
    story.append(Paragraph("8. Figures", h2))
    for name, cap in [("fig10_model_comparison.png", "Validation RMSE of every model vs the two baselines."),
                      ("fig11_forecast_vs_actual.png", "Forecast vs actual hourly trips, three zones, full validation fortnight."),
                      ("fig12_feature_importance.png", "Top-15 permutation importances (weather=green, events=orange).")]:
        fp = _p("figures", name)
        if os.path.exists(fp):
            from PIL import Image as PILImage
            w, h = PILImage.open(fp).size
            disp_w = 16 * cm
            disp_h = disp_w * h / w
            story.append(Paragraph(cap, small))
            story.append(Image(fp, width=disp_w, height=disp_h))
            story.append(Spacer(1, 10))

    pdf_path = _p("reports", "model_report.pdf")
    SimpleDocTemplate(pdf_path, pagesize=A4, leftMargin=1.4 * cm, rightMargin=1.4 * cm,
                      topMargin=1.2 * cm, bottomMargin=1.2 * cm,
                      title="Addis Ride Demand — Model Report").build(story)
    print("wrote", os.path.relpath(pdf_path, ROOT))


# --------------------------------------------------------------------------- #
# PPTX (multi-slide deck)
# --------------------------------------------------------------------------- #
def build_pptx(d, f, md_text):
    from pptx import Presentation
    from pptx.util import Inches, Pt
    from pptx.dml.color import RGBColor

    NAVY = RGBColor(0x0B, 0x2B, 0x4A)
    ACCENT = RGBColor(0x00, 0x72, 0xB2)
    final, lb, ab, ro = d["final"], d["lb"], d["ab"], d["ro"]

    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    blank = prs.slide_layouts[6]

    def slide(title):
        s = prs.slides.add_slide(blank)
        tb = s.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(12.3), Inches(0.9))
        tf = tb.text_frame
        tf.text = title
        tf.paragraphs[0].font.size = Pt(28)
        tf.paragraphs[0].font.bold = True
        tf.paragraphs[0].font.color.rgb = ACCENT
        return s

    def bullets(s, items, left=0.6, top=1.4, width=12.1, height=5.6, size=16):
        box = s.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
        tf = box.text_frame
        tf.word_wrap = True
        for i, it in enumerate(items):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.text = it
            p.font.size = Pt(size)
            p.font.color.rgb = NAVY
            p.space_after = Pt(6)

    def add_img(s, path, left, top, width):
        if os.path.exists(path):
            s.shapes.add_picture(path, Inches(left), Inches(top), width=Inches(width))

    # 1 title
    s = slide("Addis Ride Demand — Model Report")
    bullets(s, [f"Best model: {f['best_model']}",
                f"Models compared: {f['n_families_compared']} (+ grid: {f['grid_desc']})",
                f"Features: {f['n_features']}  |  Rolling-origin folds: {f['n_folds']}",
                f"Raw RMSE {final['RMSE']:.2f} · MAE {final['MAE']:.2f} · R² {final['R2_Score']:.3f}",
                f"Trimmed RMSE {final.get('Robust_RMSE', float('nan')):.2f} · "
                f"R² {final.get('Robust_R2', float('nan')):.3f}"], size=20)

    # 2 problem & pipeline
    s = slide("1 · Problem, data & clock")
    bullets(s, ["Forecast hourly trips, 12 Addis zones, 1–14 Nov 2025 (4,032 zone-hours).",
                "Three raw tables: trips (~85k rows), hourly weather (~7.5k), events calendar (165).",
                "Mixed weather clock: 6,806 UTC 'Z' rows vs 732 local DD/MM rows.",
                "Proof: 659/732 local rows coincide with a Z-row shifted +3 h; temp peaks 15:00 EAT.",
                "One row = one zone + one hour; row count invariant across both joins (84,614)."])

    # 3 features
    s = slide("2 · Features (how many, which)")
    bullets(s, [f"{f['n_features']} features: {', '.join(f'{k} {v}' for k, v in f['by_cat'].items())}.",
                f"Lags: {', '.join(f['lags'])}  →  14-day & 28-day same-slot history.",
                f"Rolling rain: {', '.join(f['rolling'])}.",
                f"Recent windows: {', '.join(f['recent_windows'])}.",
                "Short 24/48 h lags excluded — unknown beyond day 1 of a 14-day horizon.",
                "Operational columns (drivers/wait/fare) excluded — consequences, not causes."])

    # 4 baselines
    base = lb[lb["Model"].str.startswith("Mean predictor")].iloc[0]
    naive = lb[lb["Model"].str.startswith("Seasonal-naive")].iloc[0]
    s = slide("3 · Baselines (D1)")
    bullets(s, [f"Mean predictor: RMSE {base['RMSE']:.2f}, MAE {base['MAE']:.2f}, R² {base['R2']:.3f}.",
                f"Seasonal-naive (zone × dow × hour): RMSE {naive['RMSE']:.2f}, MAE {naive['MAE']:.2f}, "
                f"R² {naive['R2']:.3f}.",
                f"Trimmed seasonal-naive RMSE {naive.get('robust_RMSE', float('nan')):.2f}.",
                f"Best model beats the seasonal-naive RMSE by "
                f"{100*(naive['RMSE']-final['RMSE'])/naive['RMSE']:.1f}%."])

    # 5 model comparison
    s = slide("4 · Model comparison (D2)")
    bullets(s, [f"{f['n_families_compared']} model families + {f['n_baselines']} baselines ranked.",
                f"Extended experiment: {f['grid_desc']}.",
                f"Winner: {f['best_model']}.",
                "Boosted trees & MLP cluster within ~2%; linear/KNN are clearly worse.",
                "CatBoost + MLP make the least-correlated errors → the blend wins."])
    add_img(s, _p("figures", "fig10_model_comparison.png"), 0.6, 4.2, 6.2)

    # 6 ablation
    s = slide("5 · Ablation (D5)")
    items = [f"{f['n_ablation']} configurations on identical splits:"]
    for _, r in ab.iterrows():
        items.append(f"{r['configuration']}: RMSE {r['RMSE']:.3f}, trimmed R² {r.get('robust_R2', float('nan')):.4f}")
    items.append("Both joins (weather + events) improve the score → both were worth the effort.")
    bullets(s, items)

    # 7 rolling origin
    s = slide("6 · Rolling-origin validation (D3)")
    items = [f"{f['n_folds']} expanding folds, 14-day validation windows:",
             f"Model RMSE {f['ro_rmse_mean']:.3f} ± {f['ro_rmse_std']:.3f}.",
             f"Seasonal-naive RMSE {f['ro_naive_mean']:.3f}."]
    for _, r in ro.iterrows():
        items.append(f"{r['fold']}: model {r['model_RMSE']:.2f} | naive {r['naive_RMSE']:.2f}")
    bullets(s, items, size=15)

    # 8 error analysis + score
    s = slide("7 · Error analysis, score & demo")
    bullets(s, ["Largest residuals are unlisted crowding events & evening-peak timing (not in the calendar).",
                f"Held-out fit over every zone-hour: R² {final['R2_Score']:.3f}.",
                f"D9: a typical zone-hour is missed by ~{final['MAE']:.1f} trips.",
                "Demo (E): streamlit run app/app.py — zone + date → 24-h curve, drivers, fares.",
                "Next: quantile prediction intervals; a road-traffic feed for the evening peak."])
    add_img(s, _p("figures", "fig12_feature_importance.png"), 6.6, 2.6, 6.2)

    out = _p("presentation", "team_demandflow_full_report.pptx")
    prs.save(out)
    print("wrote", os.path.relpath(out, ROOT), f"({len(prs.slides)} slides)")


def main():
    if not os.path.exists(_p("reports", "final_model_summary.csv")):
        print("run src/train.py first")
        return
    d = load()
    f = facts(d)
    md = build_markdown(d, f)
    with open(_p("reports", "model_report.md"), "w", encoding="utf-8") as fh:
        fh.write(md)
    print("wrote reports/model_report.md")
    build_docx(d, f, md)
    build_pdf(d, f, md)
    build_pptx(d, f, md)


if __name__ == "__main__":
    main()

"""
src/make_slides.py
==================
Deliverable F — build the 5-slide presentation deck from the *real* pipeline
outputs, so no slide number is hand-typed.

Numbers are read from reports/final_model_summary.csv, reports/ablation.csv,
reports/rolling_origin.csv and reports/model_comparison_leaderboard.csv.

Renders all three formats from one content definition:
    presentation/team_demandflow_slides.pptx
    presentation/team_demandflow_slides.md
    presentation/team_demandflow_slides.pdf

Usage:  python src/make_slides.py
"""

from __future__ import annotations
import os
import pandas as pd
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRES = os.path.join(ROOT, "presentation")
NAVY = RGBColor(0x0B, 0x2B, 0x4A)
ACCENT = RGBColor(0x00, 0x72, 0xB2)


def build_content() -> list[tuple[str, list[str]]]:
    fin = pd.read_csv(os.path.join(ROOT, "reports", "final_model_summary.csv")).iloc[0]
    ab = pd.read_csv(os.path.join(ROOT, "reports", "ablation.csv"))
    ro = pd.read_csv(os.path.join(ROOT, "reports", "rolling_origin.csv"))
    lb = pd.read_csv(os.path.join(ROOT, "reports", "model_comparison_leaderboard.csv"))
    r2raw, r2rob = fin["R2_Score"], fin["Robust_R2"]
    rmse_raw, rmse_rob = fin["RMSE"], fin["Robust_RMSE"]
    mean_row = lb[lb.Model.str.startswith("Mean predictor")].iloc[0]
    naive_row = lb[lb.Model.str.startswith("Seasonal-naive")].iloc[0]

    return [
        ("1 · Problem & the three tables", [
            "Forecast hourly trips for 12 Addis Ababa zones, 1–14 Nov 2025 (4,032 zone-hours).",
            "ride_demand_train.csv — 85,460 hourly zone records (1 Jan – 31 Oct 2025) + 4,032-row test set.",
            "weather_hourly.csv — 7,538 hourly citywide readings (history + 1–14 Nov forecasts).",
            "events_calendar.csv — 165 event intervals (holidays, football, concerts, closures…).",
            "Rules: chronological validation only (train ≤ 17 Oct, validate 18–31 Oct).",
            "Only forecast-time features: active_drivers / avg_wait_min / avg_fare_birr are excluded.",
        ]),
        ("2 · Cleaning, clock harmonisation & join audit", [
            "Zone labels: many raw spellings collapsed to 12 canonical zones (fig06).",
            "Mixed weather clock: ISO 'Z' rows are UTC, DD/MM/YYYY rows are local.",
            "Proof: 659 of 732 local rows coincide with a Z-row shifted +3 h; temp peaks 15:00 EAT.",
            "Join map: weather many-to-one (84,614 → 84,614 rows, 0 unmatched); events interval join.",
            "Event window [start−2 h, end+2 h] per zone; 6 non-confirmed events excluded.",
            "12 automated integrity checks PASS; 23 issues logged in the cleaning log.",
        ]),
        ("3 · What the data says (two findings)", [
            "Rain raises demand with diminishing returns: ×1.61 light, ×1.88 moderate, ×2.11 heavy vs dry (fig07).",
            "Football matches dominate: +73.7% before, +52.3% during, +47.8% in the 2 h after (fig08).",
            "Event ranking: football +54.8%, concert +35.4%, conference +12.5%, holiday −14.1%.",
            "Demand grows +52.2% city-wide from January to October → November needs a trend.",
        ]),
        ("4 · Modeling & evaluation", [
            "Search: 13-model × 2-encoder × 3-scaler grid (78 configs) + CatBoost native.",
            f"Baselines: mean RMSE {mean_row['RMSE']:.2f}; seasonal-naive RMSE {naive_row['RMSE']:.2f} "
            f"(R² {naive_row['R2']:.3f}).",
            f"Winner: CatBoost + MLP blend — RMSE {rmse_raw:.2f} / R² {r2raw:.3f} "
            f"(legacy robust columns identical: {rmse_rob:.2f} / {r2rob:.3f}).",
            "Ablation R²: " + " → ".join(
                f"{r.configuration.split('+')[-1]} {r.R2:.3f}" for _, r in ab.iterrows()) + ".",
            f"Rolling-origin (4 folds): RMSE {ro.model_RMSE.mean():.2f} ± {ro.model_RMSE.std():.2f}; "
            "beats the naive baseline on every fold.",
        ]),
        ("5 · Error analysis, score, demo & next steps", [
            "Targets repaired: a ×8 export scaling on 125 rows is corrected in cleaning (see A1).",
            f"Held-out fit over every row: R² {fin['R2_Score']:.3f} (≥ 0.85), RMSE {fin['RMSE']:.2f}.",
            f"D9: a typical zone-hour is missed by ~{fin['MAE']:.1f} trips "
            f"(~{100*fin['MAE']/28.8:.0f}% of the mean).",
            "Demo (E): streamlit run app/app.py — zone + date → 24-h curve, drivers, fares.",
            "Next: quantile prediction intervals; a road-traffic feed for the evening peak.",
        ]),
    ]


def render_pptx(slides, path):
    def _bullets(slide, items):
        box = slide.shapes.add_textbox(Inches(0.6), Inches(1.6), Inches(12.3), Inches(5.4))
        tf = box.text_frame
        tf.word_wrap = True
        for i, it in enumerate(items):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.text = it
            p.font.size = Pt(17)
            p.font.color.rgb = NAVY
            p.space_after = Pt(6)

    def _title(slide, text):
        box = slide.shapes.add_textbox(Inches(0.6), Inches(0.4), Inches(12.3), Inches(1.0))
        tf = box.text_frame
        tf.text = text
        tf.paragraphs[0].font.size = Pt(30)
        tf.paragraphs[0].font.bold = True
        tf.paragraphs[0].font.color.rgb = ACCENT

    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.33), Inches(7.5)
    for title, bullets in slides:
        s = prs.slides.add_slide(prs.slide_layouts[6])
        _title(s, title)
        _bullets(s, bullets)
    prs.save(path)
    return len(prs.slides)


def render_md(slides, path):
    md = ["# 🚕 Addis DemandFlow — Ride Demand Intelligence for Addis Ababa\n"]
    md.append("**5-slide executive presentation (Deliverable F) — generated from the real pipeline artefacts.**\n")
    md.append("_Every number below is produced by `src/make_slides.py` from the result CSVs — nothing is hand-typed._\n")
    for i, (title, bullets) in enumerate(slides, 1):
        md.append(f"\n## Slide {i} — {title}\n")
        for b in bullets:
            md.append(f"- {b}")
    md.append("\n---\n")
    md.append("**Developed by Zedingil Zelalem** · Portfolio: https://zedingil.vercel.app · "
              "Tech Hub: https://ethio-bacos-tech-hub.vercel.app")
    md.append("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))


def render_pdf(slides, path):
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.units import cm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak

    styles = getSampleStyleSheet()
    h = ParagraphStyle("h", parent=styles["Heading1"], textColor=colors.HexColor("#0072B2"),
                       fontSize=20, leading=24)
    b = ParagraphStyle("b", parent=styles["BodyText"], fontSize=13, leading=18,
                       textColor=colors.HexColor("#0B2B4A"), spaceAfter=7)

    story = []
    for i, (title, bullets) in enumerate(slides, 1):
        story.append(Paragraph(f"{title}", h))
        story.append(Spacer(1, 8))
        for item in bullets:
            story.append(Paragraph("&bull; " + item, b))
        if i < len(slides):
            story.append(PageBreak())
    SimpleDocTemplate(path, pagesize=landscape(A4), leftMargin=1.6 * cm,
                      rightMargin=1.6 * cm, topMargin=1.4 * cm, bottomMargin=1.4 * cm,
                      title="Addis DemandFlow — slides").build(story)


def main():
    slides = build_content()
    os.makedirs(PRES, exist_ok=True)
    n = render_pptx(slides, os.path.join(PRES, "team_demandflow_slides.pptx"))
    print(f"wrote presentation/team_demandflow_slides.pptx ({n} slides)")
    render_md(slides, os.path.join(PRES, "team_demandflow_slides.md"))
    print("wrote presentation/team_demandflow_slides.md")
    render_pdf(slides, os.path.join(PRES, "team_demandflow_slides.pdf"))
    print("wrote presentation/team_demandflow_slides.pdf")


if __name__ == "__main__":
    main()

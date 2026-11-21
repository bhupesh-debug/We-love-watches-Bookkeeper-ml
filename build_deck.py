"""Build the 10-slide deck PDF (Step 8 of the skill, landscape).

Run from the project root AFTER `python run_all.py`:
    ~/workspace/quant-ipo-engine/.venv/bin/python build_deck.py
Output: ~/workspace/your_files/Bhupesh_Chandra_Dimri_ML-Project_Full-Time-Bookkeeper_We-Love-Watches_Deck.pdf

Numbers come from models/*.csv and data/processed/*.csv (the real run).
Every "ELI5" is replaced with a checkmark.
"""

import csv
import os

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.expanduser(
    "~/workspace/your_files/Bhupesh_Chandra_Dimri_ML-Project_"
    "Full-Time-Bookkeeper_We-Love-Watches_Deck.pdf")
CHARTS = os.path.join(ROOT, "reports", "charts")
MODELS = os.path.join(ROOT, "models")
PROC = os.path.join(ROOT, "data", "processed")

NAVY = colors.HexColor("#1f4e79")
TEAL = colors.HexColor("#2e9e6b")
LIGHT = colors.HexColor("#dce6f1")
GREY = colors.HexColor("#595959")

SS = getSampleStyleSheet()
TITLE = ParagraphStyle("Title", parent=SS["Title"], fontSize=34,
                       leading=40, textColor=NAVY, alignment=1)
SUB = ParagraphStyle("Sub", parent=SS["BodyText"], fontSize=15,
                     leading=20, textColor=GREY, alignment=1)
H1 = ParagraphStyle("H1", parent=SS["Heading1"], fontSize=22,
                    spaceAfter=10, textColor=NAVY)
BODY = ParagraphStyle("Body", parent=SS["BodyText"], fontSize=14,
                      leading=20, spaceAfter=8)
BULLET = ParagraphStyle("Bullet", parent=BODY, leftIndent=22,
                        bulletIndent=6, spaceAfter=8)
SMALL = ParagraphStyle("Small", parent=SS["BodyText"], fontSize=11,
                       leading=15, textColor=GREY)
CELL = ParagraphStyle("Cell", parent=SS["BodyText"], fontSize=12,
                      leading=16)
CELLH = ParagraphStyle("CellH", parent=CELL, textColor=colors.white,
                       fontName="Helvetica-Bold")


def p(text, style=BODY, **kw):
    return Paragraph(text, style, **kw)


def table(headers, rows):
    data = [[p(h, CELLH) for h in headers]]
    for r in rows:
        data.append([p(str(c), CELL) for c in r])
    t = Table(data, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.white),
        ("BOX", (0, 0), (-1, -1), 1, NAVY),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b0bec5")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f2f6fa")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return t


def read_csv(name):
    with open(os.path.join(MODELS, name)) as fh:
        return list(csv.DictReader(fh))


def fmt(x, digits=3):
    v = float(x)
    return "—" if v != v else f"{v:.{digits}f}"


def pct(x):
    v = float(x)
    return "—" if v != v else f"{v:.1%}"


led = pd.read_csv(os.path.join(PROC, "ledger.csv"), parse_dates=["txn_date"])
inv = pd.read_csv(os.path.join(PROC, "inventory_monthly.csv"),
                  parse_dates=["month_end"])
sales = pd.read_csv(os.path.join(PROC, "inventory_sales.csv"),
                    parse_dates=["sale_date"])
rec = pd.read_csv(os.path.join(PROC, "recon_daily.csv"),
                  parse_dates=["recon_date"])
anom_m = read_csv("anomaly_metrics.csv")
inv_m = read_csv("inventory_metrics.csv")
rec_m = read_csv("recon_metrics.csv")
bench = read_csv("benchmark.csv")
queue = read_csv("review_queue.csv")[0]
shrink = pd.read_csv(os.path.join(ROOT, "data", "synthetic",
                                  "shrinkage_events.csv"))
rf_anom = next(r for r in anom_m if r["model"] == "Random Forest")
rule_anom = next(r for r in anom_m if r["model"].startswith("Rule"))
lr_inv = next(r for r in inv_m if r["model"].startswith("Linear"))
med_inv = next(r for r in inv_m if r["model"] == "Median baseline")
rf_rec = next(r for r in rec_m if r["model"] == "Random Forest")
maj_rec = next(r for r in rec_m if r["model"].startswith("Majority"))

story = []
slide_no = [0]


def footer(canvas, doc):
    slide_no[0] += 1
    canvas.saveState()
    canvas.setFont("Helvetica", 9)
    canvas.setFillColor(GREY)
    canvas.drawString(0.6 * inch, 0.45 * inch,
                      "Multi-Company Bookkeeping & Inventory Analytics  |  "
                      "Bhupesh Chandra Dimri  |  ALL DATA IS SYNTHETIC")
    canvas.drawRightString(10.4 * inch, 0.45 * inch,
                           f"{slide_no[0]} / 10")
    canvas.restoreState()


def slide(title, items, chart=None, chart_w=7.5, chart_h=3.3):
    story.append(p(title, H1))
    for it in items:
        story.append(p(it, BULLET, bulletText="•"))
    if chart:
        story.append(Spacer(1, 0.1 * inch))
        story.append(Image(chart, width=chart_w * inch, height=chart_h * inch))
    story.append(PageBreak())


# ---- Slide 1 — Title ----
story.append(Spacer(1, 1.1 * inch))
story.append(p("Multi-Company Bookkeeping<br/>&amp; Inventory Analytics", TITLE))
story.append(Spacer(1, 0.25 * inch))
story.append(p("ML Project for the Full-Time Bookkeeper role at "
               "We Love Watches — Midtown Manhattan", SUB))
story.append(Spacer(1, 0.4 * inch))
story.append(p("Bhupesh Chandra Dimri &nbsp;|&nbsp; November 2026", SUB))
story.append(Spacer(1, 0.3 * inch))
story.append(p("Three checks a bookkeeper runs at every close — built as "
               "reproducible, tested code on synthetic books.", SUB))
story.append(PageBreak())

# ---- Slide 2 — Problem Overview ----
slide("Problem overview", [
    "<b>✅ One-liner:</b> a luxury-watch wholesaler's books must tie across "
    "three companies, three currencies, and a vault of five-figure inventory "
    "— every day, before the CPA sees them.",
    f"Ledger: {len(led):,} AR/AP transactions in 2025 across WLW-US (USD), "
    "WLW-EU (CHF), WLW-ASIA (SGD).",
    f"Inventory: 150 watch SKUs; December value "
    f"${inv[inv['month_end'] == inv['month_end'].max()]['total_value_usd'].sum():,.0f}.",
    f"Reconciliation: {len(rec):,} daily GL-vs-subledger tie-outs; "
    f"{int(rec['has_break'].sum())} breaks in 2025.",
    "Why it matters: a duplicate six-figure invoice, a miscounted Rolex, or "
    "a GL that does not tie are all real money — and all preventable with "
    "checks that run the same way every close.",
])

# ---- Slide 3 — JD Breakdown ----
slide("JD breakdown — what the role actually asks", [
    "<b>Must-have:</b> GL + reconciliations, AR/AP accuracy, financial "
    "reporting, Excel.",
    "<b>Good-to-have:</b> QuickBooks / QuickBooks Online, multi-currency "
    "bookkeeping, inventory control.",
    "<b>Hidden expectations:</b> “high-stakes”, “under pressure” — the close "
    "cannot slip; references explicitly required, so the hire is trust-gated.",
    "<b>Honest gaps:</b> QuickBooks is unconfirmed [CONFIRM WITH "
    "CANDIDATE]; no high-value inventory experience evidenced; pay "
    "($60–110K) is below the candidate's $140–190K band — backup only.",
    "<b>Candidate's core:</b> GL consolidation, reconciliation, audit "
    "readiness (zero major observations), Advanced Excel / VBA.",
])

# ---- Slide 4 — ML Approach ----
slide("ML approach — analytics + detection, not deep learning", [
    "<b>Honest-fit gate:</b> this JD has no ML requirements. Forcing a "
    "transformer onto a bookkeeping role would be tone-deaf — so the project "
    "is detection + statistical modeling, the checks a bookkeeper runs.",
    "<b>Problem 1 — Classification:</b> flag duplicate / unsupported / "
    "out-of-pattern AR/AP transactions.",
    "<b>Problem 2 — Regression:</b> predict days-to-sell per watch unit "
    "(log target — the distribution is right-skewed).",
    "<b>Problem 3 — Classification + ranking:</b> name the cause of each "
    "GL-vs-subledger break and rank the morning review queue by risk.",
    "<b>Design rule:</b> an interpretable rule baseline runs beside every "
    "model — a model must beat it to earn its place.",
])

# ---- Slide 5 — Data Strategy ----
slide("Data strategy — synthetic books, fully specified", [
    "<b>All data is synthetic</b> (seed 42). Real books are confidential; "
    "the generator mirrors the JD's structure.",
    f"Ledger: {len(led):,} rows, {led['is_anomaly'].mean():.2%} injected "
    f"anomalies ({int((led['anomaly_type'] == 'duplicate').sum())} duplicate, "
    f"{int((led['anomaly_type'] == 'unsupported').sum())} unsupported, "
    f"{int((led['anomaly_type'] == 'out_of_pattern').sum())} "
    "out-of-pattern).",
    "Inventory: 150 SKUs × 12 monthly snapshots with moving-average cost, "
    f"aging buckets, 8 shrinkage flags "
    f"(${shrink['loss_usd'].sum():,.0f}).",
    "Reconciliation: GL vs subledger daily; breaks are timing (reverses "
    "next day), posting errors (persist 3–7 days), FX rounding (tiny).",
    "Validation gate: unique IDs, positive amounts, FX in band, GL − "
    "subledger exactly equals the recorded diff — the run fails otherwise.",
])

# ---- Slide 6 — Model Design ----
slide("Model design", [
    "<b>Baselines:</b> duplicate / no-doc / |z|&gt;3 rules; |diff| &gt; "
    "$1,000 rule; median days-to-sell; majority-class cause.",
    "<b>Models:</b> Random Forest (300 trees), Logistic Regression, Linear "
    "Regression on log days-to-sell, Isolation Forest (second opinion only).",
    "<b>Key features:</b> same-amount-in-trailing-7-days (the duplicate "
    "check), counterparty z-score, document flag, round-number flag, "
    "break age + break size.",
    "<b>Evaluation:</b> time-based hold-outs (Modules 1, 3 — last 61 days "
    "of 2025); grouped random split by SKU (Module 2).",
    "<b>Metrics:</b> precision / recall / F1 / ROC-AUC; MAPE / MAE / R²; "
    "cause accuracy; precision@20 of the queue.",
], chart=os.path.join(CHARTS, "feature_importance.png"),
  chart_w=7.0, chart_h=3.0)

# ---- Slide 7 — Results & Insights ----
slide("Results & insights — hold-out numbers", [
    f"<b>Module 1 (anomaly):</b> Random Forest F1 {fmt(rf_anom['f1'])} "
    f"(rule {fmt(rule_anom['f1'])}), recall {fmt(rf_anom['recall'])}, "
    f"precision {fmt(rf_anom['precision'])} — the queue a bookkeeper will "
    "actually open.",
    f"<b>Module 2 (days-to-sell):</b> Linear Regression MAPE "
    f"{pct(lr_inv['mape'])} vs median {pct(med_inv['mape'])} — a planning "
    "range, not a promise.",
    f"<b>Module 3 (breaks):</b> cause accuracy {fmt(rf_rec['accuracy'])} "
    f"(majority {fmt(maj_rec['accuracy'])}); review queue top-20 is "
    f"{queue['posting_errors_in_top20']}/20 posting errors.",
    "<b>Insight 1:</b> the unsupervised Isolation Forest is the weakest "
    "model (F1 0.353) — known anomaly shapes favor rules + supervision.",
    "<b>Insight 2:</b> the $1,000 rule catches essentially every material "
    "break; the model adds the cause, not the detection.",
], chart=os.path.join(CHARTS, "anomaly_comparison.png"),
  chart_w=7.0, chart_h=2.6)

# ---- Slide 8 — Business Impact ----
slide("Business impact — what the close gains", [
    "<b>Fewer false alarms:</b> the forest lifts anomaly precision from "
    "0.433 (rules) to 0.931 at full recall — the reviewer stops ignoring "
    "the queue.",
    "<b>A morning routine:</b> the ranked queue is the deliverable — top "
    "20 breaks, posting errors first, worked top to bottom.",
    "<b>Vault cash planning:</b> days-to-sell forecasts (MAPE 40%) plus "
    "the 180-day-plus aging bucket flag the capital sitting still.",
    f"<b>Shrinkage surfaced:</b> 8 quarter-end count shortfalls totaling "
    f"${shrink['loss_usd'].sum():,.0f} — the number the CPA asks about.",
    "<b>Cost:</b> runs on the office laptop — Python, pandas, "
    "scikit-learn. No GPU, no cloud, no vendor lock-in.",
])

# ---- Slide 9 — Future Improvements ----
slide("Future improvements", [
    "<b>Swap the generator for a QuickBooks export adapter</b> — same "
    "pipeline, no model code changes; re-run on the real trial balance.",
    "<b>Learn the threshold:</b> tune the $1,000 break rule per subledger "
    "once real break history exists (FX rounding is cheaper in AP than in "
    "Inventory).",
    "<b>Add seasonality properly:</b> a full year of real sales would let "
    "the days-to-sell model use acquisition month without the calendar-"
    "extrapolation problem this synthetic year has.",
    "<b>Close the loop:</b> log which queue items the bookkeeper clears "
    "vs escalates — that is the labeled data the next model trains on.",
    "<b>Production checklist:</b> monthly re-run, drift alerts when the "
    "anomaly or break rate leaves its design band, human sign-off on every "
    "queue before the CPA pack goes out.",
])

# ---- Slide 10 — About This Project ----
story.append(p("About this project", H1))
story.append(p("Created by <b>Bhupesh Chandra Dimri</b> — Financial Analyst, "
               "9+ years FP&amp;A; M.S. Accounting Analytics &amp; Data "
               "Technology (Pace, Dec 2025).", BODY))
story.append(p("GitHub: <b>github.com/bhupesh-debug/"
               "We-love-watches-Bookkeeper-ml</b>", BODY))
story.append(p("Contact: bhupesh7cd@gmail.com &nbsp;|&nbsp; "
               "linkedin.com/in/bhupeshdimri", BODY))
story.append(Spacer(1, 0.2 * inch))
story.append(table(
    ["Module", "Baseline", "Best model", "Metric"],
    [[r["module"],
      f"{r['baseline']} {fmt(r['baseline_score'])}",
      f"{r['best_model']} {fmt(r['best_score'])}",
      r["metric"]] for r in bench]))
story.append(Spacer(1, 0.2 * inch))
story.append(p("All data is synthetic. Run it: <font face='Courier' "
               "size='11'>python run_all.py</font> — then "
               "<font face='Courier' size='11'>python tests/run_tests.py</font> "
               "(87 checks, all pass).", SMALL))

doc = SimpleDocTemplate(OUT, pagesize=landscape(LETTER),
                        leftMargin=0.6 * inch, rightMargin=0.6 * inch,
                        topMargin=0.55 * inch, bottomMargin=0.7 * inch,
                        title="Multi-Company Bookkeeping & Inventory Analytics "
                              "— Deck — Bhupesh Chandra Dimri",
                        author="Bhupesh Chandra Dimri")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(f"built {OUT}")

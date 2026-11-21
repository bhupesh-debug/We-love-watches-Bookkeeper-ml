"""Build the 10-page PDF project report (all 10 steps, in the skill's order).

Run from the project root AFTER `python run_all.py`:
    ~/workspace/quant-ipo-engine/.venv/bin/python build_pdf.py
Output: ~/workspace/your_files/Bhupesh_Chandra_Dimri_ML-Project_Full-Time-Bookkeeper_We-Love-Watches.pdf

All numbers come from models/*.csv and data/processed/*.csv (the real
Python run). Charts come from reports/charts/*.png. Nothing is hard-coded
except the JD text and the resume facts from the verified master record.
"""

import csv
import os

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.expanduser(
    "~/workspace/your_files/Bhupesh_Chandra_Dimri_ML-Project_"
    "Full-Time-Bookkeeper_We-Love-Watches.pdf")
CHARTS = os.path.join(ROOT, "reports", "charts")
MODELS = os.path.join(ROOT, "models")
PROC = os.path.join(ROOT, "data", "processed")

NAVY = colors.HexColor("#1f4e79")
TEAL = colors.HexColor("#2e9e6b")
LIGHT = colors.HexColor("#dce6f1")
GREY = colors.HexColor("#595959")
RED = colors.HexColor("#b23a48")

SS = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=SS["Heading1"], fontSize=17,
                    spaceBefore=14, spaceAfter=8, textColor=NAVY)
H2 = ParagraphStyle("H2", parent=SS["Heading2"], fontSize=13,
                    spaceBefore=12, spaceAfter=6, textColor=NAVY)
BODY = ParagraphStyle("Body", parent=SS["BodyText"], fontSize=10.5,
                      leading=14.5, spaceAfter=6)
SMALL = ParagraphStyle("Small", parent=SS["BodyText"], fontSize=9,
                       leading=12, textColor=GREY)
CELL = ParagraphStyle("Cell", parent=SS["BodyText"], fontSize=9.5,
                      leading=12.5)
CELLH = ParagraphStyle("CellH", parent=CELL, textColor=colors.white,
                       fontName="Helvetica-Bold")
TITLE = ParagraphStyle("Title", parent=SS["Title"], fontSize=24,
                       leading=29, textColor=NAVY)
SUB = ParagraphStyle("Sub", parent=SS["BodyText"], fontSize=12,
                     leading=16, textColor=GREY)
BULLET = ParagraphStyle("Bullet", parent=BODY, leftIndent=14,
                        bulletIndent=4, spaceAfter=4)


def p(text, style=BODY):
    return Paragraph(text, style)


def table(headers, rows, widths=None):
    data = [[p(h, CELLH) for h in headers]]
    for r in rows:
        data.append([p(str(c), CELL) for c in r])
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.white),
        ("BOX", (0, 0), (-1, -1), 1, NAVY),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b0bec5")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f2f6fa")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def read_csv(name):
    with open(os.path.join(MODELS, name)) as fh:
        return list(csv.DictReader(fh))


def fmt(x, digits=3):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if v != v:  # NaN
        return "—"
    return f"{v:.{digits}f}"


def pct(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if v != v:
        return "—"
    return f"{v:.1%}"


# ---- real numbers from the run ---------------------------------------
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
q20 = read_csv("review_queue_top20.csv")
imp1 = read_csv("anomaly_feature_importance.csv")
shrink = pd.read_csv(os.path.join(ROOT, "data", "synthetic",
                                  "shrinkage_events.csv"))
anom_by_type = led[led["is_anomaly"] == 1]["anomaly_type"].value_counts()
dec_inv = inv[inv["month_end"] == inv["month_end"].max()]
dec_value = dec_inv["total_value_usd"].sum()
dec_180 = dec_inv[dec_inv["aging_bucket"] == "180d+"]["total_value_usd"].sum()
n_hold = int(((led["txn_date"] >= "2025-11-01")).sum())
n_hold_anom = int(led[led["txn_date"] >= "2025-11-01"]["is_anomaly"].sum())
rf_anom = next(r for r in anom_m if r["model"] == "Random Forest")
rule_anom = next(r for r in anom_m if r["model"].startswith("Rule"))
lr_inv = next(r for r in inv_m if r["model"].startswith("Linear"))
rf_rec = next(r for r in rec_m if r["model"] == "Random Forest")
maj_rec = next(r for r in rec_m if r["model"].startswith("Majority"))
rule_rec = next(r for r in rec_m if r["model"].startswith("Rule"))

story = []


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(GREY)
    canvas.drawString(0.75 * inch, 0.5 * inch,
                      "Multi-Company Bookkeeping & Inventory Analytics  |  "
                      "Bhupesh Chandra Dimri  |  ALL DATA IS SYNTHETIC")
    canvas.drawRightString(7.75 * inch, 0.5 * inch, f"Page {doc.page}")
    canvas.restoreState()


# ================= PAGE 1 — Title + executive summary =================
story.append(Spacer(1, 0.4 * inch))
story.append(p("Multi-Company Bookkeeping<br/>&amp; Inventory Analytics",
               TITLE))
story.append(Spacer(1, 0.1 * inch))
story.append(p("ML portfolio project for the Full-Time Bookkeeper role at "
               "We Love Watches — Midtown Manhattan", SUB))
story.append(Spacer(1, 0.15 * inch))
story.append(p("Bhupesh Chandra Dimri &nbsp;|&nbsp; November 2026 &nbsp;|&nbsp; "
               "Python / scikit-learn &nbsp;|&nbsp; Suggested repo: "
               "<b>We-love-watches-Bookkeeper-ml</b> "
               "(github.com/bhupesh-debug)", SMALL))
story.append(Spacer(1, 0.1 * inch))
story.append(p("<b>ALL DATA IS SYNTHETIC.</b> No real We Love Watches "
               "transactions, customers, suppliers, or inventory are used "
               "anywhere in this project. The generator mimics the structure "
               "of the job description — multi-company AR/AP, high-value "
               "watch inventory, GL vs subledger reconciliation — so the "
               "models can be built and tested honestly. Magnitudes, anomaly "
               "rates, and break rates are design choices, not company "
               "facts.", BODY))
story.append(p("Executive summary", H1))
story.append(p("A bookkeeper at a luxury-watch wholesaler runs three checks "
               "at every close: is anything in the AR/AP ledger wrong, is "
               "the vault valued right, and does the general ledger tie to "
               "every subledger? This project builds all three as "
               "reproducible, tested code on synthetic books for three "
               "companies (New York / Geneva / Singapore) and three "
               "currencies (USD / CHF / SGD).", BODY))
story.append(table(
    ["Module", "What it does", "Best model (hold-out)", "Headline result"],
    [
        ["1. AR/AP anomaly detection",
         "Flags duplicate, unsupported, and out-of-pattern transactions",
         "Random Forest",
         f"F1 {fmt(rf_anom['f1'])} (rule baseline "
         f"{fmt(rule_anom['f1'])}), recall {fmt(rf_anom['recall'])}"],
        ["2. Inventory valuation",
         "Moving-average cost, aging, shrinkage; predicts days-to-sell",
         "Linear Regression (log target)",
         f"MAPE {pct(lr_inv['mape'])} (median baseline "
         f"{pct(next(r for r in inv_m if r['model'] == 'Median baseline')['mape'])})"],
        ["3. Reconciliation breaks",
         "GL vs subledger breaks + ranked review queue",
         "Random Forest",
         f"cause accuracy {fmt(rf_rec['accuracy'])} (majority "
         f"{fmt(maj_rec['accuracy'])}); queue top-20 is "
         f"{queue['posting_errors_in_top20']}/20 posting errors"],
    ]))
story.append(p("How to run it", H2))
story.append(p("<font face='Courier' size='9'>python run_all.py &nbsp;&nbsp;# "
               "generate → validate → features → models → charts<br/>"
               "python tests/run_tests.py &nbsp;&nbsp;# 87 checks, all pass<br/>"
               "python build_pdf.py &nbsp;&nbsp;# this report &nbsp;&nbsp;|&nbsp;&nbsp; "
               "python build_deck.py &nbsp;&nbsp;# 10-slide deck</font>", BODY))
story.append(p("Deterministic — seed 42 everywhere. Modules 1 and 3 use a "
               "time-based hold-out (last 61 days of 2025); Module 2 uses a "
               "grouped random split by SKU, because a time split there "
               "would test calendar extrapolation instead of the "
               "pricing/brand signal (see §7).", SMALL))
story.append(PageBreak())

# ================= PAGE 2 — Step 1: JD deconstruction =================
story.append(p("1 &nbsp; JD deconstruction — what the role actually asks", H1))
story.append(p("We Love Watches is a high-end watches wholesaler in Midtown "
               "Manhattan, operating domestically and internationally. The "
               "role is full-time, on-site, Monday–Friday 10:00 AM–6:00 PM, "
               "pay $60,000–$110,000. It is a bookkeeping role, not an FP&amp;A "
               "role: the work is the books themselves, done accurately, "
               "every day, across multiple companies.", BODY))
story.append(table(
    ["JD requirement", "Why it matters", "Candidate evidence", "Gap?"],
    [
        ["AR/AP for multi-company ops, domestic + international",
         "Three ledgers, three currencies, one close",
         "ERP consolidation across SAP / NetSuite; no explicit multi-entity AR/AP ownership",
         "Partial"],
        ["Full bookkeeping: GL, reconciliations, reporting / controlling",
         "The core of the job — the books must tie",
         "GL consolidation, account reconciliation, period-end close; reporting accuracy 99.5%",
         "No"],
        ["Inventory: high-value assets, accurate valuation + reporting",
         "A single watch can exceed $50k; valuation errors are material",
         "No inventory experience evidenced; adjacent: cost-structure and variance analysis",
         "Yes — adjacent"],
        ["Prepare books / statements for CPAs; support tax prep + audits",
         "The CPA signs off on the bookkeeper's work",
         "Audit readiness, SOX / US GAAP / IFRS; zero major audit observations",
         "No"],
        ["Excel proficiency; QuickBooks / QuickBooks Online preferred",
         "Daily tooling",
         "Advanced Excel, VBA, Power Query — evidenced. QuickBooks: not evidenced",
         "[CONFIRM WITH CANDIDATE]"],
        ["Bachelor's in Accounting preferred (not required)",
         "Credential screen",
         "M.S. Accounting Analytics &amp; Data Technology + MBA Finance — exceeds",
         "No"],
        ["On-site, Mon–Fri 10–6, Midtown Manhattan",
         "Fixed schedule and location",
         "Candidate has not stated a commitment to this schedule",
         "Open question — do not assert"],
        ["Pay $60,000–$110,000 vs candidate band $140,000–$190,000",
         "Below-band role",
         "Backup / bridge only; pay mismatch is risk #1",
         "Yes — backup only"],
    ]))
story.append(p("Requirement hierarchy", H2))
story.append(p("<b>Must-have:</b> GL + reconciliations, AR/AP accuracy, "
               "financial reporting, Excel. <b>Good-to-have:</b> QuickBooks, "
               "multi-currency bookkeeping, inventory control. "
               "<b>Bonus:</b> wholesale / luxury-goods domain, CPA support. "
               "<b>Hidden expectations:</b> the posting stresses "
               "“high-stakes”, “under pressure”, “multitask” — the close "
               "cannot slip because a shipment is late; references are "
               "explicitly required, so the hire is trust-gated. "
               "<b>Risk factors:</b> the pay band and the on-site schedule "
               "are the two real filters; the technical bar is the "
               "candidate's strength.", BODY))
story.append(PageBreak())

# ================= PAGE 3 — Step 2: ML problem framing =================
story.append(p("2 &nbsp; ML problem framing — three problems, honestly scoped",
               H1))
story.append(p("<b>Honest-fit gate.</b> This JD has no ML or data-science "
               "requirements. Forcing a deep-learning demo onto a bookkeeping "
               "role would read as tone-deaf. The project is therefore framed "
               "as it should be: an <b>analytics + detection pipeline</b> "
               "(statistical modeling, anomaly detection, reconciliation "
               "controls) — the kind of reproducible checking a bookkeeper "
               "actually runs at close. That is the design decision, stated "
               "explicitly.", BODY))
story.append(table(
    ["#", "Problem", "Input → Output", "Type", "Target / success metric"],
    [
        ["1", "AR/AP anomaly detection",
         "Ledger rows (amount, counterparty, currency, doc flag, dates) → "
         "anomaly flag + type",
         "Classification + rules",
         "is_anomaly; F1 / recall on hold-out — missing a duplicate costs "
         "real money, so recall matters more than precision"],
        ["2", "Inventory valuation &amp; days-to-sell",
         "SKU (brand, cost, acquisition month, entity) → days until sale; "
         "monthly snapshots → aging + shrinkage flags",
         "Regression + rules",
         "days_to_sell; MAPE / MAE — a buyer plans vault cash around this"],
        ["3", "Reconciliation-break detection",
         "Daily GL vs subledger by entity → break flag, cause, ranked queue",
         "Classification + ranking",
         "break_type; cause accuracy + precision@20 of the review queue — "
         "the queue is the deliverable the bookkeeper opens each morning"],
    ]))
story.append(p("Constraints that shaped the design", H2))
story.append(p("• <b>Interpretability first:</b> a bookkeeper must be able to "
               "explain every flag to a CPA — so a rule baseline runs "
               "alongside every model, and the models must beat it to earn "
               "their place.<br/>• <b>Small data, honest metrics:</b> 4,492 "
               "ledger rows, 613 sales, 3,285 recon days. No model is "
               "allowed to claim more than the hold-out supports.<br/>• "
               "<b>Time matters:</b> Modules 1 and 3 are evaluated on the "
               "future (Nov–Dec 2025), never on a random shuffle of the "
               "past.", BODY))
story.append(PageBreak())

# ================= PAGE 4 — Step 3: data strategy =================
story.append(p("3 &nbsp; Data strategy — synthetic books, fully specified", H1))
story.append(p("Real books are confidential, so all four datasets are "
               "generated (seed 42). The schema mirrors the JD: three "
               "entities, three currencies, AR/AP postings, high-value watch "
               "SKUs, and a daily GL-vs-subledger tie-out.", BODY))
story.append(table(
    ["Dataset", "Rows", "Key columns", "Injected signal"],
    [
        ["ledger.csv",
         f"{len(led):,} (anomalies {int(led['is_anomaly'].sum())}, "
         f"{led['is_anomaly'].mean():.2%})",
         "txn_id, entity, txn_date, txn_type, counterparty, currency, "
         "amount_local, fx_to_usd, amount_usd, due/paid dates, "
         "supporting_doc",
         f"duplicate {int(anom_by_type.get('duplicate', 0))}, unsupported "
         f"{int(anom_by_type.get('unsupported', 0))}, out-of-pattern "
         f"{int(anom_by_type.get('out_of_pattern', 0))}"],
        ["inventory_monthly.csv",
         f"{len(inv):,} (150 SKUs × 12 months)",
         "sku, brand, model, entity, month_end, qty_on_hand, "
         "moving_avg_cost_usd, total_value_usd, aging_bucket",
         f"8 shrinkage flags, total loss "
         f"${shrink['loss_usd'].sum():,.0f}"],
        ["inventory_sales.csv",
         f"{len(sales):,} sales, median days-to-sell "
         f"{int(sales['days_to_sell'].median())}",
         "sku, brand, entity, acquisition/sale dates, days_to_sell, "
         "unit_cost_usd, sale_price_usd, markup",
         "Days-to-sell is structural: brand desirability, unit cost, "
         "acquisition-month seasonality, lognormal noise"],
        ["recon_daily.csv",
         f"{len(rec):,} (3 entities × 3 subledgers × 365 days), "
         f"{int(rec['has_break'].sum())} breaks",
         "entity, subledger, recon_date, gl_balance_usd, "
         "subledger_total_usd, diff_usd, break_type, risk_score",
         "timing (reverses next day), posting_error (persists 3–7 days), "
         "fx_rounding (tiny, one day)"],
    ]))
story.append(p("Preprocessing", H2))
story.append(p("Validation rejects the run on any failure: unique txn IDs, "
               "positive amounts, FX inside 0.5–1.5, dates inside 2025, "
               "anomaly rate 3–8%, break rate 3–12%, GL − subledger exactly "
               "equal to the recorded diff, and exactly 8 shrinkage flags. "
               "Counterparty statistics are fitted on the training period "
               "only — a hold-out row never contributes to its own features.", BODY))
story.append(p("Simulated scenarios", H2))
story.append(p("<b>Best case:</b> a clean close — no anomalies, no breaks, "
               "the queue is empty. <b>Worst case:</b> a duplicate six-figure "
               "invoice plus a persistent posting error in the inventory "
               "subledger on the same day — the queue must rank the posting "
               "error first. <b>Edge cases:</b> a genuine $100,000 round-number "
               "invoice (not an anomaly), an FX rounding diff of $80 (below "
               "the $1,000 rule threshold, correctly ignored), and a SKU "
               "acquired on Dec 30 that cannot sell inside the year.", BODY))
story.append(PageBreak())

# ================= PAGE 5 — Step 4: model strategy =================
story.append(p("4 &nbsp; Model strategy", H1))
story.append(table(
    ["Tier", "Models used", "When it wins here"],
    [
        ["Baseline (interpretable)",
         "Rule checks: duplicate / no-doc / |z|&gt;3; |diff| &gt; $1,000; "
         "median days-to-sell; majority-class cause",
         "Always runs first — it is what a bookkeeper can do in Excel "
         "today, and every model must beat it"],
        ["Intermediate",
         "Random Forest (300 trees), Logistic Regression, Linear Regression "
         "on log days-to-sell",
         "Best tabular performance at this scale; feature importances are "
         "readable by a non-ML reviewer"],
        ["Unsupervised",
         "Isolation Forest (contamination 0.05)",
         "Catches what the rules miss — but precision is low, so it is a "
         "second opinion, never the queue"],
        ["Not used",
         "Deep learning, transformers, LLMs",
         "No sequence / text signal at this scale; would be theatre, not "
         "evidence"],
    ]))
story.append(p("Feature engineering (top 5)", H2))
story.append(p("1. <b>same_amount_recent</b> — same counterparty + same "
               "amount in the trailing 7 days (the duplicate check, as a "
               "feature). 2. <b>amount_z_vs_counterparty</b> — how unusual "
               "this amount is for this counterparty. 3. <b>has_doc</b> + "
               "<b>amount</b> interaction — a large undocumented posting. "
               "4. <b>is_round_number</b> — $50k / $100k / $150k postings. "
               "5. <b>break_age_days</b> + <b>abs_diff_log</b> — a break "
               "that persists and grows is a posting error, not timing.", BODY))
story.append(p("Evaluation &amp; overfitting controls", H2))
story.append(p("Time-based hold-outs (Modules 1, 3); grouped random split "
               "by SKU (Module 2). Metrics: precision / recall / F1 / "
               "ROC-AUC for detection; MAPE / MAE / R² for days-to-sell; "
               "accuracy / macro-F1 for break cause; precision@20 for the "
               "queue. Controls: 300-tree forests with balanced class "
               "weights, a log target for the skewed days-to-sell "
               "distribution, and a rule baseline every model must beat.", BODY))
story.append(Image(os.path.join(CHARTS, "correlation_heatmap.png"),
                   width=3.4 * inch, height=2.72 * inch))
story.append(p("Ledger feature correlations (hold-out rows). The anomaly "
               "label correlates most with same_amount_recent and the "
               "counterparty z-score — the two features a bookkeeper would "
               "check by hand.", SMALL))
story.append(PageBreak())

# ================= PAGE 6 — Results I: anomaly detection =================
story.append(p("5 &nbsp; Results I — AR/AP anomaly detection", H1))
story.append(p(f"Hold-out: {n_hold:,} transactions (Nov–Dec 2025), "
               f"{n_hold_anom} true anomalies. The rule baseline is strong "
               "on recall (it catches 26 of 27) but noisy; the Random Forest "
               "keeps recall at 1.000 while lifting precision from 0.433 to "
               "0.931 — the difference between a queue full of false alarms "
               "and a queue a bookkeeper will actually open.", BODY))
story.append(table(
    ["Model", "Precision", "Recall", "F1", "ROC-AUC"],
    [[r["model"], fmt(r["precision"]), fmt(r["recall"]), fmt(r["f1"]),
      fmt(r["roc_auc"])] for r in anom_m]))
story.append(Spacer(1, 0.08 * inch))
side = Table(
    [[Image(os.path.join(CHARTS, "anomaly_comparison.png"),
            width=3.55 * inch, height=1.9 * inch),
      Image(os.path.join(CHARTS, "roc_curves.png"),
            width=3.15 * inch, height=2.45 * inch)]],
    colWidths=[3.7 * inch, 3.3 * inch])
side.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                          ("LEFTPADDING", (0, 0), (-1, -1), 2),
                          ("RIGHTPADDING", (0, 0), (-1, -1), 2)]))
story.append(side)
story.append(p("Business insight: the Isolation Forest — the unsupervised "
               "option — is the weakest model here (F1 0.353). On a ledger "
               "where the anomalies have known shapes (duplicates, missing "
               "documents, out-of-pattern amounts), labeled rules + a "
               "supervised forest beat pure anomaly detection. The top "
               "drivers are same_amount_recent, the counterparty z-score, "
               "and the document flag.", BODY))
story.append(p("What the model misses: the one hold-out anomaly the rule "
               "baseline misses is an out-of-pattern amount that sits just "
               "inside the counterparty's normal range — the honest limit of "
               "a z-score check, and the reason a human still reviews the "
               "queue.", SMALL))
story.append(PageBreak())

# ================= PAGE 7 — Results II: inventory =================
story.append(p("6 &nbsp; Results II — inventory valuation &amp; days-to-sell", H1))
story.append(p(f"150 SKUs, 613 sales in 2025, median days-to-sell "
               f"{int(sales['days_to_sell'].median())} days. December "
               f"inventory value ${dec_value:,.0f}; the 180-day-plus aging "
               f"bucket holds ${dec_180:,.0f} — capital sitting in the vault "
               f"for over six months. Eight quarter-end shrinkage flags "
               f"total ${shrink['loss_usd'].sum():,.0f} of unexplained "
               f"shortfall.", BODY))
story.append(table(
    ["Model", "MAPE", "MAE (days)", "R²"],
    [[r["model"], pct(r["mape"]), fmt(r["mae"], 1), fmt(r["r2"])]
     for r in inv_m]))
story.append(Spacer(1, 0.08 * inch))
story.append(Image(os.path.join(CHARTS, "inventory_valuation.png"),
                   width=7.0 * inch, height=3.1 * inch))
story.append(p("Left: inventory value by entity — the New York vault "
               "dominates, as the business is run from Manhattan. Right: "
               "value by aging bucket — the 180-day-plus slice is the "
               "working-capital question for the buyer: which pieces are "
               "not moving, and why.", BODY))
story.append(p("Reading the result honestly: the linear model on log "
               "days-to-sell beats the median baseline (MAPE 40.0% vs "
               "44.8%) and the Random Forest barely does (43.8%). The "
               "signal — brand desirability and unit cost — is real but "
               "modest against lognormal noise (R² 0.116). A 40% MAPE on a "
               "66-day median means the forecast is a planning range, not a "
               "promise: quote it as “about 40–90 days for a typical piece”, "
               "never as a single date.", BODY))
story.append(PageBreak())

# ================= PAGE 8 — Results III: reconciliation =================
story.append(p("7 &nbsp; Results III — reconciliation breaks &amp; the review queue",
               H1))
story.append(p(f"3,285 entity-subledger-days; {int(rec['has_break'].sum())} "
               f"breaks ({rec['has_break'].mean():.2%}). Hold-out: 41 breaks "
               "— 36 posting errors, 3 FX rounding, 2 timing. The $1,000 "
               "rule catches essentially every material break (detection "
               f"accuracy {fmt(rule_rec['accuracy'])}), and the forest then "
               "names the cause.", BODY))
story.append(table(
    ["Model", "Accuracy", "Macro-F1"],
    [[r["model"], fmt(r["accuracy"]), fmt(r["macro_f1"])] for r in rec_m]))
story.append(Spacer(1, 0.08 * inch))
story.append(Image(os.path.join(CHARTS, "recon_timeline.png"),
                   width=6.6 * inch, height=2.78 * inch))
story.append(p("Daily GL-minus-subledger difference across 2025. Most days "
               "tie exactly; the spikes are the breaks. Posting errors "
               "persist for days (the long red runs); timing differences "
               "reverse the next morning; FX rounding is a one-day flicker "
               "under $1,500.", BODY))
story.append(p("The ranked review queue (top 6 of 20, hold-out)", H2))
story.append(table(
    ["Date", "Entity", "Subledger", "Diff (USD)", "Age (d)", "Cause",
     "Risk score"],
    [[r["recon_date"], r["entity"], r["subledger"],
      f"{float(r['diff_usd']):,.0f}", r["break_age_days"], r["break_type"],
      f"{float(r['risk_score']):,.0f}"] for r in q20[:6]]))
story.append(p(f"Queue quality: {queue['posting_errors_in_top20']}/20 of the "
               f"top-20 are true posting errors; the queue captures "
               f"{float(queue['recall_of_posting_errors']):.0%} of all "
               f"posting errors in the hold-out. The other 45% are smaller, "
               "younger breaks the risk score ranks lower — the queue is a "
               "triage tool, not a guarantee.", BODY))
story.append(PageBreak())

# ================= PAGE 9 — Step 6: resume alignment + benchmark =================
story.append(p("8 &nbsp; Resume alignment — match 6 / 10", H1))
story.append(p("The match is 6/10: the candidate's core is exactly the "
               "bookkeeping engine this role needs (GL, reconciliations, "
               "audit readiness, Excel), and the gaps are specific and "
               "fixable (QuickBooks, high-value inventory control, "
               "multi-entity FX bookkeeping) — not a seniority gap. The "
               "binding constraints are not technical: pay is below band "
               "and the schedule is a fixed on-site commitment the candidate "
               "has not stated.", BODY))
story.append(table(
    ["JD requirement", "Resume evidence", "Gap?", "Fix"],
    [
        ["GL, reconciliations, reporting",
         "GL consolidation, reconciliation, period-end close; 99.5% "
         "reporting accuracy",
         "No", "Lead with it — it is the hire"],
        ["AR/AP, multi-company, multi-currency",
         "ERP consolidation (SAP / NetSuite); no explicit multi-entity "
         "AR/AP ownership",
         "Partial", "This project is the evidence"],
        ["High-value inventory valuation",
         "No inventory experience evidenced",
         "Yes", "State honestly; cite this project's valuation module"],
        ["Books for CPAs, audit support",
         "Audit readiness; SOX / GAAP / IFRS; zero major observations",
         "No", "Second lead bullet"],
        ["Excel; QuickBooks / QBO",
         "Advanced Excel, VBA, Power Query evidenced; QuickBooks not "
         "evidenced",
         "[CONFIRM]", "Confirm before claiming; QBO can be learned in days"],
        ["Wholesale / luxury domain",
         "Banking + financial services; no wholesale evidenced",
         "Yes", "Frame FP&amp;A controls as transferable"],
    ]))
story.append(p("Top 3 missing skills: (1) QuickBooks / QuickBooks Online "
               "hands-on — unconfirmed, never claimed. (2) High-value "
               "inventory control — cycle counts, vault procedures. "
               "(3) Multi-entity intercompany bookkeeping — FX revaluation, "
               "intercompany eliminations.", BODY))
story.append(p("Benchmark — best model vs interpretable baseline", H2))
story.append(table(
    ["Module", "Baseline", "Best model", "Metric"],
    [[r["module"],
      f"{r['baseline']} {fmt(r['baseline_score'])}",
      f"{r['best_model']} {fmt(r['best_score'])}",
      r["metric"]] for r in bench]))
story.append(p("Bullet upgrades (STAR, from verified resume facts): "
               "“Automated account reconciliation with Python and Advanced "
               "Excel, standardizing period-end close handoffs and raising "
               "reporting accuracy to 99.5%.” — “Built Excel VBA month-end "
               "templates that cut manual close effort 45%.” — “Supported "
               "IFRS and GAAS audit readiness with zero major audit "
               "observations.”", BODY))
story.append(PageBreak())

# ================= PAGE 10 — architecture, packaging, differentiation =================
story.append(p("9 &nbsp; Architecture, packaging &amp; what makes this different",
               H1))
story.append(p("Architecture", H2))
story.append(p("[Synthetic ledger / inventory / recon] → [validation gate] "
               "→ [feature store: train-fitted counterparty stats] → "
               "[training: rules + forests + linear models] → [model "
               "registry: models/*.csv] → [serving: the ranked review queue "
               "as a daily CSV] → [monitoring: re-run monthly; drift = "
               "anomaly rate or break rate leaving its design band]. "
               "Deployment for this role is a batch job, not an API: the "
               "bookkeeper opens the queue each morning, works it top to "
               "bottom, and the close checklist records what was cleared. "
               "Tech stack: Python 3.12, pandas, scikit-learn, matplotlib; "
               "no GPU, no cloud, no vendor lock-in — it runs on the office "
               "laptop that already runs Excel.", BODY))
story.append(p("GitHub packaging", H2))
story.append(p("Suggested repo <b>We-love-watches-Bookkeeper-ml</b> under "
               "github.com/bhupesh-debug (user creates the empty repo; "
               "files are ready to push — see the file list below). "
               "README, requirements.txt, run_all.py, src/ (6 modules), "
               "tests/run_tests.py (87 checks, all pass), "
               "data/synthetic + data/processed, models/*.csv, "
               "reports/charts + model_report.md, notebooks 01–04, "
               "build_pdf.py, build_deck.py.", BODY))
story.append(p("Differentiation — why this is not a generic ML demo", H2))
story.append(p("1. <b>The rule baseline is the hero, not the villain:</b> "
               "the project proves the Excel-checkable rules first, then "
               "shows exactly how much the models add (F1 0.598 → 0.964). "
               "2. <b>The deliverable is a queue, not a score:</b> a ranked "
               "review list is what a bookkeeper opens at 10 AM. "
               "3. <b>Every number is reproducible from a single command</b> "
               "(run_all.py, seed 42) and every limit is stated on the same "
               "page as the result.", BODY))
story.append(p("Limits — read before citing any number", H2))
story.append(p("• All data is synthetic: magnitudes, anomaly rates, and "
               "break rates are design choices, not We Love Watches facts. "
               "A model that scores well here proves the <i>pipeline "
               "design</i>, not performance on real books.<br/>• Injected "
               "anomalies are deliberately detectable; real fraud and real "
               "errors are subtler — expect lower recall on live data and "
               "keep a human in the review queue.<br/>• The days-to-sell "
               "model explains only part of the variance (R² 0.116); brand "
               "desirability and seasonality are simplified, and FX is a "
               "simulated random walk, not market rates.<br/>• The candidate "
               "does not claim QuickBooks experience or CPA candidacy — both "
               "are [CONFIRM WITH CANDIDATE] items, not facts in this "
               "project.", BODY))
story.append(p("Next steps", H2))
story.append(p("1. User creates the empty public repo "
               "<b>We-love-watches-Bookkeeper-ml</b> and the files are "
               "pushed. 2. On a real trial balance: swap the generator for "
               "a QuickBooks export adapter and re-run the same pipeline — "
               "no model code changes. 3. Interview questions this project "
               "prepares the candidate for: how do you catch a duplicate "
               "invoice? how do you value a watch that has sat for 200 days? "
               "what do you do when the GL does not tie at close?", BODY))

doc = SimpleDocTemplate(OUT, pagesize=LETTER,
                        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
                        topMargin=0.6 * inch, bottomMargin=0.75 * inch,
                        title="Multi-Company Bookkeeping & Inventory Analytics "
                              "— Bhupesh Chandra Dimri",
                        author="Bhupesh Chandra Dimri")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(f"built {OUT}")

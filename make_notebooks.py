"""Generate the four walkthrough notebooks (01 EDA, 02 features,
03 modeling, 04 evaluation) as .ipynb JSON.

Run from the project root AFTER `python run_all.py`:
    ~/workspace/quant-ipo-engine/.venv/bin/python make_notebooks.py
"""

import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
NB = os.path.join(ROOT, "notebooks")


def nb(cells):
    return {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"}},
        "nbformat": 4, "nbformat_minor": 5}


def md(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": text.splitlines(True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.splitlines(True)}


def write(name, cells):
    with open(os.path.join(NB, name), "w") as fh:
        json.dump(nb(cells), fh, indent=1)
    print(f"  notebook: {name}")


os.makedirs(NB, exist_ok=True)

write("01_eda.ipynb", [
    md("# 01 — Exploratory Data Analysis\n\nAll data is synthetic. "
       "Ledger: 4,492 AR/AP transactions across WLW-US / WLW-EU / WLW-ASIA. "
       "Inventory: 150 watch SKUs. Reconciliation: daily GL vs subledger "
       "for 3 entities x 3 subledgers."),
    code("import pandas as pd\n"
         "led = pd.read_csv('data/processed/ledger.csv', parse_dates=['txn_date'])\n"
         "inv = pd.read_csv('data/processed/inventory_monthly.csv', parse_dates=['month_end'])\n"
         "rec = pd.read_csv('data/processed/recon_daily.csv', parse_dates=['recon_date'])\n"
         "print(led.shape, inv.shape, rec.shape)"),
    code("led.groupby(['entity', 'txn_type']).agg(n=('txn_id', 'count'),\n"
         "    total_usd=('amount_usd', 'sum'), median_usd=('amount_usd', 'median'))"),
    code("led['is_anomaly'].mean(), led.groupby('anomaly_type').size()"),
    code("inv.groupby('month_end')['total_value_usd'].sum().plot(title='Inventory value by month')"),
    code("rec.groupby('break_type').size()"),
])

write("02_feature_engineering.ipynb", [
    md("# 02 — Feature Engineering\n\nCounterparty statistics are fitted on "
       "the TRAIN period only (no leakage). Key features: log amount, "
       "counterparty z-score, same-amount-in-trailing-7-days (the duplicate "
       "check), days since last transaction with the same counterparty, "
       "weekend flag, supporting-document flag, round-number flag, FX "
       "deviation, overdue days."),
    code("import sys; sys.path.insert(0, 'src')\n"
         "import pandas as pd, feature_engineering as fe\n"
         "led = pd.read_csv('data/processed/ledger.csv', parse_dates=['txn_date', 'due_date', 'paid_date'])\n"
         "train = led[led['txn_date'] < '2025-11-01']\n"
         "stats, gmed, gstd = fe.fit_counterparty_stats(train)\n"
         "print(stats.head())\n"
         "f = fe.build_ledger_features(train, stats, gmed, gstd)\n"
         "print(f.shape)\n"
         "print(f.columns.tolist())"),
    code("f.groupby('is_anomaly')[['log_amount', 'amount_z_vs_counterparty', 'has_doc', 'is_round_number']].mean()"),
])

write("03_modeling.ipynb", [
    md("# 03 — Modeling\n\nModule 1: rule baseline vs Isolation Forest vs "
       "Logistic Regression vs Random Forest (classification).\n"
       "Module 2: median vs Linear Regression vs Random Forest on "
       "log(days-to-sell) (regression, grouped random split by SKU).\n"
       "Module 3: majority-class vs Random Forest for break cause, plus a "
       "risk-scored review queue."),
    code("import pandas as pd\n"
         "print(pd.read_csv('models/anomaly_metrics.csv').to_string(index=False))\n"
         "print(pd.read_csv('models/inventory_metrics.csv').to_string(index=False))\n"
         "print(pd.read_csv('models/recon_metrics.csv').to_string(index=False))\n"
         "print(pd.read_csv('models/benchmark.csv').to_string(index=False))"),
    code("print(pd.read_csv('models/review_queue_top20.csv').to_string(index=False))"),
])

write("04_evaluation.ipynb", [
    md("# 04 — Evaluation & Limits\n\nHold-out: modules 1 & 3 use the last "
       "61 days of 2025; module 2 uses a grouped random split by SKU. "
       "Headline: anomaly RF F1 0.964 (rule 0.598); days-to-sell LR MAPE "
       "40.0% (median 44.8%); break-cause RF accuracy 0.976 (majority "
       "0.878); review queue top-20 is 20/20 posting errors.\n\n"
       "Limits: synthetic data proves pipeline design, not live-book "
       "performance. Injected anomalies are deliberately detectable; real "
       "errors are subtler. Keep a human in the review queue."),
    code("import pandas as pd\n"
         "print(open('reports/model_report.md').read())"),
])
print("DONE")

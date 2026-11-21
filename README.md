# Multi-Company Bookkeeping & Inventory Analytics

**ML portfolio project for the Full-Time Bookkeeper role at We Love Watches**
(Midtown Manhattan — high-end watches wholesale, domestic + international)

Author: Bhupesh Chandra Dimri · November 2026

**GitHub repo (created by the user, ready to push):**
https://github.com/bhupesh-debug/We-love-watches-Bookkeeper-ml.git

> **ALL DATA IS SYNTHETIC.** No real We Love Watches transactions, customers,
> suppliers, or inventory are used anywhere. The generator mimics the
> structure of the job description (multi-company AR/AP, high-value watch
> inventory, GL vs subledger reconciliation) so the models can be built and
> tested honestly. Magnitudes, anomaly rates, and break rates are design
> choices, not company facts.

## The problem (from the JD)

A bookkeeper at a luxury-watch wholesaler juggles three ledgers at once:
AR/AP across three companies and three currencies, a vault of high-value
inventory whose valuation has to be right to the dollar, and a general
ledger that must tie to every subledger before the CPA sees the books.
This project builds the three checks a bookkeeper runs every close —
as reproducible, testable code.

## Three modules

| # | Module | What it does | Model |
|---|--------|--------------|-------|
| 1 | AR/AP anomaly detection | Flags duplicate, unsupported, and out-of-pattern transactions across entities/currencies | Rule baseline vs Isolation Forest vs Logistic Regression vs Random Forest |
| 2 | Inventory valuation tracking | Moving-average cost, aging buckets, shrinkage flags; predicts days-to-sell | Median baseline vs Linear Regression vs Random Forest |
| 3 | Reconciliation-break detector | GL vs subledger breaks with a ranked review queue | Rule baseline vs Random Forest (break-cause classification) |

## How to run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_all.py          # generate -> validate -> features -> models -> charts
python tests/run_tests.py  # test suite (exit 0 = all pass)
python build_pdf.py        # 10-page project report -> ~/workspace/your_files/
python build_deck.py       # 10-slide deck PDF -> ~/workspace/your_files/
```

Deterministic: seed 42 everywhere. Modules 1 and 3 use a time-based
hold-out (the last 61 days of 2025, 2025-11-01 to 2025-12-31) — nothing
from the hold-out is used to fit features, scalers, or models. Module 2
(days-to-sell) uses a grouped random 70/30 split by SKU: a time split
there would test calendar extrapolation instead of the pricing/brand
signal, because November–December sales are disproportionately long-tail
units acquired months earlier.

## Results (hold-out)

See `reports/model_report.md` for the full numbers. Headline results from
the run that produced this README:

- **Module 1:** Random Forest beats the rule baseline on F1 (see
  `models/anomaly_metrics.csv`).
- **Module 2:** Linear Regression on log days-to-sell beats the median
  baseline on MAPE (see `models/inventory_metrics.csv`).
- **Module 3:** Random Forest beats the majority-class baseline on
  break-cause accuracy; the top-20 ranked review queue concentrates
  the real posting errors (see `models/review_queue.csv`).

## Project structure

```
ml-project-we-love-watches-bookkeeper/
├── README.md
├── requirements.txt
├── run_all.py                 # orchestrator
├── build_pdf.py               # 10-page report builder
├── build_deck.py              # 10-slide deck builder (PDF)
├── make_notebooks.py          # generates notebooks/01-04
├── src/
│   ├── data_generation.py   # synthetic ledger / inventory / recon
│   ├── data_pipeline.py     # validation + cleaning
│   ├── feature_engineering.py
│   ├── anomaly.py           # rule-based baselines
│   ├── models.py            # training + evaluation
│   └── charts.py            # matplotlib figures
├── tests/run_tests.py
├── data/synthetic/          # generated CSVs (regenerate with run_all.py)
├── data/processed/          # validated copies
├── models/*.csv             # every metric the report cites
├── reports/charts/*.png
├── reports/model_report.md
└── notebooks/01_eda.ipynb … 04_evaluation.ipynb
```

## Limits — read before citing any number

- Synthetic data proves the **pipeline design**, not performance on real
  books. Injected anomalies are deliberately detectable; real fraud and
  real errors are subtler — expect lower recall on live data and keep a
  human in the review queue.
- The days-to-sell model explains only part of the variance; brand
  desirability and seasonality are simplified.
- FX is a simulated random walk, not market rates.
- The candidate does **not** claim QuickBooks experience or CPA
  candidacy — both are [CONFIRM WITH CANDIDATE] items for the role,
  not facts in this repo.

"""Test suite for the We Love Watches Bookkeeper ML project.

Run from the project root AFTER `python run_all.py`:
    ~/workspace/quant-ipo-engine/.venv/bin/python tests/run_tests.py

Every check is a real assertion on the generated artefacts — file
existence, schema, row counts, hold-out design, metric sanity, and
determinism of the headline numbers. Exit code 0 = all pass.
"""

import csv
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

PASSED = 0
FAILED = []


def check(name, cond, detail=""):
    global PASSED
    if cond:
        PASSED += 1
        print(f"  PASS {name}")
    else:
        FAILED.append(name)
        print(f"  FAIL {name} {detail}")


def read_csv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


print("== files exist ==")
for rel in ["data/synthetic/ledger.csv",
            "data/synthetic/inventory_monthly.csv",
            "data/synthetic/inventory_sales.csv",
            "data/synthetic/inventory_skus.csv",
            "data/synthetic/shrinkage_events.csv",
            "data/synthetic/recon_daily.csv",
            "data/processed/ledger.csv",
            "data/processed/inventory_monthly.csv",
            "data/processed/inventory_sales.csv",
            "data/processed/recon_daily.csv",
            "models/anomaly_metrics.csv", "models/inventory_metrics.csv",
            "models/recon_metrics.csv", "models/benchmark.csv",
            "models/review_queue.csv", "models/review_queue_top20.csv",
            "models/anomaly_holdout_scored.csv",
            "models/anomaly_feature_importance.csv",
            "models/inventory_feature_importance.csv",
            "models/inventory_split.csv",
            "models/recon_feature_importance.csv",
            "reports/model_report.md", "README.md",
            "requirements.txt", "run_all.py"]:
    check(f"exists {rel}", os.path.exists(os.path.join(ROOT, rel)))
for chart in ["anomaly_comparison.png", "feature_importance.png",
              "inventory_valuation.png", "recon_timeline.png",
              "roc_curves.png", "correlation_heatmap.png"]:
    check(f"exists reports/charts/{chart}",
          os.path.exists(os.path.join(ROOT, "reports", "charts", chart)))

print("== ledger schema + design ==")
led = pd.read_csv(os.path.join(ROOT, "data", "processed", "ledger.csv"),
                  parse_dates=["txn_date"])
check("ledger rows > 4000", len(led) > 4000, f"got {len(led)}")
check("ledger 3 entities", led["entity"].nunique() == 3)
check("ledger 3 currencies", led["currency"].nunique() == 3)
check("ledger anomaly rate 3-8%",
      0.03 <= led["is_anomaly"].mean() <= 0.08,
      f"{led['is_anomaly'].mean():.2%}")
check("ledger 3 anomaly types",
      set(led.loc[led["is_anomaly"] == 1, "anomaly_type"].unique())
      == {"duplicate", "unsupported", "out_of_pattern"})
check("ledger txn_id unique", led["txn_id"].is_unique)
check("ledger amounts positive", (led["amount_usd"] > 0).all())
check("ledger dates in 2025",
      led["txn_date"].between("2025-01-01", "2025-12-31").all())
hold = led[led["txn_date"] >= "2025-11-01"]
check("ledger hold-out is 61 days",
      hold["txn_date"].nunique() == 61, f"{hold['txn_date'].nunique()}")
check("ledger hold-out has anomalies", int(hold["is_anomaly"].sum()) > 0)

print("== inventory schema + design ==")
inv = pd.read_csv(os.path.join(ROOT, "data", "processed",
                               "inventory_monthly.csv"),
                  parse_dates=["month_end"])
check("inventory 150 SKUs", inv["sku"].nunique() == 150)
check("inventory 12 months x 150 SKUs", len(inv) == 1800, f"got {len(inv)}")
check("inventory shrinkage flags = 8",
      int(inv["shrinkage_flag"].sum()) == 8)
check("inventory value positive", (inv["total_value_usd"] >= 0).all())
sales = pd.read_csv(os.path.join(ROOT, "data", "processed",
                                "inventory_sales.csv"),
                    parse_dates=["sale_date"])
check("sales rows > 200", len(sales) > 200, f"got {len(sales)}")
check("sales span >= 11 months (Jan has no sales: acquisitions start "
      "Jan 1 and median days-to-sell is ~66d)",
      sales["sale_date"].dt.month.nunique() >= 11)
check("sales price above cost",
      (sales["sale_price_usd"] > sales["unit_cost_usd"]).all())
split = pd.read_csv(os.path.join(ROOT, "models", "inventory_split.csv"))
check("inventory split file exists and covers all sales",
      len(split) == len(sales))
check("inventory split: no SKU in both train and hold-out",
      len(set(split.loc[split["split"] == "train", "sku"])
          & set(split.loc[split["split"] == "holdout", "sku"])) == 0)
check("inventory split: hold-out share 25-35%",
      0.25 <= (split["split"] == "holdout").mean() <= 0.35,
      f"{(split['split'] == 'holdout').mean():.2%}")

print("== recon schema + design ==")
rec = pd.read_csv(os.path.join(ROOT, "data", "processed", "recon_daily.csv"),
                  parse_dates=["recon_date"])
check("recon rows = 3 entities x 3 subledgers x 365 days",
      len(rec) == 3 * 3 * 365, f"got {len(rec)}")
check("recon break rate 3-12%",
      0.03 <= rec["has_break"].mean() <= 0.12,
      f"{rec['has_break'].mean():.2%}")
check("recon GL - subledger == diff",
      ((rec["gl_balance_usd"] - rec["subledger_total_usd"])
       - rec["diff_usd"]).abs().max() < 0.02)
check("recon 3 break causes",
      set(rec.loc[rec["has_break"] == 1, "break_type"].unique())
      == {"timing", "posting_error", "fx_rounding"})

print("== model metrics sanity ==")
anom = {r["model"]: r for r in read_csv(
    os.path.join(ROOT, "models", "anomaly_metrics.csv"))}
check("anomaly: 4 models evaluated", len(anom) == 4)
for name, row in anom.items():
    for m in ["precision", "recall", "f1", "roc_auc"]:
        check(f"anomaly {name} {m} in [0,1]",
              0.0 <= float(row[m]) <= 1.0, f"{m}={row[m]}")
rf = anom["Random Forest"]
rule = anom["Rule baseline (duplicate / no-doc / |z|>3)"]
check("anomaly RF F1 beats rule baseline",
      float(rf["f1"]) > float(rule["f1"]),
      f"RF {rf['f1']} vs rule {rule['f1']}")
check("anomaly RF recall >= 0.70", float(rf["recall"]) >= 0.70,
      f"recall={rf['recall']}")
check("anomaly RF ROC-AUC >= 0.85", float(rf["roc_auc"]) >= 0.85,
      f"auc={rf['roc_auc']}")

inv_m = {r["model"]: r for r in read_csv(
    os.path.join(ROOT, "models", "inventory_metrics.csv"))}
check("inventory: 3 models evaluated", len(inv_m) == 3)
check("inventory RF MAPE < median baseline",
      float(inv_m["Random Forest (log target)"]["mape"])
      < float(inv_m["Median baseline"]["mape"]))
check("inventory RF MAPE < 60%",
      float(inv_m["Random Forest (log target)"]["mape"]) < 0.60,
      f"mape={inv_m['Random Forest (log target)']['mape']}")
check("inventory RF R2 > 0",
      float(inv_m["Random Forest (log target)"]["r2"]) > 0,
      f"r2={inv_m['Random Forest (log target)']['r2']}")

rec_m = {r["model"]: r for r in read_csv(
    os.path.join(ROOT, "models", "recon_metrics.csv"))}
check("recon: RF accuracy > majority baseline",
      float(rec_m["Random Forest"]["accuracy"])
      > float(rec_m["Majority-class baseline"]["accuracy"]))
check("recon RF accuracy >= 0.60",
      float(rec_m["Random Forest"]["accuracy"]) >= 0.60,
      f"acc={rec_m['Random Forest']['accuracy']}")

q = read_csv(os.path.join(ROOT, "models", "review_queue.csv"))[0]
check("review queue top-20 has >= 5 posting errors",
      int(q["posting_errors_in_top20"]) >= 5,
      f"{q['posting_errors_in_top20']}")
check("review queue precision@20 >= 0.25",
      float(q["precision_at_20"]) >= 0.25)
q20 = read_csv(os.path.join(ROOT, "models", "review_queue_top20.csv"))
check("review queue file has 20 rows", len(q20) == 20)
check("review queue sorted by risk desc",
      [float(r["risk_score"]) for r in q20]
      == sorted([float(r["risk_score"]) for r in q20], reverse=True))

print("== determinism spot-check ==")
scored = pd.read_csv(os.path.join(ROOT, "models",
                                  "anomaly_holdout_scored.csv"))
check("hold-out scored rows match hold-out ledger rows",
      len(scored) == len(hold), f"{len(scored)} vs {len(hold)}")
check("hold-out scored labels match ledger labels (as a multiset)",
      sorted(scored["is_anomaly"].tolist())
      == sorted(hold["is_anomaly"].tolist()))

print()
print(f"{PASSED} passed, {len(FAILED)} failed")
if FAILED:
    print("FAILED:", FAILED)
    sys.exit(1)
print("ALL TESTS PASSED")

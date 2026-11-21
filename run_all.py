"""Orchestrator: generate -> validate -> features -> models -> metrics.

Run from the project root:
    ~/workspace/quant-ipo-engine/.venv/bin/python run_all.py

Deterministic: seed 42 everywhere. Hold-out is the last 61 days
(2025-11-01 to 2025-12-31) for every module. Every metric in the PDF
and deck comes from models/*.csv written here — nothing is hard-coded.
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

import anomaly as an  # noqa: E402
import charts as ch  # noqa: E402
import data_generation as dg  # noqa: E402
import data_pipeline as dp  # noqa: E402
import feature_engineering as fe  # noqa: E402
import models as md  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

MODELS = os.path.join(ROOT, "models")
CHARTS = os.path.join(ROOT, "reports", "charts")
PROC = os.path.join(ROOT, "data", "processed")
HOLDOUT_START = pd.Timestamp("2025-11-01")

os.makedirs(MODELS, exist_ok=True)
os.makedirs(CHARTS, exist_ok=True)


def main():
    rng = np.random.default_rng(42)
    print("== 1. generate synthetic data ==")
    dg.main()
    print("== 2. validate + clean ==")
    dp.main()

    ledger = pd.read_csv(os.path.join(PROC, "ledger.csv"),
                         parse_dates=["txn_date", "due_date", "paid_date"])
    inv = pd.read_csv(os.path.join(PROC, "inventory_monthly.csv"),
                      parse_dates=["month_end"])
    sales = pd.read_csv(os.path.join(PROC, "inventory_sales.csv"),
                        parse_dates=["acquisition_date", "sale_date"])
    recon = pd.read_csv(os.path.join(PROC, "recon_daily.csv"),
                        parse_dates=["recon_date"])

    # ---------------- Module 1: AR/AP anomaly detection ----------------
    print("== 3. module 1 — AR/AP anomaly detection ==")
    train_raw = ledger[ledger["txn_date"] < HOLDOUT_START].copy()
    test_raw = ledger[ledger["txn_date"] >= HOLDOUT_START].copy()
    # lag + same-amount features on the full timeline (date/amount
    # lookups only, no label) so a hold-out row can see its October
    # predecessor — exactly what a bookkeeper checking the ledger does
    ledger_sorted = ledger.sort_values(["counterparty", "txn_date"]).copy()
    ledger_sorted["prev_txn_date"] = ledger_sorted.groupby(
        "counterparty")["txn_date"].shift(1)
    ledger_sorted["days_since_last_same_counterparty"] = (
        (ledger_sorted["txn_date"] - ledger_sorted["prev_txn_date"])
        .dt.days.fillna(365).clip(0, 365))
    lag = ledger_sorted.set_index("txn_id")["days_since_last_same_counterparty"]
    train_raw["days_since_last_same_counterparty"] = train_raw["txn_id"].map(lag)
    test_raw["days_since_last_same_counterparty"] = test_raw["txn_id"].map(lag)

    # same counterparty + same amount in the trailing 7 days (the
    # duplicate check, as a feature). Backward-looking only.
    amt_sorted = ledger.sort_values(
        ["counterparty", "amount_usd", "txn_date"]).copy()
    amt_sorted["prev_same_amt_date"] = amt_sorted.groupby(
        ["counterparty", "amount_usd"])["txn_date"].shift(1)
    gap = (amt_sorted["txn_date"] - amt_sorted["prev_same_amt_date"]).dt.days
    amt_sorted["same_amount_recent"] = ((gap >= 0) & (gap <= 7)).astype(int)
    same_amt = amt_sorted.set_index("txn_id")["same_amount_recent"]
    train_raw["same_amount_recent"] = train_raw["txn_id"].map(same_amt)
    test_raw["same_amount_recent"] = test_raw["txn_id"].map(same_amt)

    cp_stats, gmed, gstd = fe.fit_counterparty_stats(train_raw)
    train_f = fe.build_ledger_features(train_raw, cp_stats, gmed, gstd)
    test_f = fe.build_ledger_features(test_raw, cp_stats, gmed, gstd)
    # align dummy columns between train and test
    for col in train_f.columns:
        if col not in test_f.columns and col not in ("is_anomaly",):
            test_f[col] = 0
    for col in test_f.columns:
        if col not in train_f.columns and col not in ("is_anomaly",):
            train_f[col] = 0
    feat_cols = [c for c in train_f.columns if c != "is_anomaly"]
    X_train, y_train = train_f[feat_cols], train_f["is_anomaly"]
    X_test, y_test = test_f[feat_cols], test_f["is_anomaly"]
    # Rule flags are computed on the FULL ledger, then sliced: a
    # bookkeeper checking for a duplicate looks at the whole book, not
    # just November–December. Align to test_f's index (features are
    # sorted by counterparty/date inside build_ledger_features).
    rule_full = an.ledger_rule_flags(ledger)
    rule_test = rule_full.loc[test_f.index]
    rule_train = rule_full.loc[train_f.index]
    print(f"  train {len(X_train):,} rows ({int(y_train.sum())} anomalies), "
          f"hold-out {len(X_test):,} rows ({int(y_test.sum())} anomalies)")
    m1, imp1, rf1 = md.anomaly_models(X_train, y_train, X_test, y_test,
                                      rule_test)
    pd.DataFrame(m1).to_csv(os.path.join(MODELS, "anomaly_metrics.csv"),
                            index=False)
    imp1.rename_axis("feature").reset_index().rename(
        columns={0: "importance"}).to_csv(
        os.path.join(MODELS, "anomaly_feature_importance.csv"), index=False)
    # hold-out scores for the ROC chart + a naive flag-everything baseline
    test_scored = test_f[feat_cols].copy()
    test_scored["is_anomaly"] = y_test.values
    test_scored["rf_score"] = rf1.predict_proba(X_test)[:, 1]
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X_train)
    lr = LogisticRegression(max_iter=1000, class_weight="balanced",
                            random_state=42).fit(sc.transform(X_train), y_train)
    test_scored["lr_score"] = lr.predict_proba(sc.transform(X_test))[:, 1]
    from sklearn.ensemble import IsolationForest
    iso = IsolationForest(n_estimators=300, contamination=0.05,
                          random_state=42).fit(sc.transform(X_train))
    test_scored["iso_score"] = -iso.decision_function(sc.transform(X_test))
    test_scored.to_csv(os.path.join(MODELS, "anomaly_holdout_scored.csv"),
                       index=False)
    for r in m1:
        print(f"  {r['model']}: P={r['precision']:.3f} R={r['recall']:.3f} "
              f"F1={r['f1']:.3f} AUC={r['roc_auc']:.3f}")

    # ---------------- Module 2: inventory days-to-sell ----------------
    # Split: GROUPED RANDOM by SKU (GroupShuffleSplit, 70/30), not
    # time-based. Reason: a time split makes the evaluation a calendar
    # extrapolation test, not a pricing/brand test — Nov-Dec sales are
    # disproportionately long-tail units acquired months earlier, and
    # acquisition-month barely overlaps between the halves. The
    # regression question here is per-unit ("given brand + cost, how long
    # to sell?"), an independent-observation question, so a grouped
    # random split is the right evaluation. Modules 1 and 3 keep the
    # time-based hold-out.
    print("== 4. module 2 — inventory days-to-sell ==")
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    tr_idx, te_idx = next(gss.split(sales, groups=sales["sku"]))
    s_train = sales.iloc[tr_idx].copy()
    s_test = sales.iloc[te_idx].copy()
    split = pd.Series("train", index=sales.index)
    split.iloc[te_idx] = "holdout"
    pd.DataFrame({"sku": sales["sku"], "split": split.values}).to_csv(
        os.path.join(MODELS, "inventory_split.csv"), index=False)
    f_train = fe.build_inventory_features(s_train)
    f_test = fe.build_inventory_features(s_test)
    for col in f_train.columns:
        if col not in f_test.columns and col != "days_to_sell":
            f_test[col] = 0
    inv_cols = ["log_unit_cost", "markup", "acq_month"] + [
        c for c in f_train.columns if c.startswith(("brand_", "ent_"))]
    m2, imp2 = md.inventory_models(f_train[inv_cols], f_train["days_to_sell"],
                                   f_test[inv_cols], f_test["days_to_sell"])
    pd.DataFrame(m2).to_csv(os.path.join(MODELS, "inventory_metrics.csv"),
                            index=False)
    imp2.rename_axis("feature").reset_index().rename(
        columns={0: "importance"}).to_csv(
        os.path.join(MODELS, "inventory_feature_importance.csv"), index=False)
    print(f"  train {len(s_train)} sales, hold-out {len(s_test)} sales")
    for r in m2:
        print(f"  {r['model']}: MAPE={r['mape']:.2%} MAE={r['mae']:.1f}d "
              f"R2={r['r2']:.3f}")

    # ---------------- Module 3: reconciliation breaks ----------------
    print("== 5. module 3 — reconciliation breaks ==")
    r_train = recon[recon["recon_date"] < HOLDOUT_START].copy()
    r_test = recon[recon["recon_date"] >= HOLDOUT_START].copy()
    rule_r = an.recon_rule_flags(r_test)
    rule_acc = float((rule_r == r_test["has_break"]).mean())
    # break-cause classification on break rows only
    br_train = r_train[r_train["has_break"] == 1].copy()
    br_test = r_test[r_test["has_break"] == 1].copy()
    f_br_train = fe.build_recon_features(br_train)
    f_br_test = fe.build_recon_features(br_test)
    # abs_diff_log is a legitimate feature here: the bookkeeper sees the
    # size of a break when it lands, and size separates fx rounding
    # (tiny) from posting errors (large). It is not the label.
    drop = ["recon_date", "gl_balance_usd", "subledger_total_usd", "diff_usd",
            "has_break", "break_type", "subledger_weight", "risk_score"]
    feat_r = [c for c in f_br_train.columns
              if c not in drop and c != "break_type"]
    for col in feat_r:
        if col not in f_br_test.columns:
            f_br_test[col] = 0
    m3, imp3 = md.recon_models(f_br_train[feat_r], f_br_train["break_type"],
                               f_br_test[feat_r], f_br_test["break_type"])
    m3.append({"model": "Rule baseline (|diff| > $1,000) — break detection",
               "accuracy": rule_acc, "macro_f1": float("nan")})
    pd.DataFrame(m3).to_csv(os.path.join(MODELS, "recon_metrics.csv"),
                            index=False)
    imp3.rename_axis("feature").reset_index().rename(
        columns={0: "importance"}).to_csv(
        os.path.join(MODELS, "recon_feature_importance.csv"), index=False)
    print(f"  hold-out breaks: {len(br_test)} "
          f"({br_test['break_type'].value_counts().to_dict()})")
    for r in m3:
        print(f"  {r['model']}: acc={r['accuracy']:.3f} "
              f"macro-F1={r['macro_f1']:.3f}")
    # ranked review queue on hold-out breaks
    q = md.review_queue_metrics(r_test[r_test["has_break"] == 1])
    pd.DataFrame([q]).to_csv(os.path.join(MODELS, "review_queue.csv"),
                             index=False)
    print(f"  review queue top-20: {q['posting_errors_in_top20']}/20 are "
          f"posting errors; recall of posting errors "
          f"{q['recall_of_posting_errors']:.2%}")
    # save the ranked queue itself
    queue = (r_test[r_test["has_break"] == 1]
             .sort_values("risk_score", ascending=False)
             .head(20)[["recon_date", "entity", "subledger", "diff_usd",
                        "break_age_days", "break_type", "risk_score"]])
    queue.to_csv(os.path.join(MODELS, "review_queue_top20.csv"), index=False)

    # ---------------- Benchmark table ----------------
    print("== 6. benchmark ==")
    # best model per module is chosen by the metric, not assumed
    best_m1 = max(m1, key=lambda r: r["f1"])
    best_m2 = min(m2, key=lambda r: r["mape"])
    best_m3 = max((r for r in m3 if not r["model"].startswith("Rule")),
                  key=lambda r: r["accuracy"])
    bench = pd.DataFrame([
        {"module": "AR/AP anomaly", "baseline": "Rule baseline",
         "baseline_score": next(r["f1"] for r in m1
                                if r["model"].startswith("Rule")),
         "best_model": best_m1["model"],
         "best_score": best_m1["f1"],
         "metric": "F1"},
        {"module": "Days-to-sell", "baseline": "Median baseline",
         "baseline_score": next(r["mape"] for r in m2
                                if r["model"] == "Median baseline"),
         "best_model": best_m2["model"],
         "best_score": best_m2["mape"],
         "metric": "MAPE (lower is better)"},
        {"module": "Break-cause", "baseline": "Majority-class baseline",
         "baseline_score": next(r["accuracy"] for r in m3
                                if r["model"].startswith("Majority")),
         "best_model": best_m3["model"],
         "best_score": best_m3["accuracy"],
         "metric": "Accuracy"},
    ])
    bench.to_csv(os.path.join(MODELS, "benchmark.csv"), index=False)
    print(bench.to_string(index=False))

    # ---------------- Charts ----------------
    print("== 7. charts ==")
    ch.anomaly_comparison(os.path.join(MODELS, "anomaly_metrics.csv"),
                          os.path.join(CHARTS, "anomaly_comparison.png"))
    ch.feature_importance(
        os.path.join(MODELS, "anomaly_feature_importance.csv"),
        os.path.join(CHARTS, "feature_importance.png"),
        "Top drivers of AR/AP anomaly flags (Random Forest)")
    ch.inventory_valuation(os.path.join(PROC, "inventory_monthly.csv"),
                           os.path.join(CHARTS, "inventory_valuation.png"))
    ch.recon_timeline(os.path.join(PROC, "recon_daily.csv"),
                      os.path.join(CHARTS, "recon_timeline.png"))
    ch.roc_curves(os.path.join(MODELS, "anomaly_holdout_scored.csv"),
                  os.path.join(CHARTS, "roc_curves.png"))
    ch.correlation_heatmap(os.path.join(MODELS, "anomaly_holdout_scored.csv"),
                           os.path.join(CHARTS, "correlation_heatmap.png"))

    # ---------------- Model report ----------------
    print("== 8. model report ==")
    rf_row = next(r for r in m1 if r["model"] == "Random Forest")
    rf_inv = next(r for r in m2 if r["model"] == "Random Forest (log target)")
    rf_rec = next(r for r in m3 if r["model"] == "Random Forest")

    def _f(x, pct=False):
        if isinstance(x, float) and np.isnan(x):
            return "—"
        return f"{x:.2%}" if pct else f"{x:.3f}"
    shrink = pd.read_csv(os.path.join(ROOT, "data", "synthetic",
                                      "shrinkage_events.csv"))
    report = f"""# Multi-Company Bookkeeping & Inventory Analytics — Model Report

**Role:** Full-Time Bookkeeper, We Love Watches (Midtown Manhattan)
**Author:** Bhupesh Chandra Dimri — **All data is synthetic.**
**Hold-out:** Modules 1 & 3 — last 61 days of 2025 (2025-11-01 to
2025-12-31), time-based, no leakage. Module 2 — grouped random 70/30
split by SKU (see the Module 2 note below for why).
**Tests:** see `tests/run_tests.py` (run after this script).

## Module 1 — AR/AP anomaly detection (hold-out: {len(X_test):,} txns, {int(y_test.sum())} true anomalies)
| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
"""
    for r in m1:
        report += (f"| {r['model']} | {_f(r['precision'])} | {_f(r['recall'])} "
                   f"| {_f(r['f1'])} | {_f(r['roc_auc'])} |\n")
    report += f"""
Top drivers: {', '.join(imp1.head(5).index.tolist())}.

## Module 2 — Inventory days-to-sell (grouped random hold-out: {len(s_test)} sales)
Split is grouped-random by SKU, not time-based: a time split would test
calendar extrapolation (Nov-Dec sales are disproportionately long-tail
units acquired months earlier) instead of the pricing/brand signal.
Regressors are fit on log(days) and exponentiated; metrics in days.
| Model | MAPE | MAE (days) | R² |
|---|---|---|---|
"""
    for r in m2:
        report += (f"| {r['model']} | {r['mape']:.2%} | {r['mae']:.1f} "
                   f"| {r['r2']:.3f} |\n")
    report += f"""
Shrinkage detected: {len(shrink)} quarter-end count shortfalls,
total loss ${shrink['loss_usd'].sum():,.0f} (flagged by rule, not by model).
Inventory value at Dec 2025: ${inv[inv['month_end'] == inv['month_end'].max()]['total_value_usd'].sum():,.0f}
across {inv['sku'].nunique()} SKUs; 180d+ aging bucket holds
${inv[(inv['month_end'] == inv['month_end'].max()) & (inv['aging_bucket'] == '180d+')]['total_value_usd'].sum():,.0f}.

## Module 3 — Reconciliation breaks (hold-out: {len(br_test)} breaks)
| Model | Accuracy | Macro-F1 |
|---|---|---|
"""
    for r in m3:
        report += f"| {r['model']} | {_f(r['accuracy'])} | {_f(r['macro_f1'])} |\n"
    report += f"""
Rule baseline (|diff| > $1,000) break-detection accuracy on hold-out: {rule_acc:.3f}.
Ranked review queue (top 20 by risk score): {q['posting_errors_in_top20']}/20 are
true posting errors; queue recall of posting errors {q['recall_of_posting_errors']:.2%}.

## Benchmark (best model vs interpretable baseline)
| Module | Baseline | Best model | Metric |
|---|---|---|---|
"""
    for _, b in bench.iterrows():
        report += (f"| {b['module']} | {b['baseline']} {b['baseline_score']:.3f} | "
                   f"{b['best_model']} {b['best_score']:.3f} | {b['metric']} |\n")
    report += """
## Limits (read before citing any number)
- All data is synthetic: magnitudes, anomaly rates, and break rates are
  design choices, not We Love Watches facts. A model that scores well
  here proves the *pipeline design*, not performance on real books.
- Injected anomalies are deliberately detectable (duplicates, missing
  documents, 4–8x amounts). Real fraud and real errors are subtler;
  expect lower recall on live data and keep a human in the review queue.
- The days-to-sell model explains only part of the variance (see R²);
  brand desirability and seasonality are simplified.
- FX is a simulated random walk, not market rates.
"""
    with open(os.path.join(ROOT, "reports", "model_report.md"), "w") as fh:
        fh.write(report)
    print("  reports/model_report.md written")
    print("DONE. Next: python tests/run_tests.py")


if __name__ == "__main__":
    main()

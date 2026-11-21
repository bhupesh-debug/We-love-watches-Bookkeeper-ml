# Multi-Company Bookkeeping & Inventory Analytics — Model Report

**Role:** Full-Time Bookkeeper, We Love Watches (Midtown Manhattan)
**Author:** Bhupesh Chandra Dimri — **All data is synthetic.**
**Hold-out:** Modules 1 & 3 — last 61 days of 2025 (2025-11-01 to
2025-12-31), time-based, no leakage. Module 2 — grouped random 70/30
split by SKU (see the Module 2 note below for why).
**Tests:** see `tests/run_tests.py` (run after this script).

## Module 1 — AR/AP anomaly detection (hold-out: 757 txns, 27 true anomalies)
| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Rule baseline (duplicate / no-doc / |z|>3) | 0.433 | 0.963 | 0.598 | 0.958 |
| Isolation Forest | 0.375 | 0.333 | 0.353 | 0.939 |
| Logistic Regression | 0.474 | 1.000 | 0.643 | 0.997 |
| Random Forest | 0.931 | 1.000 | 0.964 | 0.998 |

Top drivers: same_amount_recent, days_since_last_same_counterparty, log_amount, has_doc, amount_usd.

## Module 2 — Inventory days-to-sell (grouped random hold-out: 181 sales)
Split is grouped-random by SKU, not time-based: a time split would test
calendar extrapolation (Nov-Dec sales are disproportionately long-tail
units acquired months earlier) instead of the pricing/brand signal.
Regressors are fit on log(days) and exponentiated; metrics in days.
| Model | MAPE | MAE (days) | R² |
|---|---|---|---|
| Median baseline | 44.81% | 25.3 | -0.000 |
| Linear Regression (log target) | 40.00% | 23.3 | 0.116 |
| Random Forest (log target) | 43.80% | 24.6 | 0.028 |

Shrinkage detected: 8 quarter-end count shortfalls,
total loss $210,325 (flagged by rule, not by model).
Inventory value at Dec 2025: $4,628,983
across 150 SKUs; 180d+ aging bucket holds
$2,232,944.

## Module 3 — Reconciliation breaks (hold-out: 41 breaks)
| Model | Accuracy | Macro-F1 |
|---|---|---|
| Majority-class baseline | 0.878 | 0.312 |
| Random Forest | 0.976 | 0.929 |
| Rule baseline (|diff| > $1,000) — break detection | 0.998 | — |

Rule baseline (|diff| > $1,000) break-detection accuracy on hold-out: 0.998.
Ranked review queue (top 20 by risk score): 20/20 are
true posting errors; queue recall of posting errors 55.56%.

## Benchmark (best model vs interpretable baseline)
| Module | Baseline | Best model | Metric |
|---|---|---|---|
| AR/AP anomaly | Rule baseline 0.598 | Random Forest 0.964 | F1 |
| Days-to-sell | Median baseline 0.448 | Linear Regression (log target) 0.400 | MAPE (lower is better) |
| Break-cause | Majority-class baseline 0.878 | Random Forest 0.976 | Accuracy |

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

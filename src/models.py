"""Model training + evaluation for the three modules.

Hold-out is TIME-BASED everywhere: train on 2025-01-01..2025-10-31,
evaluate on the last 61 days (2025-11-01..2025-12-31). Nothing from
the hold-out is used to fit features, scalers, or models.

Module 1 — AR/AP anomaly detection (classification):
    rule baseline | Isolation Forest | Logistic Regression | Random Forest
    Metrics: precision, recall, F1, ROC-AUC.
Module 2 — inventory days-to-sell (regression):
    median baseline | Linear Regression | Random Forest
    Metrics: MAPE, MAE, R^2.
Module 3 — reconciliation break-cause (3-class classification, break
    rows only) + ranked review queue:
    rule baseline | Random Forest. Metrics: accuracy, macro-F1,
    precision@20 and recall of posting errors inside the top-20 queue.
"""

import os

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, auc, f1_score, mean_absolute_error,
                             precision_score, r2_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler

HOLDOUT_START = pd.Timestamp("2025-11-01")


def _clf_metrics(y_true, y_pred, y_score=None):
    out = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_score is not None and len(np.unique(y_true)) > 1:
        out["roc_auc"] = roc_auc_score(y_true, y_score)
    else:
        out["roc_auc"] = float("nan")
    return out


def anomaly_models(X_train, y_train, X_test, y_test, rule_test):
    """Returns rows of model metrics + the RF importances."""
    results = []
    # the rule's own 0/1 flag doubles as its score, so ROC-AUC is defined
    results.append({"model": "Rule baseline (duplicate / no-doc / |z|>3)",
                    **_clf_metrics(y_test, rule_test,
                                   rule_test.astype(float))})
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(X_train)
    Xte_s = scaler.transform(X_test)

    iso = IsolationForest(n_estimators=300, contamination=0.05,
                          random_state=42)
    iso.fit(Xtr_s)
    iso_pred = (iso.predict(Xte_s) == -1).astype(int)
    iso_score = -iso.decision_function(Xte_s)
    results.append({"model": "Isolation Forest",
                    **_clf_metrics(y_test, iso_pred, iso_score)})

    lr = LogisticRegression(max_iter=1000, class_weight="balanced",
                            random_state=42)
    lr.fit(Xtr_s, y_train)
    lr_pred = lr.predict(Xte_s)
    results.append({"model": "Logistic Regression",
                    **_clf_metrics(y_test, lr_pred,
                                   lr.predict_proba(Xte_s)[:, 1])})

    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    results.append({"model": "Random Forest",
                    **_clf_metrics(y_test, rf_pred,
                                   rf.predict_proba(X_test)[:, 1])})
    importances = pd.Series(rf.feature_importances_,
                            index=X_train.columns).sort_values(ascending=False)
    return results, importances, rf


def inventory_models(X_train, y_train, X_test, y_test):
    """Days-to-sell is right-skewed (lognormal by construction), so the
    regressors are fit on log(days) and exponentiated back; all metrics
    below are reported in DAYS, against the days-space median baseline.
    """
    results = []
    med = float(np.median(y_train))
    pred_med = np.full(len(y_test), med)
    results.append({"model": "Median baseline",
                    "mape": float(np.mean(np.abs(y_test - pred_med) / y_test)),
                    "mae": float(mean_absolute_error(y_test, pred_med)),
                    "r2": float(r2_score(y_test, pred_med))})
    log_y_train = np.log(y_train)
    lr = LinearRegression()
    lr.fit(X_train, log_y_train)
    p = np.exp(lr.predict(X_test))
    results.append({"model": "Linear Regression (log target)",
                    "mape": float(np.mean(np.abs(y_test - p) / y_test)),
                    "mae": float(mean_absolute_error(y_test, p)),
                    "r2": float(r2_score(y_test, p))})
    rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(X_train, log_y_train)
    p = np.exp(rf.predict(X_test))
    results.append({"model": "Random Forest (log target)",
                    "mape": float(np.mean(np.abs(y_test - p) / y_test)),
                    "mae": float(mean_absolute_error(y_test, p)),
                    "r2": float(r2_score(y_test, p))})
    importances = pd.Series(rf.feature_importances_,
                            index=X_train.columns).sort_values(ascending=False)
    return results, importances


def recon_models(X_train, y_train, X_test, y_test):
    """3-class break-cause classification on break rows only."""
    results = []
    # rule baseline: majority class
    maj = y_train.mode().iloc[0]
    pred = np.full(len(y_test), maj)
    results.append({"model": "Majority-class baseline",
                    "accuracy": float(accuracy_score(y_test, pred)),
                    "macro_f1": float(f1_score(y_test, pred, average="macro",
                                               zero_division=0))})
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)
    results.append({"model": "Random Forest",
                    "accuracy": float(accuracy_score(y_test, pred)),
                    "macro_f1": float(f1_score(y_test, pred, average="macro",
                                               zero_division=0))})
    importances = pd.Series(rf.feature_importances_,
                            index=X_train.columns).sort_values(ascending=False)
    return results, importances


def review_queue_metrics(recon_holdout):
    """Top-20 ranked queue: how many of the 20 highest-risk open breaks
    are true posting errors, and what share of ALL posting errors the
    top 20 captures."""
    q = recon_holdout.sort_values("risk_score", ascending=False).head(20)
    n_err_top = int((q["break_type"] == "posting_error").sum())
    n_err_all = int((recon_holdout["break_type"] == "posting_error").sum())
    return {
        "queue_size": 20,
        "posting_errors_in_top20": n_err_top,
        "posting_errors_total_holdout": n_err_all,
        "precision_at_20": n_err_top / 20,
        "recall_of_posting_errors": (n_err_top / n_err_all
                                     if n_err_all else 0.0),
    }

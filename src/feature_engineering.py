"""Feature engineering.

Ledger (anomaly detection) — counterparty statistics are fitted on the
TRAIN period only and then applied to the hold-out, so no information
leaks from the future into the features. Features:

  log_amount, amount_z_vs_counterparty, amount_vs_entity_median,
  days_since_last_same_counterparty, is_weekend, has_doc,
  is_round_number, fx_deviation, overdue_days, txn_type + entity
  one-hots.

Inventory (days-to-sell regression): log_unit_cost, brand one-hots,
acq_month, entity one-hots, markup.

Reconciliation (break-cause classification): abs_diff_log, diff_sign,
break_age_days, entity + subledger one-hots, is_month_end,
prev_day_abs_diff_log, day_of_week.
"""

import numpy as np
import pandas as pd

HOLDOUT_START = pd.Timestamp("2025-11-01")


# ---------------------------------------------------------------- ledger
def fit_counterparty_stats(train):
    """Per-counterparty median / std of amount_usd, from TRAIN only."""
    stats = (train.groupby("counterparty")["amount_usd"]
             .agg(["median", "std", "count"]))
    stats["std"] = stats["std"].fillna(0.0)
    global_median = float(train["amount_usd"].median())
    global_std = float(train["amount_usd"].std())
    return stats, global_median, global_std


def build_ledger_features(df, cp_stats, global_median, global_std):
    df = df.sort_values(["counterparty", "txn_date"]).copy()
    df["log_amount"] = np.log(df["amount_usd"])
    df["is_weekend"] = (df["txn_date"].dt.dayofweek >= 5).astype(int)
    df["has_doc"] = df["supporting_doc"].astype(int)
    df["is_round_number"] = (
        (df["amount_usd"] % 5000 == 0) & (df["amount_usd"] >= 10_000)
    ).astype(int)
    # FX deviation from the entity's monthly mean FX
    df["fx_month_mean"] = df.groupby(
        ["entity", df["txn_date"].dt.to_period("M")])["fx_to_usd"].transform("mean")
    df["fx_deviation"] = (df["fx_to_usd"] - df["fx_month_mean"]).abs()
    # counterparty z-score (train-fitted stats; unseen -> global)
    med = df["counterparty"].map(cp_stats["median"]).fillna(global_median)
    sd = df["counterparty"].map(cp_stats["std"]).fillna(global_std)
    sd = sd.replace(0, global_std)
    df["amount_z_vs_counterparty"] = (df["amount_usd"] - med) / sd
    ent_med = df.groupby("entity")["amount_usd"].transform("median")
    df["amount_vs_entity_median"] = df["amount_usd"] / ent_med
    # days since the previous transaction with the same counterparty.
    # If the caller precomputed this on the full timeline (so a hold-out
    # row can see its October predecessor), keep it; otherwise compute
    # it inside this frame.
    if "days_since_last_same_counterparty" not in df.columns:
        df["prev_txn_date"] = df.groupby("counterparty")["txn_date"].shift(1)
        df["days_since_last_same_counterparty"] = (
            (df["txn_date"] - df["prev_txn_date"]).dt.days.fillna(365).clip(0, 365)
        )
    # overdue days for open invoices/bills (0 if paid or a payment row)
    df["overdue_days"] = 0
    mask = df["txn_type"].isin(["AR_INVOICE", "AP_BILL"]) & df["paid_date"].isna()
    df.loc[mask, "overdue_days"] = (
        (pd.Timestamp("2025-12-31") - df.loc[mask, "due_date"]).dt.days.clip(lower=0)
    )
    df = pd.get_dummies(df, columns=["txn_type", "entity"],
                        prefix=["type", "ent"], dtype=int)
    drop = ["txn_id", "txn_date", "counterparty", "counterparty_type",
            "currency", "amount_local", "fx_to_usd", "due_date", "paid_date",
            "supporting_doc", "anomaly_type", "fx_month_mean",
            "prev_txn_date"]
    return df.drop(columns=[c for c in drop if c in df.columns])


LEDGER_NUMERIC = [
    "log_amount", "amount_z_vs_counterparty", "amount_vs_entity_median",
    "days_since_last_same_counterparty", "same_amount_recent",
    "is_weekend", "has_doc",
    "is_round_number", "fx_deviation", "overdue_days",
]


# ------------------------------------------------------------- inventory
def build_inventory_features(sales):
    df = sales.copy()
    df["log_unit_cost"] = np.log(df["unit_cost_usd"])
    df = pd.get_dummies(df, columns=["brand", "entity"],
                        prefix=["brand", "ent"], dtype=int)
    return df


# ------------------------------------------------------------------ recon
def build_recon_features(df):
    df = df.sort_values(["entity", "subledger", "recon_date"]).copy()
    df["abs_diff_log"] = np.log1p(df["diff_usd"].abs())
    df["diff_sign"] = np.sign(df["diff_usd"]).astype(int)
    df["prev_day_abs_diff_log"] = df.groupby(
        ["entity", "subledger"])["abs_diff_log"].shift(1).fillna(0.0)
    df["day_of_week"] = df["recon_date"].dt.dayofweek
    df = pd.get_dummies(df, columns=["entity", "subledger"],
                        prefix=["ent", "sub"], dtype=int)
    return df

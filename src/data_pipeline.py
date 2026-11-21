"""Validation + cleaning pipeline.

Checks every synthetic file before modelling: required columns, no
nulls in key fields, amounts strictly positive, dates inside 2025,
FX rates inside plausible bands, and the injected anomaly/break rates
inside their design ranges. Writes cleaned copies to data/processed/.

Raises on failure — the pipeline must not silently pass bad data.
"""

import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYN = os.path.join(ROOT, "data", "synthetic")
PROC = os.path.join(ROOT, "data", "processed")


def _check(cond, msg):
    if not cond:
        raise ValueError(f"VALIDATION FAILED: {msg}")


def validate_ledger(df):
    _check(set(["txn_id", "entity", "txn_date", "txn_type", "counterparty",
                "currency", "amount_local", "fx_to_usd", "amount_usd",
                "supporting_doc", "is_anomaly",
                "anomaly_type"]).issubset(df.columns), "ledger columns")
    _check(df["txn_id"].is_unique, "txn_id must be unique")
    _check(df["amount_usd"].notna().all() and (df["amount_usd"] > 0).all(),
           "amount_usd positive, no nulls")
    _check(df["amount_local"].notna().all()
           and (df["amount_local"] > 0).all(), "amount_local positive")
    _check(df["fx_to_usd"].between(0.5, 1.5).all(), "fx_to_usd in band")
    _check(df["txn_date"].between("2025-01-01", "2025-12-31").all(),
           "txn_date inside 2025")
    _check(df["is_anomaly"].isin([0, 1]).all(), "is_anomaly binary")
    rate = df["is_anomaly"].mean()
    _check(0.03 <= rate <= 0.08, f"anomaly rate {rate:.2%} outside 3-8%")
    _check(set(df.loc[df["is_anomaly"] == 1, "anomaly_type"].unique())
           <= {"duplicate", "unsupported", "out_of_pattern"},
           "anomaly types")
    _check((df.loc[df["is_anomaly"] == 0, "anomaly_type"] == "none").all(),
           "clean rows labelled none")
    # every unsupported anomaly really has no supporting document
    unsup = df[df["anomaly_type"] == "unsupported"]
    _check((unsup["supporting_doc"] == 0).all(), "unsupported rows lack docs")
    _check((unsup["amount_usd"] > 25_000).all(),
           "unsupported rows are large amounts")
    return df


def validate_inventory(df):
    _check(set(["sku", "month_end", "qty_on_hand", "moving_avg_cost_usd",
                "total_value_usd", "aging_bucket",
                "shrinkage_flag"]).issubset(df.columns),
           "inventory columns")
    _check((df["qty_on_hand"] >= 0).all(), "qty non-negative")
    _check((df["moving_avg_cost_usd"] > 0).all(), "moving avg cost positive")
    _check(df["aging_bucket"].isin(
        ["0-30d", "31-90d", "91-180d", "180d+"]).all(), "aging buckets")
    _check(int(df["shrinkage_flag"].sum()) == 8, "exactly 8 shrinkage flags")
    _check((df.loc[df["shrinkage_flag"] == 1, "shrinkage_loss_usd"] > 0).all(),
           "shrinkage rows carry a loss")
    return df


def validate_sales(df):
    _check((df["days_to_sell"] >= 1).all(), "days_to_sell >= 1")
    _check((df["sale_price_usd"] > df["unit_cost_usd"]).all(),
           "sale price above cost")
    _check(df["markup"].between(1.10, 1.50).all(), "markup in 1.10-1.50")
    return df


def validate_recon(df):
    _check(set(["entity", "subledger", "recon_date", "gl_balance_usd",
                "subledger_total_usd", "diff_usd", "has_break",
                "break_type", "risk_score"]).issubset(df.columns),
           "recon columns")
    _check(((df["gl_balance_usd"] - df["subledger_total_usd"])
            - df["diff_usd"]).abs().max() < 0.02, "GL - subledger == diff")
    _check((df["has_break"] == (df["diff_usd"] != 0).astype(int)).all(),
           "has_break matches diff")
    _check(set(df.loc[df["has_break"] == 1, "break_type"].unique())
           <= {"timing", "posting_error", "fx_rounding"}, "break causes")
    _check((df["risk_score"] >= 0).all(), "risk score non-negative")
    rate = df["has_break"].mean()
    _check(0.03 <= rate <= 0.12, f"break rate {rate:.2%} outside 3-12%")
    return df


def main():
    os.makedirs(PROC, exist_ok=True)
    ledger = pd.read_csv(os.path.join(SYN, "ledger.csv"),
                         parse_dates=["txn_date", "due_date", "paid_date"])
    ledger = validate_ledger(ledger)
    ledger.to_csv(os.path.join(PROC, "ledger.csv"), index=False)

    inv = pd.read_csv(os.path.join(SYN, "inventory_monthly.csv"),
                      parse_dates=["month_end"])
    inv = validate_inventory(inv)
    inv.to_csv(os.path.join(PROC, "inventory_monthly.csv"), index=False)

    sales = pd.read_csv(os.path.join(SYN, "inventory_sales.csv"),
                        parse_dates=["acquisition_date", "sale_date"])
    sales = validate_sales(sales)
    sales.to_csv(os.path.join(PROC, "inventory_sales.csv"), index=False)

    recon = pd.read_csv(os.path.join(SYN, "recon_daily.csv"),
                        parse_dates=["recon_date"])
    recon = validate_recon(recon)
    recon.to_csv(os.path.join(PROC, "recon_daily.csv"), index=False)
    print("validation OK: 4/4 files clean, processed copies written.")


if __name__ == "__main__":
    main()

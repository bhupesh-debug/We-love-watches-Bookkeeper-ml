"""Rule-based detectors (the interpretable baselines every ML model must
beat — a bookkeeper can run these in Excel today, no model needed).

1. ledger_rules: duplicate (same counterparty + amount within 3 days),
   unsupported (no doc + amount > $25k), out-of-pattern (|z| > 3 vs
   counterparty history).
2. shrinkage_rules: physical-count shortfall at quarter end (already a
   flag in the data; the rule is the check itself).
3. recon_rules: |GL - subledger| > $1,000 -> break.
"""

import numpy as np
import pandas as pd


def ledger_rule_flags(df):
    """df: raw processed ledger. Returns a 0/1 flag Series."""
    flags = pd.Series(0, index=df.index)
    # duplicate: same counterparty, same amount_usd, within 3 days
    key = df.sort_values(["counterparty", "amount_usd", "txn_date"])
    dup = key.duplicated(subset=["counterparty", "amount_usd"], keep=False)
    # require the dates to be close: check consecutive rows per group
    close = pd.Series(False, index=key.index)
    for _, grp in key[dup].groupby(["counterparty", "amount_usd"]):
        dates = grp["txn_date"].sort_values()
        if (dates.max() - dates.min()).days <= 3:
            close.loc[grp.index] = True
    flags.loc[close[close].index] = 1
    # unsupported: no supporting doc and a large amount
    flags.loc[(df["supporting_doc"] == 0) & (df["amount_usd"] > 25_000)] = 1
    # out-of-pattern: |z| > 3 vs the counterparty's own history
    stats = df.groupby("counterparty")["amount_usd"].agg(["median", "std"])
    med = df["counterparty"].map(stats["median"])
    sd = df["counterparty"].map(stats["std"]).fillna(df["amount_usd"].std())
    sd = sd.replace(0, df["amount_usd"].std())
    z = (df["amount_usd"] - med) / sd
    flags.loc[z.abs() > 3] = 1
    return flags.astype(int)


def recon_rule_flags(df, threshold=1_000.0):
    """Flag a reconciliation break when |GL - subledger| > threshold."""
    return (df["diff_usd"].abs() > threshold).astype(int)


def shrinkage_rule_flags(inv_monthly):
    """A shrinkage flag is raised when the quarter-end physical count
    falls short of the system quantity (flag already computed at
    generation; this rule restates the check for the review queue)."""
    return inv_monthly["shrinkage_flag"].astype(int)

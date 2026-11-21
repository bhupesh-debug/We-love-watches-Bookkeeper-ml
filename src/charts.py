"""Charts for the report and deck. All figures are built from the real
run outputs (models/*.csv and the processed data) — nothing hand-drawn.
"""

import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

NAVY = "#1f4e79"
TEAL = "#2e9e6b"
AMBER = "#d9a441"
GREY = "#8a8a8a"
RED = "#b23a48"
plt.rcParams.update({"font.size": 10, "axes.titlesize": 12,
                     "axes.labelsize": 10, "figure.dpi": 150})


def save(fig, out):
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  chart: {os.path.basename(out)}")


def anomaly_comparison(metrics_csv, out):
    df = pd.read_csv(metrics_csv)
    x = np.arange(len(df))
    w = 0.25
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(x - w, df["precision"], w, label="Precision", color=NAVY)
    ax.bar(x, df["recall"], w, label="Recall", color=TEAL)
    ax.bar(x + w, df["f1"], w, label="F1", color=AMBER)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(" (", "\n(") for m in df["model"]], fontsize=8.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("AR/AP anomaly detection — hold-out (Nov–Dec 2025, 61 days)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    save(fig, out)


def feature_importance(importance_csv, out, title, top_n=10):
    df = pd.read_csv(importance_csv).head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.barh(df["feature"], df["importance"], color=TEAL)
    ax.set_xlabel("Random Forest feature importance")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    save(fig, out)


def inventory_valuation(inv_csv, out):
    inv = pd.read_csv(inv_csv, parse_dates=["month_end"])
    monthly = inv.groupby(["month_end", "entity"])["total_value_usd"].sum().unstack()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 4.2))
    monthly.plot(ax=a1, color=[NAVY, TEAL, AMBER], marker="o", ms=3)
    a1.set_title("Inventory value by entity (USD)")
    a1.set_ylabel("USD")
    a1.grid(alpha=0.3)
    a1.legend(fontsize=8)
    aging = (inv.groupby(["month_end", "aging_bucket"])["total_value_usd"]
             .sum().unstack().fillna(0))
    aging.plot(kind="area", ax=a2, stacked=True,
               color=[TEAL, AMBER, "#c97b2d", RED], alpha=0.85)
    a2.set_title("Inventory value by aging bucket (USD)")
    a2.set_ylabel("USD")
    a2.grid(alpha=0.3)
    a2.legend(fontsize=8)
    save(fig, out)


def recon_timeline(recon_csv, out):
    rec = pd.read_csv(recon_csv, parse_dates=["recon_date"])
    daily = rec.groupby("recon_date")["diff_usd"].sum()
    breaks = rec[rec["has_break"] == 1]
    fig, ax = plt.subplots(figsize=(9.5, 4.0))
    ax.plot(daily.index, daily.values, color=NAVY, lw=0.8,
            label="Total GL – subledger diff (all entities)")
    ax.scatter(breaks["recon_date"], breaks["diff_usd"], s=12, color=RED,
               alpha=0.6, label="Flagged break (single entity/subledger)")
    ax.axhline(1000, color=GREY, ls="--", lw=0.8, label="Rule threshold ±$1,000")
    ax.axhline(-1000, color=GREY, ls="--", lw=0.8)
    ax.set_title("Reconciliation breaks — daily GL vs subledger difference, 2025")
    ax.set_ylabel("USD")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    save(fig, out)


def roc_curves(ledger_features_csv, out):
    """ROC curves are recomputed from the saved hold-out scores."""
    df = pd.read_csv(ledger_features_csv)
    from sklearn.metrics import auc, roc_curve
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    for col, label, color in [
            ("rf_score", "Random Forest", NAVY),
            ("lr_score", "Logistic Regression", TEAL),
            ("iso_score", "Isolation Forest", AMBER)]:
        if col not in df.columns:
            continue
        fpr, tpr, _ = roc_curve(df["is_anomaly"], df[col])
        ax.plot(fpr, tpr, label=f"{label} (AUC {auc(fpr, tpr):.3f})",
                color=color, lw=1.8)
    ax.plot([0, 1], [0, 1], ls="--", color=GREY, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("AR/AP anomaly detection — ROC on hold-out")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    save(fig, out)


def correlation_heatmap(ledger_features_csv, out):
    df = pd.read_csv(ledger_features_csv)
    cols = ["log_amount", "amount_z_vs_counterparty",
            "amount_vs_entity_median", "days_since_last_same_counterparty",
            "same_amount_recent", "is_weekend", "has_doc",
            "is_round_number", "fx_deviation",
            "overdue_days", "is_anomaly"]
    cols = [c for c in cols if c in df.columns]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels([c.replace("_", "\n") for c in cols], fontsize=7.5)
    ax.set_yticklabels([c.replace("_", " ") for c in cols], fontsize=8)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center",
                    va="center", fontsize=7,
                    color="white" if abs(corr.values[i, j]) > 0.5 else "black")
    ax.set_title("Ledger feature correlations (hold-out rows)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    save(fig, out)

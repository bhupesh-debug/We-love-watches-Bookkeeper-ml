"""Synthetic data generation — Multi-Company Bookkeeping & Inventory Analytics.

ALL DATA IS SYNTHETIC. No real We Love Watches transactions, customers,
suppliers, or inventory are used anywhere in this project. The generator
mimics the structure of the JD (multi-company AR/AP, high-value watch
inventory, GL vs subledger reconciliation) so the models can be built and
tested honestly; magnitudes are plausible but invented.

Seed 42 everywhere. Deterministic output.

Three datasets:
  1. ledger.csv        — AR/AP transactions, 3 entities, 3 currencies,
                         ~4,600 rows, ~5% injected anomalies.
  2. inventory_monthly.csv — 150 watch SKUs x 12 monthly snapshots,
                         moving-average cost, aging, shrinkage flags.
  3. inventory_sales.csv   — one row per sale, with days_to_sell.
  4. recon_daily.csv   — GL vs subledger per entity x subledger x day,
                         with injected breaks and a cause label.
"""

import os

import numpy as np
import pandas as pd

SEED = 42
START = pd.Timestamp("2025-01-01")
END = pd.Timestamp("2025-12-31")
HOLDOUT_START = pd.Timestamp("2025-11-01")  # last 61 days

ENTITIES = {
    "WLW-US": {"currency": "USD", "fx_to_usd": 1.00, "city": "New York"},
    "WLW-EU": {"currency": "CHF", "fx_to_usd": 0.88, "city": "Geneva"},
    "WLW-ASIA": {"currency": "SGD", "fx_to_usd": 0.74, "city": "Singapore"},
}

SUPPLIERS = [
    "Rolex SA", "Patek Philippe SA", "Audemars Piguet", "Omega SA",
    "Cartier International", "Vacheron Constantin", "IWC Schaffhausen",
    "Jaeger-LeCoultre", "Breguet SA", "A. Lange & Sohne",
    "SwissTime Distributors AG", "Geneva Watch Traders SARL",
    "Horology Supply Co", "Timepiece Logistics Ltd", "Luxury Parts GmbH",
]
CUSTOMERS = [
    "Fifth Avenue Time Co", "Madison Watch Gallery", "Beverly Hills Jewellers",
    "London Chrono House", "Dubai Luxury Timepieces", "Tokyo Watch Salon",
    "Paris Horlogerie", "Milan Orologi SRL", "Hong Kong Time Centre",
    "Toronto Fine Watches", "Sydney Watch Merchants", "Zurich Time AG",
    "Los Angeles Watch Co", "Chicago Timepieces Inc", "Miami Luxury Goods",
    "Seoul Watch Gallery", "Shanghai Time Trading", "Mumbai Chrono Retail",
    "Amsterdam Watch House", "Vienna Uhren GmbH",
]

BRANDS = [
    ("Rolex", 0.22, 1.00), ("Patek Philippe", 0.14, 1.25),
    ("Audemars Piguet", 0.12, 1.15), ("Omega", 0.13, 0.85),
    ("Cartier", 0.11, 0.90), ("Vacheron Constantin", 0.08, 1.10),
    ("IWC", 0.08, 0.80), ("Jaeger-LeCoultre", 0.07, 0.85),
    ("Breguet", 0.03, 1.05), ("A. Lange & Sohne", 0.02, 1.20),
]
# brand, weight, desirability (1.0 = average; higher sells faster)
MODELS = {
    "Rolex": ["Submariner", "Daytona", "GMT-Master II", "Datejust", "Day-Date"],
    "Patek Philippe": ["Nautilus", "Aquanaut", "Calatrava", "Grand Complications"],
    "Audemars Piguet": ["Royal Oak", "Royal Oak Offshore", "Code 11.59"],
    "Omega": ["Speedmaster", "Seamaster", "Constellation", "De Ville"],
    "Cartier": ["Santos", "Tank", "Pasha", "Drive de Cartier"],
    "Vacheron Constantin": ["Overseas", "Patrimony", "Traditionnelle"],
    "IWC": ["Portugieser", "Pilot's Chronograph", "Portofino"],
    "Jaeger-LeCoultre": ["Reverso", "Master Ultra Thin", "Polaris"],
    "Breguet": ["Classique", "Marine", "Type XX"],
    "A. Lange & Sohne": ["Lange 1", "Zeitwerk", "Saxonia"],
}


def _fx_series(rng, base_fx, n_days):
    """Daily FX (local currency -> USD) with small random walk noise."""
    shocks = rng.normal(0, 0.002, n_days)
    walk = np.cumsum(shocks)
    return base_fx * (1 + walk - walk.mean())


def generate_ledger(rng):
    """AR/AP ledger with injected anomalies (duplicate / unsupported /
    out-of-pattern). Anomalies are extra rows, ~5% of the total."""
    dates = pd.date_range(START, END, freq="D")
    n_days = len(dates)
    fx = {e: _fx_series(rng, info["fx_to_usd"], n_days)
          for e, info in ENTITIES.items()}
    fx_map = {(e, d): fx[e][i] for e in ENTITIES for i, d in enumerate(dates)}

    rows = []
    txn_n = 0
    for ei, (entity, info) in enumerate(ENTITIES.items()):
        cur = info["currency"]
        for i, d in enumerate(dates):
            # Fewer postings on weekends (normal business pattern)
            lam = 5.0 if d.dayofweek < 5 else 0.8
            n_txn = rng.poisson(lam)
            for _ in range(n_txn):
                txn_n += 1
                is_ar = rng.random() < 0.52
                if is_ar:
                    cp = CUSTOMERS[rng.integers(len(CUSTOMERS))]
                    cp_type = "customer"
                    txn_type = "AR_INVOICE" if rng.random() < 0.7 else "AR_PAYMENT"
                    mu, sigma = 10.2, 0.75
                else:
                    cp = SUPPLIERS[rng.integers(len(SUPPLIERS))]
                    cp_type = "supplier"
                    txn_type = "AP_BILL" if rng.random() < 0.7 else "AP_PAYMENT"
                    mu, sigma = 9.9, 0.8
                amount_local = float(np.clip(rng.lognormal(mu, sigma), 500, 250_000))
                fx_rate = float(fx_map[(entity, d)])
                amount_usd = round(amount_local * fx_rate, 2)
                due = d + pd.Timedelta(days=int(rng.choice([30, 45, 60])))
                paid = pd.NaT
                if txn_type in ("AR_INVOICE", "AP_BILL") and rng.random() < 0.85:
                    paid = due + pd.Timedelta(days=int(rng.integers(-5, 21)))
                rows.append({
                    "txn_id": f"TXN-{txn_n:05d}",
                    "entity": entity, "txn_date": d, "txn_type": txn_type,
                    "counterparty": cp, "counterparty_type": cp_type,
                    "currency": cur, "amount_local": round(amount_local, 2),
                    "fx_to_usd": round(fx_rate, 4), "amount_usd": amount_usd,
                    "due_date": due, "paid_date": paid,
                    "supporting_doc": int(rng.random() < 0.97),
                    "is_anomaly": 0, "anomaly_type": "none",
                })
    ledger = pd.DataFrame(rows)

    # ---- inject anomalies as extra rows --------------------------------
    base = ledger[ledger["txn_type"].isin(["AR_INVOICE", "AP_BILL"])].copy()
    n_anom = int(round(0.05 * len(ledger)))  # ~5% of all rows
    n_dup = int(0.40 * n_anom)
    n_unsup = int(0.30 * n_anom)
    n_oop = n_anom - n_dup - n_unsup

    anom_rows = []
    # 1. duplicates: copy a real invoice/bill, new id, date +0..2 days
    dup_src = base.sample(n=n_dup, random_state=SEED, replace=False)
    for _, r in dup_src.iterrows():
        txn_n += 1
        new = r.to_dict()
        new.update({
            "txn_id": f"TXN-{txn_n:05d}",
            "txn_date": r["txn_date"] + pd.Timedelta(days=int(rng.integers(0, 3))),
            "is_anomaly": 1, "anomaly_type": "duplicate",
        })
        anom_rows.append(new)
    # 2. unsupported: no supporting document + large amount
    unsup_src = base.sample(n=n_unsup, random_state=SEED + 1, replace=False)
    for _, r in unsup_src.iterrows():
        txn_n += 1
        new = r.to_dict()
        amt = float(np.clip(r["amount_usd"] * rng.uniform(1.2, 2.0), 26_000, 250_000))
        new.update({
            "txn_id": f"TXN-{txn_n:05d}",
            "amount_usd": round(amt, 2),
            "amount_local": round(amt / r["fx_to_usd"], 2),
            "supporting_doc": 0,
            "is_anomaly": 1, "anomaly_type": "unsupported",
        })
        anom_rows.append(new)
    # 3. out-of-pattern: amount far above the counterparty's usual range,
    #    or a round-number large amount posted on a weekend
    cp_median = base.groupby("counterparty")["amount_usd"].median()
    oop_src = base.sample(n=n_oop, random_state=SEED + 2, replace=False)
    for _, r in oop_src.iterrows():
        txn_n += 1
        new = r.to_dict()
        if rng.random() < 0.6:
            amt = float(cp_median[r["counterparty"]] * rng.uniform(4.0, 8.0))
            amt = float(np.clip(amt, 5_000, 250_000))
            new_date = r["txn_date"]
        else:
            amt = float(rng.choice([50_000, 100_000, 150_000]))
            # move to the next Saturday
            new_date = r["txn_date"] + pd.Timedelta(
                days=int((5 - r["txn_date"].dayofweek) % 7))
        new.update({
            "txn_id": f"TXN-{txn_n:05d}",
            "txn_date": new_date,
            "amount_usd": round(amt, 2),
            "amount_local": round(amt / r["fx_to_usd"], 2),
            "is_anomaly": 1, "anomaly_type": "out_of_pattern",
        })
        anom_rows.append(new)

    ledger = pd.concat([ledger, pd.DataFrame(anom_rows)], ignore_index=True)
    # anomaly date shifts (+0..2 days, weekend moves) must not spill
    # into 2026
    ledger["txn_date"] = ledger["txn_date"].clip(upper=END)
    ledger = ledger.sort_values(["txn_date", "txn_id"]).reset_index(drop=True)
    ledger["txn_date"] = pd.to_datetime(ledger["txn_date"])
    return ledger


DESIRABILITY = {b[0]: b[2] for b in BRANDS}


def _draw_days_to_sell(rng, brand, unit_cost, acq_month):
    """Structural days-to-sell: desirable brands and cheaper pieces move
    faster; December (holiday gifting) is fastest, January slowest.
    Lognormal noise keeps it from being deterministic."""
    season = {1: 1.25, 2: 1.15, 3: 1.05, 4: 1.00, 5: 0.95, 6: 0.95,
              7: 1.00, 8: 1.00, 9: 1.00, 10: 0.95, 11: 0.85, 12: 0.80}
    mu = (np.log(70.0)
          - 0.55 * np.log(DESIRABILITY[brand])
          + 0.30 * np.log(unit_cost / 18_000)
          + np.log(season.get(acq_month, 1.0)))
    return int(np.clip(rng.lognormal(mu, 0.45), 5, 400))


def generate_inventory(rng):
    """150 high-value watch SKUs, simulated unit by unit.

    Every physical unit gets an acquisition (or restock) date, a cost,
    and a drawn days-to-sell; the sale date is acquisition + days_to_sell.
    Monthly snapshots, the sales table, and the shrinkage flags are all
    derived from the same unit list, so they agree with each other.
    """
    brands = [b[0] for b in BRANDS]
    weights = np.array([b[1] for b in BRANDS])
    weights = weights / weights.sum()

    skus = []
    for s in range(150):
        brand = brands[rng.choice(len(brands), p=weights)]
        model = MODELS[brand][rng.integers(len(MODELS[brand]))]
        cost = float(np.clip(rng.lognormal(np.log(18_000), 0.75), 3_000, 85_000))
        acq = START + pd.Timedelta(days=int(rng.integers(0, 300)))
        entity = list(ENTITIES)[rng.choice(3, p=[0.5, 0.3, 0.2])]
        skus.append({
            "sku": f"WLW-{s + 1:03d}", "brand": brand, "model": model,
            "entity": entity, "acquisition_date": acq,
            "unit_cost_usd": round(cost, 2), "initial_qty": int(rng.integers(1, 5)),
        })
    sku_df = pd.DataFrame(skus)

    # ---- simulate units ------------------------------------------------
    # unit: sku, brand, entity, acquired_date, cost, sale_date (or None)
    units = []
    for _, sku in sku_df.iterrows():
        cur_cost = sku["unit_cost_usd"]
        for _ in range(sku["initial_qty"]):
            dts = _draw_days_to_sell(rng, sku["brand"], cur_cost,
                                     sku["acquisition_date"].month)
            units.append({"sku": sku["sku"], "brand": sku["brand"],
                          "entity": sku["entity"],
                          "acquired": sku["acquisition_date"],
                          "cost": cur_cost,
                          "sale_date": sku["acquisition_date"]
                          + pd.Timedelta(days=dts)})
        # restocks: ~25% of months, 1-2 units at a drifted cost
        for m in pd.date_range(sku["acquisition_date"].replace(day=1)
                               + pd.offsets.MonthBegin(1),
                               END, freq="ME"):
            if rng.random() < 0.25:
                cur_cost = round(
                    cur_cost * float(rng.normal(1.02, 0.06)), 2)
                restock_day = m.replace(day=int(rng.integers(1, 28)))
                for _ in range(int(rng.integers(1, 3))):
                    dts = _draw_days_to_sell(rng, sku["brand"], cur_cost,
                                             restock_day.month)
                    units.append({"sku": sku["sku"], "brand": sku["brand"],
                                  "entity": sku["entity"],
                                  "acquired": restock_day, "cost": cur_cost,
                                  "sale_date": restock_day
                                  + pd.Timedelta(days=dts)})
    units_df = pd.DataFrame(units)

    # ---- monthly snapshots + sales table, both from the unit list -----
    months = pd.date_range(START, END, freq="ME")
    snap_rows = []
    sales_rows = []
    for _, sku in sku_df.iterrows():
        su = units_df[units_df["sku"] == sku["sku"]]
        for m in months:
            on_hand = su[(su["acquired"] <= m)
                         & ((su["sale_date"].isna())
                            | (su["sale_date"] > m))]
            qty = len(on_hand)
            ma_cost = (round(float(on_hand["cost"].mean()), 2)
                       if qty else sku["unit_cost_usd"])
            age_days = max(0, (m - sku["acquisition_date"]).days)
            if age_days <= 30:
                bucket = "0-30d"
            elif age_days <= 90:
                bucket = "31-90d"
            elif age_days <= 180:
                bucket = "91-180d"
            else:
                bucket = "180d+"
            bought = int(((su["acquired"] >= m.replace(day=1))
                          & (su["acquired"] <= m)).sum())
            sold = int(((su["sale_date"] >= m.replace(day=1))
                        & (su["sale_date"] <= m)).sum())
            snap_rows.append({
                "sku": sku["sku"], "brand": sku["brand"],
                "model": sku["model"], "entity": sku["entity"],
                "month_end": m, "qty_on_hand": qty,
                "moving_avg_cost_usd": ma_cost,
                "total_value_usd": round(float(on_hand["cost"].sum()), 2),
                "age_days": age_days, "aging_bucket": bucket,
                "purchased_qty": bought, "sold_qty": sold,
                "shrinkage_flag": 0, "shrinkage_loss_usd": 0.0,
            })
        for _, u in su.iterrows():
            if u["sale_date"] <= END:
                markup = float(rng.uniform(1.15, 1.45))
                sales_rows.append({
                    "sku": sku["sku"], "brand": sku["brand"],
                    "entity": sku["entity"],
                    "acquisition_date": u["acquired"],
                    "sale_date": u["sale_date"],
                    "days_to_sell": int((u["sale_date"] - u["acquired"]).days),
                    "unit_cost_usd": u["cost"],
                    "sale_price_usd": round(u["cost"] * markup, 2),
                    "markup": round(markup, 3),
                    "acq_month": u["acquired"].month,
                })
    inv = pd.DataFrame(snap_rows)
    sales = pd.DataFrame(sales_rows)

    # Shrinkage: 8 quarter-end physical-count discrepancies (system qty
    # exceeds the physical count by 1 unit on a high-value SKU).
    candidates = inv[(inv["qty_on_hand"] >= 1)
                     & (inv["moving_avg_cost_usd"] >= 15_000)]
    picks = candidates.sample(n=8, random_state=SEED)
    shrinkage_events = []
    for idx, row in picks.iterrows():
        loss = float(row["moving_avg_cost_usd"])
        inv.loc[idx, "shrinkage_flag"] = 1
        inv.loc[idx, "shrinkage_loss_usd"] = round(loss, 2)
        inv.loc[idx, "qty_on_hand"] = int(row["qty_on_hand"]) - 1
        inv.loc[idx, "total_value_usd"] = round(
            int(row["qty_on_hand"] - 1) * float(row["moving_avg_cost_usd"]), 2)
        shrinkage_events.append({
            "sku": row["sku"], "month_end": row["month_end"],
            "loss_usd": round(loss, 2),
        })
    return inv, sales, pd.DataFrame(shrinkage_events), sku_df


def generate_recon(rng):
    """Daily GL vs subledger per entity x subledger, with injected breaks.

    Break causes: timing (reverses next day), posting_error (persists
    3-7 days), fx_rounding (tiny, one day)."""
    dates = pd.date_range(START, END, freq="D")
    subledgers = {"AR": (1.2, 2_400_000.0), "AP": (1.0, 1_900_000.0),
                  "Inventory": (1.5, 5_500_000.0)}
    rows = []
    active_errors = {}  # (entity, subledger) -> days remaining, diff
    for entity in ENTITIES:
        for sub, (weight, base) in subledgers.items():
            level = base * float(rng.uniform(0.85, 1.15))
            for d in dates:
                level *= float(rng.normal(1.0, 0.004))
                level = max(level, base * 0.5)
                sub_total = round(level, 2)
                diff = 0.0
                cause = "none"
                key = (entity, sub)
                if key in active_errors:
                    rem, err = active_errors[key]
                    diff, cause = err, "posting_error"
                    if rem <= 1:
                        del active_errors[key]
                    else:
                        active_errors[key] = (rem - 1, err)
                elif rng.random() < 0.025:
                    # ~2.5% of days start a break; posting errors persist
                    # 3-7 days, so the row-level break rate lands near 6%.
                    roll = rng.random()
                    if roll < 0.45:
                        # timing differences are small but overlap the
                        # low end of posting errors by design
                        diff = float(rng.choice([-1, 1])
                                     * rng.uniform(200, 8_000))
                        cause = "timing"
                    elif roll < 0.80:
                        diff = float(rng.choice([-1, 1])
                                     * rng.uniform(1_500, 25_000))
                        cause = "posting_error"
                        active_errors[key] = (int(rng.integers(3, 8)), diff)
                    else:
                        diff = float(rng.choice([-1, 1])
                                     * rng.uniform(50, 1_500))
                        cause = "fx_rounding"
                rows.append({
                    "entity": entity, "subledger": sub, "recon_date": d,
                    "gl_balance_usd": round(sub_total + diff, 2),
                    "subledger_total_usd": sub_total,
                    "diff_usd": round(diff, 2),
                    "has_break": int(diff != 0.0),
                    "break_type": cause,
                    "subledger_weight": weight,
                    "is_month_end": int(d.is_month_end),
                })
    recon = pd.DataFrame(rows)
    # break age: consecutive days the same (entity, subledger) has a
    # non-zero diff of the same sign
    recon = recon.sort_values(["entity", "subledger", "recon_date"])
    recon["break_age_days"] = 0
    for (e, s), grp in recon.groupby(["entity", "subledger"]):
        age = 0
        prev_sign = 0
        for idx, r in grp.iterrows():
            sign = int(np.sign(r["diff_usd"]))
            if sign != 0 and sign == prev_sign:
                age += 1
            elif sign != 0:
                age = 1
            else:
                age = 0
            prev_sign = sign if sign != 0 else prev_sign
            recon.loc[idx, "break_age_days"] = age
    recon["risk_score"] = (
        recon["diff_usd"].abs()
        * (1 + 0.5 * recon["break_age_days"])
        * recon["subledger_weight"]
    ).round(2)
    return recon.reset_index(drop=True)


def main():
    rng = np.random.default_rng(SEED)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(root, "data", "synthetic")
    os.makedirs(out, exist_ok=True)

    ledger = generate_ledger(rng)
    ledger.to_csv(os.path.join(out, "ledger.csv"), index=False)
    print(f"ledger.csv: {len(ledger):,} rows, "
          f"{int(ledger['is_anomaly'].sum())} anomalies "
          f"({ledger['is_anomaly'].mean():.2%})")
    print("  by type:", ledger[ledger["is_anomaly"] == 1]
          ["anomaly_type"].value_counts().to_dict())

    inv, sales, shrink, skus = generate_inventory(rng)
    inv.to_csv(os.path.join(out, "inventory_monthly.csv"), index=False)
    sales.to_csv(os.path.join(out, "inventory_sales.csv"), index=False)
    shrink.to_csv(os.path.join(out, "shrinkage_events.csv"), index=False)
    skus.to_csv(os.path.join(out, "inventory_skus.csv"), index=False)
    print(f"inventory_monthly.csv: {len(inv):,} rows "
          f"({inv['sku'].nunique()} SKUs)")
    print(f"inventory_sales.csv: {len(sales):,} sales, "
          f"median days_to_sell {int(sales['days_to_sell'].median())}")
    print(f"shrinkage events: {len(shrink)}, "
          f"total loss ${shrink['loss_usd'].sum():,.0f}")

    recon = generate_recon(rng)
    recon.to_csv(os.path.join(out, "recon_daily.csv"), index=False)
    print(f"recon_daily.csv: {len(recon):,} rows, "
          f"{int(recon['has_break'].sum())} breaks "
          f"({recon['has_break'].mean():.2%})")
    print("  by cause:", recon[recon["has_break"] == 1]
          ["break_type"].value_counts().to_dict())


if __name__ == "__main__":
    main()

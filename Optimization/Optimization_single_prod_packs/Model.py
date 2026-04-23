"""
Pre-Pack Optimization – Lower Bound Solution
=============================================
Scope: Single-product packs only.
       For each product we decide:
         1. Which pack configurations (size-mix) to use
         2. How many of each pack to allocate to each city/channel

Costs (from assignment):
  - Pack creation cost : €134  per unique pack type
  - Handling cost      : €11.03 per pack passing through DC
  - Cost of capital    : 24.3% of product value for unsold units

Unit costs are loaded per product from PPP_combined.csv.
"""

import pandas as pd
import numpy as np
from pulp import (
    LpProblem, LpMinimize, LpVariable, LpBinary, lpSum,
    value, PULP_CBC_CMD, LpStatus
)
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────
PACK_CREATION_COST = 134        # € per unique pack type
HANDLING_COST      = 11.03      # € per pack through DC
COST_OF_CAPITAL    = 0.243      # fraction of unit value lost per unsold unit
# Unit costs loaded from PPP_combined.csv (per product)
_cost_df = pd.read_csv(
    "Input_Files/PPP_combined.csv", sep=";"
)
_cost_df["Cost"] = _cost_df["Cost"].str.replace(",", ".").astype(float)
UNIT_COST_MAP = _cost_df.groupby("Product ID")["Cost"].first().to_dict()
SERVICE_LEVEL      = 0.85       # minimum fraction of demand that must be covered
MAX_UNITS_IN_PACK  = 10         # upper bound per size slot in a pack

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
df = pd.read_excel(
    "Forecast/moving_average_city_product_size.xlsx",
    sheet_name="Forecast_2026"
)
df["forecast_2026"] = df["forecast_2026"].clip(lower=0).round().astype(int)

products = sorted(df["product"].unique())
cities   = sorted(df["stad"].unique())
sizes    = sorted(df["maat"].unique())

print(f"Products : {len(products)}")
print(f"Channels : {len(cities)}  → {cities}")
print(f"Sizes    : {sizes}")
print(f"Total SKUs: {len(df)}")
print()

# Build demand dict: demand[product][city][size]
demand = {}
for prod in products:
    demand[prod] = {}
    for city in cities:
        sub = df[(df["product"] == prod) & (df["stad"] == city)]
        demand[prod][city] = {}
        for sz in sizes:
            row = sub[sub["maat"] == sz]
            demand[prod][city][sz] = int(row["forecast_2026"].values[0]) if len(row) > 0 else 0

# ─────────────────────────────────────────────
# HELPER: generate candidate packs
# ─────────────────────────────────────────────
def generate_candidate_packs(prod):
    """
    Candidate pack configs for a single product.
    Includes single-size packs as fallback to guarantee feasibility.
    """
    agg = {sz: sum(demand[prod][c][sz] for c in cities) for sz in sizes}
    active_sizes = [sz for sz, d in agg.items() if d > 0]
    if not active_sizes:
        return []

    total = sum(agg[sz] for sz in active_sizes)
    candidates = []

    # 1. Ratio pack: proportional to aggregate demand
    ratios = {sz: agg[sz] / total for sz in active_sizes}
    min_ratio = min(r for r in ratios.values() if r > 0)
    ratio_pack = {sz: max(1, round(ratios[sz] / min_ratio)) for sz in active_sizes}
    ratio_pack = {sz: min(v, MAX_UNITS_IN_PACK) for sz, v in ratio_pack.items()}
    candidates.append(ratio_pack)

    # 2. Balanced: 1 of each active size
    candidates.append({sz: 1 for sz in active_sizes})

    # 3. Top-2 sizes
    top2 = sorted(active_sizes, key=lambda sz: -agg[sz])[:2]
    if len(top2) >= 2:
        candidates.append({sz: 2 for sz in top2})

    # 4. Single-size packs per active size (guarantees feasibility)
    for sz in active_sizes:
        candidates.append({sz: 1})

    # Deduplicate
    seen, unique = [], []
    for p in candidates:
        key = tuple(sorted(p.items()))
        if key not in seen:
            seen.append(key)
            unique.append(p)

    return unique


# ─────────────────────────────────────────────
# OPTIMIZATION – one product at a time
# ─────────────────────────────────────────────
results_pack_design  = []
results_allocation   = []
results_summary      = []

pack_label_to_config = {}  # maps pack_label → candidate pack config dict

for prod_idx, prod in enumerate(products):
    unit_cost  = UNIT_COST_MAP[prod]
    prod_pack_counter = 0  # resets per product
    candidates = generate_candidate_packs(prod)
    if not candidates:
        print(f"[{prod}] No active demand – skipping")
        continue

    pack_ids = list(range(len(candidates)))

    # ── Variables ────────────────────────────────────────────────────
    y = {p: LpVariable(f"y_{p}", cat=LpBinary) for p in pack_ids}
    x = {(p, c): LpVariable(f"x_{p}_{c}", lowBound=0, cat="Integer")
         for p in pack_ids for c in cities}
    o = {(c, sz): LpVariable(f"o_{c}_{sz}", lowBound=0)
         for c in cities for sz in sizes}

    prob = LpProblem(f"PrePack_{prod}", LpMinimize)

    # ── Objective ────────────────────────────────────────────────────
    prob += (
        lpSum(PACK_CREATION_COST * y[p] for p in pack_ids)
        + lpSum(HANDLING_COST * x[p, c] for p in pack_ids for c in cities)
        + lpSum(COST_OF_CAPITAL * unit_cost * o[c, sz]
                for c in cities for sz in sizes)
    )

    # ── Constraints ──────────────────────────────────────────────────
    BIG_M = 50000

    # Pack activation
    for p in pack_ids:
        prob += lpSum(x[p, c] for c in cities) <= BIG_M * y[p]

    # Overstock linearisation
    for c in cities:
        for sz in sizes:
            delivered = lpSum(candidates[p].get(sz, 0) * x[p, c] for p in pack_ids)
            prob += o[c, sz] >= delivered - demand[prod][c][sz]

    # Minimum service level (per city, per size)
    for c in cities:
        for sz in sizes:
            d = demand[prod][c][sz]
            if d > 0:
                delivered = lpSum(candidates[p].get(sz, 0) * x[p, c] for p in pack_ids)
                prob += delivered >= SERVICE_LEVEL * d

    # Cap delivery at 1.5× demand to limit overstock
    for c in cities:
        for sz in sizes:
            d = demand[prod][c][sz]
            if d > 0:
                delivered = lpSum(candidates[p].get(sz, 0) * x[p, c] for p in pack_ids)
                prob += delivered <= 1.5 * d

    # ── Solve ─────────────────────────────────────────────────────────
    prob.solve(PULP_CBC_CMD(msg=0))
    status = LpStatus[prob.status]

    if status not in ("Optimal", "Feasible"):
        print(f"[{prod}] WARNING: solver status = {status}")
        continue

    # ── Extract results ───────────────────────────────────────────────
    prod_creation = prod_handling = prod_capital = 0

    for p in pack_ids:
        if value(y[p]) is not None and value(y[p]) > 0.5:
            prod_creation += PACK_CREATION_COST
            prod_pack_counter += 1
            pack_label = f"{prod}_pack_{prod_pack_counter}"
            pack_label_to_config[pack_label] = candidates[p]

            for sz, units in candidates[p].items():
                if units > 0:
                    results_pack_design.append({
                        "product": prod, "pack_id": pack_label,
                        "size": sz, "units_in_pack": units
                    })

            for c in cities:
                n = round(value(x[p, c]) or 0)
                if n > 0:
                    prod_handling += HANDLING_COST * n
                    results_allocation.append({
                        "product": prod, "pack_id": pack_label,
                        "city": c, "n_packs": n
                    })

    for c in cities:
        for sz in sizes:
            ov = value(o[c, sz]) or 0
            if ov > 0.01:
                prod_capital += COST_OF_CAPITAL * unit_cost * ov

    prod_total = prod_creation + prod_handling + prod_capital
    results_summary.append({
        "product": prod,
        "n_pack_types":     sum(1 for p in pack_ids if (value(y[p]) or 0) > 0.5),
        "total_packs_sent": sum(round(value(x[p, c]) or 0)
                                for p in pack_ids for c in cities
                                if (value(y[p]) or 0) > 0.5),
        "creation_cost":  round(prod_creation, 2),
        "handling_cost":  round(prod_handling, 2),
        "capital_cost":   round(prod_capital, 2),
        "total_cost":     round(prod_total, 2),
    })

    if (prod_idx + 1) % 10 == 0:
        print(f"  Solved {prod_idx+1}/{len(products)} products...")

print(f"\nAll products solved.\n")

# ─────────────────────────────────────────────
# BUILD OUTPUT DATAFRAMES
# ─────────────────────────────────────────────
df_design = pd.DataFrame(results_pack_design)
df_alloc  = pd.DataFrame(results_allocation)
df_summ   = pd.DataFrame(results_summary)

# Tab 1: pack constitution – rows=pack_id, cols=SKU (product_size)
if not df_design.empty:
    df_design["sku"] = df_design["product"] + "_" + df_design["size"]
    # Include product in index so it travels through the pivot
    tab1 = df_design.pivot_table(
        index=["pack_id", "product"], columns="sku", values="units_in_pack", fill_value=0
    ).reset_index()
    # Reorder: pack_id, product first, then SKU columns
    sku_cols = [c for c in tab1.columns if c not in ("pack_id", "product")]
    tab1 = tab1[["pack_id", "product"] + sku_cols]
else:
    tab1 = pd.DataFrame()

# Tab 2: pack allocation – rows=pack_id, cols=city
if not df_alloc.empty:
    tab2 = df_alloc.pivot_table(
        index=["pack_id", "product"], columns="city", values="n_packs",
        aggfunc="sum", fill_value=0
    ).reset_index()
    # Reorder: pack_id, product first, then city columns
    city_cols = [c for c in tab2.columns if c not in ("pack_id", "product")]
    tab2 = tab2[["pack_id", "product"] + city_cols]
else:
    tab2 = pd.DataFrame()

# ─────────────────────────────────────────────
# PRINT SUMMARY
# ─────────────────────────────────────────────
print("=" * 65)
print("COST SUMMARY PER PRODUCT")
print("=" * 65)
pd.set_option("display.float_format", "{:,.2f}".format)
print(df_summ.to_string(index=False))

print()
print("=" * 65)
print("AGGREGATE COST BREAKDOWN")
print("=" * 65)
print(f"  Pack creation cost : €{df_summ['creation_cost'].sum():>10,.2f}")
print(f"  Handling cost      : €{df_summ['handling_cost'].sum():>10,.2f}")
print(f"  Cost of capital    : €{df_summ['capital_cost'].sum():>10,.2f}")
print(f"  {'─'*40}")
print(f"  TOTAL              : €{df_summ['total_cost'].sum():>10,.2f}")

# Baseline: every unit handled individually
total_units = df["forecast_2026"].sum()
baseline    = total_units * HANDLING_COST
print()
print(f"  Baseline (unit-by-unit handling)   : €{baseline:>10,.2f}")
print(f"  Pre-pack total cost                : €{df_summ['total_cost'].sum():>10,.2f}")
print(f"  NET SAVING                         : €{baseline - df_summ['total_cost'].sum():>10,.2f}")
print(f"  Saving %                           : {(1 - df_summ['total_cost'].sum()/baseline)*100:>9.1f}%")
print()
print(f"  Total unique pack types used       : {df_summ['n_pack_types'].sum()}")
print(f"  Total packs sent                   : {df_summ['total_packs_sent'].sum()}")

# ─────────────────────────────────────────────
# SAVE TO EXCEL
# ─────────────────────────────────────────────
out_path = "prepack_solution.xlsx"
with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
    tab1.to_excel(writer, sheet_name="Pack_Constitution", index=False)
    tab2.to_excel(writer, sheet_name="Pack_Allocation",   index=False)
    df_summ.to_excel(writer, sheet_name="Cost_Summary",   index=False)
    df_design.to_excel(writer, sheet_name="Detail_Design",     index=False)
    df_alloc.to_excel(writer,  sheet_name="Detail_Allocation",  index=False)

print(f"\nSolution saved to: {out_path}")

# ─────────────────────────────────────────────
# BUILD DELIVERY ANALYSIS TAB
# ─────────────────────────────────────────────
df_analysis = df[["stad", "product", "maat", "forecast_2026"]].copy()
df_analysis = df_analysis.rename(columns={"forecast_2026": "forecast_demand"})

# Compute delivered units per (stad, product, maat) from allocation results
delivered_rows = []
for alloc in results_allocation:
    prod     = alloc["product"]
    pack_lbl = alloc["pack_id"]
    city     = alloc["city"]
    n_packs  = alloc["n_packs"]
    candidates_local = generate_candidate_packs(prod)
    # Look up the original local index via the pack_label→local_idx mapping
    pack_config = pack_label_to_config[pack_lbl]
    for sz, units in pack_config.items():
        if units > 0:
            delivered_rows.append({
                "stad": city, "product": prod,
                "maat": sz,   "delivered": n_packs * units
            })

if delivered_rows:
    df_delivered = pd.DataFrame(delivered_rows)
    df_delivered = df_delivered.groupby(["stad","product","maat"], as_index=False)["delivered"].sum()
else:
    df_delivered = pd.DataFrame(columns=["stad","product","maat","delivered"])

df_analysis = df_analysis.merge(df_delivered, on=["stad","product","maat"], how="left")
df_analysis["delivered"] = df_analysis["delivered"].fillna(0).astype(int)

df_analysis["overstock"]      = (df_analysis["delivered"] - df_analysis["forecast_demand"]).clip(lower=0)
df_analysis["understock"]     = (df_analysis["forecast_demand"] - df_analysis["delivered"]).clip(lower=0)
df_analysis["service_level"]  = (
    df_analysis["delivered"] / df_analysis["forecast_demand"].replace(0, float("nan"))
).clip(upper=1.0).round(4)
df_analysis["unit_cost"]      = df_analysis["product"].map(UNIT_COST_MAP)
df_analysis["overstock_cost"] = (
    df_analysis["overstock"] * COST_OF_CAPITAL * df_analysis["unit_cost"]
).round(2)

df_analysis = df_analysis[[
    "stad", "product", "maat",
    "forecast_demand", "delivered",
    "understock", "overstock",
    "service_level", "overstock_cost"
]]

# ─────────────────────────────────────────────
# SAVE UPDATED EXCEL (with analysis tab)
# ─────────────────────────────────────────────
out_path2 = "Optimization/Optimization_single_prod_packs/prepack_solution_v2.xlsx"
with pd.ExcelWriter(out_path2, engine="openpyxl") as writer:
    tab1.to_excel(writer, sheet_name="Pack_Constitution",   index=False)
    tab2.to_excel(writer, sheet_name="Pack_Allocation",     index=False)
    df_summ.to_excel(writer, sheet_name="Cost_Summary",     index=False)
    df_analysis.to_excel(writer, sheet_name="Delivery_Analysis", index=False)
    df_design.to_excel(writer, sheet_name="Detail_Design",  index=False)
    df_alloc.to_excel(writer,  sheet_name="Detail_Allocation", index=False)

print(f"\nUpdated solution saved to: {out_path2}")

# Quick sanity check
print(f"\nDelivery Analysis – sample:")
print(df_analysis[df_analysis["product"]=="093KT7ZK38"][
    ["stad","maat","forecast_demand","delivered","service_level","overstock_cost"]
].to_string(index=False))

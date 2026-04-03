import pandas as pd
import math

# -----------------------------
# Load input
# -----------------------------
raw = pd.read_excel("stockout_analysis.xlsx", sheet_name="Raw_Merged")

# -----------------------------
# 1. Compute Counts (Non-Stockouts, Stockouts, Total)
# -----------------------------
counts = raw.groupby(["product_id", "channel_id"])["stockout_flag"].agg(
    num_non_stockouts=lambda x: (x == False).sum(),
    num_stockouts=lambda x: (x == True).sum(),
    total_observations="count"
).reset_index()

# -----------------------------
# 2. Compute Fill Rate Averages and Fallbacks
# -----------------------------
non_stockout = raw[raw["stockout_flag"] == False]

# Level 1: Product x Channel
avg_fill_rate_exact = (
    non_stockout.groupby(["product_id", "channel_id"])["fill_rate"]
    .mean()
    .reset_index()
    .rename(columns={"fill_rate": "fill_rate_exact"})
)

# Level 2: Product-level (across all channels)
product_avg_fill = (
    non_stockout.groupby("product_id")["fill_rate"]
    .mean()
    .reset_index()
    .rename(columns={"fill_rate": "product_avg_fill_rate"})
)

# Level 3: Global average
global_avg_fill = non_stockout["fill_rate"].mean()

# -----------------------------
# 3. Merge and Determine Fallback
# -----------------------------
all_combos = raw[["product_id", "channel_id"]].drop_duplicates()

# Merge all stats together
corrections = all_combos.merge(counts, on=["product_id", "channel_id"], how="left")
corrections = corrections.merge(avg_fill_rate_exact, on=["product_id", "channel_id"], how="left")
corrections = corrections.merge(product_avg_fill, on="product_id", how="left")

# Determine which fill rate to use and label the fallback
def determine_fallback(row):
    if not pd.isna(row["fill_rate_exact"]):
        return row["fill_rate_exact"], "Exact (Product x Channel)"
    elif not pd.isna(row["product_avg_fill_rate"]):
        return row["product_avg_fill_rate"], "Product Level"
    else:
        return global_avg_fill, "Global"

corrections[["avg_fill_rate", "fallback_used"]] = corrections.apply(
    lambda r: pd.Series(determine_fallback(r)), axis=1
)

# Clean up temporary columns for the final Sheet 2
avg_fill_rate_final = corrections[[
    "product_id", "channel_id", "num_non_stockouts", "num_stockouts",
    "total_observations", "avg_fill_rate", "fallback_used"
]]

# -----------------------------
# Aggregate observed demand (Same as original)
# -----------------------------
observed = (
    raw.groupby(["product_id", "channel_id", "season", "size"])
    .agg(units_sold=("units_sold", "sum"), stockout_flag=("stockout_flag", "max"))
    .reset_index()
)

# -----------------------------
# Compute True Demand (Using the merged corrections)
# -----------------------------
# We use the avg_fill_rate column from our new corrections table
true_demand = observed.merge(
    avg_fill_rate_final[["product_id", "channel_id", "avg_fill_rate"]],
    on=["product_id", "channel_id"],
    how="left"
)

true_demand["true_demand"] = true_demand.apply(
    lambda r: math.ceil(r["units_sold"] / r["avg_fill_rate"]) if r["stockout_flag"] else r["units_sold"],
    axis=1
)
true_demand["correction"] = true_demand["true_demand"] - true_demand["units_sold"]

# -----------------------------
# Summary (Same as original)
# -----------------------------
summary = (
    true_demand.groupby(["product_id", "channel_id"])
    .agg(
        total_observed=("units_sold", "sum"),
        total_true_demand=("true_demand", "sum"),
        total_correction=("correction", "sum"),
        avg_fill_rate=("avg_fill_rate", "first"),
    )
    .reset_index()
)
summary["correction_pct"] = (
    (summary["total_true_demand"] - summary["total_observed"])
    / summary["total_observed"]
    * 100
).round(2)

# -----------------------------
# Write to Excel
# -----------------------------
OUTPUT_FILE = "true_demand.xlsx"

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    # Sheet 1: granular demand
    true_demand[["product_id", "channel_id", "season", "size", "units_sold", "avg_fill_rate", "true_demand", "correction"]].to_excel(
        writer, sheet_name="true_demand_full", index=False
    )

    # Sheet 2: UPDATED corrections with counts and fallback labels
    avg_fill_rate_final.to_excel(writer, sheet_name="fill_rate_corrections", index=False)

    # Sheet 3: summary
    summary.to_excel(writer, sheet_name="summary_product_channel", index=False)

    # Sheet 4: pivot
    pivot = true_demand.groupby(["product_id", "channel_id"])["true_demand"].sum().unstack("channel_id").reset_index()
    pivot.columns.name = None
    pivot.to_excel(writer, sheet_name="true_demand_pivot", index=False)

print(f"\nAll outputs saved to {OUTPUT_FILE}")
print(f"Sheet 'fill_rate_corrections' now includes observation counts and fallback tracking.")
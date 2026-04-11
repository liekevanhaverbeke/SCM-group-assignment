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

corrections = all_combos.merge(counts, on=["product_id", "channel_id"], how="left")
corrections = corrections.merge(avg_fill_rate_exact, on=["product_id", "channel_id"], how="left")
corrections = corrections.merge(product_avg_fill, on="product_id", how="left")

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

avg_fill_rate_final = corrections[[
    "product_id", "channel_id", "num_non_stockouts", "num_stockouts",
    "total_observations", "avg_fill_rate", "fallback_used"
]]

# -----------------------------
# Aggregate observed demand
# -----------------------------
observed = (
    raw.groupby(["product_id", "channel_id", "season", "size"])
    .agg(units_sold=("units_sold", "sum"), stockout_flag=("stockout_flag", "max"))
    .reset_index()
)

# -----------------------------
# Compute True Demand
# -----------------------------
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
# Summary
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
# Demand overview: observed vs corrected
# Per season and per channel — simple totals for quick comparison
# -----------------------------

# By season
by_season = (
    true_demand.groupby("season")
    .agg(
        observed_demand=("units_sold", "sum"),
        true_demand=("true_demand", "sum"),
    )
    .reset_index()
)
by_season["correction_units"] = by_season["true_demand"] - by_season["observed_demand"]
by_season["correction_pct"] = (by_season["correction_units"] / by_season["observed_demand"] * 100).round(2)

# By channel
by_channel = (
    true_demand.groupby("channel_id")
    .agg(
        observed_demand=("units_sold", "sum"),
        true_demand=("true_demand", "sum"),
    )
    .reset_index()
)
by_channel["correction_units"] = by_channel["true_demand"] - by_channel["observed_demand"]
by_channel["correction_pct"] = (by_channel["correction_units"] / by_channel["observed_demand"] * 100).round(2)
by_channel = by_channel.sort_values("correction_pct", ascending=False)

# By product
by_product = (
    true_demand.groupby("product_id")
    .agg(
        observed_demand=("units_sold", "sum"),
        true_demand=("true_demand", "sum"),
    )
    .reset_index()
)
by_product["correction_units"] = by_product["true_demand"] - by_product["observed_demand"]
by_product["correction_pct"] = (by_product["correction_units"] / by_product["observed_demand"] * 100).round(2)
by_product = by_product.sort_values("correction_pct", ascending=False)

# Overall totals row
total_observed = true_demand["units_sold"].sum()
total_true = true_demand["true_demand"].sum()
overall = pd.DataFrame([{
    "level": "TOTAL",
    "observed_demand": total_observed,
    "true_demand": total_true,
    "correction_units": total_true - total_observed,
    "correction_pct": round((total_true - total_observed) / total_observed * 100, 2)
}])

# -----------------------------
# Write to Excel
# -----------------------------
OUTPUT_FILE = "true_demand.xlsx"

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    # Sheet 1: granular demand
    true_demand[["product_id", "channel_id", "season", "size", "units_sold", "avg_fill_rate", "true_demand", "correction"]].to_excel(
        writer, sheet_name="true_demand_full", index=False
    )

    # Sheet 2: corrections with counts and fallback labels
    avg_fill_rate_final.to_excel(writer, sheet_name="fill_rate_corrections", index=False)

    # Sheet 3: summary per product x channel
    summary.to_excel(writer, sheet_name="summary_product_channel", index=False)

    # Sheet 4: pivot
    pivot = true_demand.groupby(["product_id", "channel_id"])["true_demand"].sum().unstack("channel_id").reset_index()
    pivot.columns.name = None
    pivot.to_excel(writer, sheet_name="true_demand_pivot", index=False)

    # Sheet 5: demand overview — observed vs corrected by season, channel, product
    row = 0
    ws = writer.book.create_sheet("demand_overview")
    writer.sheets["demand_overview"] = ws

    # Overall total
    for df_part, label in [
        (overall.rename(columns={"level": ""}), "Overall"),
        (by_season.rename(columns={"season": "season"}), "By Season"),
        (by_channel.rename(columns={"channel_id": "channel"}), "By Channel"),
        (by_product.rename(columns={"product_id": "product"}), "By Product"),
    ]:
        ws.cell(row=row+1, column=1, value=label)
        for col_idx, col_name in enumerate(df_part.columns, start=1):
            ws.cell(row=row+2, column=col_idx, value=col_name)
        for r_idx, data_row in enumerate(df_part.itertuples(index=False), start=row+3):
            for col_idx, val in enumerate(data_row, start=1):
                ws.cell(row=r_idx, column=col_idx, value=val)
        row += len(df_part) + 4

print(f"\nAll outputs saved to {OUTPUT_FILE}")
print(f"\nOverall: {total_observed:,} observed → {total_true:,} true demand (+{(total_true/total_observed-1)*100:.1f}%)")
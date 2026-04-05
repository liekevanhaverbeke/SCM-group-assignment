import pandas as pd
import math

# =========================================================
# 1. DATA INLADEN & VOORBEREIDING
# =========================================================
input_file = "stockout_analysis.xlsx"
raw = pd.read_excel(input_file, sheet_name="Raw_Merged")

raw['stockout_flag'] = raw['stockout_flag'].astype(bool)

# =========================================================
# 2. BASISSTATISTIEKEN BEREKENEN
# =========================================================

stats = raw.groupby(["product_id", "channel_id"])["stockout_flag"].agg(
    n_stockouts="sum",
    total_obs="count"
).reset_index()

stats["n_non_stockouts"] = stats["total_obs"] - stats["n_stockouts"]

non_stockout_data = raw[raw["stockout_flag"] == False].copy()

local_fr = non_stockout_data.groupby(["product_id", "channel_id"])["fill_rate"].mean().reset_index()

prod_fr = non_stockout_data.groupby("product_id")["fill_rate"].mean().reset_index().rename(
    columns={"fill_rate": "prod_rate"})

global_avg_rate = non_stockout_data["fill_rate"].mean()

tab2 = stats.merge(local_fr, on=["product_id", "channel_id"], how="left")
tab2 = tab2.merge(prod_fr, on="product_id", how="left")


# =========================================================
# 3. DE SMART STRATEGIE LOGICA
# =========================================================

def apply_smart_strategy(row):
    n = row["n_non_stockouts"]
    local_val = row["fill_rate"]
    prod_val = row["prod_rate"]

    if n >= 10:
        return local_val, "Segment A: Hard Filter (>=10 obs)"
    elif n > 0:
        weight = n / 10
        blended_rate = (1.0 * (1 - weight)) + (local_val * weight)
        return blended_rate, f"Segment B: Glijdende schaal ({int(weight * 100)}% weging)"
    else:
        if pd.notna(prod_val):
            safe_rate = (1.0 + prod_val) / 2
            return safe_rate, "Segment C: Blind Spot (Product Fallback Safe)"
        else:
            return (1.0 + global_avg_rate) / 2, "Segment C: Blind Spot (Global Safe)"


tab2[["used_fill_rate", "strategy_label"]] = tab2.apply(
    lambda r: pd.Series(apply_smart_strategy(r)), axis=1
)

# =========================================================
# 4. TRUE DEMAND BEREKENEN
# =========================================================

observed = raw.groupby(["product_id", "channel_id", "season", "size"]).agg(
    units_sold=("units_sold", "sum"),
    stockout_flag=("stockout_flag", "max")
).reset_index()

true_demand_df = observed.merge(
    tab2[["product_id", "channel_id", "used_fill_rate", "strategy_label"]],
    on=["product_id", "channel_id"],
    how="left"
)

true_demand_df["true_demand"] = true_demand_df.apply(
    lambda r: math.ceil(r["units_sold"] / r["used_fill_rate"]) if r["stockout_flag"] else r["units_sold"],
    axis=1
)

true_demand_df["correction_units"] = true_demand_df["true_demand"] - true_demand_df["units_sold"]

# =========================================================
# 5. DEMAND OVERVIEW: observed vs corrected
# =========================================================

# By season
by_season = (
    true_demand_df.groupby("season")
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
    true_demand_df.groupby("channel_id")
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
    true_demand_df.groupby("product_id")
    .agg(
        observed_demand=("units_sold", "sum"),
        true_demand=("true_demand", "sum"),
    )
    .reset_index()
)
by_product["correction_units"] = by_product["true_demand"] - by_product["observed_demand"]
by_product["correction_pct"] = (by_product["correction_units"] / by_product["observed_demand"] * 100).round(2)
by_product = by_product.sort_values("correction_pct", ascending=False)

# Overall total
total_observed = true_demand_df["units_sold"].sum()
total_true = true_demand_df["true_demand"].sum()
overall = pd.DataFrame([{
    "level": "TOTAL",
    "observed_demand": total_observed,
    "true_demand": total_true,
    "correction_units": total_true - total_observed,
    "correction_pct": round((total_true - total_observed) / total_observed * 100, 2)
}])

# =========================================================
# 6. RESULTATEN OPSLAAN
# =========================================================

pivot_table = true_demand_df.pivot_table(
    index="product_id",
    columns="channel_id",
    values="true_demand",
    aggfunc="sum"
).reset_index()

output_file = "True_Demand_Results_Strategies.xlsx"

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    true_demand_df.to_excel(writer, sheet_name="1_True_Demand_Lijst", index=False)
    tab2.to_excel(writer, sheet_name="2_Factor_Uitleg", index=False)
    pivot_table.to_excel(writer, sheet_name="3_Inkoop_Matrix", index=False)

    # Sheet 4: demand overview
    ws = writer.book.create_sheet("4_Demand_Overview")
    writer.sheets["4_Demand_Overview"] = ws

    row = 0
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

print(f"Succes! De analyse is opgeslagen in: {output_file}")
print(f"\nOverall: {total_observed:,} observed → {total_true:,} true demand (+{(total_true/total_observed-1)*100:.1f}%)")
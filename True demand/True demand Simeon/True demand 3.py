import pandas as pd
import numpy as np

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

# ALPHA = 0 → alleen niet-stockout seizoenen (origineel, overcorrigeert)
# ALPHA = 1 → alle seizoenen inclusief stockouts (conservatiever)
# ALPHA = 0.5 → tussenoplossing
ALPHA = 0.55

local_fr_non_stockout = (
    non_stockout_data.groupby(["product_id", "channel_id"])["fill_rate"]
    .mean()
    .rename("fr_non_stockout")
)
local_fr_all = (
    raw.groupby(["product_id", "channel_id"])["fill_rate"]
    .mean()
    .rename("fr_all")
)

local_fr_blended = pd.concat([local_fr_non_stockout, local_fr_all], axis=1).reset_index()
local_fr_blended["fill_rate"] = (
    ALPHA * local_fr_blended["fr_all"] +
    (1 - ALPHA) * local_fr_blended["fr_non_stockout"]
)
# Als er geen niet-stockout data is, val terug op fr_all
local_fr_blended["fill_rate"] = local_fr_blended["fill_rate"].fillna(local_fr_blended["fr_all"])
local_fr_blended = local_fr_blended[["product_id", "channel_id", "fill_rate"]]

prod_fr_non_stockout = non_stockout_data.groupby("product_id")["fill_rate"].mean().rename("fr_non_stockout")
prod_fr_all = raw.groupby("product_id")["fill_rate"].mean().rename("fr_all")

prod_fr_blended = pd.concat([prod_fr_non_stockout, prod_fr_all], axis=1).reset_index()
prod_fr_blended["prod_rate"] = (
    ALPHA * prod_fr_blended["fr_all"] +
    (1 - ALPHA) * prod_fr_blended["fr_non_stockout"]
)
prod_fr_blended["prod_rate"] = prod_fr_blended["prod_rate"].fillna(prod_fr_blended["fr_all"])
prod_fr_blended = prod_fr_blended[["product_id", "prod_rate"]]

global_avg_non_stockout = non_stockout_data["fill_rate"].mean()
global_avg_all = raw["fill_rate"].mean()
global_avg_rate = ALPHA * global_avg_all + (1 - ALPHA) * global_avg_non_stockout

print(f"ALPHA: {ALPHA}")
print(f"Globaal gemiddelde fill rate (niet-stockout only): {global_avg_non_stockout:.4f}")
print(f"Globaal gemiddelde fill rate (inclusief stockouts): {global_avg_all:.4f}")
print(f"Globaal gemiddelde fill rate (blended):            {global_avg_rate:.4f}")

tab2 = stats.merge(local_fr_blended, on=["product_id", "channel_id"], how="left")
tab2 = tab2.merge(prod_fr_blended, on="product_id", how="left")

# =========================================================
# 3. SMART STRATEGY LOGICA
# =========================================================
def apply_smart_strategy(row):
    n         = row["n_non_stockouts"]
    local_val = row["fill_rate"]
    prod_val  = row["prod_rate"]

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
# 4. SIZE-VERHOUDINGEN UIT NIET-STOCKOUT SEIZOENEN
# =========================================================
size_totals = (
    non_stockout_data.groupby(["product_id", "channel_id", "size"])["units_sold"]
    .sum()
    .reset_index()
    .rename(columns={"units_sold": "size_units"})
)

combo_totals = (
    size_totals.groupby(["product_id", "channel_id"])["size_units"]
    .sum()
    .reset_index()
    .rename(columns={"size_units": "combo_total"})
)

size_shares = size_totals.merge(combo_totals, on=["product_id", "channel_id"])
size_shares["size_share"] = size_shares["size_units"] / size_shares["combo_total"]
size_shares = size_shares[["product_id", "channel_id", "size", "size_share"]]

# =========================================================
# 5. CORRECTIE OP SEIZOENSNIVEAU
# =========================================================
season_level = (
    raw.groupby(["product_id", "channel_id", "season"]).agg(
        units_sold=("units_sold", "sum"),
        stockout_flag=("stockout_flag", "max")
    ).reset_index()
)

season_level = season_level.merge(
    tab2[["product_id", "channel_id", "used_fill_rate", "strategy_label"]],
    on=["product_id", "channel_id"],
    how="left"
)

# Niet-stockout: true demand = units_sold (perfecte data, geen correctie)
# Stockout: true demand = units_sold / used_fill_rate, nooit lager dan units_sold
season_level["true_demand_season"] = season_level.apply(
    lambda r: max(r["units_sold"] / r["used_fill_rate"], r["units_sold"])
    if r["stockout_flag"] else float(r["units_sold"]),
    axis=1
)
season_level["correction_units_season"] = (
    season_level["true_demand_season"] - season_level["units_sold"]
)

# =========================================================
# 6. VERDEEL ALLEEN DE EXTRA CORRECTIE OVER SIZES
# =========================================================
raw_size = raw.groupby(["product_id", "channel_id", "season", "size"]).agg(
    units_sold_size=("units_sold", "sum")
).reset_index()

season_totals = season_level[[
    "product_id", "channel_id", "season",
    "units_sold", "true_demand_season",
    "correction_units_season", "used_fill_rate",
    "strategy_label", "stockout_flag"
]]

true_demand_df = raw_size.merge(season_totals, on=["product_id", "channel_id", "season"], how="left")
true_demand_df = true_demand_df.merge(size_shares, on=["product_id", "channel_id", "size"], how="left")

n_sizes = (
    raw.groupby(["product_id", "channel_id"])["size"]
    .nunique()
    .reset_index()
    .rename(columns={"size": "n_sizes"})
)
true_demand_df = true_demand_df.merge(n_sizes, on=["product_id", "channel_id"], how="left")
true_demand_df["size_share"] = true_demand_df["size_share"].fillna(
    1 / true_demand_df["n_sizes"]
)

# Basis units_sold_size blijft altijd intact
# Alleen de extra correctie wordt verdeeld via size_share
# Seizoenstotaal is altijd exact true_demand_season
# true_demand per size is altijd >= units_sold_size
true_demand_df["true_demand"] = true_demand_df.apply(
    lambda r: float(r["units_sold_size"]) + r["correction_units_season"] * r["size_share"]
    if r["stockout_flag"] else float(r["units_sold_size"]),
    axis=1
)
true_demand_df["units_sold"]       = true_demand_df["units_sold_size"]
true_demand_df["correction_units"] = true_demand_df["true_demand"] - true_demand_df["units_sold"]

# =========================================================
# 7. DEMAND OVERVIEW
# =========================================================
by_season = (
    true_demand_df.groupby("season")
    .agg(
        observed_demand=("units_sold", "sum"),
        true_demand=("true_demand", "sum"),
    )
    .reset_index()
)
by_season["correction_units"] = by_season["true_demand"] - by_season["observed_demand"]
by_season["correction_pct"] = (
    by_season["correction_units"] / by_season["observed_demand"] * 100
).round(2)

by_channel = (
    true_demand_df.groupby("channel_id")
    .agg(
        observed_demand=("units_sold", "sum"),
        true_demand=("true_demand", "sum"),
    )
    .reset_index()
)
by_channel["correction_units"] = by_channel["true_demand"] - by_channel["observed_demand"]
by_channel["correction_pct"] = (
    by_channel["correction_units"] / by_channel["observed_demand"] * 100
).round(2)
by_channel = by_channel.sort_values("correction_pct", ascending=False)

by_product = (
    true_demand_df.groupby("product_id")
    .agg(
        observed_demand=("units_sold", "sum"),
        true_demand=("true_demand", "sum"),
    )
    .reset_index()
)
by_product["correction_units"] = by_product["true_demand"] - by_product["observed_demand"]
by_product["correction_pct"] = (
    by_product["correction_units"] / by_product["observed_demand"] * 100
).round(2)
by_product = by_product.sort_values("correction_pct", ascending=False)

total_observed = true_demand_df["units_sold"].sum()
total_true     = true_demand_df["true_demand"].sum()
overall = pd.DataFrame([{
    "level": "TOTAL",
    "observed_demand": total_observed,
    "true_demand": total_true,
    "correction_units": total_true - total_observed,
    "correction_pct": round((total_true - total_observed) / total_observed * 100, 2)
}])

print(f"\nOverall: {total_observed:,.0f} observed → {total_true:,.0f} true demand "
      f"(+{(total_true/total_observed - 1)*100:.1f}%)")

# =========================================================
# 8. OPSLAAN
# =========================================================
pivot_table = true_demand_df.pivot_table(
    index="product_id",
    columns="channel_id",
    values="true_demand",
    aggfunc="sum"
).reset_index()

output_file = "True_Demand_Results_3.xlsx"

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    true_demand_df[[
        "product_id", "channel_id", "season", "size",
        "units_sold", "size_share", "true_demand", "correction_units",
        "used_fill_rate", "strategy_label", "stockout_flag"
    ]].to_excel(writer, sheet_name="1_True_Demand_Lijst", index=False)

    tab2.to_excel(writer, sheet_name="2_Factor_Uitleg", index=False)
    size_shares.to_excel(writer, sheet_name="3_Size_Verdeling", index=False)
    pivot_table.to_excel(writer, sheet_name="4_Inkoop_Matrix", index=False)

    ws = writer.book.create_sheet("5_Demand_Overview")
    writer.sheets["5_Demand_Overview"] = ws
    row = 0
    for df_part, label in [
        (overall.rename(columns={"level": ""}), "Overall"),
        (by_season, "By Season"),
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

print(f"Opgeslagen in: {output_file}")
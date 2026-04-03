import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os

# Set working directory to the location of this script
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ─────────────────────────────────────────────
# 1. Load data
# ─────────────────────────────────────────────
demand_df = pd.read_excel("Input_Files/PPP_stu_demand.xlsx")   # units sold
stock_df  = pd.read_excel("Input_Files/PPP_stu_stock.xlsx")    # units available
product_df = pd.read_excel("Input_Files/PPP_stu_products.xlsx") # product attributes (gender/category)
details_df = pd.read_excel("Input_Files/PPP_stu_details.xlsx")  # color & wearable type

# Rename 'units' so we can tell them apart after the merge
demand_df = demand_df.rename(columns={"units": "units_sold"})
stock_df  = stock_df.rename(columns={"units":  "units_stocked"})

# ─────────────────────────────────────────────
# 2. Merge on all shared keys
# ─────────────────────────────────────────────
merge_keys = ["product_id", "channel_id", "season", "size"]
merged = pd.merge(demand_df, stock_df, on=merge_keys, how="outer")

# ─────────────────────────────────────────────
# 3. Compute metrics
# ─────────────────────────────────────────────
merged["units_sold"]    = merged["units_sold"].fillna(0)
merged["units_stocked"] = merged["units_stocked"].fillna(0)

# Fill rate: how much of available stock was sold (1.0 = completely sold out)
merged["fill_rate"] = merged.apply(
    lambda r: r["units_sold"] / r["units_stocked"] if r["units_stocked"] > 0 else np.nan, axis=1
)

# Unsold units left on the shelf
merged["unsold_units"] = merged["units_stocked"] - merged["units_sold"]

# Stockout flag: sold == stocked AND stocked > 0 → could have sold more
merged["stockout_flag"] = (merged["units_sold"] == merged["units_stocked"]) & (merged["units_stocked"] > 0)

print(f"Total records: {len(merged)}")
print(f"Stockout events: {merged['stockout_flag'].sum()} ({merged['stockout_flag'].mean()*100:.1f}%)")

# ─────────────────────────────────────────────
# 3b. Enrich with product attributes
# ─────────────────────────────────────────────
# From products file: gender category (Menswear / Womenswear / ...) + price
attr_cols = product_df[["id", "category", "price"]].rename(
    columns={"id": "product_id", "category": "gender"}
)
merged = merged.merge(attr_cols, on="product_id", how="left")

# From details file: Color and Wearable type (Hoodie / T-Shirt / ...)
detail_cols = details_df[["id", "Color", "Wearable"]].rename(columns={"id": "product_id"})
merged = merged.merge(detail_cols, on="product_id", how="left")

print(f"Gender values: {merged['gender'].unique().tolist()}")
print(f"Unique colors: {merged['Color'].nunique()}  |  Unique wearable types: {merged['Wearable'].nunique()}")

# ─────────────────────────────────────────────
# 4. Analysis tables
# ─────────────────────────────────────────────

# --- Tab 1: Full merged raw data (now includes gender, Color, Wearable) ---
raw_merged = merged.sort_values(["product_id", "season", "channel_id", "size"])

# --- Tab 2: Stockout summary per product × season ---
# Shows % of size/channel combos that stocked out for each product each year
stockout_summary = merged.groupby(["product_id", "season"]).agg(
    total_combos      = ("stockout_flag", "count"),
    stockout_events   = ("stockout_flag", "sum"),
    avg_fill_rate     = ("fill_rate",     "mean"),
    total_sold        = ("units_sold",    "sum"),
    total_stocked     = ("units_stocked", "sum"),
    total_unsold      = ("unsold_units",  "sum"),
).reset_index()
stockout_summary["stockout_rate_pct"] = (
    stockout_summary["stockout_events"] / stockout_summary["total_combos"] * 100
).round(1)
stockout_summary["avg_fill_rate"] = stockout_summary["avg_fill_rate"].round(3)
stockout_summary = stockout_summary.sort_values("stockout_rate_pct", ascending=False)

# --- Tab 3: Lost sales analysis by size ---
# Sizes with fill_rate == 1 most often may be systematically under-stocked
size_analysis = merged.groupby("size").agg(
    total_combos    = ("stockout_flag", "count"),
    stockout_events = ("stockout_flag", "sum"),
    avg_fill_rate   = ("fill_rate",     "mean"),
    avg_unsold      = ("unsold_units",  "mean"),
    total_sold      = ("units_sold",    "sum"),
    total_stocked   = ("units_stocked", "sum"),
).reset_index()
size_analysis["stockout_rate_pct"] = (
    size_analysis["stockout_events"] / size_analysis["total_combos"] * 100
).round(1)
size_analysis["avg_fill_rate"]  = size_analysis["avg_fill_rate"].round(3)
size_analysis["avg_unsold"]     = size_analysis["avg_unsold"].round(1)

# Sort sizes logically
size_order = ["XS", "S", "M", "L", "XL", "XXL"]
size_analysis["size_rank"] = size_analysis["size"].map(
    {s: i for i, s in enumerate(size_order)}
).fillna(99)
size_analysis = size_analysis.sort_values("size_rank").drop(columns="size_rank")

# --- Tab 4: Overstocked items (consistently low fill rate → excess stock) ---
# Products × channel combos where avg fill_rate < 0.5 (sold less than half of stock on average)
overstock = merged.groupby(["product_id", "channel_id"]).agg(
    avg_fill_rate  = ("fill_rate",    "mean"),
    total_unsold   = ("unsold_units", "sum"),
    total_stocked  = ("units_stocked","sum"),
    total_sold     = ("units_sold",   "sum"),
    seasons_active = ("season",       "nunique"),
).reset_index()
overstock["avg_fill_rate"] = overstock["avg_fill_rate"].round(3)
overstock["waste_pct"] = (overstock["total_unsold"] / overstock["total_stocked"] * 100).round(1)
overstock = overstock[overstock["avg_fill_rate"] < 0.8].sort_values("avg_fill_rate")

print(f"\nOverstocked product×channel combos (avg fill rate < 50%): {len(overstock)}")

# --- Tab 5: Stockout rate per location (channel_id) ---
location_summary = merged.groupby("channel_id").agg(
    total_combos      = ("stockout_flag", "count"),
    stockout_events   = ("stockout_flag", "sum"),
    avg_fill_rate     = ("fill_rate",     "mean"),
    total_sold        = ("units_sold",    "sum"),
    total_stocked     = ("units_stocked", "sum"),
    total_unsold      = ("unsold_units",  "sum"),
).reset_index()
location_summary["stockout_rate_pct"] = (
    location_summary["stockout_events"] / location_summary["total_combos"] * 100
).round(1)
location_summary["avg_fill_rate"] = location_summary["avg_fill_rate"].round(3)
location_summary = location_summary.sort_values("stockout_rate_pct", ascending=False)

# --- Tab 6: Fill rate pivot – location × size (avg across all products & seasons) ---
# Shows if a specific city consistently runs out of a specific size
loc_size_pivot = (
    merged.groupby(["channel_id", "size"])["fill_rate"]
    .mean()
    .unstack("size")
)
ordered_sizes_present = [s for s in size_order if s in loc_size_pivot.columns]
loc_size_pivot = loc_size_pivot[ordered_sizes_present].round(3)
loc_size_pivot_reset = loc_size_pivot.reset_index()

# --- Tab 7: Stockout rate per product × channel × size (most granular) ---
# Directly answers: "does Stockholm always stock out on XS for product A?"
prod_ch_size = merged.groupby(["product_id", "channel_id", "size"]).agg(
    seasons_active    = ("season",        "nunique"),
    stockout_seasons  = ("stockout_flag",  "sum"),
    avg_fill_rate     = ("fill_rate",      "mean"),
    total_sold        = ("units_sold",     "sum"),
    total_stocked     = ("units_stocked",  "sum"),
).reset_index()
prod_ch_size["stockout_rate_pct"] = (
    prod_ch_size["stockout_seasons"] / prod_ch_size["seasons_active"] * 100
).round(1)
prod_ch_size["avg_fill_rate"] = prod_ch_size["avg_fill_rate"].round(3)
# Add size sort order
prod_ch_size["size_rank"] = prod_ch_size["size"].map(
    {s: i for i, s in enumerate(size_order)}
).fillna(99)
prod_ch_size = prod_ch_size.sort_values(
    ["product_id", "channel_id", "size_rank"]
).drop(columns="size_rank")

print(f"Unique product × channel × size combos: {len(prod_ch_size)}")
print(f"Combos that stock out 100% of seasons: {(prod_ch_size['stockout_rate_pct'] == 100).sum()}")

def stockout_agg(df, group_col):
    """Helper: compute stockout rate table grouped by a single attribute column."""
    g = df.groupby(group_col).agg(
        total_combos    = ("stockout_flag", "count"),
        stockout_events = ("stockout_flag", "sum"),
        avg_fill_rate   = ("fill_rate",     "mean"),
        total_sold      = ("units_sold",    "sum"),
        total_stocked   = ("units_stocked", "sum"),
        total_unsold    = ("unsold_units",  "sum"),
    ).reset_index()
    g["stockout_rate_pct"] = (g["stockout_events"] / g["total_combos"] * 100).round(1)
    g["avg_fill_rate"]     = g["avg_fill_rate"].round(3)
    return g.sort_values("stockout_rate_pct", ascending=False)

# --- Tab 8: Stockout by gender (Menswear vs Womenswear) ---
gender_summary = stockout_agg(merged, "gender")

# --- Tab 9: Stockout by color ---
color_summary = stockout_agg(merged, "Color")

# --- Tab 10: Stockout by wearable type (Hoodie, T-Shirt, ...) ---
wearable_summary = stockout_agg(merged, "Wearable")

print(f"\nGender breakdown:")
print(gender_summary[["gender", "stockout_rate_pct", "avg_fill_rate"]].to_string(index=False))

# --- Tab 11: Stockout by price tier ---
# Bucket each product into a price tier, then check stockout rate per tier
price_bins   = [0, 50, 100, 150, 10_000]
price_labels = ["Budget (<=EUR50)", "Mid (EUR51-100)", "Premium (EUR101-150)", "Luxury (>EUR150)"]
merged["price_tier"] = pd.cut(
    merged["price"], bins=price_bins, labels=price_labels, right=True
)
merged["price_tier"] = merged["price_tier"].astype(str)  # make Excel-friendly

price_tier_summary = stockout_agg(merged, "price_tier")
# Keep tiers in logical order
price_tier_summary["_order"] = price_tier_summary["price_tier"].map(
    {l: i for i, l in enumerate(price_labels)}
).fillna(99)
price_tier_summary = price_tier_summary.sort_values("_order").drop(columns="_order")

# Per-product scatter data: avg price vs stockout rate
prod_price_stockout = (
    merged.groupby("product_id")
    .agg(
        price           = ("price",        "first"),
        stockout_rate   = ("stockout_flag", "mean"),
        avg_fill_rate   = ("fill_rate",     "mean"),
    )
    .reset_index()
)
prod_price_stockout["stockout_rate_pct"] = (prod_price_stockout["stockout_rate"] * 100).round(1)
prod_price_stockout["avg_fill_rate"]     = prod_price_stockout["avg_fill_rate"].round(3)
prod_price_stockout = prod_price_stockout.sort_values("price")

print(f"\nPrice tier breakdown:")
print(price_tier_summary[["price_tier", "stockout_rate_pct", "avg_fill_rate"]].to_string(index=False))

# ─────────────────────────────────────────────
# 4b. Combo / pattern mining
# ─────────────────────────────────────────────

# --- Tab 13: Chronic stockouts ---
# Product × city × size combos that stock out in EVERY season they appear
# These are the most urgent: no matter what year, this combo always runs out
prod_lookup = product_df[["id", "name"]].rename(columns={"id": "product_id"})

chronic = (
    merged.groupby(["product_id", "channel_id", "size"]).agg(
        seasons_active   = ("season",        "nunique"),
        stockout_seasons = ("stockout_flag",  "sum"),
        avg_fill_rate    = ("fill_rate",      "mean"),
        total_sold       = ("units_sold",     "sum"),
        total_stocked    = ("units_stocked",  "sum"),
        gender           = ("gender",         "first"),
        wearable         = ("Wearable",       "first"),
        color            = ("Color",          "first"),
        price            = ("price",          "first"),
    ).reset_index()
)
chronic["stockout_rate_pct"] = (
    chronic["stockout_seasons"] / chronic["seasons_active"] * 100
).round(1)
chronic["avg_fill_rate"] = chronic["avg_fill_rate"].round(3)
# Only keep combos that ALWAYS stock out AND appeared in at least 2 seasons
chronic = chronic[
    (chronic["stockout_rate_pct"] == 100) & (chronic["seasons_active"] >= 2)
].merge(prod_lookup, on="product_id", how="left")
# Reorder columns to be human-readable
chronic = chronic[[
    "product_id", "name", "gender", "wearable", "color", "price",
    "channel_id", "size", "seasons_active", "stockout_seasons",
    "stockout_rate_pct", "avg_fill_rate", "total_sold", "total_stocked"
]].sort_values(["channel_id", "wearable", "size"])

print(f"\nChronic stockouts (100%% of ≥2 seasons): {len(chronic)} combos")
if len(chronic) > 0:
    print(chronic[["name", "channel_id", "size", "seasons_active"]].head(10).to_string(index=False))

# --- Tab 14: Gender stockout rate by city ---
# Menswear vs Womenswear within each city — does Amsterdam skew female?
gender_city = (
    merged.groupby(["channel_id", "gender"]).agg(
        total_combos    = ("stockout_flag", "count"),
        stockout_events = ("stockout_flag", "sum"),
        avg_fill_rate   = ("fill_rate",     "mean"),
        total_sold      = ("units_sold",    "sum"),
        total_stocked   = ("units_stocked", "sum"),
    ).reset_index()
)
gender_city["stockout_rate_pct"] = (
    gender_city["stockout_events"] / gender_city["total_combos"] * 100
).round(1)
gender_city["avg_fill_rate"] = gender_city["avg_fill_rate"].round(3)
gender_city = gender_city.sort_values(["channel_id", "gender"])

# Wide pivot for easy comparison in Excel
gender_city_pivot = gend_fill = gender_city.pivot(
    index="channel_id", columns="gender", values="stockout_rate_pct"
).reset_index()
gender_city_pivot.columns.name = None

# --- Tab 15: City × gender × wearable × size — the "crazy combo" table ---
# Answers: "do Males in Stockholm always buy out XL hoodies?"
city_combo = (
    merged.groupby(["channel_id", "gender", "Wearable", "size"]).agg(
        total_combos    = ("stockout_flag", "count"),
        stockout_events = ("stockout_flag", "sum"),
        avg_fill_rate   = ("fill_rate",     "mean"),
        total_sold      = ("units_sold",    "sum"),
        total_stocked   = ("units_stocked", "sum"),
    ).reset_index()
)
city_combo["stockout_rate_pct"] = (
    city_combo["stockout_events"] / city_combo["total_combos"] * 100
).round(1)
city_combo["avg_fill_rate"] = city_combo["avg_fill_rate"].round(3)
city_combo["size_rank"] = city_combo["size"].map(
    {s: i for i, s in enumerate(size_order)}
).fillna(99)
city_combo = city_combo.sort_values(
    ["stockout_rate_pct", "channel_id", "gender", "Wearable", "size_rank"],
    ascending=[False, True, True, True, True]
).drop(columns="size_rank")

print(f"\nTop 10 city × gender × wearable × size combos by stockout rate:")
print(city_combo[["channel_id","gender","Wearable","size","stockout_rate_pct","avg_fill_rate"]]
      .head(10).to_string(index=False))

# ─────────────────────────────────────────────
# 5. Export to Excel
# ─────────────────────────────────────────────

# --- README sheet (first tab) ---
doc_rows = [
    # EXCEL TABS
    ("EXCEL TABS", "", ""),
    ("Sheet name",               "Type",     "What it contains"),
    ("README",                   "Info",     "This sheet. Overview of all tabs and PNG files."),
    ("Raw_Merged",               "Raw data", "Full merged table of demand vs stock for every product / city / year / size. Includes fill_rate (sold/stocked), unsold_units, stockout_flag, gender, Color, Wearable, price."),
    ("Stockout_By_Product",      "Summary",  "Stockout rate (%) and avg fill rate per product x season. Sorted by highest stockout rate. Use to identify which products ran out most across all sizes and cities."),
    ("Lost_Sales_By_Size",       "Summary",  "Stockout rate and fill rate aggregated by clothing size (XS to XXL). Shows if certain sizes are systematically under-stocked across all products and cities."),
    ("Overstocked_Items",        "Summary",  "Product x city combinations with avg fill rate below 50%, meaning more than half of stocked units went unsold. Empty if no combos qualify."),
    ("Stockout_By_Location",     "Summary",  "Overall stockout rate per city / channel. Ranked highest to lowest. Answers: which stores run out of stock the most?"),
    ("Location_Size_Heatmap",    "Pivot",    "Pivot table: city (rows) x size (columns), values = avg fill rate (0-1). A value near 1.0 means that city consistently sells out of that size. Good for spotting city-level sizing mismatches."),
    ("Product_Channel_Size",     "Detail",   "Stockout rate per product x city x size. Shows in how many seasons each specific combination stocked out. Filter stockout_rate_pct = 100 to find chronic issues."),
    ("Stockout_By_Gender",       "Summary",  "Overall stockout rate comparing Menswear vs Womenswear across all products, cities, and seasons."),
    ("Stockout_By_Color",        "Summary",  "Stockout rate per product color (31 colors). Identifies if certain colors (e.g. Black) sell out more than others."),
    ("Stockout_By_Wearable",     "Summary",  "Stockout rate per garment type (Hoodie, T-Shirt, Dress, etc., 11 types). Shows which clothing categories are most under-stocked."),
    ("Stockout_By_Price_Tier",   "Summary",  "Stockout rate grouped by 4 price brackets: Budget (<=EUR50), Mid (EUR51-100), Premium (EUR101-150), Luxury (>EUR150). Answers: do expensive products stock out more?"),
    ("Price_vs_Stockout",        "Detail",   "One row per product: its price, stockout rate %, and avg fill rate. Use to plot or manually inspect the price-stockout relationship."),
    ("Chronic_Stockouts",        "Action",   "PRIORITY LIST: product x city x size combos that stocked out in 100% of the seasons they were active (min 2 seasons). Includes product name, gender, wearable type, color, and price. These are structural issues requiring immediate reallocation."),
    ("Gender_By_City",           "Detail",   "Stockout rate for Menswear and Womenswear separately, per city. Long format (one row per city x gender combination)."),
    ("Gender_By_City_Pivot",     "Pivot",    "Same data as Gender_By_City but wide format: city in rows, Menswear / Womenswear as columns. Easy to compare which gender sells out more per city at a glance."),
    ("City_Gender_Wearable_Size","Detail",   "The most granular combo table: city x gender x garment type x size, sorted by stockout rate descending. Directly answers questions like 'do Males in Stockholm always buy out XL Hoodies?'. Filter or sort as needed."),
    # blank separator
    ("", "", ""),
    # PNG FILES
    ("PNG OUTPUT FILES", "", ""),
    ("File name",                          "Type",    "What it shows"),
    ("top_stockout_products.png",          "Bar",     "Top 15 products ranked by stockout rate (% of size x city combos that sold out). Identifies the most critically under-stocked products."),
    ("fillrate_heatmap.png",               "Heatmap", "Average fill rate per size (XS-XXL) per season (year). Green = sells out, Red = excess stock. Shows if size distribution is getting better or worse over time."),
    ("stockout_by_location.png",           "Bar",     "Stockout rate per city / channel ranked highest to lowest. Quick visual of which stores have the biggest stockout problem."),
    ("location_size_heatmap.png",          "Heatmap", "Average fill rate: city (rows) x size (columns). Green cell = that city consistently runs out of that size. Use to reallocate sizes between cities."),
    ("stockout_by_gender.png",             "Bar",     "Side-by-side stockout rate for Menswear vs Womenswear overall."),
    ("stockout_by_color_wearable.png",     "Bar x2",  "Two charts side by side: (left) stockout rate by color, (right) stockout rate by garment type. Identifies which product attributes drive stockouts."),
    ("price_vs_stockout_scatter.png",      "Scatter", "Each dot = one product. X axis = price, Y axis = stockout rate. Dots colored by price tier. Trend line shows direction: slope tells you the change in stockout rate per EUR of price."),
    ("stockout_by_price_tier.png",         "Bar",     "Stockout rate per price tier (Budget / Mid / Premium / Luxury). Answers whether cheap or expensive products run out more often."),
    ("gender_stockout_by_city.png",        "Bar",     "Grouped bar chart: for each city, Menswear (yellow) vs Womenswear (purple) stockout rate side by side. Shows if certain cities skew heavily toward one gender."),
    ("city_wearable_heatmap.png",          "Heatmap", "Stockout rate: city (rows) x garment type (columns). Green = high stockout, red = excess. Reveals structural patterns e.g. hoodies selling out in all cities or dresses only moving in specific locations."),
]

doc_df = pd.DataFrame(doc_rows, columns=["Name", "Type", "Description"])

output_file = "stockout_analysis.xlsx"
with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    doc_df.to_excel(writer,               sheet_name="README",                 index=False)
    raw_merged.to_excel(writer,           sheet_name="Raw_Merged",           index=False)
    stockout_summary.to_excel(writer,     sheet_name="Stockout_By_Product",  index=False)
    size_analysis.to_excel(writer,        sheet_name="Lost_Sales_By_Size",   index=False)
    overstock.to_excel(writer,            sheet_name="Overstocked_Items",    index=False)
    location_summary.to_excel(writer,     sheet_name="Stockout_By_Location", index=False)
    loc_size_pivot_reset.to_excel(writer, sheet_name="Location_Size_Heatmap",index=False)
    prod_ch_size.to_excel(writer,         sheet_name="Product_Channel_Size", index=False)
    gender_summary.to_excel(writer,       sheet_name="Stockout_By_Gender",   index=False)
    color_summary.to_excel(writer,        sheet_name="Stockout_By_Color",    index=False)
    wearable_summary.to_excel(writer,     sheet_name="Stockout_By_Wearable", index=False)
    price_tier_summary.to_excel(writer,   sheet_name="Stockout_By_Price_Tier",index=False)
    prod_price_stockout.to_excel(writer,  sheet_name="Price_vs_Stockout",    index=False)
    chronic.to_excel(writer,              sheet_name="Chronic_Stockouts",    index=False)
    gender_city.to_excel(writer,          sheet_name="Gender_By_City",       index=False)
    gender_city_pivot.to_excel(writer,    sheet_name="Gender_By_City_Pivot", index=False)
    city_combo.to_excel(writer,           sheet_name="City_Gender_Wearable_Size", index=False)

print(f"\nExcel saved → {output_file}")
print("  Tabs: Raw_Merged | Stockout_By_Product | Lost_Sales_By_Size | Overstocked_Items")
print("        Stockout_By_Location | Location_Size_Heatmap | Product_Channel_Size")
print("        Stockout_By_Gender | Stockout_By_Color | Stockout_By_Wearable")
print("        Stockout_By_Price_Tier | Price_vs_Stockout")
print("        Chronic_Stockouts | Gender_By_City | Gender_By_City_Pivot | City_Gender_Wearable_Size")

# ─────────────────────────────────────────────
# 6. Plot 1 – Top 15 products by stockout frequency
# ─────────────────────────────────────────────
top_stockout = (
    merged.groupby("product_id")["stockout_flag"]
    .agg(["sum", "count"])
    .rename(columns={"sum": "stockout_events", "count": "total"})
    .assign(stockout_rate=lambda d: d["stockout_events"] / d["total"] * 100)
    .sort_values("stockout_rate", ascending=False)
    .head(15)
    .reset_index()
)

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.barh(top_stockout["product_id"], top_stockout["stockout_rate"], color="#E05A2B")
ax.set_xlabel("Stockout Rate (%)")
ax.set_title("Top 15 Products by Stockout Frequency\n(% of size×channel combos that sold out)", fontsize=13)
ax.invert_yaxis()
ax.xaxis.set_major_formatter(mticker.PercentFormatter())
ax.bar_label(bars, fmt="%.1f%%", padding=3)
ax.grid(axis="x", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("top_stockout_products.png", dpi=150)
plt.show()

# ─────────────────────────────────────────────
# 7. Plot 2 – Average fill rate per size per season (heatmap)
# ─────────────────────────────────────────────
heatmap_data = merged.groupby(["season", "size"])["fill_rate"].mean().unstack("size")

# Reorder sizes if all present
ordered_sizes = [s for s in size_order if s in heatmap_data.columns]
heatmap_data = heatmap_data[ordered_sizes]

fig, ax = plt.subplots(figsize=(10, 6))
cax = ax.imshow(heatmap_data.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
plt.colorbar(cax, ax=ax, label="Average Fill Rate (0 = nothing sold, 1 = fully sold out)")

ax.set_xticks(range(len(heatmap_data.columns)))
ax.set_xticklabels(heatmap_data.columns)
ax.set_yticks(range(len(heatmap_data.index)))
ax.set_yticklabels(heatmap_data.index)
ax.set_title("Average Fill Rate per Size per Season\n(green = sold out / potential stockout, red = excess stock)", fontsize=12)
ax.set_xlabel("Size")
ax.set_ylabel("Season (Year)")

# Annotate cells
for i in range(len(heatmap_data.index)):
    for j in range(len(heatmap_data.columns)):
        val = heatmap_data.values[i, j]
        if not np.isnan(val):
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color="black" if 0.3 < val < 0.8 else "white", fontsize=9)

plt.tight_layout()
plt.savefig("fillrate_heatmap.png", dpi=150)
plt.show()

print("\nPlots saved: top_stockout_products.png | fillrate_heatmap.png")

# ─────────────────────────────────────────────
# 8. Plot 3 – Stockout rate per city (horizontal bar)
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, max(4, len(location_summary) * 0.5 + 1)))
bars = ax.barh(
    location_summary["channel_id"],
    location_summary["stockout_rate_pct"],
    color="#3B82F6"
)
ax.set_xlabel("Stockout Rate (%)")
ax.set_title("Stockout Rate per City / Channel\n(% of product×size combos that sold out across all seasons)", fontsize=13)
ax.invert_yaxis()
ax.xaxis.set_major_formatter(mticker.PercentFormatter())
ax.bar_label(bars, fmt="%.1f%%", padding=3)
ax.grid(axis="x", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("stockout_by_location.png", dpi=150)
plt.show()

# ─────────────────────────────────────────────
# 9. Plot 4 – Heatmap: location × size fill rate
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, max(5, len(loc_size_pivot) * 0.45 + 1.5)))
cax = ax.imshow(loc_size_pivot.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
plt.colorbar(cax, ax=ax, label="Avg Fill Rate  (1.0 = always sold out → likely losing sales)")

ax.set_xticks(range(len(loc_size_pivot.columns)))
ax.set_xticklabels(loc_size_pivot.columns, fontsize=10)
ax.set_yticks(range(len(loc_size_pivot.index)))
ax.set_yticklabels(loc_size_pivot.index, fontsize=9)
ax.set_title(
    "Average Fill Rate per City × Size\n"
    "(green = sells out everywhere → potential lost sales  |  red = unsold stock)",
    fontsize=12
)
ax.set_xlabel("Size")
ax.set_ylabel("City / Channel")

for i in range(len(loc_size_pivot.index)):
    for j in range(len(loc_size_pivot.columns)):
        val = loc_size_pivot.values[i, j]
        if not np.isnan(val):
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color="black" if 0.25 < val < 0.85 else "white", fontsize=8)

plt.tight_layout()
plt.savefig("location_size_heatmap.png", dpi=150)
plt.show()

print("\nNew plots saved: stockout_by_location.png | location_size_heatmap.png")

# ─────────────────────────────────────────────
# 10. Plot 5 – Stockout rate: Menswear vs Womenswear
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
colors_gender = ["#6366F1" if "Women" in str(g) else "#F59E0B" for g in gender_summary["gender"]]
bars = ax.bar(gender_summary["gender"], gender_summary["stockout_rate_pct"], color=colors_gender, width=0.5)
ax.set_ylabel("Stockout Rate (%)")
ax.set_title("Stockout Rate by Gender Category\n(Menswear vs Womenswear)", fontsize=13)
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=11)
ax.set_ylim(0, min(100, gender_summary["stockout_rate_pct"].max() * 1.2))
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("stockout_by_gender.png", dpi=150)
plt.show()

# ─────────────────────────────────────────────
# 11. Plot 6 – Stockout rate by color and by wearable type (side-by-side)
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left: by color
color_plot = color_summary.sort_values("stockout_rate_pct", ascending=True)
axes[0].barh(color_plot["Color"], color_plot["stockout_rate_pct"], color="#10B981")
axes[0].set_xlabel("Stockout Rate (%)")
axes[0].set_title("Stockout Rate by Color", fontsize=12)
axes[0].xaxis.set_major_formatter(mticker.PercentFormatter())
for bar, val in zip(axes[0].patches, color_plot["stockout_rate_pct"]):
    axes[0].text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                 f"{val:.1f}%", va="center", fontsize=8)
axes[0].grid(axis="x", linestyle="--", alpha=0.5)

# Right: by wearable type
wear_plot = wearable_summary.sort_values("stockout_rate_pct", ascending=True)
axes[1].barh(wear_plot["Wearable"], wear_plot["stockout_rate_pct"], color="#8B5CF6")
axes[1].set_xlabel("Stockout Rate (%)")
axes[1].set_title("Stockout Rate by Garment Type", fontsize=12)
axes[1].xaxis.set_major_formatter(mticker.PercentFormatter())
for bar, val in zip(axes[1].patches, wear_plot["stockout_rate_pct"]):
    axes[1].text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                 f"{val:.1f}%", va="center", fontsize=8)
axes[1].grid(axis="x", linestyle="--", alpha=0.5)

plt.suptitle("Stockout Rate by Product Attribute", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("stockout_by_color_wearable.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nAttribute plots saved: stockout_by_gender.png | stockout_by_color_wearable.png")

# ─────────────────────────────────────────────
# 12. Plot 7 – Scatter: product price vs stockout rate
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

# Colour dots by price tier for extra clarity
tier_color_map = {
    "Budget (<=EUR50)":     "#10B981",
    "Mid (EUR51-100)":      "#3B82F6",
    "Premium (EUR101-150)": "#F59E0B",
    "Luxury (>EUR150)":     "#EF4444",
}
for tier, grp in prod_price_stockout.groupby(
    pd.cut(prod_price_stockout["price"], bins=price_bins, labels=price_labels)
):
    ax.scatter(
        grp["price"], grp["stockout_rate_pct"],
        label=str(tier), color=tier_color_map.get(str(tier), "gray"),
        alpha=0.8, s=80, edgecolors="white", linewidths=0.5,
    )

# Trend line
if prod_price_stockout["price"].notna().sum() > 2:
    z = np.polyfit(
        prod_price_stockout["price"].dropna(),
        prod_price_stockout.loc[prod_price_stockout["price"].notna(), "stockout_rate_pct"],
        1
    )
    p = np.poly1d(z)
    x_line = np.linspace(prod_price_stockout["price"].min(), prod_price_stockout["price"].max(), 200)
    ax.plot(x_line, p(x_line), "--", color="#1F2937", linewidth=1.5, label=f"Trend (slope={z[0]:.3f}%/EUR)")

ax.set_xlabel("Product Price (EUR)")
ax.set_ylabel("Stockout Rate (%)")
ax.set_title("Product Price vs Stockout Rate\n(each dot = one product; trend line shows direction)", fontsize=13)
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
ax.legend(title="Price Tier", fontsize=9)
ax.grid(linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("price_vs_stockout_scatter.png", dpi=150)
plt.show()

# ─────────────────────────────────────────────
# 13. Plot 8 – Bar: stockout rate per price tier
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
bar_colors = [tier_color_map.get(t, "gray") for t in price_tier_summary["price_tier"]]
bars = ax.bar(
    price_tier_summary["price_tier"],
    price_tier_summary["stockout_rate_pct"],
    color=bar_colors, width=0.55, edgecolor="white"
)
ax.set_ylabel("Stockout Rate (%)")
ax.set_title("Stockout Rate per Price Tier\n(do cheaper / more expensive products stock out more?)", fontsize=13)
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=11)
ax.set_ylim(0, min(105, price_tier_summary["stockout_rate_pct"].max() * 1.2))
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("stockout_by_price_tier.png", dpi=150)
plt.show()

print("\nPrice plots saved: price_vs_stockout_scatter.png | stockout_by_price_tier.png")

# ─────────────────────────────────────────────
# 14. Plot 9 – Grouped bar: Menswear vs Womenswear stockout rate per city
# ─────────────────────────────────────────────
genders   = sorted(gender_city["gender"].unique())
cities    = sorted(gender_city["channel_id"].unique())
n_cities  = len(cities)
n_genders = len(genders)
bar_w     = 0.35
gender_palette = {"Menswear": "#F59E0B", "Womenswear": "#6366F1"}

fig, ax = plt.subplots(figsize=(max(10, n_cities * 1.2), 6))
x = np.arange(n_cities)

for i, gender in enumerate(genders):
    vals = []
    for city in cities:
        row = gender_city[
            (gender_city["channel_id"] == city) & (gender_city["gender"] == gender)
        ]
        vals.append(row["stockout_rate_pct"].values[0] if len(row) else 0)
    offset = (i - (n_genders - 1) / 2) * bar_w
    bars = ax.bar(x + offset, vals, bar_w,
                  label=gender, color=gender_palette.get(gender, "gray"),
                  edgecolor="white")
    ax.bar_label(bars, fmt="%.0f%%", padding=2, fontsize=7)

ax.set_xticks(x)
ax.set_xticklabels(cities, rotation=30, ha="right", fontsize=9)
ax.set_ylabel("Stockout Rate (%)")
ax.set_title("Menswear vs Womenswear Stockout Rate per City\n"
             "(which gender sells out more, and does it vary by city?)", fontsize=13)
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
ax.legend(title="Gender")
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("gender_stockout_by_city.png", dpi=150)
plt.show()

# ─────────────────────────────────────────────
# 15. Plot 10 – Heatmap: city × wearable type stockout rate
# ─────────────────────────────────────────────
city_wear_pivot = (
    merged.groupby(["channel_id", "Wearable"])["stockout_flag"]
    .mean()
    .mul(100)
    .unstack("Wearable")
    .round(1)
)

fig, ax = plt.subplots(figsize=(max(10, len(city_wear_pivot.columns) * 1.1),
                                max(5,  len(city_wear_pivot) * 0.55 + 1.5)))
cax = ax.imshow(city_wear_pivot.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)
plt.colorbar(cax, ax=ax, label="Stockout Rate % (green = always sold out | red = excess stock)")

ax.set_xticks(range(len(city_wear_pivot.columns)))
ax.set_xticklabels(city_wear_pivot.columns, rotation=35, ha="right", fontsize=9)
ax.set_yticks(range(len(city_wear_pivot.index)))
ax.set_yticklabels(city_wear_pivot.index, fontsize=9)
ax.set_title("Stockout Rate: City × Garment Type\n"
             "(e.g. do Stockholm shoppers always buy out hoodies?)", fontsize=12)
ax.set_xlabel("Garment Type")
ax.set_ylabel("City / Channel")

for i in range(len(city_wear_pivot.index)):
    for j in range(len(city_wear_pivot.columns)):
        val = city_wear_pivot.values[i, j]
        if not np.isnan(val):
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    color="black" if 25 < val < 80 else "white", fontsize=8)

plt.tight_layout()
plt.savefig("city_wearable_heatmap.png", dpi=150)
plt.show()

print("\nCombo plots saved: gender_stockout_by_city.png | city_wearable_heatmap.png")

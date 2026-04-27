import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings

warnings.filterwarnings("ignore")


INPUT_PATH = "True demand/True demand Simeon/True_Demand_Results.xlsx"
SHEET_NAME = "1_True_Demand_Lijst"

df = pd.read_excel(INPUT_PATH, sheet_name=SHEET_NAME)

DEMAND_COL = "true_demand"

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 0 — prepare a clean yearly panel at each level
# ═══════════════════════════════════════════════════════════════════════════════

def aggregate_level(df, group_cols, demand_col=DEMAND_COL):
    """Sum demand over group_cols × year."""
    cols = group_cols + ["season", demand_col]
    return (
        df[cols]
        .groupby(group_cols + ["season"], as_index=False)[demand_col]
        .sum()
    )

total_df        = aggregate_level(df, [])
city_df         = aggregate_level(df, ["channel_id"])
city_prod_df    = aggregate_level(df, ["channel_id", "product_id"])
city_prod_size  = aggregate_level(df, ["channel_id", "product_id", "size"])

YEARS    = sorted(df["season"].unique())
HORIZON  = 3          # how many years ahead to forecast
FC_YEARS = list(range(max(YEARS) + 1, max(YEARS) + 1 + HORIZON))

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — ETS at TOTAL level
# ═══════════════════════════════════════════════════════════════════════════════

def fit_ets(series: pd.Series, horizon: int) -> np.ndarray:
    """
    Fit ETS(A,A,N) — additive error, additive trend, no seasonality.
    With yearly data there is no within-year seasonality to model.
    Falls back to simple linear extrapolation if ETS fails.
    """
    try:
        model = ExponentialSmoothing(
            series,
            trend="add",
            seasonal=None,
            initialization_method="estimated"
        ).fit(optimized=True)
        return model.forecast(horizon).values
    except Exception:
        # Linear fallback
        x = np.arange(len(series)).reshape(-1, 1)
        lr = LinearRegression().fit(x, series.values)
        x_fc = np.arange(len(series), len(series) + horizon).reshape(-1, 1)
        return lr.predict(x_fc)

total_series = total_df.set_index("year")[DEMAND_COL]
fc_total = fit_ets(total_series, HORIZON)

total_forecast = pd.DataFrame({
    "year": FC_YEARS,
    "forecast": fc_total
})
print("Total forecast:")
print(total_forecast)

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Global ETS at CITY level
# One ETS per store, but all share the same model structure.
# Store cluster label is used to decide whether to trust the trend or dampen it.
# ═══════════════════════════════════════════════════════════════════════════════


# ── 2. Aggregate to CITY level (one row per store per year) ─────────────────────

city_year = (
    df.groupby(["channel_id", "season"])[DEMAND_COL]
    .sum()
    .reset_index()
    .rename(columns={DEMAND_COL: "sales"})
)

# ── 3. Pivot to wide format: rows = stores, columns = years ────────────────────

pivot = (
    city_year.pivot(index="channel_id", columns="season", values="sales")
    .sort_index(axis=1)   # ensure years are in chronological order
)

years = pivot.columns.tolist()
n_years = len(years)

# ── 4. Handle missing years (stores that opened mid-period) ────────────────────
# Flag stores with incomplete history so you can treat them separately.
pivot["n_years_active"] = pivot[years].notna().sum(axis=1)
incomplete = pivot[pivot["n_years_active"] < n_years].index.tolist()
if incomplete:
    print(f"Stores with incomplete history ({len(incomplete)}): {incomplete}")

# Fill remaining NaNs with 0 for feature computation (or use interpolation).
pivot[years] = pivot[years].fillna(0)

# ── 5. Compute volume features ─────────────────────────────────────────────────

feats = pd.DataFrame(index=pivot.index)

sales = pivot[years].values  # shape: (n_stores, n_years)

# --- Scale ---
feats["mean_sales"]  = sales.mean(axis=1)
feats["log_mean"]    = np.log1p(feats["mean_sales"])   # compressed scale for clustering
feats["peak_sales"]  = sales.max(axis=1)

# --- Growth ---
first = np.where(sales[:, 0] > 0, sales[:, 0], np.nan)
last  = sales[:, -1]

feats["growth_rate"] = (last - first) / np.where(first > 0, first, np.nan)

# CAGR: (last/first)^(1/n) - 1  — safer with log to avoid overflow
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    feats["cagr"] = np.where(
        first > 0,
        np.exp(np.log(last / first) / (n_years - 1)) - 1,
        np.nan
    )

# Linear trend slope via least squares across years (uses all data points)
year_idx = np.arange(n_years)  # [0, 1, 2, ...]
year_idx_centered = year_idx - year_idx.mean()

def ls_slope(row):
    """Least-squares slope of sales over time."""
    mask = row > 0  # ignore leading zeros from stores not yet open
    if mask.sum() < 2:
        return np.nan
    x = year_idx_centered[mask]
    y = row[mask]
    return np.dot(x, y - y.mean()) / np.dot(x, x)

feats["trend_slope"] = [ls_slope(row) for row in sales]
# Normalise slope by mean sales so it's comparable across store sizes
feats["trend_slope_norm"] = feats["trend_slope"] / feats["mean_sales"].replace(0, np.nan)

# --- Stability ---
feats["cv"] = sales.std(axis=1) / np.where(feats["mean_sales"] > 0, feats["mean_sales"], np.nan)

yoy_changes = np.diff(sales, axis=1) / np.where(sales[:, :-1] > 0, sales[:, :-1], np.nan)
feats["max_yoy_drop"] = np.nanmin(yoy_changes, axis=1)   # most negative YoY change

# ── 6. Select features for clustering ─────────────────────────────────────────
# log_mean handles scale; cagr captures growth direction; cv captures volatility.
# trend_slope_norm adds shape nuance without duplicating cagr too much.

CLUSTER_FEATURES = ["log_mean", "cagr", "cv", "trend_slope_norm"]

X = feats[CLUSTER_FEATURES].copy()

# Drop stores where any feature couldn't be computed (e.g. always-zero stores)
X = X.dropna()
print(f"\nStores used for clustering: {len(X)} / {len(feats)}")

# ── 7. Scale features ──────────────────────────────────────────────────────────

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── 8. Choose k with silhouette scores ────────────────────────────────────────

print("\nSilhouette scores by k:")
scores = {}
for k in range(2, min(11, len(X))):
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = km.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    scores[k] = score
    print(f"  k={k}: {score:.3f}")

best_k = max(scores, key=scores.get)
print(f"\nBest k by silhouette: {best_k}")

km_final = KMeans(n_clusters=best_k, random_state=42, n_init=20)
X["cluster"] = km_final.fit_predict(X_scaled)

# Merge cluster label (from earlier volume clustering)
cluster_map = X[["cluster"]].rename(columns={"cluster": "volume_cluster"})

city_pivot = (
    city_df.pivot(index="channel_id", columns="season", values=DEMAND_COL)
    .sort_index(axis=1)
)

city_forecasts = {}

for store in city_pivot.index:
    series = city_pivot.loc[store].dropna()
    vol_cluster = cluster_map.loc[store, "volume_cluster"] if store in cluster_map.index else 0

    # Dampen trend for the declining store (cluster 1)
    trend_type = "add" if vol_cluster == 0 else "add"
    damped     = (vol_cluster == 1)   # only dampen the outlier store

    try:
        model = ExponentialSmoothing(
            series,
            trend="add",
            damped_trend=damped,
            seasonal=None,
            initialization_method="estimated"
        ).fit(optimized=True)
        fc = model.forecast(HORIZON).values
    except Exception:
        fc = fit_ets(series, HORIZON)   # fallback

    # Clip negative forecasts — demand can't go below zero
    fc = np.maximum(fc, 0)
    city_forecasts[store] = fc

city_forecast_df = pd.DataFrame(
    city_forecasts,
    index=FC_YEARS
).T.reset_index().rename(columns={"index": "channel_id"})
city_forecast_df = city_forecast_df.melt(
    id_vars="channel_id", var_name="year", value_name="forecast"
)
print("\nCity-level forecasts (first rows):")
print(city_forecast_df.head(10))

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Global model at CITY × PRODUCT level
# Too many series (11 stores × 50 products = 550) for individual ETS.
# Use proportional disaggregation from city-level forecast instead,
# adjusted by product trend observed in the data.
# ═══════════════════════════════════════════════════════════════════════════════

# 3a. Compute each product's share within each store, per year
city_prod_pivot = city_prod_df.copy()
store_year_totals = (
    city_prod_pivot.groupby(["channel_id", "year"])[DEMAND_COL]
    .transform("sum")
)
city_prod_pivot["share"] = city_prod_pivot[DEMAND_COL] / store_year_totals

# 3b. For each store × product, fit a trend to the share over time
#     This captures products that are growing or declining in relative importance.

def forecast_share(share_series: pd.Series, horizon: int) -> np.ndarray:
    """Extrapolate a share series with a dampened linear trend."""
    x = np.arange(len(share_series)).reshape(-1, 1)
    lr = LinearRegression().fit(x, share_series.values)
    x_fc = np.arange(len(share_series), len(share_series) + horizon).reshape(-1, 1)
    fc = lr.predict(x_fc)
    # Dampen extrapolation: blend 70% trend, 30% last observed share
    last_share = share_series.iloc[-1]
    fc = 0.7 * fc + 0.3 * last_share
    return np.clip(fc, 0, 1)   # shares must stay in [0, 1]

share_pivot = (
    city_prod_pivot.pivot_table(
        index=["channel_id", "product_id"],
        columns="year",
        values="share",
        aggfunc="mean"
    ).sort_index(axis=1)
    .fillna(0)
)

share_forecasts = {}
for (store, product), row in share_pivot.iterrows():
    share_series = row[row > 0]   # skip leading zeros
    if len(share_series) < 2:
        fc_share = np.full(HORIZON, row.mean())
    else:
        fc_share = forecast_share(share_series, HORIZON)
    share_forecasts[(store, product)] = fc_share

# 3c. Normalise shares per store × year so they sum to 1
share_fc_df = pd.DataFrame(share_forecasts, index=FC_YEARS).T
share_fc_df.index = pd.MultiIndex.from_tuples(share_fc_df.index, names=["channel_id", "product_id"])
share_fc_df.columns.name = "year"

# Normalise each store's product shares to sum to 1 per forecast year
share_fc_norm = share_fc_df.div(
    share_fc_df.groupby(level="channel_id").transform("sum")
)

# 3d. Multiply normalised shares by city-level forecast
city_fc_lookup = city_forecast_df.set_index(["channel_id", "year"])["forecast"]

records = []
for (store, product), row in share_fc_norm.iterrows():
    for yr in FC_YEARS:
        city_fc = city_fc_lookup.get((store, yr), np.nan)
        prod_fc = row[yr] * city_fc if not np.isnan(city_fc) else np.nan
        records.append({
            "channel_id": store,
            "product_id": product,
            "year": yr,
            "forecast": prod_fc
        })

city_prod_forecast_df = pd.DataFrame(records)
print("\nCity × Product forecasts (first rows):")
print(city_prod_forecast_df.head(10))

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Leaf level: disaggregate by fixed size ratios
# Size curves are nearly identical across stores (silhouette ~0.08),
# so use a global size ratio per product rather than store-specific ratios.
# ═══════════════════════════════════════════════════════════════════════════════

# 4a. Compute historical size ratio per product (averaged over all stores and years)
size_ratio = (
    city_prod_size.groupby(["product_id", "size"])[DEMAND_COL].sum()
    .reset_index()
)
product_totals = size_ratio.groupby("product_id")[DEMAND_COL].transform("sum")
size_ratio["ratio"] = size_ratio[DEMAND_COL] / product_totals

size_ratio_lookup = size_ratio.set_index(["product_id", "size"])["ratio"]

# 4b. Expand city × product forecast down to size level
leaf_records = []
for _, row in city_prod_forecast_df.iterrows():
    store, product, yr, prod_fc = (
        row["channel_id"], row["product_id"], row["year"], row["forecast"]
    )
    # Get sizes for this product
    sizes = size_ratio[size_ratio["product_id"] == product]["size"].tolist()
    for size in sizes:
        ratio = size_ratio_lookup.get((product, size), np.nan)
        if not np.isnan(ratio) and not np.isnan(prod_fc):
            leaf_records.append({
                "channel_id": store,
                "product_id": product,
                "size": size,
                "year": yr,
                "forecast_raw": prod_fc * ratio
            })

leaf_df = pd.DataFrame(leaf_records)

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Reconciliation (OLS / proportional)
# Force all levels to be consistent:
#   leaf sums → city×product → city → total
# We use bottom-up reconciliation here (sum leaf to get all upper levels)
# because our leaf forecasts are derived from city-level anyway —
# they are already internally consistent by construction.
# ═══════════════════════════════════════════════════════════════════════════════

# 5a. Check: do city-level forecasts sum to total?
city_sum_by_year = city_forecast_df.groupby("year")["forecast"].sum()
print("\nCity sum vs total forecast:")
comparison = pd.DataFrame({
    "city_sum": city_sum_by_year,
    "total_fc": total_forecast.set_index("year")["forecast"]
})
comparison["ratio"] = comparison["city_sum"] / comparison["total_fc"]
print(comparison)

# 5b. Scale city forecasts proportionally to match total
#     This is proportional (top-down) reconciliation — simple and robust
#     for your case since you trust the total-level forecast most.
scaling_factors = (
    total_forecast.set_index("year")["forecast"] / city_sum_by_year
)

city_forecast_df["forecast_reconciled"] = (
    city_forecast_df.apply(
        lambda r: r["forecast"] * scaling_factors.get(r["year"], 1.0),
        axis=1
    )
)

# 5c. Re-derive lower levels from reconciled city forecasts
#     (they were built proportionally so the ratios stay the same)
city_prod_forecast_df = city_prod_forecast_df.merge(
    city_forecast_df[["channel_id", "year", "forecast_reconciled"]],
    on=["channel_id", "year"]
)

# Recompute product share within reconciled city total
city_prod_forecast_df["prod_share"] = (
    city_prod_forecast_df["forecast"]
    / city_prod_forecast_df.groupby(["channel_id", "year"])["forecast"].transform("sum")
)
city_prod_forecast_df["forecast_reconciled"] = (
    city_prod_forecast_df["prod_share"]
    * city_prod_forecast_df["forecast_reconciled"]
)

# 5d. Apply same scaling to leaf
leaf_df = leaf_df.merge(
    city_prod_forecast_df[["channel_id", "product_id", "year", "forecast_reconciled"]],
    on=["channel_id", "product_id", "year"]
)
leaf_size_sums = leaf_df.groupby(["channel_id", "product_id", "year"])["forecast_raw"].transform("sum")
leaf_df["size_ratio_actual"] = leaf_df["forecast_raw"] / leaf_size_sums
leaf_df["forecast_reconciled"] = leaf_df["size_ratio_actual"] * leaf_df["forecast_reconciled"]

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Final output
# ═══════════════════════════════════════════════════════════════════════════════

print("\n=== FINAL RECONCILED FORECASTS ===")

print("\n-- Total --")
print(total_forecast)

print("\n-- By store --")
print(
    city_forecast_df[["channel_id", "year", "forecast_reconciled"]]
    .pivot(index="channel_id", columns="year", values="forecast_reconciled")
    .round(0)
)

print("\n-- Leaf level (first 20 rows) --")
print(
    leaf_df[["channel_id", "product_id", "size", "year", "forecast_reconciled"]]
    .sort_values(["channel_id", "product_id", "size", "year"])
    .head(20)
    .round(1)
)

# Save to Excel
with pd.ExcelWriter("Forecast/reconciled_forecasts.xlsx") as writer:
    total_forecast.to_excel(writer, sheet_name="total", index=False)
    city_forecast_df[["channel_id", "year", "forecast_reconciled"]].to_excel(
        writer, sheet_name="city", index=False
    )
    city_prod_forecast_df[["channel_id", "product_id", "year", "forecast_reconciled"]].to_excel(
        writer, sheet_name="city_product", index=False
    )
    leaf_df[["channel_id", "product_id", "size", "year", "forecast_reconciled"]].to_excel(
        writer, sheet_name="city_product_size", index=False
    )

print("\nSaved to Forecast/reconciled_forecasts.xlsx")
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings

# ── 1. Load data ────────────────────────────────────────────────────────────────

INPUT_PATH = "True demand/True demand Simeon/True_Demand_Results.xlsx"
SHEET_NAME = "1_True_Demand_Lijst"

df = pd.read_excel(INPUT_PATH, sheet_name=SHEET_NAME)

# Expecting columns: channel_id, year, and a demand/quantity column.
# Adjust 'demand' to whatever your actual quantity column is called.
DEMAND_COL = "true_demand"

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

# ── 9. Fit final model ────────────────────────────────────────────────────────

km_final = KMeans(n_clusters=best_k, random_state=42, n_init=20)
X["cluster"] = km_final.fit_predict(X_scaled)

# ── 10. Inspect cluster profiles ──────────────────────────────────────────────

profile = (
    X.groupby("cluster")[CLUSTER_FEATURES]
    .agg(["mean", "count"])
)
print("\nCluster profiles:")
print(profile)

# ── 11. Merge cluster labels back to your main dataframe ──────────────────────

cluster_map = X[["cluster"]]   # index = channel_id
df = df.merge(cluster_map, on="channel_id", how="left")

# Stores that couldn't be clustered (all-zero, new stores, etc.) get NaN.
# You can assign them to the nearest cluster manually or treat as their own group.
print(f"\nStores without cluster: {df[df['cluster'].isna()]['channel_id'].nunique()}")

print(X[X["cluster"] == 1])
print("\nRaw sales for that store:")
store_id = X[X["cluster"] == 1].index[0]
print(pivot.loc[store_id, years])
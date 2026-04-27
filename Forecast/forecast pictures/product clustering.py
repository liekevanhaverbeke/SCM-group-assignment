import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
import seaborn as sns

# ── Assumes df is already loaded with columns:
#    channel_id, product_id, size, year, demand
# ── and cluster labels are already merged in from the volume step


INPUT_PATH = "True demand/True demand Simeon/True_Demand_Results.xlsx"
SHEET_NAME = "1_True_Demand_Lijst"

df = pd.read_excel(INPUT_PATH, sheet_name=SHEET_NAME)

DEMAND_COL = "true_demand"

# ═══════════════════════════════════════════════════════════
# PART 1: PRODUCT MIX SIMILARITY
# ═══════════════════════════════════════════════════════════

# ── 1a. Share of each product in each store's total sales ──

product_store = (
    df.groupby(["channel_id", "product_id"])[DEMAND_COL]
    .sum()
    .reset_index()
)

store_totals = product_store.groupby("channel_id")[DEMAND_COL].sum()
product_store["share"] = (
    product_store[DEMAND_COL]
    / product_store["channel_id"].map(store_totals)
)

# Pivot: rows = stores, columns = products, values = share (0–1)
product_mix = product_store.pivot(
    index="channel_id", columns="product_id", values="share"
).fillna(0)

print("Product mix matrix (share of each product per store):")
print(product_mix.round(3))

# ── 1b. Cluster stores on product mix ─────────────────────

scaler_p = StandardScaler()
X_product = scaler_p.fit_transform(product_mix)

print("\nSilhouette scores for product mix clustering:")
for k in range(2, min(8, len(product_mix))):
    labels = KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(X_product)
    score = silhouette_score(X_product, labels)
    print(f"  k={k}: {score:.3f}")

# ── 1c. Heatmap: which stores sell which products? ─────────

fig, ax = plt.subplots(figsize=(12, 5))
sns.heatmap(
    product_mix,
    cmap="YlOrRd",
    annot=True,
    fmt=".2f",
    linewidths=0.5,
    ax=ax,
    cbar_kws={"label": "share of store total sales"}
)
ax.set_title("Product mix by store")
ax.set_xlabel("Product")
ax.set_ylabel("Store (channel_id)")
plt.tight_layout()
plt.savefig("product_mix_heatmap.png", dpi=150)
plt.show()

# ── 1d. Dendrogram: which stores have similar product mixes? 

# Cosine distance works well for share vectors — ignores volume, pure mix shape
dist_matrix = pdist(product_mix.values, metric="cosine")
linkage_matrix = linkage(dist_matrix, method="ward")

fig, ax = plt.subplots(figsize=(10, 4))
dendrogram(
    linkage_matrix,
    labels=product_mix.index.tolist(),
    ax=ax,
    color_threshold=0.3
)
ax.set_title("Store similarity by product mix (cosine distance, Ward linkage)")
ax.set_xlabel("Store (channel_id)")
ax.set_ylabel("Distance")
plt.tight_layout()
plt.savefig("product_mix_dendrogram.png", dpi=150)
plt.show()

# ── 1e. Top products per store — easy to read summary ─────

top_n = 3
top_products = (
    product_store.sort_values("share", ascending=False)
    .groupby("channel_id")
    .head(top_n)
    .groupby("channel_id")
    .apply(lambda g: list(zip(g["product_id"], g["share"].round(3))))
    .rename("top_products")
)
print("\nTop products per store:")
print(top_products.to_string())


# ═══════════════════════════════════════════════════════════
# PART 2: SIZE CURVE SIMILARITY
# ═══════════════════════════════════════════════════════════

# ── 2a. Size distribution per store (collapsed across products and years) ──
# This gives you the overall size "fingerprint" of each store's customers.

size_store = (
    df.groupby(["channel_id", "size"])[DEMAND_COL]
    .sum()
    .reset_index()
)

size_store["share"] = (
    size_store[DEMAND_COL]
    / size_store["channel_id"].map(
        size_store.groupby("channel_id")[DEMAND_COL].sum()
    )
)

size_mix = size_store.pivot(
    index="channel_id", columns="size", values="share"
).fillna(0)

# Sort size columns in natural order if they're S/M/L etc.
size_order = ["XXS", "XS", "S", "M", "L", "XL", "XXL", "XXXL"]
size_cols_ordered = [s for s in size_order if s in size_mix.columns]
remaining = [s for s in size_mix.columns if s not in size_order]
size_mix = size_mix[size_cols_ordered + remaining]

print("\nSize mix by store:")
print(size_mix.round(3))

# ── 2b. Heatmap: size curves per store ────────────────────

fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(
    size_mix,
    cmap="Blues",
    annot=True,
    fmt=".2f",
    linewidths=0.5,
    ax=ax,
    cbar_kws={"label": "share of store total sales"}
)
ax.set_title("Size curve by store")
ax.set_xlabel("Size")
ax.set_ylabel("Store (channel_id)")
plt.tight_layout()
plt.savefig("size_mix_heatmap.png", dpi=150)
plt.show()

# ── 2c. Size curve per store per product ──────────────────
# More granular: does size curve shift by product type?

size_product_store = (
    df.groupby(["channel_id", "product_id", "size"])[DEMAND_COL]
    .sum()
    .reset_index()
)

# Normalise within each store × product combination
grp_totals = size_product_store.groupby(["channel_id", "product_id"])[DEMAND_COL].transform("sum")
size_product_store["share"] = size_product_store[DEMAND_COL] / grp_totals

# Plot one facet per product — how does the size curve differ across stores?
products = size_product_store["product_id"].unique()

n_cols = 3
n_rows = int(np.ceil(len(products) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharey=False)
axes = axes.flatten()

for i, prod in enumerate(sorted(products)):
    ax = axes[i]
    subset = size_product_store[size_product_store["product_id"] == prod]
    pivot_prod = subset.pivot(
        index="channel_id", columns="size", values="share"
    ).fillna(0)
    pivot_prod = pivot_prod[[s for s in size_cols_ordered if s in pivot_prod.columns]]

    for store in pivot_prod.index:
        ax.plot(
            pivot_prod.columns,
            pivot_prod.loc[store],
            marker="o",
            linewidth=1.2,
            label=str(store),
            alpha=0.7
        )
    ax.set_title(f"Product: {prod}", fontsize=11)
    ax.set_xlabel("Size")
    ax.set_ylabel("Share")
    ax.set_ylim(0, None)

# Hide unused subplots
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, title="Store", loc="lower right", ncol=3)
fig.suptitle("Size curves per product across stores", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("size_curves_by_product.png", dpi=150)
plt.show()

# ── 2d. Combine volume + product mix + size mix clusters ──
# Merge all cluster signals into one summary per store

summary = pd.DataFrame(index=product_mix.index)

# Volume cluster from earlier
if "cluster" in df.columns:
    summary["volume_cluster"] = (
        df.drop_duplicates("channel_id")
        .set_index("channel_id")["cluster"]
    )

# Product mix cluster
km_prod = KMeans(n_clusters=2, random_state=42, n_init=20)
summary["product_cluster"] = km_prod.fit_predict(X_product)

# Size mix cluster
X_size = StandardScaler().fit_transform(size_mix.reindex(product_mix.index).fillna(0))
km_size = KMeans(n_clusters=2, random_state=42, n_init=20)
summary["size_cluster"] = km_size.fit_predict(X_size)

print("\nCombined cluster summary per store:")
print(summary)
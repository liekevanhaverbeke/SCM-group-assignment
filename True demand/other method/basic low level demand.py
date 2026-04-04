import pandas as pd
import numpy as np
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# 0. LOAD & CLEAN
# ─────────────────────────────────────────────────────────────
df = pd.read_excel("stockout_analysis.xlsx", sheet_name="Raw_Merged")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
df["units_sold"]    = pd.to_numeric(df["units_sold"],    errors="coerce")
df["units_stocked"] = pd.to_numeric(df["units_stocked"], errors="coerce")

# Always recompute the flag from raw numbers — never trust a pre-computed column
df["stockout_flag"] = df["units_sold"] >= df["units_stocked"]
df["fill_rate"]     = df["units_sold"] / df["units_stocked"].clip(lower=1)

print(f"Loaded {len(df):,} rows | "
      f"{df['product_id'].nunique()} products | "
      f"{df['channel_id'].nunique()} channels | "
      f"{df['season'].nunique()} seasons")
print(f"Overall stockout rate: {df['stockout_flag'].mean():.1%}\n")


# ─────────────────────────────────────────────────────────────
# 1. PRE-COMPUTE CROSS-SEASON HISTORICAL STATS
#    For every (product, channel, size) we store:
#      - the max units_stocked ever seen          → realistic volume ceiling
#      - the median sell-through rate (uncensored) → conservative STR reference
#      - the max units_sold in any uncensored season → hard historical peak
#
#    These become the guardrails that prevent inter-season explosions.
# ─────────────────────────────────────────────────────────────

uncensored = df[~df["stockout_flag"]].copy()

history = (
    df.groupby(["product_id", "channel_id", "size"])
    .agg(
        max_stocked       = ("units_stocked", "max"),
        median_stocked    = ("units_stocked", "median"),
    )
    .reset_index()
)

history_uncens = (
    uncensored.groupby(["product_id", "channel_id", "size"])
    .agg(
        max_sold_uncens   = ("units_sold",  "max"),
        median_str        = ("fill_rate",   "median"),
        n_uncens_seasons  = ("season",      "nunique"),
    )
    .reset_index()
)

history = history.merge(history_uncens, on=["product_id","channel_id","size"], how="left")

# Fill products/sizes with no uncensored history at all
history["max_sold_uncens"]  = history["max_sold_uncens"].fillna(history["max_stocked"])
history["median_str"]       = history["median_str"].fillna(0.85)   # conservative default
history["n_uncens_seasons"] = history["n_uncens_seasons"].fillna(0)

# Merge back onto main df so every row knows its own history
df = df.merge(history, on=["product_id","channel_id","size"], how="left")


# ─────────────────────────────────────────────────────────────
# 2. COMPUTE UPPER BOUND PER ROW
#
#    The key insight: we allow demand to exceed observed history,
#    but only by a controlled growth factor.
#    - If the product has ≥2 uncensored seasons → trust history more
#    - If the product is new (1 or 0 uncensored seasons) → allow more headroom
#    - The absolute ceiling is always units_stocked × max_str_allowed
#      (you can't have sold more than what was there × some reasonable rate)
# ─────────────────────────────────────────────────────────────

MAX_STR_ALLOWED    = 1.30   # imputed demand won't exceed 130% of stock
                             # (the 30% headroom covers demand > supply scenarios)
GROWTH_CAP_WEAK    = 1.50   # ≤1 uncensored season: allow up to 1.5x historical max
GROWTH_CAP_STRONG  = 1.25   # ≥2 uncensored seasons: tighter, 1.25x historical max

def compute_upper_bound(row) -> float:
    """
    Returns the maximum plausible demand for a single censored row.
    Three constraints, the tightest one wins.
    """
    # Constraint A: stock-based ceiling (always applies)
    stock_ceil = row["units_stocked"] * MAX_STR_ALLOWED

    # Constraint B: historical growth cap
    growth_cap = GROWTH_CAP_STRONG if row["n_uncens_seasons"] >= 2 else GROWTH_CAP_WEAK
    hist_ceil  = row["max_sold_uncens"] * growth_cap

    # Constraint C: absolute minimum is the observed lower bound
    lower = row["units_stocked"]   # we know demand >= stock

    return max(lower, min(stock_ceil, hist_ceil))

df["upper_bound"] = df.apply(compute_upper_bound, axis=1)


# ─────────────────────────────────────────────────────────────
# 3. SELL-THROUGH RATE IMPUTATION  (replaces the broken share-division)
#
#    Within a cell, uncensored sizes give us an observed STR.
#    We apply that STR to the censored size's stock to get a
#    point estimate, then let EM refine it — all within bounds.
#
#    STR-based estimate: demand_est = stock × reference_str
#    where reference_str = median STR of uncensored sizes in this cell,
#    falling back to the product-channel historical STR if cell is thin.
# ─────────────────────────────────────────────────────────────

def cell_reference_str(group: pd.DataFrame) -> float:
    """
    Best estimate of 'what sell-through rate would a size achieve
    here if it hadn't stocked out', from uncensored sizes in this cell.
    """
    uncens = group[~group["stockout_flag"]]
    if len(uncens) >= 1:
        # Use the uncensored sizes' actual STR in this cell
        return uncens["fill_rate"].median()
    else:
        # All stocked out — fall back to each size's own historical STR
        return None   # handled per-row below


# ─────────────────────────────────────────────────────────────
# 4. EM UNCONSTRAINING — bounded and STR-anchored
# ─────────────────────────────────────────────────────────────

def truncated_normal_mean(mu: float, sigma: float, lower: float) -> float:
    if sigma < 1e-6:
        return max(mu, lower)
    alpha = (lower - mu) / sigma
    phi, Phi = norm.pdf(alpha), norm.cdf(alpha)
    denom = 1.0 - Phi
    if denom < 1e-9:
        return lower
    return mu + sigma * (phi / denom)


def em_unconstrain_cell(
    group: pd.DataFrame,
    max_iter: int = 80,
    tol: float    = 1e-5,
) -> pd.DataFrame:

    group    = group.copy().reset_index(drop=True)
    censored = group["stockout_flag"].values
    sales    = group["units_sold"].values.astype(float)
    stock    = group["units_stocked"].values.astype(float)
    upper    = group["upper_bound"].values
    n_uncens = (~censored).sum()

    # ── Initialise demand ────────────────────────────────────
    demand = sales.copy()

    # Get the cell-level reference STR
    ref_str = cell_reference_str(group)

    for i in np.where(censored)[0]:
        if ref_str is not None:
            # STR-anchored starting point: stock × ref_str, clipped to bounds
            str_est = stock[i] * ref_str
        else:
            # All censored: use each size's own historical STR
            hist_str = group.loc[i, "median_str"]
            str_est  = stock[i] * hist_str

        # Clip to [lower_bound, upper_bound] immediately
        demand[i] = float(np.clip(str_est, stock[i], upper[i]))

    # ── EM loop ─────────────────────────────────────────────
    mu    = demand.mean()
    sigma = max(demand.std(), 0.5)

    for iteration in range(max_iter):
        mu_old = mu

        for i in np.where(censored)[0]:
            # E-step: truncated normal expectation
            e_val = truncated_normal_mean(mu, sigma, stock[i])

            # Hard clip to pre-computed bounds — this is the critical fix
            demand[i] = float(np.clip(e_val, stock[i], upper[i]))

        # M-step
        mu    = demand.mean()
        sigma = max(demand.std(), 0.5)

        if abs(mu - mu_old) < tol:
            break

    # ── Write results ────────────────────────────────────────
    group["demand_est"] = np.round(demand, 2)
    # Final safety check: demand must always be >= observed sales
    group["demand_est"] = group[["demand_est","units_sold"]].max(axis=1)
    group["uplift"]     = (group["demand_est"] - group["units_sold"]).round(2)
    group["n_uncens"]   = n_uncens
    group["ref_str"]    = ref_str if ref_str is not None else np.nan
    group["em_iters"]   = iteration + 1
    group["method"]     = "all_censored_str" if n_uncens == 0 else "em_bounded"
    return group


# ─────────────────────────────────────────────────────────────
# 5. APPLY ACROSS ALL GROUPS
# ─────────────────────────────────────────────────────────────

GROUP_KEYS = ["product_id", "channel_id", "season"]
results    = []

for keys, group in df.groupby(GROUP_KEYS):
    results.append(em_unconstrain_cell(group))

result_df = pd.concat(results, ignore_index=True)


# ─────────────────────────────────────────────────────────────
# 6. DIAGNOSTICS  — the checks that caught the original bug
# ─────────────────────────────────────────────────────────────

print("── Sanity checks ────────────────────────────────────────")
# Check 1: no imputation below observed sales
violations_lower = (result_df["demand_est"] < result_df["units_sold"]).sum()
print(f"Below-sales violations:     {violations_lower}  (must be 0)")

# Check 2: no imputation above upper bound
violations_upper = (result_df["demand_est"] > result_df["upper_bound"] + 0.01).sum()
print(f"Above-upper-bound violations: {violations_upper}  (must be 0)")

# Check 3: uplift distribution on censored rows — flag extreme values
cens = result_df[result_df["stockout_flag"]]
print(f"\nUplift on censored rows (units above observed sales):")
print(cens["uplift"].describe().round(2))

extreme = cens[cens["uplift"] > cens["units_stocked"] * 0.5]
if len(extreme):
    print(f"\nRows with uplift > 50% of stock ({len(extreme)} rows) — review these:")
    print(extreme[["product_id","channel_id","season","size",
                   "units_sold","units_stocked","upper_bound",
                   "demand_est","uplift"]].head(10).to_string(index=False))
else:
    print("\nNo extreme uplifts found — estimates look reasonable.")

# Check 4: cross-season consistency for your specific example
print("\n── Cross-season check: product 093KT7ZK38, Amsterdam, S ──")
ex = result_df[
    (result_df["product_id"] == "093KT7ZK38") &
    (result_df["channel_id"] == "Amsterdam") &
    (result_df["size"]       == "S")
][["season","units_sold","units_stocked","upper_bound","demand_est","uplift"]]
print(ex.sort_values("season").to_string(index=False))


# ─────────────────────────────────────────────────────────────
# 7. AGGREGATE OUTPUTS
# ─────────────────────────────────────────────────────────────

detail_agg = (
    result_df
    .groupby(["product_id","channel_id","season","size"])
    .agg(
        observed  = ("units_sold",    "sum"),
        stocked   = ("units_stocked", "sum"),
        demand    = ("demand_est",    "sum"),
        uplift    = ("uplift",        "sum"),
        oos_rate  = ("stockout_flag", "mean"),
        ref_str   = ("ref_str",       "first"),
    )
    .assign(uplift_pct=lambda x: (x["uplift"] / x["observed"].clip(lower=1) * 100).round(1))
    .reset_index()
)

product_agg = (
    result_df
    .groupby(["product_id","season"])
    .agg(observed=("units_sold","sum"), demand=("demand_est","sum"),
         uplift=("uplift","sum"), oos_rate=("stockout_flag","mean"))
    .assign(uplift_pct=lambda x: (x["uplift"] / x["observed"].clip(lower=1) * 100).round(1))
    .reset_index()
    .sort_values("uplift", ascending=False)
)

print("\n── Top products by uplift ───────────────────────────────")
print(product_agg.head(10).to_string(index=False))

result_df.to_excel("unconstrained_detail.xlsx",     index=False)
detail_agg.to_excel("unconstrained_by_size.xlsx",   index=False)
product_agg.to_excel("unconstrained_by_product.xlsx", index=False)
print("\nExported: unconstrained_detail.xlsx, unconstrained_by_size.xlsx, unconstrained_by_product.xlsx")
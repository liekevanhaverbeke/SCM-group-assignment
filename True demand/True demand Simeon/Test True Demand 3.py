import pandas as pd
import numpy as np
import math

# =========================================================
# 1. DATA INLADEN
# =========================================================
raw = pd.read_excel("stockout_analysis.xlsx", sheet_name="Raw_Merged")
raw['stockout_flag'] = raw['stockout_flag'].astype(bool)

# =========================================================
# 2. SPLITS: TRAIN vs TEST
# =========================================================
CENSOR_RATE = 0.30
SEED = 42
ALPHA = 0.55

known = raw[raw["stockout_flag"] == False].copy()

rng = np.random.default_rng(SEED)
censor_mask = rng.random(len(known)) < CENSOR_RATE

train = known[~censor_mask].copy()
test  = known[censor_mask].copy()

print(f"Train set: {len(train):,} rijen")
print(f"Test set (nep-stockouts): {len(test):,} rijen")

# Lokaal gemiddelde fill rate per product × kanaal op train set (voor censuur)
# Onafhankelijk van strategie — gebaseerd op historisch gemiddelde van die combinatie
local_censor_rate = (
    train.groupby(["product_id", "channel_id"])["fill_rate"]
    .mean()
    .reset_index()
    .rename(columns={"fill_rate": "censor_rate"})
)
global_censor_rate = train["fill_rate"].mean()
print(f"\nGlobale fallback censuur rate: {global_censor_rate:.4f}")

# =========================================================
# 3. FILL RATE BEREKENEN OP TRAIN SET (MET ALPHA BLENDING)
# =========================================================
stats_train = train.groupby(["product_id", "channel_id"])["stockout_flag"].agg(
    n_stockouts="sum",
    total_obs="count"
).reset_index()
stats_train["n_non_stockouts"] = stats_train["total_obs"] - stats_train["n_stockouts"]

local_fr_non_stockout = (
    train.groupby(["product_id", "channel_id"])["fill_rate"]
    .mean()
    .rename("fr_non_stockout")
)

raw_train_combos = train[["product_id", "channel_id"]].drop_duplicates()
raw_with_stockouts = raw.merge(raw_train_combos, on=["product_id", "channel_id"], how="inner")

local_fr_all = (
    raw_with_stockouts.groupby(["product_id", "channel_id"])["fill_rate"]
    .mean()
    .rename("fr_all")
)

local_fr_blended = pd.concat([local_fr_non_stockout, local_fr_all], axis=1).reset_index()
local_fr_blended["fill_rate_local"] = (
    ALPHA * local_fr_blended["fr_all"] +
    (1 - ALPHA) * local_fr_blended["fr_non_stockout"]
)
local_fr_blended["fill_rate_local"] = local_fr_blended["fill_rate_local"].fillna(
    local_fr_blended["fr_all"]
)
local_fr_blended = local_fr_blended[["product_id", "channel_id", "fill_rate_local"]]

prod_fr_non_stockout = train.groupby("product_id")["fill_rate"].mean().rename("fr_non_stockout")
prod_fr_all = raw.groupby("product_id")["fill_rate"].mean().rename("fr_all")

prod_fr_blended = pd.concat([prod_fr_non_stockout, prod_fr_all], axis=1).reset_index()
prod_fr_blended["fill_rate_product"] = (
    ALPHA * prod_fr_blended["fr_all"] +
    (1 - ALPHA) * prod_fr_blended["fr_non_stockout"]
)
prod_fr_blended["fill_rate_product"] = prod_fr_blended["fill_rate_product"].fillna(
    prod_fr_blended["fr_all"]
)
prod_fr_blended = prod_fr_blended[["product_id", "fill_rate_product"]]

global_fr_non_stockout = train["fill_rate"].mean()
global_fr_all = raw["fill_rate"].mean()
global_fr = ALPHA * global_fr_all + (1 - ALPHA) * global_fr_non_stockout

print(f"ALPHA: {ALPHA}")
print(f"Globaal fill rate (niet-stockout only): {global_fr_non_stockout:.4f}")
print(f"Globaal fill rate (inclusief stockouts): {global_fr_all:.4f}")
print(f"Globaal fill rate (blended):             {global_fr:.4f}")

# =========================================================
# 4. SMART STRATEGY TOEPASSEN OP TRAIN COMBINATIES
# =========================================================
tab_train = stats_train.merge(local_fr_blended, on=["product_id", "channel_id"], how="left")
tab_train = tab_train.merge(prod_fr_blended, on="product_id", how="left")

def apply_smart_strategy(row):
    n         = row["n_non_stockouts"]
    local_val = row["fill_rate_local"]
    prod_val  = row["fill_rate_product"]

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
            return (1.0 + global_fr) / 2, "Segment C: Blind Spot (Global Safe)"

tab_train[["used_fill_rate", "strategy_label"]] = tab_train.apply(
    lambda r: pd.Series(apply_smart_strategy(r)), axis=1
)

# =========================================================
# 5. CORRECTIE TOEPASSEN OP TEST SET
# =========================================================
test = test.merge(
    tab_train[["product_id", "channel_id", "used_fill_rate", "strategy_label"]],
    on=["product_id", "channel_id"],
    how="left"
)
test["used_fill_rate"] = test["used_fill_rate"].fillna(global_fr)
test["strategy_label"] = test["strategy_label"].fillna("Fallback: Global (niet in train)")

# Voeg lokale censuur rate toe aan test set
test = test.merge(local_censor_rate, on=["product_id", "channel_id"], how="left")
test["censor_rate"] = test["censor_rate"].fillna(global_censor_rate)

# CENSUUR: lokaal gemiddelde fill rate per product × kanaal (train set, niet-stockout)
# Volledig onafhankelijk van de strategie
test["censored_units"] = test.apply(
    lambda r: math.floor(r["units_sold"] * r["censor_rate"]),
    axis=1
)

# HERSTEL: strategie fill_rate (onafhankelijk van censuur)
test["recovered_demand"] = test.apply(
    lambda r: math.ceil(r["censored_units"] / r["used_fill_rate"]),
    axis=1
)

# =========================================================
# 6. EVALUATIE
# =========================================================
test["true_units"]  = test["units_sold"]
test["error_units"] = test["recovered_demand"] - test["true_units"]
test["error_pct"]   = (test["error_units"] / test["true_units"] * 100).round(2)

total_true      = test["true_units"].sum()
total_censored  = test["censored_units"].sum()
total_recovered = test["recovered_demand"].sum()

print("\n========== CALIBRATIE RESULTAAT ==========")
print(f"Werkelijke vraag (bekend):      {total_true:,}")
print(f"Gecensureerde vraag (nep):      {total_censored:,}  ({(total_censored/total_true - 1)*100:+.1f}%)")
print(f"Herstelde vraag (model):        {total_recovered:,}  ({(total_recovered/total_true - 1)*100:+.1f}%)")
print(f"Netto fout op test set:         {total_recovered - total_true:+,} eenheden")
print(f"Gemiddelde absolute fout/rij:   {test['error_pct'].abs().mean():.1f}%")
print(f"Mediaan fout/rij:               {test['error_pct'].median():.1f}%")

print("\n--- Fout per segment ---")
segment_eval = (
    test.groupby("strategy_label")
    .agg(
        n_rijen=("true_units", "count"),
        true_demand=("true_units", "sum"),
        recovered=("recovered_demand", "sum"),
    )
    .reset_index()
)
segment_eval["fout_pct"] = (
    (segment_eval["recovered"] - segment_eval["true_demand"])
    / segment_eval["true_demand"] * 100
).round(2)
print(segment_eval.to_string(index=False))

# =========================================================
# 7. OPSLAAN
# =========================================================
output_file = "Test_true_demand_3.xlsx"

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    test[[
        "product_id", "channel_id", "season", "size",
        "true_units", "censored_units", "recovered_demand",
        "error_units", "error_pct",
        "censor_rate", "used_fill_rate", "strategy_label"
    ]].to_excel(writer, sheet_name="1_Test_Rijen", index=False)

    tab_train.to_excel(writer, sheet_name="2_Fill_Rate_Train", index=False)
    segment_eval.to_excel(writer, sheet_name="3_Segment_Evaluatie", index=False)

print(f"\nResultaten opgeslagen in: {output_file}")
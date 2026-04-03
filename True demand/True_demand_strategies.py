import pandas as pd
import math

# =========================================================
# 1. DATA INLADEN & VOORBEREIDING
# =========================================================
# Zorg dat 'stockout_analysis.xlsx' in dezelfde map staat als dit script.
input_file = "stockout_analysis.xlsx"
raw = pd.read_excel(input_file, sheet_name="Raw_Merged")

# Zorg dat de stockout_flag als Boolean (True/False) wordt gelezen
raw['stockout_flag'] = raw['stockout_flag'].astype(bool)

# =========================================================
# 2. BASISSTATISTIEKEN BEREKENEN
# =========================================================

# A. Aantallen berekenen per Product x Winkel (Channel)
stats = raw.groupby(["product_id", "channel_id"])["stockout_flag"].agg(
    n_stockouts="sum",
    total_obs="count"
).reset_index()

# Bereken het aantal keer dat een product WEL op voorraad was
stats["n_non_stockouts"] = stats["total_obs"] - stats["n_stockouts"]

# B. Fill Rates berekenen op verschillende niveaus
# We kijken alleen naar regels waar GEEN stockout was (natuurlijke verkoop)
non_stockout_data = raw[raw["stockout_flag"] == False].copy()

# Niveau 1: Lokaal gemiddelde per winkel
local_fr = non_stockout_data.groupby(["product_id", "channel_id"])["fill_rate"].mean().reset_index()

# Niveau 2: Product-gemiddelde (over alle winkels heen)
prod_fr = non_stockout_data.groupby("product_id")["fill_rate"].mean().reset_index().rename(
    columns={"fill_rate": "prod_rate"})

# Niveau 3: Globaal gemiddelde (als laatste noodgreep)
global_avg_rate = non_stockout_data["fill_rate"].mean()

# Voeg alle bronnen samen in één overzichtstabel (Tab 2)
tab2 = stats.merge(local_fr, on=["product_id", "channel_id"], how="left")
tab2 = tab2.merge(prod_fr, on="product_id", how="left")


# =========================================================
# 3. DE SMART STRATEGIE LOGICA
# =========================================================

def apply_smart_strategy(row):
    """
    Bepaalt de correctiefactor op basis van de hoeveelheid bewijs (n).
    """
    n = row["n_non_stockouts"]
    local_val = row["fill_rate"]
    prod_val = row["prod_rate"]

    # --- SEGMENT A: Hard Filter (Vol vertrouwen) ---
    if n >= 10:
        return local_val, "Segment A: Hard Filter (>=10 obs)"

    # --- SEGMENT B: Glijdende Schaal (Voorzichtig) ---
    elif n > 0:
        # We berekenen een gewicht (W) tussen 0 en 1
        weight = n / 10
        # Mengvorm: 1.0 (geen correctie) en de lokale gemeten waarde
        blended_rate = (1.0 * (1 - weight)) + (local_val * weight)
        return blended_rate, f"Segment B: Glijdende schaal ({int(weight * 100)}% weging)"

    # --- SEGMENT C: Blind Spot (Geen lokale data) ---
    else:
        # Gebruik product-gemiddelde van andere winkels, maar halveer de impact (Safe)
        if pd.notna(prod_val):
            safe_rate = (1.0 + prod_val) / 2
            return safe_rate, "Segment C: Blind Spot (Product Fallback Safe)"
        else:
            # Ultieme fallback naar het totale gemiddelde
            return (1.0 + global_avg_rate) / 2, "Segment C: Blind Spot (Global Safe)"


# Pas de strategie toe op de tabel met factoren
tab2[["used_fill_rate", "strategy_label"]] = tab2.apply(
    lambda r: pd.Series(apply_smart_strategy(r)), axis=1
)

# =========================================================
# 4. TRUE DEMAND BEREKENEN (Tab 1)
# =========================================================

# We groeperen de originele data (bijv. per maat/seizoen)
observed = raw.groupby(["product_id", "channel_id", "season", "size"]).agg(
    units_sold=("units_sold", "sum"),
    stockout_flag=("stockout_flag", "max")
).reset_index()

# Koppel de berekende factor aan de data
true_demand_df = observed.merge(
    tab2[["product_id", "channel_id", "used_fill_rate", "strategy_label"]],
    on=["product_id", "channel_id"],
    how="left"
)

# Berekening: Verkoop / Factor (alleen bij stockouts)
# We gebruiken math.ceil omdat je geen halve producten kunt inkopen
true_demand_df["true_demand"] = true_demand_df.apply(
    lambda r: math.ceil(r["units_sold"] / r["used_fill_rate"]) if r["stockout_flag"] else r["units_sold"],
    axis=1
)

# Bereken het verschil (de gemiste verkoop)
true_demand_df["correction_units"] = true_demand_df["true_demand"] - true_demand_df["units_sold"]

# =========================================================
# 5. RESULTATEN OPSLAAN
# =========================================================

# Pivot tabel maken voor een snel inkoopoverzicht
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

print(f"Succes! De analyse is opgeslagen in: {output_file}")
"""
VERENIGDE SIMPLE EXPONENTIAL SMOOTHING VOORSPELLING (Alle Niveaus)
- Consolideert SES voorspellingen voor: Total, City, City/Product en City/Product/Size
- Genereert 4 aparte Excel resultaten
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import os
import warnings
import time

# Onderdruk warnings
warnings.filterwarnings("ignore")

# ── 1. Configuratie ──────────────────────────────────────────────────────────
INPUT_PATH = "True demand/True demand Simeon/True_Demand_Results.xlsx"
SHEET_NAME = "1_True_Demand_Lijst"

LEVELS = [
    {
        "name": "TOTAL",
        "group_cols": [],
        "output_xlsx": "Forecast/simple_exp_total.xlsx"
    },
    {
        "name": "CITY",
        "group_cols": ["channel_id"],
        "output_xlsx": "Forecast/simple_exp_city.xlsx"
    },
    {
        "name": "CITY_PRODUCT",
        "group_cols": ["channel_id", "product_id"],
        "output_xlsx": "Forecast/simple_exp_city_product.xlsx"
    },
    {
        "name": "CITY_PRODUCT_SIZE",
        "group_cols": ["channel_id", "product_id", "size"],
        "output_xlsx": "Forecast/simple_exp_city_product_size.xlsx"
    }
]

# ── 2. SES Hulpfunctie ───────────────────────────────────────────────────────
def ses_forecast(waarden, predict_steps=1):
    y = pd.Series(waarden, dtype=float)
    try:
        model = SimpleExpSmoothing(y, initialization_method="estimated")
        fit = model.fit()
        forecast = fit.forecast(predict_steps).iloc[-1]
        return max(0.0, float(forecast))
    except:
        return float(waarden[-1])

# ── 3. Main Processing Functie ───────────────────────────────────────────────
def run_level(df, config):
    level_name = config["name"]
    group_cols = config["group_cols"]
    output_path = config["output_xlsx"]
    
    print(f"\n" + "="*80)
    print(f"VERWERKEN: {level_name} (SES)")
    print("="*80)
    
    group_by = group_cols + ["season"]
    agg = df.groupby(group_by)["true_demand"].sum().reset_index().rename(columns={"season": "jaar"})
    
    if "channel_id" in agg.columns: agg = agg.rename(columns={"channel_id": "stad"})
    else: agg["stad"] = "TOTAL_COMPANY"
    
    if "product_id" in agg.columns: agg = agg.rename(columns={"product_id": "product"})
    else: agg["product"] = "ALL"
    
    if "size" in agg.columns: agg = agg.rename(columns={"size": "maat"})
    else: agg["maat"] = "ALL"

    id_cols = ["stad", "product", "maat"]
    combinaties = agg[id_cols].drop_duplicates()
    
    val_records = []
    fc_records = []
    start_time = time.time()
    
    for i, (_, row) in enumerate(combinaties.iterrows()):
        subset = agg[
            (agg["stad"] == row["stad"]) & 
            (agg["product"] == row["product"]) & 
            (agg["maat"] == row["maat"])
        ].sort_values("jaar")
        
        # Validatie 2025
        train_val = subset[subset["jaar"] <= 2024]
        actual_val_row = subset[subset["jaar"] == 2025]
        
        if not train_val.empty and not actual_val_row.empty:
            pred_2025 = ses_forecast(train_val["true_demand"].tolist())
            actual_2025 = actual_val_row["true_demand"].values[0]
            val_records.append({
                "stad": row["stad"],
                "product": f"{row['product']} - {row['maat']}" if row['maat'] != "ALL" else row['product'],
                "actual_2025": actual_2025,
                "predicted_2025": round(pred_2025, 1),
                "abs_error": abs(pred_2025 - actual_2025),
                "pct_error": abs(pred_2025 - actual_2025) / actual_2025 * 100 if actual_2025 > 0 else np.nan
            })

        # Forecast 2026
        if not subset.empty:
            pred_2026 = ses_forecast(subset["true_demand"].tolist())
            demand_2025 = subset[subset["jaar"] == 2025]["true_demand"].values[0] if not subset[subset["jaar"] == 2025].empty else np.nan
            
            fc_records.append({
                "stad": row["stad"],
                "product": row["product"],
                "maat": row["maat"],
                "true_demand_2025": demand_2025,
                "forecast_2026": round(pred_2026, 1)
            })
            
        if (i+1) % 1000 == 0:
            elapsed = time.time() - start_time
            print(f"    Voortgang: {i+1}/{len(combinaties)} ({elapsed:.1f}s)")

    val_df = pd.DataFrame(val_records)
    fc_df = pd.DataFrame(fc_records)
    summary_2026 = fc_df.groupby("stad")[["true_demand_2025", "forecast_2026"]].sum()
    
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        fc_df.to_excel(writer, sheet_name="Forecast_2026", index=False)
        val_df.to_excel(writer, sheet_name="Validatie_2025", index=False)
        summary_2026.reset_index().to_excel(writer, sheet_name="Samenvatting_per_stad", index=False)

    print(f"  GEREED: {output_path}")
    if not val_df.empty:
        print(f"  Gemiddelde MAPE: {val_df['pct_error'].mean():.1f}%")

# ── 4. Main Executie ────────────────────────────────────────────────────────
def main():
    df = pd.read_excel(INPUT_PATH, sheet_name=SHEET_NAME)
    for config in LEVELS:
        run_level(df, config)

if __name__ == "__main__":
    main()

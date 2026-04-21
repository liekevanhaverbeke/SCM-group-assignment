"""
VERENIGDE HOLT'S VOORSPELLING (Alle Niveaus)
- Consolideert Holt's voorspellingen voor: Total, City, City/Product en City/Product/Size
- Genereert 4 aparte Excel resultaten
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
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
        "output_xlsx": "Forecast/holt_total.xlsx"
    },
    {
        "name": "CITY",
        "group_cols": ["channel_id"],
        "output_xlsx": "Forecast/holts_city.xlsx"
    },
    {
        "name": "CITY_PRODUCT",
        "group_cols": ["channel_id", "product_id"],
        "output_xlsx": "Forecast/holts_city_product.xlsx"
    },
    {
        "name": "CITY_PRODUCT_SIZE",
        "group_cols": ["channel_id", "product_id", "size"],
        "output_xlsx": "Forecast/holts_city_product_size.xlsx"
    }
]

# ── 2. Holt's Hulpfunctie (met SSE optimalisatie) ────────────────────────────
def holts_forecast(waarden, jaar_labels, alpha=None, beta=None, predict_steps=1):
    y = np.array(waarden, dtype=float)
    n = len(y)

    if n < 3:
        return float(y[-1]), 0.5, 0.1, {}

    def sse(params):
        a, b = params
        if not (0 < a < 1 and 0 < b < 1):
            return 1e15
        L = y[0]
        T = y[1] - y[0]
        err = 0.0
        for i in range(1, n):
            L_prev, T_prev = L, T
            fitted = L_prev + T_prev
            L = a * y[i] + (1 - a) * fitted
            T = b * (L - L_prev) + (1 - b) * T_prev
            err += (y[i] - fitted) ** 2
        return err

    try:
        if alpha is None or beta is None:
            res = minimize(sse, x0=[0.3, 0.1], method="Nelder-Mead",
                           options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 500})
            alpha, beta = res.x
            alpha = max(0.01, min(0.99, alpha))
            beta  = max(0.01, min(0.99, beta))

        # Reconstruct fit history for residual calculation
        fits = {}
        L = y[0]
        T = y[1] - y[0]
        for i in range(1, n):
            L_prev, T_prev = L, T
            fitted = L_prev + T_prev
            fits[jaar_labels[i]] = fitted
            L = alpha * y[i] + (1 - alpha) * fitted
            T = beta  * (L - L_prev) + (1 - beta) * T_prev

        forecast = L + predict_steps * T
        return max(0.0, forecast), alpha, beta, fits
    except:
        return float(y[-1]), 0.5, 0.1, {}

# ── 3. Main Processing Functie ───────────────────────────────────────────────
def run_level(df, config):
    level_name = config["name"]
    group_cols = config["group_cols"]
    output_path = config["output_xlsx"]
    
    print(f"\n" + "="*80)
    print(f"VERWERKEN: {level_name} (Holt's)")
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
    
    print(f"  Aantal combinaties: {len(combinaties)}")

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
        
        # We halen hier de residuals op voor de volledige subset om de gewichten te bepalen
        jaar_labels = subset["jaar"].tolist()
        pred_val, _, _, fits = holts_forecast(train_val["true_demand"].tolist(), train_val["jaar"].tolist())
        
        # Bereken Residual MSE voor 2023, 2024, 2025
        # We gebruiken de fit op de volledige reeks voor de meest actuele weights
        _, a_final, b_final, final_fits = holts_forecast(subset["true_demand"].tolist(), jaar_labels)
        
        residu_sq = []
        for j in [2023, 2024, 2025]:
            if j in final_fits and not subset[subset["jaar"] == j].empty:
                act = subset[subset["jaar"] == j]["true_demand"].values[0]
                fit = final_fits[j]
                residu_sq.append((act - fit) ** 2)
        
        residual_mse = np.mean(residu_sq) if residu_sq else 1.0 # Fallback naar 1.0 als er geen data is

        if len(train_val) >= 3 and not actual_val_row.empty:
            actual_2025 = actual_val_row["true_demand"].values[0]
            val_records.append({
                "stad": row["stad"],
                "product": f"{row['product']} - {row['maat']}" if row['maat'] != "ALL" else row['product'],
                "actual_2025": actual_2025,
                "predicted_2025": round(pred_val, 1),
                "abs_error": abs(pred_val - actual_2025),
                "pct_error": abs(pred_val - actual_2025) / actual_2025 * 100 if actual_2025 > 0 else np.nan
            })

        # Forecast 2026
        if len(subset) >= 3:
            pred_2026 = L_plus_T = pred_val # placeholder, wordt hieronder overschreven
            # We hebben a_final, b_final al van hierboven
            pred_2026, _, _, _ = holts_forecast(subset["true_demand"].tolist(), jaar_labels)
            
            demand_2025 = subset[subset["jaar"] == 2025]["true_demand"].values[0] if not subset[subset["jaar"] == 2025].empty else np.nan
            
            fc_records.append({
                "stad": row["stad"],
                "product": row["product"],
                "maat": row["maat"],
                "true_demand_2025": demand_2025,
                "forecast_2026": round(pred_2026, 1),
                "residual_mse_23_25": residual_mse,
                "alpha": round(a_final, 4),
                "beta": round(b_final, 4)
            })
            
        if (i+1) % 1000 == 0:
            elapsed = time.time() - start_time
            print(f"    Voortgang: {i+1}/{len(combinaties)} ({elapsed:.1f}s)")

    val_df = pd.DataFrame(val_records)
    fc_df = pd.DataFrame(fc_records)
    
    summary_2026 = fc_df.groupby("stad")[["true_demand_2025", "forecast_2026"]].sum()
    summary_2026["groei_%"] = ((summary_2026["forecast_2026"] - summary_2026["true_demand_2025"]) / summary_2026["true_demand_2025"] * 100).round(1)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        fc_df.to_excel(writer, sheet_name="Forecast_2026", index=False)
        val_df.to_excel(writer, sheet_name="Validatie_2025", index=False)
        summary_2026.reset_index().to_excel(writer, sheet_name="Samenvatting_per_stad", index=False)

    print(f"  GEREED: {output_path}")
    if not val_df.empty:
        print(f"  Gemiddelde MAPE: {val_df['pct_error'].mean():.1f}%")

# ── 4. Main Executie ────────────────────────────────────────────────────────
def main():
    print("="*80)
    print("UNIFIED HOLT'S FORECASTING PIPELINE")
    print("="*80)
    
    df = pd.read_excel(INPUT_PATH, sheet_name=SHEET_NAME)
    for config in LEVELS:
        run_level(df, config)
    print("\nAlle Holt's niveaus zijn succesvol verwerkt!")

if __name__ == "__main__":
    main()

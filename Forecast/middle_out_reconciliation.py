"""
MIDDLE-OUT RECONCILIATION WITH RATIO CORRECTION & BLENDING
- Segments 'Platform' (outlier) vs 'Normal' stores.
- Anchors forecast at the City/Product level (captures product lifecycle).
- Disaggregates to the Size level using historical ratios.
- Blends the disaggregated forecast with the independent granular model (alpha = 0.6).
- Re-aggregates back to Total and City levels for coherence.
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import os
import warnings
import time

# Onderdruk warnings
warnings.filterwarnings("ignore")

# ── 1. Configuratie ──────────────────────────────────────────────────────────
INPUT_PATH = "True demand/True demand Simeon/True_Demand_Results.xlsx"
SHEET_NAME = "1_True_Demand_Lijst"
ALPHA = 0.6  # Weight toward the top-down/middle-out signal

FORECAST_FOLDER = "Forecast/Results"
OUTPUT_TOTAL = f"{FORECAST_FOLDER}/middle_out_total.xlsx"
OUTPUT_REGION = f"{FORECAST_FOLDER}/middle_out_region.xlsx"
OUTPUT_CITY = f"{FORECAST_FOLDER}/middle_out_city.xlsx"
OUTPUT_CITY_PRODUCT = f"{FORECAST_FOLDER}/middle_out_city_product.xlsx"
OUTPUT_CITY_PRODUCT_SIZE = f"{FORECAST_FOLDER}/middle_out_city_product_size.xlsx"

# Region mapping
REGION_MAP = {
    'Platform': 'Online',
    'Webshop': 'Online',
    'Helsinki': 'Scandinavian',
    'Copenhagen': 'Scandinavian',
    'Stockholm': 'Scandinavian',
    'Amsterdam': 'European',
    'Berlin': 'European',
    'Brussels': 'European',
    'Madrid': 'European',
    'Paris': 'European',
    'Rome': 'European'
}

# ── 2. ARIMA Hulpfunctie ─────────────────────────────────────────────────────
def arima_forecast(waarden, order=(1, 1, 0), predict_steps=1):
    y = pd.Series(waarden, dtype=float)
    if len(y) < 3:
        return float(waarden[-1]) if len(waarden) > 0 else 0.0
    try:
        model = ARIMA(y, order=order)
        fit = model.fit()
        forecast = fit.forecast(predict_steps).iloc[-1]
        return max(0.0, float(forecast))
    except:
        return float(waarden[-1])

def run_middle_out_reconciliation():
    print("="*80)
    print("STARTING MIDDLE-OUT RECONCILIATION (CITY/PRODUCT ANCHOR)")
    print("="*80)

    # 3.1 Data laden
    print(f"Loading data from {INPUT_PATH}...")
    df = pd.read_excel(INPUT_PATH, sheet_name=SHEET_NAME)
    
    # 3.2 Segmentatie (Isolate Platform)
    df_platform = df[df['channel_id'] == 'Platform'].copy()
    df_normal   = df[df['channel_id'] != 'Platform'].copy()
    
    segments = [
        {"name": "NORMAL_STORES", "df": df_normal},
        {"name": "PLATFORM", "df": df_platform}
    ]
    
    all_final_forecasts = []
    all_validation_results = []

    for seg in segments:
        seg_name = seg["name"]
        seg_df = seg["df"]
        if seg_df.empty: continue
            
        print(f"\nProcessing segment: {seg_name}")
        
        # A. Middle-Level Forecast (City/Product)
        print(f"  Generating Anchor forecasts (City/Product)...")
        cp_year = seg_df.groupby(['channel_id', 'product_id', 'season'])['true_demand'].sum().reset_index()
        cp_combos = cp_year[['channel_id', 'product_id']].drop_duplicates()
        
        cp_fc_list = []
        cp_val_list = []
        
        for _, row in cp_combos.iterrows():
            subset = cp_year[(cp_year['channel_id'] == row['channel_id']) & (cp_year['product_id'] == row['product_id'])].sort_values('season')
            
            # 2026 Forecast
            fc_2026 = arima_forecast(subset['true_demand'].tolist())
            actual_2025 = subset[subset['season'] == 2025]['true_demand'].values[0] if not subset[subset['season'] == 2025].empty else 0
            cp_fc_list.append({**row, 'cp_fc_2026': fc_2026, 'actual_2025': actual_2025})
            
            # 2025 Validation
            train_val = subset[subset['season'] <= 2024]
            if not train_val.empty:
                pred_2025 = arima_forecast(train_val['true_demand'].tolist())
                cp_val_list.append({**row, 'cp_pred_2025': pred_2025, 'actual_2025': actual_2025})

        df_cp_fc = pd.DataFrame(cp_fc_list)
        df_cp_val = pd.DataFrame(cp_val_list)

        # B. Bottom-Level Forecast (Independent Granular Models)
        print(f"  Generating Independent Bottom forecasts (City/Product/Size)...")
        cps_year = seg_df.groupby(['channel_id', 'product_id', 'size', 'season'])['true_demand'].sum().reset_index()
        cps_combos = cps_year[['channel_id', 'product_id', 'size']].drop_duplicates()
        
        cps_fc_list = []
        cps_val_list = []
        
        for _, row in cps_combos.iterrows():
            subset = cps_year[(cps_year['channel_id'] == row['channel_id']) & 
                              (cps_year['product_id'] == row['product_id']) & 
                              (cps_year['size'] == row['size'])].sort_values('season')
            
            # 2026 Forecast
            fc_2026 = arima_forecast(subset['true_demand'].tolist())
            cps_fc_list.append({**row, 'indep_fc_2026': fc_2026})
            
            # 2025 Validation
            train_val = subset[subset['season'] <= 2024]
            actual_2025 = subset[subset['season'] == 2025]['true_demand'].values[0] if not subset[subset['season'] == 2025].empty else 0
            if not train_val.empty:
                pred_2025 = arima_forecast(train_val['true_demand'].tolist())
                cps_val_list.append({**row, 'indep_pred_2025': pred_2025, 'actual_2025': actual_2025})

        df_cps_fc = pd.DataFrame(cps_fc_list)
        df_cps_val = pd.DataFrame(cps_val_list)

        # C. Size Ratios (Historical)
        print(f"  Computing Size Ratios...")
        # Correct calculation: total demand for (size) / total demand for (product)
        size_totals = seg_df.groupby(['channel_id', 'product_id', 'size'])['true_demand'].sum().reset_index()
        product_totals = seg_df.groupby(['channel_id', 'product_id'])['true_demand'].sum().reset_index().rename(columns={'true_demand': 'total_product_demand'})
        
        ratios = size_totals.merge(product_totals, on=['channel_id', 'product_id'])
        ratios['size_ratio'] = ratios['true_demand'] / ratios['total_product_demand']
        ratios = ratios[['channel_id', 'product_id', 'size', 'size_ratio']]
        
        # D. Disaggregation & Blending (2026)
        print(f"  Blending Forecasts (alpha={ALPHA})...")
        merged_fc = df_cps_fc.merge(df_cp_fc, on=['channel_id', 'product_id'])
        merged_fc = merged_fc.merge(ratios, on=['channel_id', 'product_id', 'size'])
        
        merged_fc['disagg_fc_2026'] = merged_fc['cp_fc_2026'] * merged_fc['size_ratio']
        merged_fc['blended_fc_2026'] = (ALPHA * merged_fc['disagg_fc_2026'] + (1 - ALPHA) * merged_fc['indep_fc_2026']).round(1)
        
        all_final_forecasts.append(merged_fc)

        # E. Disaggregation & Blending (Validation 2025)
        if not df_cp_val.empty:
            # We merge cps_val (which now has size-level actuals) with cp_val (which has product-level predictions)
            merged_val = df_cps_val.merge(df_cp_val.drop(columns='actual_2025'), on=['channel_id', 'product_id'])
            merged_val = merged_val.merge(ratios, on=['channel_id', 'product_id', 'size'])
            
            merged_val['disagg_pred_2025'] = merged_val['cp_pred_2025'] * merged_val['size_ratio']
            merged_val['blended_pred_2025'] = (ALPHA * merged_val['disagg_pred_2025'] + (1 - ALPHA) * merged_val['indep_pred_2025']).round(1)
            all_validation_results.append(merged_val)

    # 4. Final Aggregation & Export
    print("\nFinalizing results and aggregating to all levels...")
    final_df = pd.concat(all_final_forecasts, ignore_index=True)
    val_df = pd.concat(all_validation_results, ignore_index=True)

    # Rename for consistency
    final_df = final_df.rename(columns={'channel_id': 'stad', 'product_id': 'product', 'size': 'maat', 'blended_fc_2026': 'forecast_2026'})
    val_df = val_df.rename(columns={'channel_id': 'stad', 'product_id': 'product', 'size': 'maat', 'blended_pred_2025': 'predicted_2025'})

    # Map regions for aggregation
    final_df['region'] = final_df['stad'].map(REGION_MAP).fillna('Other')
    val_df['region'] = val_df['stad'].map(REGION_MAP).fillna('Other')

    def save_level(df_fc, df_val, path, name):
        print(f"  Saving Level: {name} -> {path}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            df_fc.to_excel(writer, sheet_name="Forecast_2026", index=False)
            df_val.to_excel(writer, sheet_name="Validatie_2025", index=False)
            summary = df_fc.groupby('stad')[['actual_2025', 'forecast_2026']].sum().reset_index()
            summary['growth_pct'] = ((summary['forecast_2026'] - summary['actual_2025']) / summary['actual_2025'] * 100).round(1)
            summary.to_excel(writer, sheet_name="Summary_City", index=False)

    # 1. Lowest Level (City/Product/Size)
    save_level(final_df, val_df, OUTPUT_CITY_PRODUCT_SIZE, "CITY_PRODUCT_SIZE")

    # 2. City/Product Level
    cp_fc = final_df.groupby(['stad', 'product'])[['actual_2025', 'forecast_2026']].sum().reset_index()
    cp_val = val_df.groupby(['stad', 'product'])[['actual_2025', 'predicted_2025']].sum().reset_index()
    save_level(cp_fc, cp_val, OUTPUT_CITY_PRODUCT, "CITY_PRODUCT")

    # 3. City Level
    c_fc = final_df.groupby(['stad'])[['actual_2025', 'forecast_2026']].sum().reset_index()
    c_val = val_df.groupby(['stad'])[['actual_2025', 'predicted_2025']].sum().reset_index()
    save_level(c_fc, c_val, OUTPUT_CITY, "CITY")

    # 3.5 Region Level
    r_fc = final_df.groupby(['region'])[['actual_2025', 'forecast_2026']].sum().reset_index().rename(columns={'region': 'stad'})
    r_val = val_df.groupby(['region'])[['actual_2025', 'predicted_2025']].sum().reset_index().rename(columns={'region': 'stad'})
    save_level(r_fc, r_val, OUTPUT_REGION, "REGION")

    # 4. Total Level
    t_fc = final_df[['actual_2025', 'forecast_2026']].sum().to_frame().T
    t_fc['stad'] = 'TOTAL_COMPANY'
    t_val = val_df[['actual_2025', 'predicted_2025']].sum().to_frame().T
    t_val['stad'] = 'TOTAL_COMPANY'
    save_level(t_fc, t_val, OUTPUT_TOTAL, "TOTAL")

    print("\nDONE!")
    val_df['abs_error'] = abs(val_df['predicted_2025'] - val_df['actual_2025'])
    val_df['pct_error'] = (val_df['abs_error'] / val_df['actual_2025'] * 100).replace([np.inf, -np.inf], np.nan)
    print(f"Overall Validation MAPE (Lowest Level): {val_df['pct_error'].mean():.1f}%")

if __name__ == "__main__":
    run_middle_out_reconciliation()

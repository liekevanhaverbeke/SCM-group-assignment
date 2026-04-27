"""
TOP-DOWN PROPORTIONAL RECONCILIATION
- Splits data into 'Platform' (outlier) and 'Normal' stores.
- Forecasts at the City level (channel_id).
- Disaggregates to the lowest level (Product/Size) using historical proportions.
- Ensures coherence and isolates the declining trend of 'Platform'.
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import os
import warnings

# Onderdruk warnings
warnings.filterwarnings("ignore")

# ── 1. Configuratie ──────────────────────────────────────────────────────────
INPUT_PATH = "True demand/True demand Simeon/True_Demand_Results.xlsx"
SHEET_NAME = "1_True_Demand_Lijst"
FORECAST_FOLDER = "Forecast/Results"
OUTPUT_TOTAL = f"{FORECAST_FOLDER}/top_down_total.xlsx"
OUTPUT_REGION = f"{FORECAST_FOLDER}/top_down_region.xlsx"
OUTPUT_CITY = f"{FORECAST_FOLDER}/top_down_city.xlsx"
OUTPUT_CITY_PRODUCT = f"{FORECAST_FOLDER}/top_down_city_product.xlsx"
OUTPUT_CITY_PRODUCT_SIZE = f"{FORECAST_FOLDER}/top_down_city_product_size.xlsx"

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
    if len(y) < 3: # Not enough data for ARIMA
        return float(waarden[-1]) if len(waarden) > 0 else 0.0
    try:
        model = ARIMA(y, order=order)
        fit = model.fit()
        forecast = fit.forecast(predict_steps).iloc[-1]
        return max(0.0, float(forecast))
    except:
        return float(waarden[-1])

def run_top_down_reconciliation():
    print("="*80)
    print("STARTING TOP-DOWN PROPORTIONAL RECONCILIATION")
    print("="*80)

    # 3.1 Data laden
    print(f"Loading data from {INPUT_PATH}...")
    df = pd.read_excel(INPUT_PATH, sheet_name=SHEET_NAME)
    
    # 3.2 Segmentatie (Step 2: Handle the Outlier Store First)
    print("Segmenting data: 'Platform' (Outlier) vs Others...")
    df_platform = df[df['channel_id'] == 'Platform'].copy()
    df_normal   = df[df['channel_id'] != 'Platform'].copy()
    
    # Lijst met segmenten voor verwerking
    segments = [
        {"name": "NORMAL_STORES", "df": df_normal},
        {"name": "PLATFORM", "df": df_platform}
    ]
    
    all_final_forecasts = []
    all_validation_results = []

    for seg in segments:
        seg_name = seg["name"]
        seg_df = seg["df"]
        
        if seg_df.empty:
            continue
            
        print(f"\nProcessing segment: {seg_name}")
        
        # A. Calculate Historical Shares (Step 3, Method A)
        # We use all historical data to determine the proportions of product x size within each city
        print(f"  Calculating historical shares for {seg_name}...")
        total_city_sales = seg_df.groupby('channel_id')['true_demand'].sum()
        
        shares = (
            seg_df.groupby(['channel_id', 'product_id', 'size'])['true_demand']
            .sum()
            .reset_index()
        )
        
        # Merge with city totals to compute ratio
        shares = shares.merge(total_city_sales.rename('city_total'), on='channel_id')
        shares['share'] = shares['true_demand'] / shares['city_total']
        shares = shares.drop(columns=['true_demand', 'city_total'])
        
        # B. City-Level Forecasts
        # Aggregate to city/year level for forecasting
        city_year = seg_df.groupby(['channel_id', 'season'])['true_demand'].sum().reset_index()
        cities = city_year['channel_id'].unique()
        
        city_forecasts_2026 = []
        city_validation_2025 = []
        
        print(f"  Generating city-level forecasts for {len(cities)} cities...")
        for city in cities:
            city_data = city_year[city_year['channel_id'] == city].sort_values('season')
            
            # Forecast 2026 (using data up to 2025)
            fc_2026 = arima_forecast(city_data['true_demand'].tolist())
            actual_2025 = city_data[city_data['season'] == 2025]['true_demand'].values[0] if not city_data[city_data['season'] == 2025].empty else 0
            
            city_forecasts_2026.append({
                'channel_id': city,
                'forecast_city_2026': fc_2026,
                'actual_city_2025': actual_2025
            })
            
            # Validation 2025 (using data up to 2024)
            train_val = city_data[city_data['season'] <= 2024]
            if not train_val.empty:
                pred_2025 = arima_forecast(train_val['true_demand'].tolist())
                city_validation_2025.append({
                    'channel_id': city,
                    'predicted_city_2025': pred_2025,
                    'actual_city_2025': actual_2025
                })

        df_city_fc = pd.DataFrame(city_forecasts_2026)
        df_city_val = pd.DataFrame(city_validation_2025)
        
        # C. Disaggregate Forecasts (Step 3.3)
        print(f"  Disaggregating city-level forecasts to Product/Size level...")
        bottom_forecast = shares.merge(df_city_fc, on='channel_id')
        bottom_forecast['forecast_2026'] = (bottom_forecast['share'] * bottom_forecast['forecast_city_2026']).round(1)
        bottom_forecast['actual_2025'] = (bottom_forecast['share'] * bottom_forecast['actual_city_2025']).round(1)
        
        all_final_forecasts.append(bottom_forecast)
        
        # Disaggregate Validation
        if not df_city_val.empty:
            bottom_val = shares.merge(df_city_val, on='channel_id')
            bottom_val['predicted_2025'] = (bottom_val['share'] * bottom_val['predicted_city_2025']).round(1)
            
            # GET REAL ACTUALS for 2025
            real_act_2025 = seg_df[seg_df['season'] == 2025].groupby(['channel_id', 'product_id', 'size'])['true_demand'].sum().reset_index()
            real_act_2025 = real_act_2025.rename(columns={'true_demand': 'actual_2025'})
            
            # Merge with real actuals instead of re-sharing city actuals
            bottom_val = bottom_val.drop(columns=['actual_city_2025'])
            bottom_val = bottom_val.merge(real_act_2025, on=['channel_id', 'product_id', 'size'], how='left').fillna(0)
            
            all_validation_results.append(bottom_val)

    # 3.4 Combineer resultaten (Lowest Level)
    final_df = pd.concat(all_final_forecasts, ignore_index=True)
    val_df = pd.concat(all_validation_results, ignore_index=True)
    
    # Calculate errors for validation
    val_df['abs_error'] = abs(val_df['predicted_2025'] - val_df['actual_2025'])
    val_df['pct_error'] = np.where(val_df['actual_2025'] > 0, (val_df['abs_error'] / val_df['actual_2025'] * 100), np.nan)
    
    # ── 3.5 Aggregatie naar hogere niveaus ───────────────────────────────────
    # We hebben nu het laagste niveau (stad, product, maat). 
    # De rest aggregeren we hieruit om consistentie te garanderen.
    
    # Rename for consistency with other scripts
    final_df = final_df.rename(columns={'channel_id': 'stad', 'product_id': 'product', 'size': 'maat'})
    val_df = val_df.rename(columns={'channel_id': 'stad', 'product_id': 'product', 'size': 'maat'})

    # Map regions for aggregation
    # Note: 'stad' now contains the city name
    final_df['region'] = final_df['stad'].map(REGION_MAP).fillna('Other')
    val_df['region'] = val_df['stad'].map(REGION_MAP).fillna('Other')

    # Helper function to save a level
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
    print(f"Total combinations forecasted: {len(final_df)}")
    print(f"Overall Validation MAPE (Lowest Level): {val_df['pct_error'].mean():.1f}%")

if __name__ == "__main__":
    run_top_down_reconciliation()

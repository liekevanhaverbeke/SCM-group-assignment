"""
PRODUCT-CITY LEVEL SES FORECAST
- Forecasts at the Product x City level using Simple Exponential Smoothing (SES).
- Disaggregates to the Size level using direct historical size proportions.
- Includes automated validation (2020-2025) and final 2026 forecast.
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# ── 0. Configuration ──────────────────────────────────────────────────────────
INPUT_PATH = "True demand/True demand Simeon/True_Demand_Results.xlsx"
SHEET_NAME = "1_True_Demand_Lijst"

# ── 1. SES Forecast Function ──────────────────────────────────────────────────

def ses_forecast(values):
    y = np.array(values, dtype=float)
    if len(y) < 2: return float(y[-1]) if len(y) > 0 else 0.0
    def sse(a):
        if not (0 < a < 1): return 1e15
        L = y[0]; err = 0.0
        for i in range(1, len(y)):
            fitted = L
            L = a * y[i] + (1 - a) * fitted
            err += (y[i] - fitted)**2
        return err
    try:
        res = minimize(sse, x0=[0.5], method="Nelder-Mead")
        alpha = max(0.01, min(0.99, res.x[0]))
        L = y[0]
        for i in range(1, len(y)): L = alpha * y[i] + (1 - alpha) * L
        return max(0.0, float(L))
    except: return float(y[-1])

# ── 2. Pipeline Logic ────────────────────────────────────────────────────────

def run_product_city_ses_pipeline(df, target_year):
    history = df[df['season'] < target_year].copy()
    if history.empty: return pd.DataFrame()
    
    # A. Calculate Size Proportions per Product per City
    prod_city_total = history.groupby(['channel_id', 'product_id'])['true_demand'].sum().reset_index()
    size_shares = history.groupby(['channel_id', 'product_id', 'size'])['true_demand'].sum().reset_index()
    size_shares = size_shares.merge(prod_city_total.rename(columns={'true_demand':'pc_total'}), on=['channel_id', 'product_id'])
    size_shares['size_share'] = (size_shares['true_demand'] / size_shares['pc_total']).fillna(0)
    size_shares = size_shares.drop(columns=['true_demand', 'pc_total'])
    
    # B. Product-City Level Forecasts using SES
    pc_history = history.groupby(['channel_id', 'product_id', 'season'])['true_demand'].sum().reset_index()
    
    # Filter to only products that have history
    combos = pc_history[['channel_id', 'product_id']].drop_duplicates()
    
    pc_forecasts = []
    for _, row in combos.iterrows():
        sub = pc_history[(pc_history['channel_id'] == row['channel_id']) & (pc_history['product_id'] == row['product_id'])].sort_values('season')
        vals = sub['true_demand'].tolist()
        
        fc = ses_forecast(vals)
        if fc > 0:
            pc_forecasts.append({
                'channel_id': row['channel_id'],
                'product_id': row['product_id'],
                'forecast_pc': fc
            })
            
    df_pc_fc = pd.DataFrame(pc_forecasts)
    if df_pc_fc.empty: return pd.DataFrame()
    
    # C. Disaggregate to Size Level
    res = size_shares.merge(df_pc_fc, on=['channel_id', 'product_id'])
    res['forecast_sku'] = res['size_share'] * res['forecast_pc']
    res['season'] = target_year
    
    return res[['channel_id', 'season', 'product_id', 'size', 'forecast_sku']]

def main():
    print("Loading raw True Demand data...")
    raw_demand = pd.read_excel(INPUT_PATH, sheet_name=SHEET_NAME)
    
    sku_actuals = raw_demand.groupby(['channel_id', 'season', 'product_id', 'size'])['true_demand'].sum().reset_index().rename(columns={'true_demand':'actual_demand'})
    
    # ── 3. Validation Loop (2020-2025) ───────────────────────────────────────
    years = [2020, 2021, 2022, 2023, 2024, 2025]
    all_val = []
    for y in years:
        print(f"Running Product-City SES Pipeline for {y}...")
        all_val.append(run_product_city_ses_pipeline(raw_demand, y))
    
    val_df = pd.concat(all_val, ignore_index=True)
    val_df = pd.merge(val_df, sku_actuals, on=['channel_id', 'season', 'product_id', 'size'], how='outer')
    val_df = val_df[val_df['season'].isin(years)].fillna(0)
    
    val_df['abs_error'] = abs(val_df['forecast_sku'] - val_df['actual_demand'])
    val_df['sq_error'] = (val_df['forecast_sku'] - val_df['actual_demand'])**2
    
    # ── 4. Print Performance ──────────────────────────────────────────────────
    print("\n" + "="*50 + "\nPRODUCT-CITY SES PERFORMANCE\n" + "="*50)
    yearly = val_df.groupby('season').agg(MAE=('abs_error', 'mean'), MSE=('sq_error', 'mean')).reset_index()
    print(yearly.to_string(index=False))
    print(f"\nCOMBINED MAE: {val_df['abs_error'].mean():.2f}")
    
    product_metrics = val_df.groupby('product_id').agg(MSE=('sq_error', 'mean'), MAE=('abs_error', 'mean')).reset_index()
    city_metrics = val_df.groupby('channel_id').agg(MSE=('sq_error', 'mean'), MAE=('abs_error', 'mean')).reset_index()
    
    print("\n" + "="*50)
    print("TOP 5 WORST PRODUCTS (By MAE, Product-City SES 2020-2025)")
    print("="*50)
    print(product_metrics.sort_values(by='MAE', ascending=False).head(5).to_string(index=False))
    
    print("\n" + "="*50)
    print("CITY PERFORMANCE (Sorted by MAE, Product-City SES 2020-2025)")
    print("="*50)
    print(city_metrics.sort_values(by='MAE', ascending=False).to_string(index=False))
    
    # ── 5. Generate Forecast for 2026 ────────────────────────────────────────
    #print("\nGenerating Forecast for 2026...")
    #fc_2026 = run_product_city_ses_pipeline(raw_demand, 2026)
    
    # ── 6. Save Results ───────────────────────────────────────────────────────
    os.makedirs('Validation files', exist_ok=True)
    
    with pd.ExcelWriter('Validation files/product_city_ses_validation_all_years.xlsx') as writer:
        val_df.to_excel(writer, sheet_name='SKU_Validation', index=False)
        yearly.to_excel(writer, sheet_name='Yearly_Metrics', index=False)
        product_metrics.to_excel(writer, sheet_name='Metrics_per_Product', index=False)
        city_metrics.to_excel(writer, sheet_name='Metrics_per_City', index=False)

    #output_2026 = fc_2026.rename(columns={
    #    'channel_id': 'stad',
    #    'product_id': 'product',
    #    'size': 'maat',
    #    'forecast_sku': 'forecast_2026'
    #})[['stad', 'product', 'maat', 'forecast_2026']]
    
    #output_2026 = output_2026[output_2026['forecast_2026'] > 0]
    #output_2026.to_excel('Validation files/product_city_ses_sku_forecast_2026.xlsx', sheet_name='Forecast_2026', index=False)
    
    #print(f"\nDONE! Final 2026 SKU Forecast Units: {output_2026['forecast_2026'].sum():.2f}")

if __name__ == "__main__":
    main()

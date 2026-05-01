"""
TOP-DOWN PROPORTIONAL RECONCILIATION (LINEAR VERSION)
- Forecasts at the City level (channel_id) using Linear Regression.
- Disaggregates to the lowest level (Product/Size) using direct historical proportions.
- Includes automated validation (2020-2025) and final 2026 forecast.
"""

import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# ── 0. Configuration ──────────────────────────────────────────────────────────
INPUT_PATH = "True demand/True demand Simeon/True_Demand_Results.xlsx"
SHEET_NAME = "1_True_Demand_Lijst"
REGION_MAP = {
    'Platform': 'Online', 'Webshop': 'Online',
    'Helsinki': 'Scandinavian', 'Copenhagen': 'Scandinavian', 'Stockholm': 'Scandinavian',
    'Amsterdam': 'European', 'Berlin': 'European', 'Brussels': 'European',
    'Madrid': 'European', 'Paris': 'European', 'Rome': 'European'
}

# ── 1. Forecast Functions ─────────────────────────────────────────────────────

def lr_forecast(jaren, waarden, target_year):
    jaars = [j for j, w in zip(jaren, waarden) if not np.isnan(w)]
    waard = [w for w in waarden if not np.isnan(w)]
    if len(waard) < 2: return float(waard[-1]) if waard else 0.0
    model = LinearRegression().fit(np.array(jaars).reshape(-1, 1), np.array(waard))
    return max(0.0, float(model.predict([[target_year]])[0]))

# ── 2. Pipeline Logic ────────────────────────────────────────────────────────

def run_top_down_pipeline(df, target_year):
    history = df[df['season'] < target_year].copy()
    if history.empty: return pd.DataFrame()
    
    # A. Calculate Historical Shares
    # (Direct share for every unique product x size combination within each city)
    total_city_sales = history.groupby('channel_id')['true_demand'].sum()
    shares = (
        history.groupby(['channel_id', 'product_id', 'size'])['true_demand']
        .sum()
        .reset_index()
    )
    shares = shares.merge(total_city_sales.rename('city_total'), on='channel_id')
    shares['share'] = shares['true_demand'] / shares['city_total']
    shares = shares.drop(columns=['true_demand', 'city_total'])
    
    # B. City-Level Forecasts
    city_year = history.groupby(['channel_id', 'season'])['true_demand'].sum().reset_index()
    cities = city_year['channel_id'].unique()
    
    city_forecasts = []
    for city in cities:
        city_data = city_year[city_year['channel_id'] == city].sort_values('season')
        jaren = city_data['season'].tolist()
        waard = city_data['true_demand'].tolist()
        
        fc = lr_forecast(jaren, waard, target_year)
        city_forecasts.append({'channel_id': city, 'forecast_city': fc})
    
    df_city_fc = pd.DataFrame(city_forecasts)
    
    # C. Disaggregate
    res = shares.merge(df_city_fc, on='channel_id')
    res['forecast_sku'] = res['share'] * res['forecast_city']
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
        print(f"Running Top-Down Linear Pipeline for {y}...")
        all_val.append(run_top_down_pipeline(raw_demand, y))
    
    val_df = pd.concat(all_val, ignore_index=True)
    val_df = pd.merge(val_df, sku_actuals, on=['channel_id', 'season', 'product_id', 'size'], how='outer')
    val_df = val_df[val_df['season'].isin(years)].fillna(0)
    
    val_df['abs_error'] = abs(val_df['forecast_sku'] - val_df['actual_demand'])
    val_df['sq_error'] = (val_df['forecast_sku'] - val_df['actual_demand'])**2
    
    # ── 4. Print Performance (Same as Hybrid Layered) ────────────────────────
    print("\n" + "="*50 + "\nOVERALL TOP-DOWN LINEAR PERFORMANCE\n" + "="*50)
    yearly = val_df.groupby('season').agg(MAE=('abs_error', 'mean'), MSE=('sq_error', 'mean')).reset_index()
    print(yearly.to_string(index=False))
    print(f"\nCOMBINED MAE: {val_df['abs_error'].mean():.2f}")
    
    product_metrics = val_df.groupby('product_id').agg(MSE=('sq_error', 'mean'), MAE=('abs_error', 'mean')).reset_index()
    city_metrics = val_df.groupby('channel_id').agg(MSE=('sq_error', 'mean'), MAE=('abs_error', 'mean')).reset_index()
    
    print("\n" + "="*50)
    print("TOP 5 WORST PRODUCTS (By MAE, Top-Down Linear 2020-2025)")
    print("="*50)
    print(product_metrics.sort_values(by='MAE', ascending=False).head(5).to_string(index=False))
    
    print("\n" + "="*50)
    print("CITY PERFORMANCE (Sorted by MAE, Top-Down Linear 2020-2025)")
    print("="*50)
    print(city_metrics.sort_values(by='MAE', ascending=False).to_string(index=False))
    
    # ── 5. Generate Forecast for 2026 ────────────────────────────────────────
    #print("\nGenerating Forecast for 2026...")
    #fc_2026 = run_top_down_pipeline(raw_demand, 2026)
    
    # ── 6. Save Results ───────────────────────────────────────────────────────
    os.makedirs('Validation files', exist_ok=True)
    
    # Save the full validation details
    with pd.ExcelWriter('Validation files/top_down_linear_validation_all_years.xlsx') as writer:
        val_df.to_excel(writer, sheet_name='SKU_Validation', index=False)
        yearly.to_excel(writer, sheet_name='Yearly_Metrics', index=False)
        product_metrics.to_excel(writer, sheet_name='Metrics_per_Product', index=False)
        city_metrics.to_excel(writer, sheet_name='Metrics_per_City', index=False)

    # Prepare final output format: stad, product, maat, forecast_2026
    #output_2026 = fc_2026.rename(columns={
    #    'channel_id': 'stad',
    #    'product_id': 'product',
    #    'size': 'maat',
    #    'forecast_sku': 'forecast_2026'
    #})[['stad', 'product', 'maat', 'forecast_2026']]
    
    # Filter out 0 forecast
    #output_2026 = output_2026[output_2026['forecast_2026'] > 0]

    #output_2026.to_excel('Validation files/top_down_linear_sku_forecast_2026.xlsx', sheet_name='Forecast_2026', index=False)
    
    #print(f"\nDONE! Final 2026 SKU Forecast Units: {output_2026['forecast_2026'].sum():.2f}")

if __name__ == "__main__":
    main()

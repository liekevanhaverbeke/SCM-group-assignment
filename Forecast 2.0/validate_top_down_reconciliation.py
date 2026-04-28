import pandas as pd
import numpy as np
import os
import warnings
from sklearn.linear_model import LinearRegression
warnings.filterwarnings('ignore')

def lr_forecast(jaren, waarden, target_year):
    if len(jaren) < 2:
        return float(waarden[-1]) if len(waarden) > 0 else 0.0
    model = LinearRegression().fit(np.array(jaren).reshape(-1, 1), np.array(waarden))
    return max(0.0, float(model.predict([[target_year]])[0]))

def generate_reconciliation_forecast(df, target_year):
    """Replicates top_down_reconciliation linear.py logic for a specific year."""
    history = df[df['season'] < target_year].copy()
    if history.empty: return pd.DataFrame()
    
    segments = [
        history[history['channel_id'] != 'Platform'].copy(),
        history[history['channel_id'] == 'Platform'].copy()
    ]
    
    all_forecasts = []
    
    for seg_df in segments:
        if seg_df.empty: continue
        
        # 1. Calculate historical shares (SKU Total / City Total)
        total_city_sales = seg_df.groupby('channel_id')['true_demand'].sum()
        shares = seg_df.groupby(['channel_id', 'product_id', 'size'])['true_demand'].sum().reset_index()
        shares = shares.merge(total_city_sales.rename('city_total'), on='channel_id')
        shares['share'] = shares['true_demand'] / shares['city_total']
        shares = shares.drop(columns=['true_demand', 'city_total'])
        
        # 2. City-Level Forecast using Linear Regression
        city_year = seg_df.groupby(['channel_id', 'season'])['true_demand'].sum().reset_index()
        city_forecasts = []
        for city in city_year['channel_id'].unique():
            city_data = city_year[city_year['channel_id'] == city].sort_values('season')
            fc = lr_forecast(city_data['season'].tolist(), city_data['true_demand'].tolist(), target_year)
            city_forecasts.append({'channel_id': city, 'forecast_city': fc})
            
        df_city_fc = pd.DataFrame(city_forecasts)
        
        # 3. Disaggregate Forecasts
        bottom_forecast = shares.merge(df_city_fc, on='channel_id')
        bottom_forecast['forecast_sku'] = bottom_forecast['share'] * bottom_forecast['forecast_city']
        
        all_forecasts.append(bottom_forecast[['channel_id', 'product_id', 'size', 'forecast_sku']])
        
    if not all_forecasts: return pd.DataFrame()
    
    res = pd.concat(all_forecasts, ignore_index=True)
    res['season'] = target_year
    return res

def main():
    print("Loading raw True Demand data...")
    raw_demand = pd.read_excel('True demand/True demand Simeon/True_Demand_Results.xlsx', sheet_name='1_True_Demand_Lijst')
    
    # Actuals at SKU level
    sku_actuals = raw_demand.groupby(['channel_id', 'season', 'product_id', 'size'])['true_demand'].sum().reset_index()
    sku_actuals = sku_actuals.rename(columns={'true_demand': 'actual_demand'})
    
    years_to_validate = [2020, 2021, 2022, 2023, 2024, 2025]
    all_forecasts = []
    
    for year in years_to_validate:
        print(f"Running Original Top-Down Reconciliation Logic for {year}...")
        forecast_df = generate_reconciliation_forecast(raw_demand, year)
        all_forecasts.append(forecast_df)
        
    all_forecasts = pd.concat(all_forecasts, ignore_index=True)
    
    print("\nCalculating Errors...")
    validation_df = pd.merge(all_forecasts, sku_actuals, on=['channel_id', 'season', 'product_id', 'size'], how='outer')
    
    # We only care about validation years
    validation_df = validation_df[validation_df['season'].isin(years_to_validate)]
    
    validation_df['forecast_sku'] = validation_df['forecast_sku'].fillna(0)
    validation_df['actual_demand'] = validation_df['actual_demand'].fillna(0)
    
    validation_df['absolute_error'] = abs(validation_df['forecast_sku'] - validation_df['actual_demand'])
    validation_df['squared_error'] = (validation_df['forecast_sku'] - validation_df['actual_demand'])**2
    
    print("\n" + "="*50)
    print("YEARLY ORIGINAL RECONCILIATION PERFORMANCE (LR City + Direct SKU Share)")
    print("="*50)
    
    yearly_metrics = validation_df.groupby('season').agg(
        MAE=('absolute_error', 'mean'),
        MSE=('squared_error', 'mean')
    ).reset_index()
    
    for _, row in yearly_metrics.iterrows():
        print(f"--- {int(row['season'])} ---")
        print(f"MAE: {row['MAE']:.2f}")
        print(f"MSE: {row['MSE']:.2f}\n")
        
    print(f"OVERALL COMBINED MAE (2020-2025): {validation_df['absolute_error'].mean():.2f}")
    print(f"OVERALL COMBINED MSE (2020-2025): {validation_df['squared_error'].mean():.2f}")
    
    # Aggregations per Product
    product_metrics = validation_df.groupby('product_id').agg(
        MSE=('squared_error', 'mean'),
        MAE=('absolute_error', 'mean')
    ).reset_index()
    
    # Aggregations per City
    city_metrics = validation_df.groupby('channel_id').agg(
        MSE=('squared_error', 'mean'),
        MAE=('absolute_error', 'mean')
    ).reset_index()
    
    print("\n" + "="*50)
    print("TOP 5 WORST PRODUCTS (By MAE, Original Reconciliation 2020-2025)")
    print("="*50)
    print(product_metrics.sort_values(by='MAE', ascending=False).head(5).to_string(index=False))
    
    print("\n" + "="*50)
    print("CITY PERFORMANCE (Sorted by MAE, Original Reconciliation 2020-2025)")
    print("="*50)
    print(city_metrics.sort_values(by='MAE', ascending=False).to_string(index=False))
    
    output_file = 'Forecast/reconciliation_validation_all_years.xlsx'
    with pd.ExcelWriter(output_file) as writer:
        validation_df.to_excel(writer, sheet_name='SKU_Validation', index=False)
        yearly_metrics.to_excel(writer, sheet_name='Yearly_Metrics', index=False)
        product_metrics.to_excel(writer, sheet_name='Metrics_per_Product', index=False)
        city_metrics.to_excel(writer, sheet_name='Metrics_per_City', index=False)
        
    print(f"\nDetailed reconciliation reports saved to: {output_file}")

if __name__ == "__main__":
    main()

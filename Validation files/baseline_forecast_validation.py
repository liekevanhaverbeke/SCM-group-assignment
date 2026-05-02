import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

def main():
    print("Loading raw True Demand data...")
    raw_demand = pd.read_excel('True demand/True demand Simeon/True_Demand_Results.xlsx', sheet_name='1_True_Demand_Lijst')
    
    # 1. Get Actuals at SKU level
    sku_actuals = raw_demand.groupby(['channel_id', 'season', 'product_id', 'size'])['true_demand'].sum().reset_index()
    sku_actuals = sku_actuals.rename(columns={'true_demand': 'actual_demand'})
    
    # 2. Generate Naive Baseline Forecast (Forecast_Year = Actuals_Year_Minus_1)
    naive_forecast = sku_actuals.copy()
    naive_forecast['season'] = naive_forecast['season'] + 1  # Shift the year forward by 1
    naive_forecast = naive_forecast.rename(columns={'actual_demand': 'forecast_sku'})
    
    # 3. Validation Phase
    years_to_validate = [2020, 2021, 2022, 2023, 2024, 2025]
    print(f"Validating Naive Baseline for years: {years_to_validate}...")
    
    # Merge actuals and forecast
    validation_df = pd.merge(naive_forecast, sku_actuals, on=['channel_id', 'season', 'product_id', 'size'], how='outer')
    
    # We only care about validation years
    validation_df = validation_df[validation_df['season'].isin(years_to_validate)]
    
    # If a SKU wasn't sold last year, forecast is 0. If it wasn't sold this year, actual is 0.
    validation_df['forecast_sku'] = validation_df['forecast_sku'].fillna(0)
    validation_df['actual_demand'] = validation_df['actual_demand'].fillna(0)
    
    # Calculate errors
    validation_df['raw_error'] = validation_df['actual_demand'] - validation_df['forecast_sku']
    validation_df['absolute_error'] = abs(validation_df['raw_error'])
    validation_df['squared_error'] = (validation_df['raw_error'])**2
    
    # MAPE_val calculation
    validation_df['MAPE_val'] = np.where(validation_df['actual_demand'] != 0, 
                                         validation_df['absolute_error'] / validation_df['actual_demand'], 
                                         0)
    
    # MPE_val calculation (shows over/under forecast bias)
    validation_df['MPE_val'] = np.where(validation_df['actual_demand'] != 0, 
                                        validation_df['raw_error'] / validation_df['actual_demand'], 
                                        0)


    print("\n" + "="*50)
    print("YEARLY NAIVE BASELINE PERFORMANCE (Last Year = This Year)")
    print("="*50)
    
    yearly_metrics = validation_df.groupby('season').agg(
        MAE=('absolute_error', 'mean'),
        MSE=('squared_error', 'mean'),
        MAPE=('MAPE_val', 'mean'),
        MPE=('MPE_val', 'mean')
    ).reset_index()
    
    for _, row in yearly_metrics.iterrows():
        print(f"--- {int(row['season'])} ---")
        print(f"MAE: {row['MAE']:.2f}")
        print(f"MSE: {row['MSE']:.2f}\n")
        print(f"MAPE: {row['MAPE']:.2%}")
        print(f"MPE: {row['MPE']:.2%}\n")
        
    print(f"OVERALL COMBINED MAE (2020-2025): {validation_df['absolute_error'].mean():.2f}")
    print(f"OVERALL COMBINED MSE (2020-2025): {validation_df['squared_error'].mean():.2f}")
    
    # Aggregations per Product
    product_metrics = validation_df.groupby('product_id').agg(
        MSE=('squared_error', 'mean'),
        MAE=('absolute_error', 'mean'),
        MAPE=('MAPE_val', 'mean'),
        MPE=('MPE_val', 'mean')
    ).reset_index()
    
    # Aggregations per City
    city_metrics = validation_df.groupby('channel_id').agg(
        MSE=('squared_error', 'mean'),
        MAE=('absolute_error', 'mean'),
        MAPE=('MAPE_val', 'mean'),
        MPE=('MPE_val', 'mean')
    ).reset_index()
    
    print("\n" + "="*50)
    print("TOP 5 WORST PRODUCTS (By MAE, Naive Baseline 2020-2025)")
    print("="*50)
    print(product_metrics.sort_values(by='MAE', ascending=False).head(5).to_string(index=False))
    
    print("\n" + "="*50)
    print("CITY PERFORMANCE (Sorted by MAE, Naive Baseline 2020-2025)")
    print("="*50)
    print(city_metrics.sort_values(by='MAE', ascending=False).to_string(index=False))
    
    output_file = 'Validation files/baseline_validation_all_years.xlsx'
    with pd.ExcelWriter(output_file) as writer:
        validation_df.to_excel(writer, sheet_name='SKU_Validation', index=False)
        yearly_metrics.to_excel(writer, sheet_name='Yearly_Metrics', index=False)
        product_metrics.to_excel(writer, sheet_name='Metrics_per_Product', index=False)
        city_metrics.to_excel(writer, sheet_name='Metrics_per_City', index=False)
        
    print(f"\nDetailed naive baseline reports saved to: {output_file}")

if __name__ == "__main__":
    main()

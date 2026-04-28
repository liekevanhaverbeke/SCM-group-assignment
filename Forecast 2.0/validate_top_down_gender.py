import pandas as pd
import numpy as np
import os
import warnings
from sklearn.linear_model import LinearRegression
warnings.filterwarnings('ignore')

def lr_forecast(jaren, waarden, forecast_jaar):
    """Linear regression helper. Falls back to naive if < 2 points."""
    if len(jaren) < 2:
        return float(waarden[-1]) if len(waarden) > 0 else 0.0
        
    X = np.array(jaren).reshape(-1, 1)
    y = np.array(waarden)
    
    model = LinearRegression()
    model.fit(X, y)
    
    pred = model.predict(np.array([[forecast_jaar]]))
    return max(0.0, float(pred[0]))

def load_proportions_without_gender(target_year):
    """Loads proportions for Price, Product, and Size (skipping Gender)."""
    
    # 1. PRICE
    price_xl = pd.ExcelFile('Bert Map/price_proportions_by_gender_channel.xlsx')
    price_dfs = []
    for sheet in price_xl.sheet_names:
        df = price_xl.parse(sheet)
        df['category'] = sheet
        price_dfs.append(df)
    price_prop = pd.concat(price_dfs, ignore_index=True)
    price_prop = price_prop[price_prop['season'] < target_year].copy()
    
    if price_prop.empty: 
        return None, None, None
    
    bracket_cols = [c for c in price_prop.columns if c not in ['channel_id', 'season', 'category']]
    price_prop_melted = price_prop.melt(id_vars=['channel_id', 'season', 'category'],
                                        value_vars=bracket_cols,
                                        var_name='price_bracket', value_name='price_pct')
    price_avg = price_prop_melted.groupby(['channel_id', 'category', 'price_bracket'])['price_pct'].sum().reset_index()
    price_avg['price_pct'] = (price_avg['price_pct'] / price_avg.groupby(['channel_id', 'category'])['price_pct'].transform('sum') * 100).fillna(0)
    
    # 2. PRODUCT
    prod_xl = pd.ExcelFile('Bert Map/product_shares_within_price_brackets.xlsx')
    prod_dfs = []
    for sheet in prod_xl.sheet_names:
        df = prod_xl.parse(sheet)
        df['category'] = sheet
        prod_dfs.append(df)
    prod_share = pd.concat(prod_dfs, ignore_index=True)
    prod_share = prod_share[prod_share['season'] < target_year].copy()
    
    prod_avg = prod_share.groupby(['channel_id', 'category', 'price_bracket', 'product_id'])['share_within_bracket'].sum().reset_index()
    prod_avg['share_within_bracket'] = (prod_avg['share_within_bracket'] / prod_avg.groupby(['channel_id', 'category', 'price_bracket'])['share_within_bracket'].transform('sum') * 100).fillna(0)
    
    # 3. SIZE
    size_prop = pd.read_excel('Bert Map/size_proportions_by_channel_gender.xlsx')
    size_prop = size_prop[size_prop['season'] < target_year].copy()
    size_cols = [c for c in size_prop.columns if c not in ['channel_id', 'season', 'category']]
    size_prop_melted = size_prop.melt(id_vars=['channel_id', 'season', 'category'],
                                      value_vars=size_cols,
                                      var_name='size', value_name='size_pct')
    size_avg = size_prop_melted.groupby(['channel_id', 'category', 'size'])['size_pct'].sum().reset_index()
    size_avg['size_pct'] = (size_avg['size_pct'] / size_avg.groupby(['channel_id', 'category'])['size_pct'].transform('sum') * 100).fillna(0)
                                      
    return price_avg, prod_avg, size_avg

def generate_gender_pipeline_forecast(target_year, raw_demand_df):
    """Calculates Linear Regression for City/Gender, then Disaggregates."""
    
    # 1. Calculate LR City/Gender Forecast
    historical_gender = raw_demand_df[raw_demand_df['season'] < target_year].groupby(['channel_id', 'category', 'season'])['true_demand'].sum().reset_index()
    
    if historical_gender.empty:
        return pd.DataFrame()
        
    gender_forecasts = []
    for channel in historical_gender['channel_id'].unique():
        for cat in ['Menswear', 'Womenswear']:
            subset = historical_gender[(historical_gender['channel_id'] == channel) & (historical_gender['category'] == cat)].sort_values('season')
            if subset.empty:
                fc = 0.0
            else:
                fc = lr_forecast(subset['season'].tolist(), subset['true_demand'].tolist(), target_year)
            gender_forecasts.append({'channel_id': channel, 'category': cat, 'forecast_gender': fc})
            
    df = pd.DataFrame(gender_forecasts)
    
    # 2. Load Proportions
    price_avg, prod_avg, size_avg = load_proportions_without_gender(target_year)
    if price_avg is None:
        return pd.DataFrame()
        
    # 3. Disaggregate
    df = pd.merge(df, price_avg, on=['channel_id', 'category'], how='left')
    df['forecast_price'] = df['forecast_gender'] * (df['price_pct'] / 100.0)
    
    df = pd.merge(df, prod_avg, on=['channel_id', 'category', 'price_bracket'], how='left')
    df['forecast_product'] = df['forecast_price'] * (df['share_within_bracket'] / 100.0)
    df = df.dropna(subset=['product_id'])
    
    df = pd.merge(df, size_avg, on=['channel_id', 'category'], how='left')
    df['forecast_sku'] = df['forecast_product'] * (df['size_pct'] / 100.0)
    
    df = df[df['forecast_sku'] > 0].copy()
    df['season'] = target_year
    
    return df[['channel_id', 'season', 'product_id', 'size', 'forecast_sku']]

def main():
    print("Loading raw True Demand data and Product details...")
    raw_demand = pd.read_excel('True demand/True demand Simeon/True_Demand_Results.xlsx', sheet_name='1_True_Demand_Lijst')
    products = pd.read_excel('Input_Files/PPP_stu_products.xlsx')
    
    # Merge category into raw demand
    raw_demand = pd.merge(raw_demand, products[['id', 'category']], left_on='product_id', right_on='id', how='left')
    
    # Actuals at SKU level
    sku_actuals = raw_demand.groupby(['channel_id', 'season', 'product_id', 'size'])['true_demand'].sum().reset_index()
    sku_actuals = sku_actuals.rename(columns={'true_demand': 'actual_demand'})
    
    years_to_validate = [2020, 2021, 2022, 2023, 2024, 2025]
    all_forecasts = []
    
    for year in years_to_validate:
        print(f"Running Gender-Level Top-Down Pipeline for {year}...")
        forecast_df = generate_gender_pipeline_forecast(year, raw_demand)
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
    print("YEARLY PERFORMANCE (LR Gender + Top-Down SKU)")
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
    print("TOP 5 WORST PRODUCTS (By MAE, Gender Model 2020-2025)")
    print("="*50)
    print(product_metrics.sort_values(by='MAE', ascending=False).head(5).to_string(index=False))
    
    print("\n" + "="*50)
    print("CITY PERFORMANCE (Sorted by MAE, Gender Model 2020-2025)")
    print("="*50)
    print(city_metrics.sort_values(by='MAE', ascending=False).to_string(index=False))
    
    output_file = 'Forecast/top_down_gender_validation_all_years.xlsx'
    with pd.ExcelWriter(output_file) as writer:
        validation_df.to_excel(writer, sheet_name='SKU_Validation', index=False)
        yearly_metrics.to_excel(writer, sheet_name='Yearly_Metrics', index=False)
        product_metrics.to_excel(writer, sheet_name='Metrics_per_Product', index=False)
        city_metrics.to_excel(writer, sheet_name='Metrics_per_City', index=False)
        
    print(f"\nDetailed Gender Pipeline reports saved to: {output_file}")

if __name__ == "__main__":
    main()

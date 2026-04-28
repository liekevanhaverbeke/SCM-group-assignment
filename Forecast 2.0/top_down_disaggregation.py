import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

def load_proportions(target_year):
    """Loads and averages all necessary proportions for all seasons prior to the target_year"""
    
    # ------------------
    # 1. GENDER
    # ------------------
    gender_prop = pd.read_excel('Bert Map/gender_proportions_by_channel.xlsx')
    gender_prop = gender_prop[gender_prop['season'] < target_year].copy()
    gender_prop_melted = gender_prop.melt(id_vars=['channel_id', 'season'], 
                                          value_vars=['Menswear', 'Womenswear'],
                                          var_name='category', value_name='gender_pct')
    
    gender_avg = gender_prop_melted.groupby(['channel_id', 'category'])['gender_pct'].sum().reset_index()
    gender_avg['gender_pct'] = (gender_avg['gender_pct'] / gender_avg.groupby('channel_id')['gender_pct'].transform('sum') * 100).fillna(0)
    
    # ------------------
    # 2. PRICE
    # ------------------
    price_xl = pd.ExcelFile('Bert Map/price_proportions_by_gender_channel.xlsx')
    price_dfs = []
    for sheet in price_xl.sheet_names:
        df = price_xl.parse(sheet)
        df['category'] = sheet
        price_dfs.append(df)
    price_prop = pd.concat(price_dfs, ignore_index=True)
    price_prop = price_prop[price_prop['season'] < target_year].copy()
    
    bracket_cols = [c for c in price_prop.columns if c not in ['channel_id', 'season', 'category']]
    price_prop_melted = price_prop.melt(id_vars=['channel_id', 'season', 'category'],
                                        value_vars=bracket_cols,
                                        var_name='price_bracket', value_name='price_pct')
                                        
    price_avg = price_prop_melted.groupby(['channel_id', 'category', 'price_bracket'])['price_pct'].sum().reset_index()
    price_avg['price_pct'] = (price_avg['price_pct'] / price_avg.groupby(['channel_id', 'category'])['price_pct'].transform('sum') * 100).fillna(0)
    
    # ------------------
    # 3. PRODUCT
    # ------------------
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
    
    # ------------------
    # 4. SIZE
    # ------------------
    size_prop = pd.read_excel('Bert Map/size_proportions_by_channel_gender.xlsx')
    size_prop = size_prop[size_prop['season'] < target_year].copy()
    size_cols = [c for c in size_prop.columns if c not in ['channel_id', 'season', 'category']]
    size_prop_melted = size_prop.melt(id_vars=['channel_id', 'season', 'category'],
                                      value_vars=size_cols,
                                      var_name='size', value_name='size_pct')
                                      
    size_avg = size_prop_melted.groupby(['channel_id', 'category', 'size'])['size_pct'].sum().reset_index()
    size_avg['size_pct'] = (size_avg['size_pct'] / size_avg.groupby(['channel_id', 'category'])['size_pct'].transform('sum') * 100).fillna(0)
                                      
    return gender_avg, price_avg, prod_avg, size_avg

def main():
    target_year = 2026
    print(f"Loading {target_year} City Forecast...")
    city_forecast = pd.read_excel('Forecast/linear_regression_city.xlsx')
    city_forecast = city_forecast[['stad', f'forecast_{target_year}']].rename(columns={'stad': 'channel_id'})
    
    print(f"Loading and averaging historical proportions (2018 - {target_year-1})...")
    gender_avg, price_avg, prod_avg, size_avg = load_proportions(target_year)
    
    # Apply Gender
    df = pd.merge(city_forecast, gender_avg, on='channel_id', how='left')
    df['forecast_gender'] = df[f'forecast_{target_year}'] * (df['gender_pct'] / 100.0)
    
    # Apply Price
    df = pd.merge(df, price_avg, on=['channel_id', 'category'], how='left')
    df['forecast_price'] = df['forecast_gender'] * (df['price_pct'] / 100.0)
    
    # Apply Product
    df = pd.merge(df, prod_avg, on=['channel_id', 'category', 'price_bracket'], how='left')
    df['forecast_product'] = df['forecast_price'] * (df['share_within_bracket'] / 100.0)
    df = df.dropna(subset=['product_id'])
    
    # Apply Size
    df = pd.merge(df, size_avg, on=['channel_id', 'category'], how='left')
    df['forecast_sku'] = df['forecast_product'] * (df['size_pct'] / 100.0)
    
    # Clean up output
    df = df[df['forecast_sku'] > 0].copy()
    final_output = df[['channel_id', 'category', 'price_bracket', 'product_id', 'size', 'forecast_sku']].copy()
    final_output = final_output.rename(columns={'forecast_sku': f'forecast_{target_year}_units'})
    final_output[f'forecast_{target_year}_units'] = final_output[f'forecast_{target_year}_units'].round(2)
    final_output = final_output.sort_values(by=['channel_id', 'category', 'product_id', 'size'])
    
    output_path = f'Forecast/disaggregated_sku_forecast_{target_year}_averaged.xlsx'
    final_output.to_excel(output_path, index=False)
    
    print(f"\nTop-down disaggregation complete! Saved to {output_path}")
    print("\nSummary of Forecast Units:")
    print(f"Original City Total ({target_year}): {city_forecast[f'forecast_{target_year}'].sum():.2f}")
    print(f"Disaggregated SKU Total ({target_year}): {final_output[f'forecast_{target_year}_units'].sum():.2f}")

if __name__ == "__main__":
    main()

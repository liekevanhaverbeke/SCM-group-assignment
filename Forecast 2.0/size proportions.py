import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
import os
import warnings

warnings.filterwarnings("ignore")

# ── 0. Configuratie ──────────────────────────────────────────────────────────
INPUT_PATH = "True demand/True demand Simeon/True_Demand_Results.xlsx"
SHEET_NAME = "1_True_Demand_Lijst"
INPUT_PATH_Detaillist = "Input_Files/PPP_stu_products.xlsx"
SHEET_NAME_Detaillist = "Sheet1"

REGION_MAP = {
    'Platform': 'Online', 'Webshop': 'Online',
    'Helsinki': 'Scandinavian', 'Copenhagen': 'Scandinavian', 'Stockholm': 'Scandinavian',
    'Amsterdam': 'European', 'Berlin': 'European', 'Brussels': 'European',
    'Madrid': 'European', 'Paris': 'European', 'Rome': 'European'
}

def append_details(df_demand, df_details):
    """
    Appends product details to the demand dataframe.
    """
    return pd.merge(df_demand, df_details, left_on="product_id", right_on="id", how="left")

def calculate_gender(df):
    """
    Calculates the percentage of sales for each category (Womenswear/Menswear) 
    per channel and per season.
    """
    gender_stats = df.groupby(['channel_id', 'season', 'category'])['units_sold'].sum().reset_index()
    gender_stats['total_units'] = gender_stats.groupby(['channel_id', 'season'])['units_sold'].transform('sum')
    gender_stats['percentage'] = (gender_stats['units_sold'] / gender_stats['total_units']) * 100
    
    report = gender_stats.pivot(index=['channel_id', 'season'], columns='category', values='percentage').reset_index()
    report = report.fillna(0)
    
    os.makedirs("Bert Map", exist_ok=True)
    report.to_excel("Bert Map/gender_proportions_by_channel.xlsx", index=False)
    print("Gender proportions report saved.")
    return report

def calculate_price_proportions(df):
    """
    Groups products into specific price brackets and calculates the percentage 
    of sales per channel and season for each bracket, split by gender.
    """
    def get_bracket(price):
        if 140 <= price <= 170: return "140-170"
        elif 110 <= price <= 125: return "110-125"
        elif 70 <= price <= 95: return "70-95"
        elif 39 <= price <= 45: return "39-45"
        elif 0 <= price <= 10: return "0-10"
        else: return "Other"

    df['price_bracket'] = df['price'].apply(get_bracket)
    
    os.makedirs("Bert Map", exist_ok=True)
    output_file = "Bert Map/price_proportions_by_gender_channel.xlsx"
    
    with pd.ExcelWriter(output_file) as writer:
        for category in ['Menswear', 'Womenswear']:
            df_cat = df[df['category'] == category]
            if df_cat.empty: continue

            price_stats = df_cat.groupby(['channel_id', 'season', 'price_bracket'])['units_sold'].sum().reset_index()
            price_stats['total_units'] = price_stats.groupby(['channel_id', 'season'])['units_sold'].transform('sum')
            price_stats['percentage'] = (price_stats['units_sold'] / price_stats['total_units']) * 100
            
            report = price_stats.pivot(index=['channel_id', 'season'], columns='price_bracket', values='percentage').reset_index().fillna(0)
            
            required_brackets = ["140-170", "110-125", "70-95", "39-45", "0-10"]
            for bracket in required_brackets:
                if bracket not in report.columns: report[bracket] = 0.0
            
            cols = ['channel_id', 'season'] + required_brackets
            if 'Other' in report.columns: cols.append('Other')
            
            report[cols].to_excel(writer, sheet_name=category, index=False)
            
    print(f"Price proportions by gender saved to: {output_file}")

def calculate_product_shares_within_brackets(df):
    """
    Calculates the share of each product within its specific price bracket.
    For example: of the 25% of sales in bracket 70-95, what % is Product X?
    """
    def get_bracket(price):
        if 140 <= price <= 170: return "140-170"
        elif 110 <= price <= 125: return "110-125"
        elif 70 <= price <= 95: return "70-95"
        elif 39 <= price <= 45: return "39-45"
        elif 0 <= price <= 10: return "0-10"
        else: return "Other"

    df['price_bracket'] = df['price'].apply(get_bracket)
    
    os.makedirs("Bert Map", exist_ok=True)
    output_file = "Bert Map/product_shares_within_price_brackets.xlsx"
    
    with pd.ExcelWriter(output_file) as writer:
        for category in ['Menswear', 'Womenswear']:
            df_cat = df[df['category'] == category]
            if df_cat.empty: continue

            # Group by channel, season, bracket AND product
            prod_stats = df_cat.groupby(['channel_id', 'season', 'price_bracket', 'product_id'])['units_sold'].sum().reset_index()
            
            # Calculate total units sold WITHIN that specific bracket
            prod_stats['total_bracket_units'] = prod_stats.groupby(['channel_id', 'season', 'price_bracket'])['units_sold'].transform('sum')
            
            # Calculate product's share of that bracket
            prod_stats['share_within_bracket'] = (prod_stats['units_sold'] / prod_stats['total_bracket_units']) * 100
            
            # Final output for this gender
            report = prod_stats[['channel_id', 'season', 'price_bracket', 'product_id', 'share_within_bracket']]
            report.to_excel(writer, sheet_name=category, index=False)
            
    print(f"Detailed product shares saved to: {output_file}")

def calculate_size_proportions(df):
    """
    Calculates the percentage of sales for each size per channel, season, and category.
    """
    df_filtered = df[df['size'].str.lower() != 'onesize']
    size_stats = df_filtered.groupby(['channel_id', 'season', 'category', 'size'])['true_demand'].sum().reset_index()
    size_stats['total_units'] = size_stats.groupby(['channel_id', 'season', 'category'])['true_demand'].transform('sum')
    size_stats['percentage'] = (size_stats['true_demand'] / size_stats['total_units']) * 100
    
    report = size_stats.pivot(index=['channel_id', 'season', 'category'], columns='size', values='percentage').reset_index().fillna(0)
    
    os.makedirs("Bert Map", exist_ok=True)
    report.to_excel("Bert Map/size_proportions_by_channel_gender.xlsx", index=False)
    print("Size proportions report saved.")

def calculate_product_size_proportions(df):
    """
    Calculates the size proportions for each product across all cities.
    This aggregates demand over all cities to find the global size distribution for each product.
    """
    # Filter out onesize
    df_filtered = df[df['size'].str.lower() != 'onesize']
    
    # Group by product, season and category to keep metadata
    # Sum true_demand across all cities (channels)
    product_stats = df_filtered.groupby(['product_id', 'season', 'category', 'size'])['true_demand'].sum().reset_index()
    
    # Calculate total demand per product-season
    product_stats['total_product_demand'] = product_stats.groupby(['product_id', 'season'])['true_demand'].transform('sum')
    
    # Calculate proportion
    product_stats['size_proportion'] = (product_stats['true_demand'] / product_stats['total_product_demand']) * 100
    
    # Pivot for output
    report = product_stats.pivot(
        index=['product_id', 'season', 'category'], 
        columns='size', 
        values='size_proportion'
    ).reset_index().fillna(0)
    
    os.makedirs("Bert Map", exist_ok=True)
    report.to_excel("Bert Map/size_proportions_by_product.xlsx", index=False)
    print("Product size proportions (global aggregate) saved to: Bert Map/size_proportions_by_product.xlsx")

def calculate_product_channel_size_proportions(df):
    """
    Calculates the size proportions for each product per channel, aggregated over all years (seasons).
    """
    # Filter out onesize
    df_filtered = df[df['size'].str.lower() != 'onesize']
    
    # Group by product, channel, category and size (ignoring season)
    # Sum true_demand over all years
    product_stats = df_filtered.groupby(['product_id', 'channel_id', 'category', 'size'])['true_demand'].sum().reset_index()
    
    # Calculate total demand per product-channel
    product_stats['total_demand'] = product_stats.groupby(['product_id', 'channel_id'])['true_demand'].transform('sum')
    
    # Calculate proportion
    product_stats['size_proportion'] = (product_stats['true_demand'] / product_stats['total_demand']) * 100
    
    # Pivot for output
    report = product_stats.pivot(
        index=['product_id', 'channel_id', 'category'], 
        columns='size', 
        values='size_proportion'
    ).reset_index().fillna(0)
    
    os.makedirs("Bert Map", exist_ok=True)
    report.to_excel("Bert Map/size_proportions_by_product_channel.xlsx", index=False)
    print("Product x Channel size proportions saved to: Bert Map/size_proportions_by_product_channel.xlsx")

def main():
    df_true_demand = pd.read_excel(INPUT_PATH, sheet_name=SHEET_NAME)
    df_detaillist = pd.read_excel(INPUT_PATH_Detaillist, sheet_name=SHEET_NAME_Detaillist)
    
    df = append_details(df_true_demand, df_detaillist)
    
    calculate_gender(df)
    calculate_price_proportions(df)
    calculate_product_shares_within_brackets(df) # New detailed share function
    calculate_size_proportions(df)
    calculate_product_size_proportions(df)
    calculate_product_channel_size_proportions(df)

if __name__ == "__main__":
    main()
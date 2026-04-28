import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
import os
import warnings
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

def lr_forecast(jaren, waarden, target_year):
    jaars = [j for j, w in zip(jaren, waarden) if not np.isnan(w)]
    waard = [w for w in waarden if not np.isnan(w)]
    if len(waard) < 2: return float(waard[-1]) if waard else 0.0
    model = LinearRegression().fit(np.array(jaars).reshape(-1, 1), np.array(waard))
    return max(0.0, float(model.predict([[target_year]])[0]))

def holts_forecast(values, jaar_labels):
    y = np.array(values, dtype=float)
    if len(y) < 3: return float(y[-1]) if len(y) > 0 else 0.0
    def sse(params):
        a, b = params
        if not (0 < a < 1 and 0 < b < 1): return 1e15
        L = y[0]; T = y[1] - y[0]; err = 0.0
        for i in range(1, len(y)):
            fitted = L + T
            L_new = a * y[i] + (1 - a) * fitted
            T = b * (L_new - L) + (1 - b) * T
            L = L_new
            err += (y[i] - fitted)**2
        return err
    try:
        res = minimize(sse, x0=[0.3, 0.1], method="Nelder-Mead")
        a, b = res.x
        L = y[0]; T = y[1] - y[0]
        for i in range(1, len(y)):
            L_new = a * y[i] + (1 - a) * (L + T)
            T = b * (L_new - L) + (1 - b) * T
            L = L_new
        return max(0.0, L + T)
    except: return float(y[-1])

# ── 2. Reconciliation Functions ───────────────────────────────────────────────

def mint_reconcile(y_hat, S, variances):
    variances = np.clip(variances, 1e-6, None)
    W_inv = sparse.diags(1.0 / variances)
    STS = (S.T @ W_inv @ S).toarray()
    P = np.linalg.inv(STS) @ (S.T @ W_inv)
    return (S @ (P @ y_hat)).ravel()

def build_s_matrix(nodes_df, bottom_nodes_df):
    m, n = len(bottom_nodes_df), len(nodes_df)
    city_map = {}; region_map = {}
    for idx, row in bottom_nodes_df.reset_index(drop=True).iterrows():
        city_map.setdefault(row["stad"], []).append(idx)
        region_map.setdefault(row["region"], []).append(idx)
    
    rows, cols, data = [], [], []
    for i, node in nodes_df.iterrows():
        lvl, s, r = node["level"], node["stad"], node["region"]
        if lvl == 0: idxs = range(m)
        elif lvl == 1: idxs = region_map.get(s, [])
        elif lvl == 2: idxs = city_map.get(s, [])
        else: continue
        for j in idxs:
            rows.append(i); cols.append(j); data.append(1.0)
    return sparse.csr_matrix((data, (rows, cols)), shape=(n, m))

# ── 3. Proportions Loading ───────────────────────────────────────────────────

def load_layered_proportions(target_year):
    # 1. GENDER
    gender_prop = pd.read_excel('Bert Map/gender_proportions_by_channel.xlsx')
    gender_prop = gender_prop[gender_prop['season'] < target_year].copy()
    gender_prop_melted = gender_prop.melt(id_vars=['channel_id', 'season'], 
                                          value_vars=['Menswear', 'Womenswear'],
                                          var_name='category', value_name='gender_pct')
    gender_avg = gender_prop_melted.groupby(['channel_id', 'category'])['gender_pct'].sum().reset_index()
    gender_avg['gender_pct'] = (gender_avg['gender_pct'] / gender_avg.groupby('channel_id')['gender_pct'].transform('sum') * 100).fillna(0)
    
    # 2. PRICE
    price_xl = pd.ExcelFile('Bert Map/price_proportions_by_gender_channel.xlsx')
    price_dfs = []
    for sheet in price_xl.sheet_names:
        df = price_xl.parse(sheet); df['category'] = sheet; price_dfs.append(df)
    price_prop = pd.concat(price_dfs, ignore_index=True)
    price_prop = price_prop[price_prop['season'] < target_year].copy()
    bracket_cols = [c for c in price_prop.columns if c not in ['channel_id', 'season', 'category']]
    price_avg = price_prop.melt(id_vars=['channel_id', 'season', 'category'], value_vars=bracket_cols, var_name='price_bracket', value_name='price_pct')
    price_avg = price_avg.groupby(['channel_id', 'category', 'price_bracket'])['price_pct'].sum().reset_index()
    price_avg['price_pct'] = (price_avg['price_pct'] / price_avg.groupby(['channel_id', 'category'])['price_pct'].transform('sum') * 100).fillna(0)
    
    # 3. PRODUCT
    prod_xl = pd.ExcelFile('Bert Map/product_shares_within_price_brackets.xlsx')
    prod_dfs = []
    for sheet in prod_xl.sheet_names:
        df = prod_xl.parse(sheet); df['category'] = sheet; prod_dfs.append(df)
    prod_share = pd.concat(prod_dfs, ignore_index=True)
    prod_share = prod_share[prod_share['season'] < target_year].copy()
    prod_avg = prod_share.groupby(['channel_id', 'category', 'price_bracket', 'product_id'])['share_within_bracket'].sum().reset_index()
    prod_avg['share_within_bracket'] = (prod_avg['share_within_bracket'] / prod_avg.groupby(['channel_id', 'category', 'price_bracket'])['share_within_bracket'].transform('sum') * 100).fillna(0)
    
    # 4. SIZE
    size_prop = pd.read_excel('Bert Map/size_proportions_by_channel_gender.xlsx')
    size_prop = size_prop[size_prop['season'] < target_year].copy()
    size_cols = [c for c in size_prop.columns if c not in ['channel_id', 'season', 'category']]
    size_avg = size_prop.melt(id_vars=['channel_id', 'season', 'category'], value_vars=size_cols, var_name='size', value_name='size_pct')
    size_avg = size_avg.groupby(['channel_id', 'category', 'size'])['size_pct'].sum().reset_index()
    size_avg['size_pct'] = (size_avg['size_pct'] / size_avg.groupby(['channel_id', 'category'])['size_pct'].transform('sum') * 100).fillna(0)
                                      
    return gender_avg, price_avg, prod_avg, size_avg

# ── 4. Pipeline Logic ────────────────────────────────────────────────────────

def run_hybrid_layered_pipeline(df, target_year):
    history = df[df['season'] < target_year].copy()
    if history.empty: return pd.DataFrame()
    
    # 1. Base Forecasts (Total/Region/City)
    level_configs = [("TOTAL", [], "HOLTS", 0), ("REGION", ["region"], "LR", 1), ("CITY", ["channel_id"], "LR", 2)]
    recs = {"stad":[], "region":[], "level":[], "forecast":[], "mse":[]}
    
    for name, gcols, method, lvl in level_configs:
        agg = history.groupby(gcols + ["season"])["true_demand"].sum().reset_index().rename(columns={"season":"jaar"})
        if "channel_id" in agg.columns: agg["stad"] = agg["channel_id"]; agg["region"] = agg["channel_id"].map(REGION_MAP)
        elif "region" in agg.columns: agg["stad"] = agg["region"]; agg["region"] = agg["region"]
        else: agg["stad"] = "TOTAL_COMPANY"; agg["region"] = "TOTAL_COMPANY"
        
        combos = agg[["stad", "region"]].drop_duplicates()
        for _, row in combos.iterrows():
            sub = agg[agg["stad"]==row["stad"]].sort_values("jaar")
            vals, jaren = sub["true_demand"].tolist(), sub["jaar"].tolist()
            if method=="HOLTS": fc = holts_forecast(vals, jaren)
            elif method=="LR": fc = lr_forecast(jaren, vals, target_year)
            else: fc = ses_forecast(vals)
            
            resids = []
            unique_years = sorted(list(set(jaren)))
            if len(unique_years) > 1:
                for y in (unique_years[-3:] if len(unique_years) >= 3 else unique_years[1:]):
                    tr = sub[sub["jaar"] < y]; ac = sub[sub["jaar"] == y]
                    if not tr.empty and not ac.empty:
                        if method=="HOLTS": p = holts_forecast(tr["true_demand"].tolist(), tr["jaar"].tolist())
                        elif method=="LR": p = lr_forecast(tr["jaar"].tolist(), tr["true_demand"].tolist(), y)
                        else: p = ses_forecast(tr["true_demand"].tolist())
                        resids.append((ac["true_demand"].values[0] - p)**2)
            recs["stad"].append(row["stad"]); recs["region"].append(row["region"]); recs["level"].append(lvl)
            recs["forecast"].append(fc); recs["mse"].append(np.mean(resids) if resids else 1.0)
            
    nodes_df = pd.DataFrame(recs)
    bottom_nodes = nodes_df[nodes_df["level"] == 2].copy()
    
    # 2. MinT Reconciliation
    S = build_s_matrix(nodes_df, bottom_nodes)
    nodes_df["reconciled"] = mint_reconcile(nodes_df["forecast"].to_numpy(), S, nodes_df["mse"].to_numpy())
    
    # 3. Layered Disaggregation
    city_rec = nodes_df[nodes_df["level"] == 2][['stad', 'reconciled']].rename(columns={'stad':'channel_id'})
    gender_avg, price_avg, prod_avg, size_avg = load_layered_proportions(target_year)
    
    # Apply Gender
    res = pd.merge(city_rec, gender_avg, on='channel_id', how='left')
    res['forecast_gender'] = res['reconciled'] * (res['gender_pct'] / 100.0)
    # Apply Price
    res = pd.merge(res, price_avg, on=['channel_id', 'category'], how='left')
    res['forecast_price'] = res['forecast_gender'] * (res['price_pct'] / 100.0)
    # Apply Product
    res = pd.merge(res, prod_avg, on=['channel_id', 'category', 'price_bracket'], how='left')
    res['forecast_product'] = res['forecast_price'] * (res['share_within_bracket'] / 100.0)
    res = res.dropna(subset=['product_id'])
    # Apply Size
    res = pd.merge(res, size_avg, on=['channel_id', 'category'], how='left')
    res['forecast_sku'] = res['forecast_product'] * (res['size_pct'] / 100.0)
    
    res['season'] = target_year
    return res[['channel_id', 'season', 'product_id', 'size', 'forecast_sku']]

def main():
    print("Loading raw True Demand data...")
    raw_demand = pd.read_excel(INPUT_PATH, sheet_name=SHEET_NAME)
    raw_demand['region'] = raw_demand['channel_id'].map(REGION_MAP).fillna('Other')
    sku_actuals = raw_demand.groupby(['channel_id', 'season', 'product_id', 'size'])['true_demand'].sum().reset_index().rename(columns={'true_demand':'actual_demand'})
    
    years = [2020, 2021, 2022, 2023, 2024, 2025]
    all_val = []
    for y in years:
        print(f"Running Hybrid Layered Pipeline for {y}...")
        all_val.append(run_hybrid_layered_pipeline(raw_demand, y))
    
    val_df = pd.concat(all_val, ignore_index=True)
    val_df = pd.merge(val_df, sku_actuals, on=['channel_id', 'season', 'product_id', 'size'], how='outer')
    val_df = val_df[val_df['season'].isin(years)].fillna(0)
    val_df['abs_error'] = abs(val_df['forecast_sku'] - val_df['actual_demand'])
    val_df['sq_error'] = (val_df['forecast_sku'] - val_df['actual_demand'])**2
    
    print("\n" + "="*50 + "\nOVERALL HYBRID LAYERED PERFORMANCE\n" + "="*50)
    yearly = val_df.groupby('season').agg(MAE=('abs_error', 'mean'), MSE=('sq_error', 'mean')).reset_index()
    print(yearly.to_string(index=False))
    print(f"\nCOMBINED MAE: {val_df['abs_error'].mean():.2f}")
    
    # Aggregations per Product
    product_metrics = val_df.groupby('product_id').agg(
        MSE=('sq_error', 'mean'),
        MAE=('abs_error', 'mean')
    ).reset_index()
    
    # Aggregations per City
    city_metrics = val_df.groupby('channel_id').agg(
        MSE=('sq_error', 'mean'),
        MAE=('abs_error', 'mean')
    ).reset_index()
    
    print("\n" + "="*50)
    print("TOP 5 WORST PRODUCTS (By MAE, Hybrid Layered 2020-2025)")
    print("="*50)
    print(product_metrics.sort_values(by='MAE', ascending=False).head(5).to_string(index=False))
    
    print("\n" + "="*50)
    print("CITY PERFORMANCE (Sorted by MAE, Hybrid Layered 2020-2025)")
    print("="*50)
    print(city_metrics.sort_values(by='MAE', ascending=False).to_string(index=False))
    
    print("\nGenerating Forecast for 2026...")
    fc_2026 = run_hybrid_layered_pipeline(raw_demand, 2026)
    
    with pd.ExcelWriter('Forecast/hybrid_layered_validation_all_years.xlsx') as writer:
        val_df.to_excel(writer, sheet_name='SKU_Validation', index=False)
        yearly.to_excel(writer, sheet_name='Yearly_Metrics', index=False)
        product_metrics.to_excel(writer, sheet_name='Metrics_per_Product', index=False)
        city_metrics.to_excel(writer, sheet_name='Metrics_per_City', index=False)

    fc_2026.to_excel('Forecast/hybrid_layered_sku_forecast_2026.xlsx', index=False)
    print(f"DONE! SKU Forecast Total: {fc_2026['forecast_sku'].sum():.2f}")

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
import os
import warnings
warnings.filterwarnings("ignore")

REGION_MAP = {
    'Platform': 'Online', 'Webshop': 'Online',
    'Helsinki': 'Scandinavian', 'Copenhagen': 'Scandinavian', 'Stockholm': 'Scandinavian',
    'Amsterdam': 'European', 'Berlin': 'European', 'Brussels': 'European',
    'Madrid': 'European', 'Paris': 'European', 'Rome': 'European'
}

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

def mint_reconcile(y_hat, S, variances):
    variances = np.clip(variances, 1e-6, None)
    W_inv = sparse.diags(1.0 / variances)
    STS = (S.T @ W_inv @ S).toarray()
    P = np.linalg.inv(STS) @ (S.T @ W_inv)
    return (S @ (P @ y_hat)).ravel()

def build_s_matrix(nodes_df, bottom_nodes_df):
    m, n = len(bottom_nodes_df), len(nodes_df)
    city_map = {}; region_map = {}; cp_map = {}
    for idx, row in bottom_nodes_df.reset_index(drop=True).iterrows():
        city_map.setdefault(row["stad"], []).append(idx)
        region_map.setdefault(row["region"], []).append(idx)
        cp_map.setdefault((row["stad"], row["product"]), []).append(idx)
    
    rows, cols, data = [], [], []
    for i, node in nodes_df.iterrows():
        lvl, s, p, r = node["level"], node["stad"], node["product"], node["region"]
        if lvl == 0: idxs = range(m)
        elif lvl == 1: idxs = region_map.get(s, [])
        elif lvl == 2: idxs = city_map.get(s, [])
        elif lvl == 3: idxs = cp_map.get((s, p), [])
        else: continue
        for j in idxs:
            rows.append(i); cols.append(j); data.append(1.0)
    return sparse.csr_matrix((data, (rows, cols)), shape=(n, m))

def run_hybrid_city_product_yearly(df, target_year):
    history = df[df['season'] < target_year].copy()
    if history.empty: return pd.DataFrame()
    
    level_configs = [
        ("TOTAL", [], "HOLTS", 0),
        ("REGION", ["region"], "LR", 1),
        ("CITY", ["channel_id"], "LR", 2),
        ("CITY_PRODUCT", ["channel_id", "product_id"], "SES", 3)
    ]
    
    recs = {"stad":[], "product":[], "region":[], "level":[], "forecast":[], "mse":[]}
    
    for name, gcols, method, lvl in level_configs:
        agg = history.groupby(gcols + ["season"])["true_demand"].sum().reset_index().rename(columns={"season":"jaar"})
        
        if "channel_id" in agg.columns:
            agg["stad"] = agg["channel_id"]
            agg["region"] = agg["channel_id"].map(REGION_MAP)
        elif "region" in agg.columns:
            agg["stad"] = agg["region"]
            agg["region"] = agg["region"]
        else:
            agg["stad"] = "TOTAL_COMPANY"
            agg["region"] = "TOTAL_COMPANY"
            
        if "product_id" in agg.columns: agg["product"] = agg["product_id"]
        else: agg["product"] = "ALL"
        
        combos = agg[["stad", "product", "region"]].drop_duplicates()
        for _, row in combos.iterrows():
            sub = agg[(agg["stad"]==row["stad"]) & (agg["product"]==row["product"])].sort_values("jaar")
            vals, jaren = sub["true_demand"].tolist(), sub["jaar"].tolist()
            
            if method=="HOLTS": fc = holts_forecast(vals, jaren)
            elif method=="LR": fc = lr_forecast(jaren, vals, target_year)
            else: fc = ses_forecast(vals)
            
            # Historical residuals for variances (calculate in-sample 1-step ahead)
            resids = []
            unique_years = sorted(list(set(jaren)))
            if len(unique_years) > 1:
                # Test on the last 3 available years in history
                test_years = unique_years[-3:] if len(unique_years) >= 3 else unique_years[1:]
                for y in test_years:
                    tr = sub[sub["jaar"] < y]; ac = sub[sub["jaar"] == y]
                    if not tr.empty and not ac.empty:
                        if method=="HOLTS": p = holts_forecast(tr["true_demand"].tolist(), tr["jaar"].tolist())
                        elif method=="LR": p = lr_forecast(tr["jaar"].tolist(), tr["true_demand"].tolist(), y)
                        else: p = ses_forecast(tr["true_demand"].tolist())
                        resids.append((ac["true_demand"].values[0] - p)**2)
                
            recs["stad"].append(row["stad"])
            recs["product"].append(row["product"])
            recs["region"].append(row["region"])
            recs["level"].append(lvl)
            recs["forecast"].append(fc)
            recs["mse"].append(np.mean(resids) if resids else 1.0)
            
    nodes_df = pd.DataFrame(recs)
    bottom_nodes = nodes_df[nodes_df["level"] == 3].copy()
    
    if bottom_nodes.empty: return pd.DataFrame()
    
    # 2. MinT Reconciliation
    S = build_s_matrix(nodes_df, bottom_nodes)
    y_hat = nodes_df["forecast"].to_numpy()
    variances = nodes_df["mse"].to_numpy()
    
    nodes_df["reconciled"] = mint_reconcile(y_hat, S, variances)
    
    # 3. Disaggregation (City/Product -> Size)
    size_agg = history.groupby(['channel_id', 'product_id', 'size'])['true_demand'].sum().reset_index()
    cp_total_agg = history.groupby(['channel_id', 'product_id'])['true_demand'].sum().reset_index().rename(columns={'true_demand':'total_cp'})
    ratios = size_agg.merge(cp_total_agg, on=['channel_id', 'product_id'])
    
    # Avoid div by zero
    ratios = ratios[ratios['total_cp'] > 0].copy()
    ratios['share'] = ratios['true_demand'] / ratios['total_cp']
    ratios = ratios.drop(columns=['true_demand', 'total_cp'])
    
    cp_rec = nodes_df[nodes_df["level"] == 3][['stad', 'product', 'reconciled']].rename(columns={'stad':'channel_id', 'product':'product_id'})
    
    final = ratios.merge(cp_rec, on=['channel_id', 'product_id'])
    final['forecast_sku'] = final['reconciled'] * final['share']
    final['season'] = target_year
    
    return final[['channel_id', 'season', 'product_id', 'size', 'forecast_sku']]

def main():
    print("Loading raw True Demand data...")
    raw_demand = pd.read_excel('True demand/True demand Simeon/True_Demand_Results.xlsx', sheet_name='1_True_Demand_Lijst')
    raw_demand['region'] = raw_demand['channel_id'].map(REGION_MAP).fillna('Other')
    
    # Actuals at SKU level
    sku_actuals = raw_demand.groupby(['channel_id', 'season', 'product_id', 'size'])['true_demand'].sum().reset_index()
    sku_actuals = sku_actuals.rename(columns={'true_demand': 'actual_demand'})
    
    years_to_validate = [2020, 2021, 2022, 2023, 2024, 2025]
    all_forecasts = []
    
    for year in years_to_validate:
        print(f"Running Hybrid City/Product Reconciliation Logic for {year}...")
        forecast_df = run_hybrid_city_product_yearly(raw_demand, year)
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
    print("YEARLY HYBRID CITY/PRODUCT PERFORMANCE (MinT Coherent + CP-to-Size)")
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
    print("TOP 5 WORST PRODUCTS (By MAE, Hybrid CP MinT 2020-2025)")
    print("="*50)
    print(product_metrics.sort_values(by='MAE', ascending=False).head(5).to_string(index=False))
    
    print("\n" + "="*50)
    print("CITY PERFORMANCE (Sorted by MAE, Hybrid CP MinT 2020-2025)")
    print("="*50)
    print(city_metrics.sort_values(by='MAE', ascending=False).to_string(index=False))
    
    output_file = 'Forecast/hybrid_city_product_validation_all_years.xlsx'
    with pd.ExcelWriter(output_file) as writer:
        validation_df.to_excel(writer, sheet_name='SKU_Validation', index=False)
        yearly_metrics.to_excel(writer, sheet_name='Yearly_Metrics', index=False)
        product_metrics.to_excel(writer, sheet_name='Metrics_per_Product', index=False)
        city_metrics.to_excel(writer, sheet_name='Metrics_per_City', index=False)
        
    print(f"\nDetailed hybrid CP validation reports saved to: {output_file}")

if __name__ == "__main__":
    main()

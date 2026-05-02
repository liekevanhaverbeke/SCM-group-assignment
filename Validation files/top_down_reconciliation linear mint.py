"""
TOP-DOWN PROPORTIONAL RECONCILIATION WITH MINT
- Uses MinT to reconcile forecasts across Total, Region, and City levels.
- All base forecasts use Linear Regression.
- Disaggregates from reconciled city forecasts to Product/Size using direct historical shares.
- Includes automated validation (2020-2025) and final 2026 forecast.
"""

import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.linear_model import LinearRegression
import os
import warnings
from scipy.optimize import minimize

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

# ── 1. Forecast & Reconciliation Functions ────────────────────────────────────

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

# ── 2. Pipeline Logic ────────────────────────────────────────────────────────

def run_top_down_mint_pipeline(df, target_year):
    history = df[df['season'] < target_year].copy()
    if history.empty: return pd.DataFrame()
    
    # A. Calculate Historical Shares (Direct Product x Size within City)
    total_city_sales = history.groupby('channel_id')['true_demand'].sum()
    shares = (
        history.groupby(['channel_id', 'product_id', 'size'])['true_demand']
        .sum()
        .reset_index()
    )
    shares = shares.merge(total_city_sales.rename('city_total'), on='channel_id')
    shares['share'] = shares['true_demand'] / shares['city_total']
    shares = shares.drop(columns=['true_demand', 'city_total'])
    
    # B. Base Forecasts (Total/Region/City)
    # We use HOLTS for Total and LR for Region/Cities to create a difference for MinT to reconcile
    level_configs = [("TOTAL", [], 0, "HOLTS"), ("REGION", ["region"], 1, "LR"), ("CITY", ["channel_id"], 2, "LR")]
    recs = {"stad":[], "region":[], "level":[], "forecast":[], "mse":[]}
    
    for name, gcols, lvl, method in level_configs:
        agg = history.groupby(gcols + ["season"])["true_demand"].sum().reset_index().rename(columns={"season":"jaar"})
        if "channel_id" in agg.columns: agg["stad"] = agg["channel_id"]; agg["region"] = agg["channel_id"].map(REGION_MAP)
        elif "region" in agg.columns: agg["stad"] = agg["region"]; agg["region"] = agg["region"]
        else: agg["stad"] = "TOTAL_COMPANY"; agg["region"] = "TOTAL_COMPANY"
        
        combos = agg[["stad", "region"]].drop_duplicates()
        for _, row in combos.iterrows():
            sub = agg[agg["stad"]==row["stad"]].sort_values("jaar")
            vals, jaren = sub["true_demand"].tolist(), sub["jaar"].tolist()
            
            # Forecast based on chosen method
            if method == "HOLTS":
                fc = holts_forecast(vals, jaren)
            else:
                fc = lr_forecast(jaren, vals, target_year)
            
            # Calculate MSE for weights
            resids = []
            unique_years = sorted(list(set(jaren)))
            if len(unique_years) > 1:
                # Use sliding window validation for MSE
                for y_val in (unique_years[-3:] if len(unique_years) >= 3 else unique_years[1:]):
                    tr = sub[sub["jaar"] < y_val]; ac = sub[sub["jaar"] == y_val]
                    if not tr.empty and not ac.empty:
                        if method == "HOLTS":
                            p = holts_forecast(tr["true_demand"].tolist(), tr["jaar"].tolist())
                        else:
                            p = lr_forecast(tr["jaar"].tolist(), tr["true_demand"].tolist(), y_val)
                        resids.append((ac["true_demand"].values[0] - p)**2)
            
            recs["stad"].append(row["stad"]); recs["region"].append(row["region"]); recs["level"].append(lvl)
            recs["forecast"].append(fc); recs["mse"].append(np.mean(resids) if resids else 1.0)
            
    nodes_df = pd.DataFrame(recs)
    bottom_nodes = nodes_df[nodes_df["level"] == 2].copy()
    
    # C. MinT Reconciliation
    S = build_s_matrix(nodes_df, bottom_nodes)
    nodes_df["reconciled"] = mint_reconcile(nodes_df["forecast"].to_numpy(), S, nodes_df["mse"].to_numpy())
    
    # D. Disaggregate Reconciled City Forecasts
    city_rec = nodes_df[nodes_df["level"] == 2][['stad', 'reconciled']].rename(columns={'stad':'channel_id'})
    res = shares.merge(city_rec, on='channel_id')
    res['forecast_sku'] = res['share'] * res['reconciled']
    res['season'] = target_year
    
    return res[['channel_id', 'season', 'product_id', 'size', 'forecast_sku']]

def main():
    print("Loading raw True Demand data...")
    raw_demand = pd.read_excel(INPUT_PATH, sheet_name=SHEET_NAME)
    raw_demand['region'] = raw_demand['channel_id'].map(REGION_MAP).fillna('Other')
    
    sku_actuals = raw_demand.groupby(['channel_id', 'season', 'product_id', 'size'])['true_demand'].sum().reset_index().rename(columns={'true_demand':'actual_demand'})
    
    # ── 3. Validation Loop (2020-2025) ───────────────────────────────────────
    years = [2020, 2021, 2022, 2023, 2024, 2025]
    all_val = []
    for y in years:
        print(f"Running Top-Down Linear MinT Pipeline for {y}...")
        all_val.append(run_top_down_mint_pipeline(raw_demand, y))
    
    val_df = pd.concat(all_val, ignore_index=True)
    val_df = pd.merge(val_df, sku_actuals, on=['channel_id', 'season', 'product_id', 'size'], how='outer')
    val_df = val_df[val_df['season'].isin(years)].fillna(0)
    
    val_df['raw_error'] = val_df['actual_demand'] - val_df['forecast_sku'] 
    val_df['abs_error'] = abs(val_df['raw_error'])
    val_df['sq_error'] = val_df['raw_error']**2

    # MAPE_val: Mean Absolute Percentage Error
    val_df['MAPE_val'] = np.where(val_df['actual_demand'] != 0, 
                                  val_df['abs_error'] / val_df['actual_demand'], 
                                  0)
    
    # MPE_val: Mean Percentage Error to identify systematic bias
    val_df['MPE_val'] = np.where(val_df['actual_demand'] != 0, 
                                 val_df['raw_error'] / val_df['actual_demand'], 
                                 0)
    
    # ── 4. Print Performance ──────────────────────────────────────────────────
    print("\n" + "="*50 + "\nOVERALL TOP-DOWN LINEAR MINT PERFORMANCE\n" + "="*50)
    yearly = val_df.groupby('season').agg(MAE=('abs_error', 'mean'), MSE=('sq_error', 'mean'), MAPE=('MAPE_val', 'mean'), MPE=('MPE_val', 'mean')).reset_index()
    print(yearly.to_string(index=False))
    print(f"\nCOMBINED MAE: {val_df['abs_error'].mean():.2f}")
    
    product_metrics = val_df.groupby('product_id').agg(MSE=('sq_error', 'mean'), MAE=('abs_error', 'mean'), MAPE=('MAPE_val', 'mean'), MPE=('MPE_val', 'mean')).reset_index()
    city_metrics = val_df.groupby('channel_id').agg(MSE=('sq_error', 'mean'), MAE=('abs_error', 'mean'), MAPE=('MAPE_val', 'mean'), MPE=('MPE_val', 'mean')).reset_index()
    
    print("\n" + "="*50)
    print("TOP 5 WORST PRODUCTS (By MAE, Top-Down Linear MinT 2020-2025)")
    print("="*50)
    print(product_metrics.sort_values(by='MAE', ascending=False).head(5).to_string(index=False))
    
    print("\n" + "="*50)
    print("CITY PERFORMANCE (Sorted by MAE, Top-Down Linear MinT 2020-2025)")
    print("="*50)
    print(city_metrics.sort_values(by='MAE', ascending=False).to_string(index=False))
    
    # ── 5. Generate Forecast for 2026 ────────────────────────────────────────
    #print("\nGenerating Forecast for 2026...")
    #fc_2026 = run_top_down_mint_pipeline(raw_demand, 2026)
    
    # ── 6. Save Results ───────────────────────────────────────────────────────
    os.makedirs('Validation files', exist_ok=True)
    
    with pd.ExcelWriter('Validation files/top_down_linear_mint_validation_all_years.xlsx') as writer:
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
    #output_2026.to_excel('Validation files/top_down_linear_mint_sku_forecast_2026.xlsx', sheet_name='Forecast_2026', index=False)
    
    #print(f"\nDONE! Final 2026 SKU Forecast Units: {output_2026['forecast_2026'].sum():.2f}")

if __name__ == "__main__":
    main()

"""
HYBRID MIDDLE-OUT RECONCILIATION (MinT + Top-Down Disaggregation)
================================================================
Steps:
1.  Independent Base Forecasts (Total: Holt, Region: LR, City: LR, City/Product: SES).
2.  MinT Reconciliation across these 4 levels for coherence.
3.  Historical Size Shares calculation.
4.  Proportional Disaggregation from City/Product reconciled results down to Sizes.
"""

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
FORECAST_FOLDER = "Forecast/Results"
RESIDUAL_YEARS = [2023, 2024, 2025]
FORECAST_YEAR = 2026

REGION_MAP = {
    'Platform': 'Online', 'Webshop': 'Online',
    'Helsinki': 'Scandinavian', 'Copenhagen': 'Scandinavian', 'Stockholm': 'Scandinavian',
    'Amsterdam': 'European', 'Berlin': 'European', 'Brussels': 'European',
    'Madrid': 'European', 'Paris': 'European', 'Rome': 'European'
}

# ── 1. Forecast Functies ─────────────────────────────────────────────────────

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

# ── 2. Reconciliatie Functies ────────────────────────────────────────────────

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

# ── 3. Main Logic ────────────────────────────────────────────────────────────

def main():
    print("="*80)
    print("HYBRID MIDDLE-OUT RECONCILIATION")
    print("="*80)

    df = pd.read_excel(INPUT_PATH, sheet_name=SHEET_NAME)
    df['region'] = df['channel_id'].map(REGION_MAP).fillna('Other')
    df['season'] = df['season'].astype(int)

    # 3.1 Base Forecasts
    level_configs = [
        ("TOTAL", [], "HOLTS", 0),
        ("REGION", ["region"], "LR", 1),
        ("CITY", ["channel_id"], "LR", 2)
    ]
    
    recs = {"stad":[], "product":[], "region":[], "level":[], "forecast_2026":[], 
            "predicted_2025":[], "actual_2025":[], "predicted_2024":[], "actual_2024":[], "mse":[]}
    
    for name, gcols, method, lvl in level_configs:
        print(f"Verwerken: {name}...")
        agg = df.groupby(gcols + ["season"])["true_demand"].sum().reset_index().rename(columns={"season":"jaar"})
        
        # Add level-specific columns
        if "channel_id" in agg.columns:
            agg["stad"] = agg["channel_id"]
            agg["region"] = agg["channel_id"].map(REGION_MAP)
        elif "region" in agg.columns:
            agg["stad"] = agg["region"]
            agg["region"] = agg["region"]
        else:
            agg["stad"] = "TOTAL_COMPANY"
            agg["region"] = "TOTAL_COMPANY"
        
        agg["product"] = "ALL"
        
        id_cols = ["stad", "product", "region"]
        combos = agg[id_cols].drop_duplicates()
        for _, row in combos.iterrows():
            sub = agg[(agg["stad"]==row["stad"]) & (agg["product"]==row["product"])].sort_values("jaar")
            vals, jaren = sub["true_demand"].tolist(), sub["jaar"].tolist()
            
            # Forecast 2026
            if method=="HOLTS": fc26 = holts_forecast(vals, jaren)
            elif method=="LR": fc26 = lr_forecast(jaren, vals, 2026)
            else: fc26 = ses_forecast(vals)
            
            # Validation 2025 (trained up to 2024)
            v25_sub = sub[sub["jaar"] <= 2024]
            if not v25_sub.empty:
                v25_vals, v25_jaren = v25_sub["true_demand"].tolist(), v25_sub["jaar"].tolist()
                if method=="HOLTS": pr25 = holts_forecast(v25_vals, v25_jaren) if len(v25_vals)>2 else np.nan
                elif method=="LR": pr25 = lr_forecast(v25_jaren, v25_vals, 2025) if len(v25_vals)>1 else np.nan
                else: pr25 = ses_forecast(v25_vals) if len(v25_vals)>0 else np.nan
            else: pr25 = np.nan
            
            # Validation 2024 (trained up to 2023)
            v24_sub = sub[sub["jaar"] <= 2023]
            if not v24_sub.empty:
                v24_vals, v24_jaren = v24_sub["true_demand"].tolist(), v24_sub["jaar"].tolist()
                if method=="HOLTS": pr24 = holts_forecast(v24_vals, v24_jaren) if len(v24_vals)>2 else np.nan
                elif method=="LR": pr24 = lr_forecast(v24_jaren, v24_vals, 2024) if len(v24_vals)>1 else np.nan
                else: pr24 = ses_forecast(v24_vals) if len(v24_vals)>0 else np.nan
            else: pr24 = np.nan
            
            # Residuals (MSE)
            resids = []
            for y in RESIDUAL_YEARS:
                tr = sub[sub["jaar"] < y]; ac = sub[sub["jaar"] == y]
                if not tr.empty and not ac.empty:
                    if method=="HOLTS": p = holts_forecast(tr["true_demand"].tolist(), tr["jaar"].tolist())
                    elif method=="LR": p = lr_forecast(tr["jaar"].tolist(), tr["true_demand"].tolist(), y)
                    else: p = ses_forecast(tr["true_demand"].tolist())
                    resids.append((ac["true_demand"].values[0] - p)**2)
            
            recs["stad"].append(row["stad"]); recs["product"].append(row["product"]); recs["region"].append(row["region"])
            recs["level"].append(lvl); recs["forecast_2026"].append(fc26)
            recs["predicted_2025"].append(pr25)
            recs["actual_2025"].append(sub[sub["jaar"]==2025]["true_demand"].values[0] if not sub[sub["jaar"]==2025].empty else np.nan)
            recs["predicted_2024"].append(pr24)
            recs["actual_2024"].append(sub[sub["jaar"]==2024]["true_demand"].values[0] if not sub[sub["jaar"]==2024].empty else np.nan)
            recs["mse"].append(np.mean(resids) if resids else 1.0)

    nodes_df = pd.DataFrame(recs)
    bottom_nodes = nodes_df[nodes_df["level"] == 2].copy()
    
    # 3.2 MinT Reconciliation
    print("Reconciliatie MinT (Levels 0-2)...")
    S = build_s_matrix(nodes_df, bottom_nodes)
    y_26 = nodes_df["forecast_2026"].to_numpy(); mse = nodes_df["mse"].to_numpy()
    y_25 = nodes_df["predicted_2025"].fillna(nodes_df["actual_2025"]).to_numpy()
    y_24 = nodes_df["predicted_2024"].fillna(nodes_df["actual_2024"]).to_numpy()
    
    nodes_df["reconciled_2026"] = mint_reconcile(y_26, S, mse)
    nodes_df["reconciled_2025"] = mint_reconcile(y_25, S, mse)
    nodes_df["reconciled_2024"] = mint_reconcile(y_24, S, mse)

    # 3.3 Disaggregation from City Level
    print("Disaggregatie vanaf City niveau naar Product/Sizes...")
    # Calculate historical ratios: (Product x Size) / City_Total
    size_agg = df.groupby(['channel_id', 'product_id', 'size'])['true_demand'].sum().reset_index()
    city_total_agg = df.groupby(['channel_id'])['true_demand'].sum().reset_index().rename(columns={'true_demand':'total_city'})
    ratios = size_agg.merge(city_total_agg, on=['channel_id'])
    ratios['share'] = ratios['true_demand'] / ratios['total_city']
    
    # Map reconciled City Forecasts to Product/Sizes
    city_rec = nodes_df[nodes_df["level"] == 2][['stad', 'reconciled_2026', 'reconciled_2025', 'reconciled_2024']].rename(columns={'stad':'channel_id'})
    final = ratios.merge(city_rec, on=['channel_id'])
    final['forecast_2026'] = final['reconciled_2026'] * final['share']
    final['predicted_2025'] = final['reconciled_2025'] * final['share']
    final['predicted_2024'] = final['reconciled_2024'] * final['share']
    
    # Actuals for validation
    act_all = df[df["season"].isin([2024, 2025])].groupby(['channel_id', 'product_id', 'size', 'season'])['true_demand'].sum().reset_index()
    act25 = act_all[act_all["season"]==2025].rename(columns={'true_demand':'actual_2025'})
    act24 = act_all[act_all["season"]==2024].rename(columns={'true_demand':'actual_2024'})
    
    final = final.merge(act25[['channel_id', 'product_id', 'size', 'actual_2025']], on=['channel_id', 'product_id', 'size'], how='left').fillna(0)
    final = final.merge(act24[['channel_id', 'product_id', 'size', 'actual_2024']], on=['channel_id', 'product_id', 'size'], how='left').fillna(0)
    
    # 3.4 Export
    print("Exporteren...")
    os.makedirs(FORECAST_FOLDER, exist_ok=True)
    final = final.rename(columns={'channel_id':'stad', 'product_id':'product', 'size':'maat'})
    
    with pd.ExcelWriter(f"{FORECAST_FOLDER}/hybrid_middle_out_city_product_size.xlsx") as writer:
        final[['stad', 'product', 'maat', 'forecast_2026']].to_excel(writer, sheet_name="Forecast_2026", index=False)
        # Validation Sheet
        val = final[['stad', 'product', 'maat', 'actual_2025', 'predicted_2025', 'actual_2024', 'predicted_2024']].copy()
        val['mape_2025'] = np.where(val['actual_2025']>0, (abs(val['predicted_2025'] - val['actual_2025'])/val['actual_2025'])*100, np.nan)
        val['mape_2024'] = np.where(val['actual_2024']>0, (abs(val['predicted_2024'] - val['actual_2024'])/val['actual_2024'])*100, np.nan)
        # Final column for sorting or display
        val['pct_error'] = val['mape_2025'] # Keep for compatibility with evaluate script
        val.to_excel(writer, sheet_name="Validatie_Meerdere_Jaren", index=False)
    
    # Aggregates for other files
    for lvl_name, gcols, fname in [("City_Product", ["stad", "product"], "hybrid_middle_out_city_product.xlsx"),
                                   ("City", ["stad"], "hybrid_middle_out_city.xlsx")]:
        agg_val = final.groupby(gcols)[['actual_2025', 'predicted_2025', 'actual_2024', 'predicted_2024']].sum().reset_index()
        agg_val['mape_2025'] = np.where(agg_val['actual_2025']>0, (abs(agg_val['predicted_2025'] - agg_val['actual_2025'])/agg_val['actual_2025'])*100, np.nan)
        agg_val['mape_2024'] = np.where(agg_val['actual_2024']>0, (abs(agg_val['predicted_2024'] - agg_val['actual_2024'])/agg_val['actual_2024'])*100, np.nan)
        agg_val['pct_error'] = agg_val['mape_2025']
        
        agg_fc = final.groupby(gcols)['forecast_2026'].sum().reset_index()
        with pd.ExcelWriter(f"{FORECAST_FOLDER}/{fname}") as writer:
            agg_fc.to_excel(writer, sheet_name="Forecast_2026", index=False)
            agg_val.to_excel(writer, sheet_name="Validatie_Meerdere_Jaren", index=False)

    print("KLAAR!")

if __name__ == "__main__":
    main()

"""
GEMENGDE METHODE FORECAST RECONCILIATIE via MinT (Minimum Trace / WLS)
=======================================================================
Hiërarchie:
  - Total (Level 0)            → Holt's Exponential Smoothing
  - City (Level 1)             → Linear Regression
  - City × Product (Level 2)   → Moving Average (N=3)
  - City × Product × Size (Level 3) [bottom] → Moving Average (N=3)

Residual-berekenings-strategie (Rolling Forecast):
  - Train op 2018–2022 → voorspel 2023 → residu_2023
  - Train op 2018–2023 → voorspel 2024 → residu_2024
  - Train op 2018–2024 → voorspel 2025 → residu_2025
  - MSE over de 3 residuals → W-matrix diagonaal voor MinT

Output:
  - mixed_reconciled_total.xlsx
  - mixed_reconciled_city.xlsx
  - mixed_reconciled_city_product.xlsx
  - mixed_reconciled_city_product_size.xlsx
"""

import pandas as pd
import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
import os
import time
import warnings

warnings.filterwarnings("ignore")

# ── 0. Configuratie ──────────────────────────────────────────────────────────
INPUT_PATH = "True demand/True demand Simeon/True_Demand_Results.xlsx"
SHEET_NAME = "1_True_Demand_Lijst"
FORECAST_FOLDER = "Forecast/Results"
MA_N = 3          # Moving average window
RESIDUAL_YEARS = [2023, 2024, 2025]   # Rolling residuals window
FORECAST_YEAR = 2026

# Region mapping
REGION_MAP = {
    'Platform': 'Online',
    'Webshop': 'Online',
    'Helsinki': 'Scandinavian',
    'Copenhagen': 'Scandinavian',
    'Stockholm': 'Scandinavian',
    'Amsterdam': 'European',
    'Berlin': 'European',
    'Brussels': 'European',
    'Madrid': 'European',
    'Paris': 'European',
    'Rome': 'European'
}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTIE 1: Hulpfuncties per methode
# ═══════════════════════════════════════════════════════════════════════════════




def ses_forecast(values, predict_steps=1):
    """Simple Exponential Smoothing met SSE-optimalisatie."""
    y = np.array(values, dtype=float)
    n = len(y)
    if n < 2:
        return float(y[-1]) if n > 0 else 0.0
    
    def sse(a):
        if not (0 < a < 1): return 1e15
        L = y[0]; err = 0.0
        for i in range(1, n):
            fitted = L
            L = a * y[i] + (1 - a) * fitted
            err += (y[i] - fitted)**2
        return err
        
    try:
        res = minimize(sse, x0=[0.5], method="Nelder-Mead")
        alpha = max(0.01, min(0.99, res.x[0]))
        L = y[0]
        for i in range(1, n):
            L = alpha * y[i] + (1 - alpha) * L
        return max(0.0, float(L))
    except:
        return float(y[-1])


def lr_forecast(jaren, waarden, forecast_jaar):
    """Lineaire regressie voorspelling voor een enkel forecast jaar."""
    jaars = [j for j, w in zip(jaren, waarden) if not np.isnan(w)]
    waard = [w for w in waarden if not np.isnan(w)]
    if len(waard) < 2:
        return float(waard[-1]) if waard else 0.0
    X = np.array(jaars).reshape(-1, 1)
    y = np.array(waard)
    model = LinearRegression().fit(X, y)
    pred = model.predict(np.array([[forecast_jaar]]))[0]
    return max(0.0, float(pred))


def holts_forecast(values, jaar_labels, predict_steps=1):
    """
    Holt's (double exponential smoothing) met SSE-optimalisatie.
    Geeft: (forecast, alpha, beta, {jaar: fitted_value})
    """
    y = np.array(values, dtype=float)
    n = len(y)

    if n < 3:
        return float(y[-1]) if n > 0 else 0.0, 0.5, 0.1, {}

    def sse(params):
        a, b = params
        if not (0 < a < 1 and 0 < b < 1):
            return 1e15
        L = y[0]; T = y[1] - y[0]; err = 0.0
        for i in range(1, n):
            L_prev, T_prev = L, T
            fitted = L_prev + T_prev
            L = a * y[i] + (1 - a) * fitted
            T = b * (L - L_prev) + (1 - b) * T_prev
            err += (y[i] - fitted) ** 2
        return err

    try:
        res = minimize(sse, x0=[0.3, 0.1], method="Nelder-Mead",
                       options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 500})
        alpha = max(0.01, min(0.99, res.x[0]))
        beta  = max(0.01, min(0.99, res.x[1]))

        fits = {}
        L = y[0]; T = y[1] - y[0]
        for i in range(1, n):
            L_prev, T_prev = L, T
            fitted = L_prev + T_prev
            fits[jaar_labels[i]] = fitted
            L = alpha * y[i] + (1 - alpha) * fitted
            T = beta  * (L - L_prev) + (1 - beta) * T_prev

        forecast = max(0.0, L + predict_steps * T)
        return forecast, alpha, beta, fits
    except Exception:
        return float(y[-1]), 0.5, 0.1, {}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTIE 2: Rolling-Residuals berekening
# ═══════════════════════════════════════════════════════════════════════════════

def rolling_residuals(subset_sorted, method):
    """
    Berekent residuals via het rolling-forecast protocol:
      Train op data t.o.m. jaar-1  →  voorspel jaar  →  residu = actual - pred
    Doet dit voor elk jaar in RESIDUAL_YEARS (2023, 2024, 2025).

    Geeft een dict: {jaar: residu} terug (en None als data ontbreekt).
    """
    residuals = {}
    for target_year in RESIDUAL_YEARS:
        train = subset_sorted[subset_sorted["jaar"] < target_year]
        actual_row = subset_sorted[subset_sorted["jaar"] == target_year]

        if train.empty or actual_row.empty:
            continue

        actual = actual_row["true_demand"].values[0]
        train_vals = train["true_demand"].tolist()
        train_jaren = train["jaar"].tolist()

        
        if method == "SES":
            pred = ses_forecast(train_vals)
        elif method == "LR":
            if len(train_vals) < 2:
                continue
            pred = lr_forecast(train_jaren, train_vals, target_year)
        elif method == "HOLTS":
            if len(train_vals) < 3:
                continue
            pred, _, _, _ = holts_forecast(train_vals, train_jaren)
        else:
            continue

        residuals[target_year] = actual - pred

    return residuals


def residual_mse(residuals_dict):
    """MSE van de beschikbare residuals; fallback naar 1.0."""
    vals = list(residuals_dict.values())
    if not vals:
        return 1.0
    return float(np.mean(np.array(vals) ** 2)) or 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# SECTIE 3: Verwerk één niveau → geeft lijst van node-dicts terug
# ═══════════════════════════════════════════════════════════════════════════════

def process_level(df, level_name, group_cols, method):
    """
    Berekent voor elk knooppunt op dit niveau:
      - forecast_2026
      - predicted_2025  (voor validatie)
      - actual_2025
      - residual_mse_23_25  (voor W-matrix)
    Geeft een lijst van dicts terug.
    """
    print(f"\n{'='*80}")
    print(f"VERWERKEN: {level_name}  (methode: {method})")
    print("="*80)

    group_by = group_cols + ["season"]
    agg = (df.groupby(group_by)["true_demand"]
             .sum()
             .reset_index()
             .rename(columns={"season": "jaar"}))

    # Kolommen uniformiseren
    if "channel_id" in agg.columns:
        agg = agg.rename(columns={"channel_id": "stad"})
    elif "region" in agg.columns:
        agg = agg.rename(columns={"region": "stad"})
    else:
        agg["stad"] = "TOTAL_COMPANY"

    if "product_id" in agg.columns:
        agg = agg.rename(columns={"product_id": "product"})
    else:
        agg["product"] = "ALL"

    if "size" in agg.columns:
        agg = agg.rename(columns={"size": "maat"})
    else:
        agg["maat"] = "ALL"

    agg["maat"] = agg["maat"].astype(str)

    id_cols = ["stad", "product", "maat"]
    combinaties = agg[id_cols].drop_duplicates().reset_index(drop=True)
    print(f"  Combinaties: {len(combinaties)}")

    records = []
    start = time.time()

    for idx, row in combinaties.iterrows():
        subset = agg[
            (agg["stad"]    == row["stad"]) &
            (agg["product"] == row["product"]) &
            (agg["maat"]    == row["maat"])
        ].sort_values("jaar")

        train_vals  = subset["true_demand"].tolist()
        train_jaren = subset["jaar"].tolist()

        # ── Forecast 2026 (train op alle beschikbare data)
        if method == "SES":
            fc_2026 = ses_forecast(train_vals)
        elif method == "LR":
            fc_2026 = lr_forecast(train_jaren, train_vals, FORECAST_YEAR) if len(train_vals) >= 2 else 0.0
        elif method == "HOLTS":
            fc_2026, _, _, _ = holts_forecast(train_vals, train_jaren) if len(train_vals) >= 3 else (0.0, 0, 0, {})
        else:
            fc_2026 = 0.0

        # ── Predicted 2025 (train t.o.m. 2024)
        train_val = subset[subset["jaar"] <= 2024]
        actual_2025_row = subset[subset["jaar"] == 2025]
        actual_2025 = actual_2025_row["true_demand"].values[0] if not actual_2025_row.empty else np.nan

        
        if method == "SES":
            pred_2025 = ses_forecast(train_val["true_demand"].tolist()) if not train_val.empty else np.nan
        elif method == "LR":
            pred_2025 = (lr_forecast(train_val["jaar"].tolist(),
                                      train_val["true_demand"].tolist(), 2025)
                         if len(train_val) >= 2 else np.nan)
        elif method == "HOLTS":
            pred_2025 = (holts_forecast(train_val["true_demand"].tolist(),
                                         train_val["jaar"].tolist())[0]
                         if len(train_val) >= 3 else np.nan)
        else:
            pred_2025 = np.nan

        # ── Rolling residuals 2023-2025 voor W-matrix
        resids = rolling_residuals(subset, method)
        mse    = residual_mse(resids)

        records.append({
            "stad":               row["stad"],
            "product":            row["product"],
            "maat":               row["maat"],
            "forecast_2026":      round(float(fc_2026), 1),
            "predicted_2025":     round(float(pred_2025), 1) if not np.isnan(pred_2025) else np.nan,
            "actual_2025":        actual_2025,
            "residual_mse_23_25": mse,
            "method":             method,
        })

        if (idx + 1) % 500 == 0:
            print(f"    {idx+1}/{len(combinaties)}  ({time.time()-start:.1f}s)")

    print(f"  Klaar in {time.time()-start:.1f}s")
    return records


# ═══════════════════════════════════════════════════════════════════════════════
# SECTIE 4: Suming-matrix (S) bouwen
# ═══════════════════════════════════════════════════════════════════════════════

def build_summing_matrix(all_nodes_df, bottom_nodes_df):
    """
    S is (n × m) met S[i,j] = 1 als node i ancestor is van bottom-node j.
    all_nodes_df:   alle n knopen in volgorde (level 0 t/m 3)
    bottom_nodes_df: de m onderste knopen
    Geeft sparse CSR matrix terug.
    """
    m = len(bottom_nodes_df)
    n = len(all_nodes_df)

    # Snelle lookup dictionaries
    region_to_bot   = bottom_nodes_df.groupby("region").indices
    city_to_bot     = bottom_nodes_df.groupby("stad").indices
    cp_to_bot       = bottom_nodes_df.groupby(["stad","product"]).indices

    S_rows, S_cols, S_data = [], [], []

    for i, node in all_nodes_df.iterrows():
        lvl   = node["level"]
        s, p, m_val = node["stad"], node["product"], str(node["maat"])
        r = node.get("region", None)

        if lvl == 0:   # Total → alle bottom nodes
            indices = range(m)
        elif lvl == 1: # Region
            indices = region_to_bot.get(s, []) # Note: at region level, 'stad' contains region name
        elif lvl == 2: # City
            indices = city_to_bot.get(s, [])
        elif lvl == 3: # City × Product
            indices = cp_to_bot.get((s, p), [])
        elif lvl == 4: # Bottom
            res = bottom_nodes_df[
                (bottom_nodes_df["stad"]    == s) &
                (bottom_nodes_df["product"] == p) &
                (bottom_nodes_df["maat"]    == m_val)
            ].index
            indices = [res[0]] if not res.empty else []
        else:
            indices = []

        for j in indices:
            S_rows.append(i)
            S_cols.append(j)
            S_data.append(1.0)

    return sparse.csr_matrix((S_data, (S_rows, S_cols)), shape=(n, m))


# ═══════════════════════════════════════════════════════════════════════════════
# SECTIE 5: MinT reconciliatie
# ═══════════════════════════════════════════════════════════════════════════════

def mint_reconcile(y_hat, S, variances):
    """
    MinT (WLS): y_tilde = S (S' W^-1 S)^-1 S' W^-1 y_hat
    W is diagonaal met variances.
    """
    variances = np.clip(variances, 1e-6, None)
    W_inv = sparse.diags(1.0 / variances)
    STS   = (S.T @ W_inv @ S).toarray()
    STS_inv = np.linalg.inv(STS)
    P = STS_inv @ (S.T @ W_inv)
    return (S @ (P @ y_hat)).ravel()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTIE 6: MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("GEMENGDE METHODE FORECAST RECONCILIATIE (MinT)")
    print("  Bottom / City x Product : Simple Exp Smoothing")
    print("  City / Region           : Linear Regression")
    print("  Total                   : Holt's")
    print("=" * 80)

    # ── 6.1 Data inladen
    print("\nData inladen...")
    df = pd.read_excel(INPUT_PATH, sheet_name=SHEET_NAME)
    
    # Apply region mapping
    df['region'] = df['channel_id'].map(REGION_MAP).fillna('Other')
    
    print(f"  {len(df)} rijen geladen.")

    # ── 6.2 Forecast per niveau berekenen
    # Niveau-configuraties: (naam, group_cols, methode, level_int)
    level_configs = [
        ("TOTAL",             [],                                           "HOLTS", 0),
        ("REGION",            ["region"],                                   "LR",    1),
        ("CITY",              ["channel_id"],                               "LR",    2),
        ("CITY_PRODUCT",      ["channel_id", "product_id"],                 "SES",   3),
        ("CITY_PRODUCT_SIZE", ["channel_id", "product_id", "size"],         "SES",   4),
    ]

    all_records = []
    for (name, gcols, method, lvl) in level_configs:
        recs = process_level(df, name, gcols, method)
        for r in recs:
            r["level"] = lvl
        all_records.extend(recs)

    all_nodes = pd.DataFrame(all_records).reset_index(drop=True)
    
    # Add region to all_nodes for summing matrix lookup
    all_nodes['region'] = all_nodes['stad'].where(all_nodes['level'] == 1) # For region level, region is in 'stad'
    # For city level and below, we need to map from 'stad' (which is channel_id)
    all_nodes.loc[all_nodes['level'] >= 2, 'region'] = all_nodes['stad'].map(REGION_MAP)
    
    print(f"\nTotaal nodes: {len(all_nodes)}")

    # ── 6.3 Bottom nodes bepalen (level 4)
    bottom_nodes = (all_nodes[all_nodes["level"] == 4][["stad","product","maat","region"]]
                    .sort_values(["stad","product","maat"])
                    .reset_index(drop=True))
    m = len(bottom_nodes)
    n = len(all_nodes)
    print(f"Bottom nodes (m): {m},  Alle nodes (n): {n}")

    # ── 6.4 Summing matrix bouwen
    print("\nSumming Matrix (S) construeren...")
    S = build_summing_matrix(all_nodes, bottom_nodes)

    # ── 6.5 Weight matrix + reconciliatie (2026 forecast)
    print("MinT reconciliatie 2026...")
    variances_2026 = all_nodes["residual_mse_23_25"].values
    y_hat_2026     = all_nodes["forecast_2026"].values
    y_tilde_2026   = mint_reconcile(y_hat_2026, S, variances_2026)
    all_nodes["reconciled_2026"] = np.round(y_tilde_2026, 1)

    # ── 6.6 Validatie reconciliatie (2025 predicted)
    print("MinT reconciliatie 2025 (validatie)...")
    # Gebruik actual_2025 als fallback voor nodes zonder predicted (te weinig traindata)
    y_hat_2025 = np.where(
        all_nodes["predicted_2025"].isna(),
        all_nodes["actual_2025"].fillna(0),
        all_nodes["predicted_2025"]
    )
    y_tilde_2025 = mint_reconcile(y_hat_2025, S, variances_2026)   # zelfde W
    all_nodes["reconciled_2025"] = np.round(y_tilde_2025, 1)

    # ── 6.7 Coherentie check
    rec_total = all_nodes[all_nodes["level"] == 0].copy()
    rec_bot   = all_nodes[all_nodes["level"] == 4].copy()
    total_sum = rec_total["reconciled_2026"].sum()
    bot_sum   = rec_bot["reconciled_2026"].sum()
    print(f"\nCoherentie Check 2026:")
    print(f"  Reconciled Total:       {total_sum:,.1f}")
    print(f"  Som Bottom Level:       {bot_sum:,.1f}")
    print(f"  Verschil:               {abs(total_sum - bot_sum):.4f}  ({'OK' if abs(total_sum - bot_sum) < 0.5 else 'AFWIJKING'})")

    # ── 6.8 Validatie MAPE
    val_mask = all_nodes["actual_2025"].notna() & (all_nodes["actual_2025"] > 0)
    all_nodes_val = all_nodes[val_mask].copy()
    all_nodes_val["mape_base"] = (
        (all_nodes_val["predicted_2025"] - all_nodes_val["actual_2025"]).abs()
        / all_nodes_val["actual_2025"] * 100
    )
    all_nodes_val["mape_rec"] = (
        (all_nodes_val["reconciled_2025"] - all_nodes_val["actual_2025"]).abs()
        / all_nodes_val["actual_2025"] * 100
    )
    for lvl_n, lvl_i in [("Total", 0), ("Region", 1), ("City", 2), ("City x Product", 3), ("Bottom", 4)]:
        sub = all_nodes_val[all_nodes_val["level"] == lvl_i]
        if not sub.empty:
            print(f"  MAPE {lvl_n:15s}  Base: {sub['mape_base'].mean():.1f}%  ->  Reconciled: {sub['mape_rec'].mean():.1f}%")

    # ── 6.9 Exporteren
    print(f"\nExporteren naar {FORECAST_FOLDER}/...")
    level_map = {
        0: ("total",             "mixed_reconciled_total.xlsx"),
        1: ("region",            "mixed_reconciled_region.xlsx"),
        2: ("city",              "mixed_reconciled_city.xlsx"),
        3: ("city_product",      "mixed_reconciled_city_product.xlsx"),
        4: ("city_product_size", "mixed_reconciled_city_product_size.xlsx"),
    }

    for lvl_i, (lvl_key, fname) in level_map.items():
        df_fc  = all_nodes[all_nodes["level"] == lvl_i].copy()
        df_val = all_nodes[all_nodes["level"] == lvl_i].copy()

        # Forecast sheet – kolommen opschonen
        fc_cols = ["stad", "product", "maat", "forecast_2026", "reconciled_2026",
                   "residual_mse_23_25", "method"]
        # Voeg true_demand_2025 toe als kolom voor vergelijking
        fc_out = df_fc[fc_cols].rename(columns={"forecast_2026": "base_forecast_2026"}).copy()
        fc_out.insert(fc_out.columns.get_loc("base_forecast_2026") + 1,
                      "true_demand_2025",
                      df_fc["actual_2025"].values)

        # Validatie sheet
        val_cols = ["stad", "product", "maat", "actual_2025",
                    "predicted_2025", "reconciled_2025"]
        val_out = df_val[val_cols].copy()
        val_out["abs_error_base"] = (val_out["predicted_2025"] - val_out["actual_2025"]).abs()
        val_out["abs_error_rec"]  = (val_out["reconciled_2025"] - val_out["actual_2025"]).abs()
        val_out["pct_error_base"] = np.where(
            val_out["actual_2025"] > 0,
            val_out["abs_error_base"] / val_out["actual_2025"] * 100,
            np.nan
        )
        val_out["pct_error_rec"] = np.where(
            val_out["actual_2025"] > 0,
            val_out["abs_error_rec"] / val_out["actual_2025"] * 100,
            np.nan
        )

        out_path = os.path.join(FORECAST_FOLDER, fname)
        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            fc_out.to_excel(writer, sheet_name="Forecast_2026", index=False)
            val_out.to_excel(writer, sheet_name="Validatie_2025", index=False)

        print(f"  GEREED: {out_path}")

    print("\nAlle bestanden opgeslagen. Reconciliatie voltooid!")


if __name__ == "__main__":
    main()

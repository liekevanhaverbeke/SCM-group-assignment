"""
FORECAST RECONCILIATION via MinT (Minimum Trace / WLS)
- Gebruikt Holt's Independent Base Forecasts
- Gebruikt Variantie van Residuals (2023-2025) als gewichten (W matrix)
- Garandeert hirarchische coherentie (Size -> Product -> City -> Total)
"""

import pandas as pd
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import inv
import os

# ── 1. Data laden ────────────────────────────────────────────────────────────
forecast_folder = "Forecast"
print("Base forecasts laden...")

df_total = pd.read_excel(os.path.join(forecast_folder, "holt_total.xlsx"), sheet_name="Forecast_2026")
df_city = pd.read_excel(os.path.join(forecast_folder, "holts_city.xlsx"), sheet_name="Forecast_2026")
df_cp = pd.read_excel(os.path.join(forecast_folder, "holts_city_product.xlsx"), sheet_name="Forecast_2026")
df_cps = pd.read_excel(os.path.join(forecast_folder, "holts_city_product_size.xlsx"), sheet_name="Forecast_2026")

# We extraheren ook de validatie data om de gereconcilieerde MAPE te berekenen
val_total = pd.read_excel(os.path.join(forecast_folder, "holt_total.xlsx"), sheet_name="Validatie_2025")
val_city = pd.read_excel(os.path.join(forecast_folder, "holts_city.xlsx"), sheet_name="Validatie_2025")
val_cp = pd.read_excel(os.path.join(forecast_folder, "holts_city_product.xlsx"), sheet_name="Validatie_2025")
val_cps = pd.read_excel(os.path.join(forecast_folder, "holts_city_product_size.xlsx"), sheet_name="Validatie_2025")

# ── 2. Hirarchie opbouwen ───────────────────────────────────────────────────
# 2.1 Keys normaliseren voor Validatie data
# De Lowest Level validatie sheet heeft een gecombineerde 'product' kolom (bijv. "ID - MAAT")
# Die moeten we weer splitsen om de hirarchie te kunnen matchen.
def split_product_size(df):
    if 'product' in df.columns and 'maat' not in df.columns:
        # Splits op ' - ' als het aanwezig is
        new_cols = df['product'].str.split(' - ', expand=True)
        if new_cols.shape[1] == 2:
            df['product'] = new_cols[0]
            df['maat'] = new_cols[1]
        else:
            df['maat'] = 'ALL'
    return df

val_cps = split_product_size(val_cps)
val_cp = split_product_size(val_cp)
val_city = split_product_size(val_city) # Voor de zekerheid
val_total = split_product_size(val_total)

# Alle 'maat' kolommen uniform als string behandelen
for df in [df_cps, val_cps, df_cp, val_cp, df_city, val_city, df_total, val_total]:
    if 'maat' in df.columns: df['maat'] = df['maat'].astype(str)

# Hirarchie nodes bepalen
bottom_nodes = df_cps[['stad', 'product', 'maat']].copy()
bottom_nodes = bottom_nodes.sort_values(['stad', 'product', 'maat']).reset_index(drop=True)
m = len(bottom_nodes) 

# Level indicators
df_total['level'], val_total['level'] = 0, 0
df_city['level'], val_city['level'] = 1, 1
df_cp['level'], val_cp['level'] = 2, 2
df_cps['level'], val_cps['level'] = 3, 3

# Voorbereiden voor samenvoeging: zorg dat kolommen matchen
def prepare_df(df, target_col):
    cols = ['stad', 'product', 'maat', target_col, 'residual_mse_23_25', 'level']
    if 'actual_2025' in df.columns:
        cols.append('actual_2025')
        
    for c in cols:
        if c not in df.columns:
            if c == 'maat': df[c] = 'ALL'
            elif c == 'product': df[c] = 'ALL'
            else: df[c] = 0
    return df[cols]

all_nodes_2026 = pd.concat([
    prepare_df(df_total, 'forecast_2026'),
    prepare_df(df_city, 'forecast_2026'),
    prepare_df(df_cp, 'forecast_2026'),
    prepare_df(df_cps, 'forecast_2026')
]).reset_index(drop=True)

# Ook validatie nodes verzamelen
all_nodes_2025 = pd.concat([
    prepare_df(val_total, 'predicted_2025'),
    prepare_df(val_city, 'predicted_2025'),
    prepare_df(val_cp, 'predicted_2025'),
    prepare_df(val_cps, 'predicted_2025')
]).reset_index(drop=True)

n = len(all_nodes_2026) # 3587 totaal
print(f"  Totaal aantal nodes (n): {n}")
print(f"  Bottom nodes (m): {m}")

# ── 3. Summing Matrix (S) bouwen ──────────────────────────────────────────────
print("Summing Matrix (S) construeren...")
# S is (n x m)
# S_ij = 1 als node i een ancestor is van bottom node j

# Maak dictionaries voor snelle lookup
city_to_bottom = bottom_nodes.groupby('stad').indices
cp_to_bottom = bottom_nodes.groupby(['stad', 'product']).indices

S_rows = []
S_cols = []
S_data = []

for i, node in all_nodes_2026.iterrows():
    level = node['level']
    s, p, m_val = node['stad'], node['product'], str(node['maat'])
    
    indices = []
    if level == 0: # Total
        indices = range(m)
    elif level == 1: # City
        indices = city_to_bottom.get(s, [])
    elif level == 2: # City x Product
        indices = cp_to_bottom.get((s, p), [])
    elif level == 3: # Bottom
        # We zoeken de exacte bottom node index
        res = bottom_nodes[(bottom_nodes['stad'] == s) & 
                           (bottom_nodes['product'] == p) & 
                           (bottom_nodes['maat'] == m_val)].index
        if not res.empty:
            indices = [res[0]]
            
    for idx in indices:
        S_rows.append(i)
        S_cols.append(idx)
        S_data.append(1.0)

S = sparse.csr_matrix((S_data, (S_rows, S_cols)), shape=(n, m))

# ── 4. Weight Matrix (W) bouwen ───────────────────────────────────────────────
print("Weight Matrix (W) construeren (WLS/MinT)...")
# W is diagonaal met residual variances
variances = all_nodes_2026['residual_mse_23_25'].values
# Voorkom zero variances (voor stabiliteit)
variances = np.clip(variances, 1e-6, None)
W_inv = sparse.diags(1.0 / variances)

# ── 5. Reconciliatie berekenen ────────────────────────────────────────────────
print("Reconciliatie uitvoeren (2025 & 2026)...")
# Formule: y_tilde = S * (S.T * W^-1 * S)^-1 * S.T * W^-1 * y_hat

# Compute projection matrix P = (S.T * W^-1 * S)^-1 * S.T * W^-1
STS = (S.T @ W_inv @ S).toarray()
STS_inv = np.linalg.inv(STS)
P = STS_inv @ (S.T @ W_inv)

# 2026 Forecast
y_hat_2026 = all_nodes_2026['forecast_2026'].values
y_tilde_2026 = S @ (P @ y_hat_2026)

# 2025 Validation
y_hat_2025 = all_nodes_2025['predicted_2025'].values
y_tilde_2025 = S @ (P @ y_hat_2025)

# ── 6. Resultaten Verwerken ──────────────────────────────────────────────────
all_nodes_2026['reconciled_2026'] = y_tilde_2026.round(1)
all_nodes_2025['reconciled_2025'] = y_tilde_2025.round(1)

# Splitsen terug naar de originele levels voor export
rec_total = all_nodes_2026[all_nodes_2026['level'] == 0].copy()
rec_city = all_nodes_2026[all_nodes_2026['level'] == 1].copy()
rec_cp = all_nodes_2026[all_nodes_2026['level'] == 2].copy()
rec_cps = all_nodes_2026[all_nodes_2026['level'] == 3].copy()

# Validatie data hernoemen voor plot script compatibiliteit
val_output = all_nodes_2025.copy().rename(columns={'reconciled_2025': 'predicted_2025'})

# Coherentie check
print("\nCoherentie Check (2026):")
total_sum = rec_total['reconciled_2026'].sum()
bottom_sum = rec_cps['reconciled_2026'].sum()
print(f"  Reconciled Total: {total_sum}")
print(f"  Sum of Reconciled Sizes: {bottom_sum}")
print(f"  Verschil: {abs(total_sum - bottom_sum):.4f}")

# ── 7. Export ────────────────────────────────────────────────────────────────
# We splitsen dit nu op in 4 aparte files zodat het plot script ze herkent
print("\nExporteren van gereconcilieerde resultaten...")

mapping = {
    "total": rec_total,
    "city": rec_city,
    "city_product": rec_cp,
    "city_product_size": rec_cps
}

for level_key, df_rec in mapping.items():
    level_output_path = os.path.join(forecast_folder, f"holts_reconciled_{level_key}.xlsx")
    
    # Filter validatie data voor dit specifieke level
    lvl_num = {"total": 0, "city": 1, "city_product": 2, "city_product_size": 3}[level_key]
    val_subset = val_output[val_output['level'] == lvl_num].copy()
    
    with pd.ExcelWriter(level_output_path, engine="openpyxl") as writer:
        df_rec.to_excel(writer, sheet_name="Forecast_2026", index=False)
        val_subset.to_excel(writer, sheet_name="Validatie_2025", index=False)
    
    print(f"  GEREED: {level_output_path}")

print(f"\nAlle gereconcilieerde bestanden zijn opgeslagen in {forecast_folder}/")

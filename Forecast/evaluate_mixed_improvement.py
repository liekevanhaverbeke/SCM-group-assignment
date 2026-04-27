"""
EVALUATIE VAN RECONCILIATIE-VERBETERING (Mixed Method)
- Analyseert de City x Product resultaten (Level 3).
- Vergelijkt de MAPE van de basis-voorspelling (SES) met de MinT-reconcilieerde voorspelling.
- Genereert een overzichtelijk Excel bestand met de verbetering per City/Product.
"""

import pandas as pd
import numpy as np
import os

INPUT_FILE_CP = "Forecast/Results/mixed_reconciled_city_product.xlsx"
INPUT_FILE_CPS = "Forecast/Results/mixed_reconciled_city_product_size.xlsx"
INPUT_FILE_TD = "Forecast/Results/top_down_city_product_size.xlsx"
INPUT_FILE_HY = "Forecast/Results/hybrid_middle_out_city_product_size.xlsx"
OUTPUT_FILE = "Forecast/Results/mixed_improvement_analysis.xlsx"

def run_evaluation():
    print("="*80)
    print("ANALYSING IMPROVEMENT & METHOD COMPARISON")
    print("="*80)

    if not all(os.path.exists(f) for f in [INPUT_FILE_CP, INPUT_FILE_CPS, INPUT_FILE_TD]):
        print(f"ERROR: Bestanden niet gevonden. Voer alle reconciliatie scripts uit.")
        return

    # 1. City x Product data laden
    print("  Processing Level: City x Product...")
    df_cp = pd.read_excel(INPUT_FILE_CP, sheet_name="Validatie_2025")
    analysis_cp = df_cp[['stad', 'product', 'actual_2025', 'predicted_2025', 'reconciled_2025', 'pct_error_base', 'pct_error_rec']].copy()
    analysis_cp['improvement_pts'] = analysis_cp['pct_error_base'] - analysis_cp['pct_error_rec']
    
    # 2. City x Product x Size data laden (Lowest Level)
    print("  Processing Level: City x Product x Size...")
    df_cps = pd.read_excel(INPUT_FILE_CPS, sheet_name="Validatie_2025")
    analysis_cps = df_cps[['stad', 'product', 'maat', 'actual_2025', 'predicted_2025', 'reconciled_2025', 'pct_error_base', 'pct_error_rec']].copy()
    analysis_cps['improvement_pts'] = analysis_cps['pct_error_base'] - analysis_cps['pct_error_rec']

    # 3. Top-Down data laden voor vergelijking
    print("  Comparing with Top-Down & Hybrid Methods...")
    df_td = pd.read_excel(INPUT_FILE_TD, sheet_name="Validatie_2025")
    df_td = df_td[['stad', 'product', 'maat', 'predicted_2025', 'pct_error']].rename(columns={
        'pct_error': 'mape_top_down',
        'predicted_2025': 'pred_top_down'
    })
    
    # 4. Hybrid data laden
    df_hy = pd.read_excel(INPUT_FILE_HY, sheet_name="Validatie_Meerdere_Jaren")
    df_hy = df_hy[['stad', 'product', 'maat', 'actual_2025', 'predicted_2025']].rename(columns={
        'predicted_2025': 'pred_hybrid'
    })
    
    # 5. Merge all
    print("  Merging all methods for 2025 analysis...")
    comparison = analysis_cps[['stad', 'product', 'maat', 'actual_2025', 'reconciled_2025']].copy()
    comparison = comparison.rename(columns={'reconciled_2025': 'pred_mixed'})
    
    # Merge TD
    df_td_clean = df_td[['stad', 'product', 'maat', 'pred_top_down']]
    comparison = comparison.merge(df_td_clean, on=['stad', 'product', 'maat'], how='inner')
    
    # Merge Hybrid
    df_hy_clean = df_hy[['stad', 'product', 'maat', 'pred_hybrid']]
    comparison = comparison.merge(df_hy_clean, on=['stad', 'product', 'maat'], how='inner')
    
    # Metric Calculations
    methods = ['top_down', 'hybrid'] # Removing 'mixed'
    for m in methods:
        pred_col = f'pred_{m}'
        # Percentage Off: (Pred - Act) / Act * 100
        comparison[f'pct_off_{m}'] = np.where(comparison['actual_2025'] > 0, 
                                             ((comparison[pred_col] - comparison['actual_2025']) / comparison['actual_2025']) * 100, 
                                             0)
        # MSE: (Pred - Act)^2
        comparison[f'mse_{m}'] = (comparison[pred_col] - comparison['actual_2025'])**2
        
    # Closest Method (Lowest MSE)
    mse_cols = [f'mse_{m}' for m in methods]
    comparison['closest_method'] = comparison[mse_cols].idxmin(axis=1).str.replace('mse_', '')
    
    # Best Positive Method Logic
    def find_best_positive(row):
        # 1. Which methods are above actual?
        pos_methods = []
        for m in methods:
            if row[f'pct_off_{m}'] >= 0:
                pos_methods.append(m)
        
        if not pos_methods:
            return row['closest_method']
        
        # 2. Of those above, which has lowest MSE?
        best_pos = pos_methods[0]
        min_mse = row[f'mse_{best_pos}']
        for m in pos_methods[1:]:
            if row[f'mse_{m}'] < min_mse:
                min_mse = row[f'mse_{m}']
                best_pos = m
        return best_pos

    print("  Determining Best Positive Method (Hybrid vs Top-Down)...")
    comparison['best_positive_method'] = comparison.apply(find_best_positive, axis=1)
    
    # Summarize win rates
    best_counts = comparison['closest_method'].value_counts()
    pos_counts = comparison['best_positive_method'].value_counts()

    # 5. Exporteren
    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        comparison.to_excel(writer, sheet_name="Hybrid_vs_TopDown", index=False)
        # Also keep original sheets for reference if they existed
        analysis_cp.to_excel(writer, sheet_name="Comparison_Per_Product", index=False)
    
    print(f"\nAnalyse voltooid!")
    print(f"Resultaten opgeslagen in: {OUTPUT_FILE}")
    
    # 6. Highlights
    print("\nHEAD-TO-HEAD: HYBRID vs TOP-DOWN")
    print("-" * 35)
    print("CLOSEST METHOD (Pure Accuracy):")
    for m, count in best_counts.items():
        print(f"  {m:15s}: {count:5d} items ({count/len(comparison)*100:4.1f}%)")
        
    print("\nBEST POSITIVE METHOD (Inventory Safe):")
    for m, count in pos_counts.items():
        print(f"  {m:15s}: {count:5d} items ({count/len(comparison)*100:4.1f}%)")

if __name__ == "__main__":
    run_evaluation()

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
INPUT_FILE_TD = "Forecast/Results/top_down_city_product_size_linear.xlsx"
INPUT_FILE_HY = "Forecast/Results/hybrid_middle_out_city_product_size.xlsx"
OUTPUT_FILE = "Forecast/Results/linear_method_comparison_analysis.xlsx"

def run_evaluation():
    print("="*80)
    print("ANALYSING LINEAR METHOD COMPARISON (Hybrid vs Linear TD)")
    print("="*80)

    for f in [INPUT_FILE_TD, INPUT_FILE_HY]:
        if not os.path.exists(f):
            print(f"ERROR: {f} niet gevonden.")
            return

    # 1. City x Product data laden
    print("  Processing Level: City x Product...")
    # 1. Hybrid data laden
    print("  Loading Hybrid data...")
    df_hy = pd.read_excel(INPUT_FILE_HY, sheet_name="Validatie_Meerdere_Jaren")
    df_hy = df_hy[['stad', 'product', 'maat', 'actual_2025', 'predicted_2025']].rename(columns={
        'predicted_2025': 'pred_hybrid'
    })
    
    # 2. Linear Top-Down data laden
    print("  Loading Linear Top-Down data...")
    df_td = pd.read_excel(INPUT_FILE_TD, sheet_name="Validatie_2025")
    df_td = df_td[['stad', 'product', 'maat', 'predicted_2025']].rename(columns={
        'predicted_2025': 'pred_top_down_linear'
    })
    
    # 3. Merge methods for head-to-head analysis
    print("  Merging methods...")
    comparison = df_hy.merge(df_td, on=['stad', 'product', 'maat'], how='inner')
    
    # Metric Calculations
    methods = ['top_down_linear', 'hybrid']
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
        pos_methods = [m for m in methods if row[f'pct_off_{m}'] >= 0]
        if not pos_methods: return row['closest_method']
        return min(pos_methods, key=lambda m: row[f'mse_{m}'])

    print("  Determining Best Positive Method...")
    comparison['best_positive_method'] = comparison.apply(find_best_positive, axis=1)
    
    # Summarize win rates
    best_counts = comparison['closest_method'].value_counts()
    pos_counts = comparison['best_positive_method'].value_counts()

    # 4. Exporteren
    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        comparison.to_excel(writer, sheet_name="Hybrid_vs_Linear_TD", index=False)
    
    print(f"\nAnalyse voltooid!")
    print(f"Resultaten opgeslagen in: {OUTPUT_FILE}")
    
    # 5. Highlights
    print("\nHEAD-TO-HEAD: HYBRID vs LINEAR TOP-DOWN")
    print("-" * 45)
    print("CLOSEST METHOD (Pure Accuracy):")
    for m, count in best_counts.items():
        print(f"  {m:20s}: {count:5d} items ({count/len(comparison)*100:4.1f}%)")
        
    print("\nBEST POSITIVE METHOD (Inventory Safe):")
    for m, count in pos_counts.items():
        print(f"  {m:20s}: {count:5d} items ({count/len(comparison)*100:4.1f}%)")

if __name__ == "__main__":
    run_evaluation()

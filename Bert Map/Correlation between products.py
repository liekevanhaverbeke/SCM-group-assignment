import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# ── 1. Configuration ──────────────────────────────────────────────────────────
INPUT_PATH = "True demand/True demand Simeon/True_Demand_Results.xlsx"
SHEET_NAME = "1_True_Demand_Lijst"
OUTPUT_DIR = "Bert Map/analysis_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def analyze_product_correlation():
    print("="*80)
    print("PRODUCT CORRELATION ANALYSIS")
    print("="*80)

    # ── 2. Load Data ──────────────────────────────────────────────────────────
    print(f"Loading data from {INPUT_PATH}...")
    try:
        df = pd.read_excel(INPUT_PATH, sheet_name=SHEET_NAME)
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return

    # ── 3. Data Preparation ───────────────────────────────────────────────────
    # We want to see if products sell together. 
    # We treat each combination of (channel_id, season) as an observation.
    # This gives us more data points than just looking at yearly totals.
    
    print("Preparing data for correlation...")
    # Aggregate demand per product, channel, and season
    prod_demand = df.groupby(['channel_id', 'season', 'product_id'])['true_demand'].sum().reset_index()
    
    # Create a Pivot Table: Rows = (Store, Season), Columns = Product IDs
    # This matrix shows how much of each product was demanded in each store during each season.
    pivot_df = prod_demand.pivot_table(
        index=['channel_id', 'season'], 
        columns='product_id', 
        values='true_demand'
    ).fillna(0)
    
    print(f"Matrix shape: {pivot_df.shape[0]} observations of {pivot_df.shape[1]} products.")

    # ── 4. Calculate Correlation ──────────────────────────────────────────────
    # Pearson correlation coefficient
    corr_matrix = pivot_df.corr()

    # ── 5. Visualization 1: Heatmap ───────────────────────────────────────────
    print("Generating correlation heatmap...")
    plt.figure(figsize=(16, 12))
    
    # Use a diverging color map (RdBu_r) to highlight positive/negative correlations
    sns.heatmap(
        corr_matrix, 
        annot=False, # Too many products for annotations
        cmap='RdBu_r', 
        center=0,
        linewidths=0.1,
        cbar_kws={"shrink": .8}
    )
    
    plt.title('Product Demand Correlation Matrix\n(Based on Store/Season demand patterns)', fontsize=16, pad=20)
    plt.xlabel('Product ID', fontsize=12)
    plt.ylabel('Product ID', fontsize=12)
    plt.tight_layout()
    
    heatmap_path = os.path.join(OUTPUT_DIR, "product_correlation_heatmap.png")
    plt.savefig(heatmap_path, dpi=200)
    print(f"Heatmap saved to: {heatmap_path}")

    # ── 6. Visualization 2: Clustermap ────────────────────────────────────────
    # A clustermap automatically groups correlated products together
    print("Generating clustermap...")
    cg = sns.clustermap(
        corr_matrix, 
        cmap='RdBu_r', 
        center=0, 
        figsize=(15, 15),
        linewidths=0.1
    )
    plt.setp(cg.ax_heatmap.get_yticklabels(), rotation=0)
    cg.fig.suptitle('Hierarchical Clustering of Product Correlations', fontsize=16, y=1.02)
    
    clustermap_path = os.path.join(OUTPUT_DIR, "product_correlation_clustermap.png")
    plt.savefig(clustermap_path, dpi=200)
    print(f"Clustermap saved to: {clustermap_path}")

    # ── 7. Extract Top Correlations ───────────────────────────────────────────
    print("\n" + "-"*40)
    print("TOP CORRELATED PRODUCTS")
    print("-"*40)
    
    # Unstack the correlation matrix and remove self-correlations
    sol = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    corr_pairs = sol.unstack().dropna().sort_values(ascending=False)
    
    print("\nTop 10 Strongest POSITIVE Correlations:")
    top_pos = corr_pairs.head(10)
    for (p1, p2), val in top_pos.items():
        print(f"  {p1} <---> {p2}: {val:.3f}")
        
    print("\nTop 10 Strongest NEGATIVE Correlations:")
    top_neg = corr_pairs.tail(10).sort_values()
    for (p1, p2), val in top_neg.items():
        print(f"  {p1} <---> {p2}: {val:.3f}")

    # ── 8. Insights Summary ──────────────────────────────────────────────────
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"1. Identified {len(corr_pairs)} unique product pairs.")
    high_corr = corr_pairs[abs(corr_pairs) > 0.7]
    print(f"2. Found {len(high_corr)} pairs with high correlation (|r| > 0.7).")
    print(f"3. Visualizations available in '{OUTPUT_DIR}' folder.")
    print("="*80)

if __name__ == "__main__":
    analyze_product_correlation()
    plt.show()

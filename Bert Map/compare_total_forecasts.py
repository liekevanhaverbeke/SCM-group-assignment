import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
import warnings

warnings.filterwarnings("ignore")

# Configuration
INPUT_PATH = "True demand/True demand Simeon/True_Demand_Results.xlsx"
SHEET_NAME = "1_True_Demand_Lijst"

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

def main():
    print("Loading data...")
    df = pd.read_excel(INPUT_PATH, sheet_name=SHEET_NAME)
    
    # Aggregate to Total Level
    total_history = df.groupby('season')['true_demand'].sum().reset_index().sort_values('season')
    jaren = total_history['season'].tolist()
    waard = total_history['true_demand'].tolist()
    
    print("\nHistorical Data (Total Units):")
    for j, w in zip(jaren, waard):
        print(f"  {j}: {w:.2f}")
    
    # Forecasts for 2026
    fc_lr = lr_forecast(jaren, waard, 2026)
    fc_holt = holts_forecast(waard, jaren)
    
    print("\n" + "="*40)
    print("2026 TOTAL COMPANY FORECAST COMPARISON")
    print("="*40)
    print(f"Linear Regression:  {fc_lr:.2f} units")
    print(f"Holt's Linear:      {fc_holt:.2f} units")
    print(f"Difference:         {abs(fc_lr - fc_holt):.2f} units")
    print("="*40)

if __name__ == "__main__":
    main()

"""
Vraagvoorspelling 2026 via Holt's Methode (Double Exponential Smoothing)
- Aggregeert true_demand per stad, jaar en product
- Validatie: train op 2018-2024, voorspel 2025 en vergelijk met actuals
- Forecast: train op alle data (2018-2025), voorspel 2026
- Holt's methode modelleert zowel niveau als trend → geschikt voor data met trend
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

# ── 1. Data laden & aggregeren ───────────────────────────────────────────────
df = pd.read_excel("True demand/true_demand.xlsx", sheet_name=0)
agg = (
    df.groupby(["channel_id", "season", "product_id"])["true_demand"]
    .sum()
    .reset_index()
    .rename(columns={"channel_id": "stad", "season": "jaar", "product_id": "product"})
)

combinaties = agg[["stad", "product"]].drop_duplicates()

# ── 2. Holt's methode (handmatige implementatie) ────────────────────────────
def holts_forecast(waarden, alpha=None, beta=None, predict_steps=1):
    """
    Double Exponential Smoothing (Holt's Linear Trend Method).
    Als alpha/beta niet opgegeven: optimaliseer via minimalisatie van SSE.
    Geeft de forecast 'predict_steps' stappen vooruit.
    """
    y = np.array(waarden, dtype=float)
    n = len(y)

    def sse(params):
        a, b = params
        if not (0 < a < 1 and 0 < b < 1):
            return 1e10
        L = y[0]
        T = y[1] - y[0]
        err = 0.0
        for i in range(1, n):
            L_prev, T_prev = L, T
            L = a * y[i] + (1 - a) * (L_prev + T_prev)
            T = b * (L - L_prev) + (1 - b) * T_prev
            err += (y[i] - (L_prev + T_prev)) ** 2
        return err

    if alpha is None or beta is None:
        from scipy.optimize import minimize
        res = minimize(sse, x0=[0.3, 0.1], method="Nelder-Mead",
                       options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 2000})
        alpha, beta = res.x
        alpha = max(0.01, min(0.99, alpha))
        beta  = max(0.01, min(0.99, beta))

    # Herbereken met optimale parameters
    L = y[0]
    T = y[1] - y[0]
    for i in range(1, n):
        L_prev, T_prev = L, T
        L = alpha * y[i] + (1 - alpha) * (L_prev + T_prev)
        T = beta  * (L - L_prev) + (1 - beta) * T_prev

    forecast = L + predict_steps * T
    return max(0.0, forecast), alpha, beta

# ── 3. VALIDATIE: train 2018-2024, voorspel 2025 ────────────────────────────
print("=" * 65)
print("VALIDATIE  –  Voorspelling 2025 vs. Actuals  (Holt's methode)")
print("=" * 65)

val_records = []
for _, row in combinaties.iterrows():
    subset = agg[(agg["stad"] == row["stad"]) & (agg["product"] == row["product"])]
    train = subset[subset["jaar"] <= 2024].sort_values("jaar")
    actual_row = subset[subset["jaar"] == 2025]

    if len(train) < 3 or actual_row.empty:   # minimaal 3 punten voor Holt
        continue

    pred_2025, alpha, beta = holts_forecast(train["true_demand"].tolist(), predict_steps=1)
    actual_2025 = actual_row["true_demand"].values[0]

    val_records.append({
        "stad": row["stad"],
        "product": row["product"],
        "actual_2025": actual_2025,
        "predicted_2025": round(pred_2025, 1),
        "abs_error": abs(pred_2025 - actual_2025),
        "pct_error": abs(pred_2025 - actual_2025) / actual_2025 * 100 if actual_2025 > 0 else np.nan,
        "alpha": round(alpha, 4),
        "beta": round(beta, 4),
    })

val_df = pd.DataFrame(val_records)

print("\nGemiddelde Absolute Fout (MAE) per stad:")
print("-" * 40)
for stad, grp in val_df.groupby("stad"):
    mae = grp["abs_error"].mean()
    mape = grp["pct_error"].mean()
    print(f"  {stad:<15} MAE = {mae:>7.1f}  |  MAPE = {mape:>6.1f}%")

overall_mae = val_df["abs_error"].mean()
overall_mape = val_df["pct_error"].mean()
print(f"\n  {'TOTAAL':<15} MAE = {overall_mae:>7.1f}  |  MAPE = {overall_mape:>6.1f}%")

print("\nTop 10 grootste afwijkingen (2025 validatie):")
cols = ["stad", "product", "actual_2025", "predicted_2025", "abs_error", "pct_error"]
print(val_df.nlargest(10, "abs_error")[cols].to_string(index=False))

# ── 4. FORECAST: train 2018-2025, voorspel 2026 ─────────────────────────────
print("\n" + "=" * 65)
print("FORECAST  –  Voorspelling 2026  (Holt's methode)")
print("=" * 65)

forecast_records = []
for _, row in combinaties.iterrows():
    subset = agg[(agg["stad"] == row["stad"]) & (agg["product"] == row["product"])].sort_values("jaar")

    if len(subset) < 3:
        continue

    pred_2026, alpha, beta = holts_forecast(subset["true_demand"].tolist(), predict_steps=1)
    last_actual = subset[subset["jaar"] == 2025]
    demand_2025 = last_actual["true_demand"].values[0] if not last_actual.empty else np.nan

    forecast_records.append({
        "stad": row["stad"],
        "product": row["product"],
        "true_demand_2025": demand_2025,
        "forecast_2026": round(pred_2026, 1),
        "verschil": round(pred_2026 - demand_2025, 1) if not np.isnan(demand_2025) else np.nan,
        "alpha": round(alpha, 4),
        "beta": round(beta, 4),
    })

fc_df = pd.DataFrame(forecast_records)

print("\nVoorspelling 2026 per stad (som over alle producten):")
print("-" * 50)
summary = fc_df.groupby("stad")[["true_demand_2025", "forecast_2026"]].sum()
summary["groei_%"] = ((summary["forecast_2026"] - summary["true_demand_2025"]) / summary["true_demand_2025"] * 100).round(1)
print(summary.to_string())

# ── 5. Exporteren naar Excel ─────────────────────────────────────────────────
output_path = "Forecast/forecast_holts_method_2026.xlsx"
with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    fc_df.to_excel(writer, sheet_name="Forecast_2026", index=False)
    val_df.to_excel(writer, sheet_name="Validatie_2025", index=False)
    summary.reset_index().to_excel(writer, sheet_name="Samenvatting_per_stad", index=False)

print(f"\n✔  Resultaten opgeslagen in: {output_path}")
print("    Sheets: Forecast_2026 | Validatie_2025 | Samenvatting_per_stad")

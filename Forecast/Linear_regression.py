"""
Vraagvoorspelling 2026 via Lineaire Regressie
- Aggregeert true_demand per stad, jaar en product
- Validatie: train op 2018-2024, voorspel 2025 en vergelijk met actuals
- Forecast: train op alle data (2018-2025), voorspel 2026
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import os

# ── 1. Data laden & aggregeren ───────────────────────────────────────────────
input_path = "True demand/True demand Simeon/True_Demand_Results.xlsx"
sheet_name = "1_True_Demand_Lijst"
demand_col = "true_demand"

print("=" * 70)
print("Dataset laden:", os.path.basename(input_path))
print("=" * 70)

df = pd.read_excel(input_path, sheet_name=sheet_name)
agg = (
    df.groupby(["channel_id", "season", "product_id"])["true_demand"]
    .sum()
    .reset_index()
    .rename(columns={"channel_id": "stad", "season": "jaar", "product_id": "product"})
)

combinaties = agg[["stad", "product"]].drop_duplicates()

# ── 2. Hulpfunctie: lineaire regressie op één tijdreeks ─────────────────────
def lr_forecast(jaren, waarden, predict_jaar):
    X = np.array(jaren).reshape(-1, 1)
    y = np.array(waarden)
    model = LinearRegression()
    model.fit(X, y)
    pred = model.predict([[predict_jaar]])[0]
    return max(0, pred), model.coef_[0], model.intercept_

# ── 3. VALIDATIE: train 2018-2024, voorspel 2025 ────────────────────────────
print("=" * 65)
print("VALIDATIE  –  Voorspelling 2025 vs. Actuals")
print("=" * 65)

val_records = []
for _, row in combinaties.iterrows():
    subset = agg[(agg["stad"] == row["stad"]) & (agg["product"] == row["product"])]
    train = subset[subset["jaar"] <= 2024].sort_values("jaar")
    actual_row = subset[subset["jaar"] == 2025]

    if len(train) < 2 or actual_row.empty:
        continue

    pred_2025, _, _ = lr_forecast(train["jaar"].tolist(), train["true_demand"].tolist(), 2025)
    actual_2025 = actual_row["true_demand"].values[0]

    val_records.append({
        "stad": row["stad"],
        "product": row["product"],
        "actual_2025": actual_2025,
        "predicted_2025": round(pred_2025, 1),
        "abs_error": abs(pred_2025 - actual_2025),
        "pct_error": abs(pred_2025 - actual_2025) / actual_2025 * 100 if actual_2025 > 0 else np.nan,
    })

val_df = pd.DataFrame(val_records)

# Metrics per stad
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
print(val_df.nlargest(10, "abs_error")[["stad", "product", "actual_2025", "predicted_2025", "abs_error", "pct_error"]].to_string(index=False))

# ── 4. FORECAST: train 2018-2025, voorspel 2026 ─────────────────────────────
print("\n" + "=" * 65)
print("FORECAST  –  Voorspelling 2026")
print("=" * 65)

forecast_records = []
for _, row in combinaties.iterrows():
    subset = agg[(agg["stad"] == row["stad"]) & (agg["product"] == row["product"])].sort_values("jaar")

    if len(subset) < 2:
        continue

    pred_2026, slope, intercept = lr_forecast(subset["jaar"].tolist(), subset["true_demand"].tolist(), 2026)
    last_actual = subset[subset["jaar"] == 2025]
    demand_2025 = last_actual["true_demand"].values[0] if not last_actual.empty else np.nan

    forecast_records.append({
        "stad": row["stad"],
        "product": row["product"],
        "true_demand_2025": demand_2025,
        "forecast_2026": round(pred_2026, 1),
        "verschil": round(pred_2026 - demand_2025, 1) if not np.isnan(demand_2025) else np.nan,
        "slope": round(slope, 2),
    })

fc_df = pd.DataFrame(forecast_records)

print("\nVoorspelling 2026 per stad (som over alle producten):")
print("-" * 50)
summary = fc_df.groupby("stad")[["true_demand_2025", "forecast_2026"]].sum()
summary["groei_%"] = ((summary["forecast_2026"] - summary["true_demand_2025"]) / summary["true_demand_2025"] * 100).round(1)
print(summary.to_string())

# ── 5. Exporteren naar Excel ─────────────────────────────────────────────────
output_path = "Forecast/forecast_linear_regression_2026.xlsx"
with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    fc_df.to_excel(writer, sheet_name="Forecast_2026", index=False)
    val_df.to_excel(writer, sheet_name="Validatie_2025", index=False)
    summary.reset_index().to_excel(writer, sheet_name="Samenvatting_per_stad", index=False)

print(f"\n✔  Resultaten opgeslagen in: {output_path}")
print("    Sheets: Forecast_2026 | Validatie_2025 | Samenvatting_per_stad")

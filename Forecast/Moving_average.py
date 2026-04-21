"""
Vraagvoorspelling 2026 via Moving Average met vaste venstergrootte n=3
- Gebruikt de 'True Demand Results' data (aggregatie per stad, jaar, product)
- Validatie: train op 2018–2024, voorspel 2025 en vergelijk met actuals
- Forecast: train op alle data (2018–2025), voorspel 2026
- Vaste n=3: elke voorspelling is het gemiddelde van de 3 meest recente jaren
"""

import pandas as pd
import numpy as np
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
    df.groupby(["channel_id", "season", "product_id"])[demand_col]
    .sum()
    .reset_index()
    .rename(columns={"channel_id": "stad", "season": "jaar", "product_id": "product"})
)

combinaties = agg[["stad", "product"]].drop_duplicates()

# ── 2. Hulpfuncties ──────────────────────────────────────────────────────────
N = 3  # vaste venstergrootte


def moving_average_forecast(waarden: list, n: int = N) -> float:
    """Voorspel de volgende periode als het gemiddelde van de laatste n waarden."""
    if len(waarden) < n:
        raise ValueError(f"Te weinig datapunten ({len(waarden)}) voor n={n}")
    return float(np.mean(waarden[-n:]))


# ── 3. VALIDATIE: train 2018–2024, voorspel 2025 ────────────────────────────
print("\n" + "=" * 65)
print("VALIDATIE – Voorspelling 2025 vs. Actuals (Moving Average)")
print("=" * 65)

val_records = []
for _, row in combinaties.iterrows():
    subset = (
        agg[(agg["stad"] == row["stad"]) & (agg["product"] == row["product"])]
        .sort_values("jaar")
    )

    train = subset[subset["jaar"] <= 2024]
    actual_row = subset[subset["jaar"] == 2025]

    if len(train) < 3 or actual_row.empty:
        continue

    train_vals = train[demand_col].tolist()
    pred_2025 = moving_average_forecast(train_vals)
    actual_2025 = actual_row[demand_col].values[0]

    val_records.append({
        "stad": row["stad"],
        "product": row["product"],
        "actual_2025": actual_2025,
        "predicted_2025": round(pred_2025, 1),
        "abs_error": abs(pred_2025 - actual_2025),
        "pct_error": (
            abs(pred_2025 - actual_2025) / actual_2025 * 100
            if actual_2025 > 0 else np.nan
        ),
        "n": N,
    })

val_df = pd.DataFrame(val_records)

print("\nGemiddelde Absolute Fout (MAE) per stad:")
print("-" * 40)
for stad, grp in val_df.groupby("stad"):
    mae = grp["abs_error"].mean()
    mape = grp["pct_error"].mean()
    print(f"  {stad:<15} MAE = {mae:>7.1f} | MAPE = {mape:>6.1f}%")

overall_mae = val_df["abs_error"].mean()
overall_mape = val_df["pct_error"].mean()
print(f"\n  {'TOTAAL':<15} MAE = {overall_mae:>7.1f} | MAPE = {overall_mape:>6.1f}%")

print("\nTop 10 grootste afwijkingen (2025 validatie):")
cols = ["stad", "product", "actual_2025", "predicted_2025", "abs_error", "pct_error", "n"]
print(val_df.nlargest(10, "abs_error")[cols].to_string(index=False))

# ── 4. FORECAST: train 2018–2025, voorspel 2026 ─────────────────────────────
print("\n" + "=" * 65)
print(f"FORECAST – Voorspelling 2026 (Moving Average, n={N})")
print("=" * 65)

forecast_records = []
for _, row in combinaties.iterrows():
    subset = (
        agg[(agg["stad"] == row["stad"]) & (agg["product"] == row["product"])]
        .sort_values("jaar")
    )

    if len(subset) < N:
        continue

    all_vals = subset[demand_col].tolist()
    pred_2026 = moving_average_forecast(all_vals)

    last_actual = subset[subset["jaar"] == 2025]
    demand_2025 = last_actual[demand_col].values[0] if not last_actual.empty else np.nan

    forecast_records.append({
        "stad": row["stad"],
        "product": row["product"],
        "true_demand_2025": demand_2025,
        "forecast_2026": round(pred_2026, 1),
        "verschil": round(pred_2026 - demand_2025, 1) if not np.isnan(demand_2025) else np.nan,
        "n": N,
    })

fc_df = pd.DataFrame(forecast_records)

print("\nVoorspelling 2026 per stad (som over alle producten):")
print("-" * 50)
summary = fc_df.groupby("stad")[["true_demand_2025", "forecast_2026"]].sum()
summary["groei_%"] = (
    (summary["forecast_2026"] - summary["true_demand_2025"])
    / summary["true_demand_2025"] * 100
).round(1)
print(summary.to_string())

# ── 5. Exporteren naar Excel ─────────────────────────────────────────────────
output_path = "Forecast/forecast_moving_average_2026.xlsx"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    fc_df.to_excel(writer, sheet_name="Forecast_2026", index=False)
    val_df.to_excel(writer, sheet_name="Validatie_2025", index=False)
    summary.reset_index().to_excel(writer, sheet_name="Samenvatting_per_stad", index=False)

print(f"\n✔  Resultaten opgeslagen in: {output_path}")
print("    Sheets: Forecast_2026 | Validatie_2025 | Samenvatting_per_stad")
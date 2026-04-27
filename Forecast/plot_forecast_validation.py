import pandas as pd
import matplotlib.pyplot as plt
import os

# Settings
forecast_folder = "Forecast"
plot_folder = os.path.join(forecast_folder, "forecast pictures")
os.makedirs(plot_folder, exist_ok=True)

# Unified levels and methods
methods = {
    "ARIMA": "arima",
    "ARIMA (No Platform)": "arima_no_platform",
    "Holts": "holts",
    "Holts Reconciled": "holts_reconciled",
    "Linear Regression": "linear_regression",
    "Moving Average": "moving_average",
    "Simple Exp Smoothing": "simple_exp",
    "Mixed Reconciled": "mixed_reconciled",
    "Top-Down Proportional": "top_down",
    "Middle-Out Ratio": "middle_out"
}

levels = {
    "Total Level": "total",
    "Region Level": "region",
    "City Level": "city",
    "City/Product": "city_product",
    "Lowest Level (Size)": "city_product_size"
}

def generate_plots():
    for method_label, method_file in methods.items():
        for level_label, level_file in levels.items():
            # Standard construction
            filename = f"{method_file}_{level_file}.xlsx"
            
            # Special case for 'No Platform' naming convention: arima_{level}_no_platform.xlsx
            if method_file == "arima_no_platform":
                filename = f"arima_{level_file}_no_platform.xlsx"

            file_path = os.path.join(forecast_folder, filename)

            if not os.path.exists(file_path):
                # Specifieke uitzondering voor holts_total vs holt_total (consistentie fix)
                if filename == "holts_total.xlsx" and os.path.exists(os.path.join(forecast_folder, "holt_total.xlsx")):
                    filename = "holt_total.xlsx"
                    file_path = os.path.join(forecast_folder, filename)
                else:
                    continue

            display_name = f"{method_label} - {level_label}"

            # Load the validation data
            try:
                df_val = pd.read_excel(file_path, sheet_name="Validatie_2025")
            except Exception:
                print(f"Sheet Validatie_2025 not found in {filename}")
                continue

            # ── Mixed reconciled: grouped bar (Base vs Reconciled) ──────────────
            has_reconciled = "reconciled_2025" in df_val.columns

            if has_reconciled:
                agg_cols = ["actual_2025", "predicted_2025", "reconciled_2025"]
                city_summary = df_val.groupby("stad")[agg_cols].sum()
            else:
                city_summary = df_val.groupby("stad")[["actual_2025", "predicted_2025"]].sum()

            # Check for data presence
            if city_summary.empty or city_summary["actual_2025"].sum() == 0:
                print(f"Skipping empty or zero-data plot: {filename}")
                continue

            if has_reconciled:
                # ── Grouped bar: Base error + Reconciled error ──────────────────
                city_summary["Error_Base"] = city_summary["predicted_2025"] - city_summary["actual_2025"]
                city_summary["Error_Rec"]  = city_summary["reconciled_2025"] - city_summary["actual_2025"]
                city_summary["Pct_Base"] = (city_summary["Error_Base"] / city_summary["actual_2025"] * 100).round(1)
                city_summary["Pct_Rec"]  = (city_summary["Error_Rec"]  / city_summary["actual_2025"] * 100).round(1)

                cities = city_summary.index.tolist()
                x = range(len(cities))
                width = 0.35

                fig, ax = plt.subplots(figsize=(14, 8))

                colors_base = ['#3498db' if v >= 0 else '#e67e22' for v in city_summary["Error_Base"]]
                colors_rec  = ['#2ecc71' if v >= 0 else '#e74c3c' for v in city_summary["Error_Rec"]]

                bars_base = ax.bar(
                    [xi - width / 2 for xi in x], city_summary["Error_Base"],
                    width=width, color=colors_base, alpha=0.85,
                    edgecolor='black', linewidth=0.5, label="Base Forecast Error"
                )
                bars_rec = ax.bar(
                    [xi + width / 2 for xi in x], city_summary["Error_Rec"],
                    width=width, color=colors_rec, alpha=0.85,
                    edgecolor='black', linewidth=0.5, label="Reconciled Forecast Error"
                )

                ax.axhline(0, color='black', linewidth=1, linestyle='--')
                fig.suptitle(f"METHOD: {display_name.upper()}", fontsize=18,
                             fontweight='bold', color='#1a5276', y=0.98)
                ax.set_title("Forecast Error 2025 per City (Predicted - Actual)\nBase vs. MinT-Reconciled",
                             fontsize=12, pad=10)
                ax.set_ylabel("Absolute Error (Units)", fontsize=12)
                ax.set_xlabel("City", fontsize=12)
                ax.set_xticks(list(x))
                ax.set_xticklabels(cities, rotation=45, ha='right')
                ax.grid(axis='y', linestyle=':', alpha=0.6)
                ax.legend(fontsize=11, loc='upper right')

                # Percentage labels
                for bar, pct in zip(bars_base, city_summary["Pct_Base"]):
                    yval = bar.get_height()
                    va = 'bottom' if yval >= 0 else 'top'
                    ax.text(bar.get_x() + bar.get_width() / 2, yval,
                            f"{pct}%", ha='center', va=va, fontsize=8, fontweight='bold', color='#2c3e50')
                for bar, pct in zip(bars_rec, city_summary["Pct_Rec"]):
                    yval = bar.get_height()
                    va = 'bottom' if yval >= 0 else 'top'
                    ax.text(bar.get_x() + bar.get_width() / 2, yval,
                            f"{pct}%", ha='center', va=va, fontsize=8, fontweight='bold', color='#1a5276')

                plt.tight_layout()

            else:
                # ── Standard single-bar plot ────────────────────────────────────
                city_summary["Error"] = city_summary["predicted_2025"] - city_summary["actual_2025"]
                city_summary["Pct_Error"] = (city_summary["Error"] / city_summary["actual_2025"] * 100).round(1)

                plt.figure(figsize=(12, 8))
                colors = ['#2ecc71' if x >= 0 else '#e74c3c' for x in city_summary["Error"]]

                bars = plt.bar(city_summary.index, city_summary["Error"],
                               color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

                plt.axhline(0, color='black', linewidth=1, linestyle='--')
                plt.suptitle(f"METHOD: {display_name.upper()}", fontsize=18,
                             fontweight='bold', color='#1a5276', y=0.98)
                plt.title("Forecast Error 2025 per City (Predicted - Actual)", fontsize=12, pad=10)
                plt.ylabel("Absolute Error (Units)", fontsize=12)
                plt.xlabel("City", fontsize=12)
                plt.xticks(rotation=45)
                plt.grid(axis='y', linestyle=':', alpha=0.6)

                for bar, pct in zip(bars, city_summary["Pct_Error"]):
                    yval = bar.get_height()
                    va = 'bottom' if yval >= 0 else 'top'
                    plt.text(bar.get_x() + bar.get_width() / 2, yval,
                             f"{pct}%", ha='center', va=va, fontsize=10, fontweight='bold')

                plt.tight_layout()

            # Save plot
            plot_filename = filename.replace(".xlsx", ".png")
            plot_path = os.path.join(plot_folder, plot_filename)
            plt.savefig(plot_path, dpi=300)
            plt.close()
            print(f"Generated plot: {plot_path}")


if __name__ == "__main__":
    generate_plots()

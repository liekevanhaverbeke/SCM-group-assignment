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
    "Holts": "holts",
    "Holts Reconciled": "holts_reconciled",
    "Linear Regression": "linear_regression",
    "Moving Average": "moving_average",
    "Simple Exp Smoothing": "simple_exp"
}

levels = {
    "Total Level": "total",
    "City Level": "city",
    "City/Product": "city_product",
    "Lowest Level (Size)": "city_product_size"
}

def generate_plots():
    for method_label, method_file in methods.items():
        for level_label, level_file in levels.items():
            filename = f"{method_file}_{level_file}.xlsx"
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
            except:
                print(f"Sheet Validatie_2025 not found in {filename}")
                continue
            
            # Aggregate per city
            city_summary = df_val.groupby("stad")[["actual_2025", "predicted_2025"]].sum()
            
            # Check for data presence
            if city_summary.empty or city_summary["actual_2025"].sum() == 0:
                print(f"Skipping empty or zero-data plot: {filename}")
                continue
            
            # Calculate Error (Predicted - Actual)
            city_summary["Error"] = city_summary["predicted_2025"] - city_summary["actual_2025"]
            city_summary["Pct_Error"] = (city_summary["Error"] / city_summary["actual_2025"] * 100).round(1)
            
            # Plotting
            plt.figure(figsize=(12, 8))
            colors = ['#2ecc71' if x >= 0 else '#e74c3c' for x in city_summary["Error"]]
            
            bars = plt.bar(city_summary.index, city_summary["Error"], color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
            
            plt.axhline(0, color='black', linewidth=1, linestyle='--')
            plt.suptitle(f"METHOD: {display_name.upper()}", fontsize=18, fontweight='bold', color='#1a5276', y=0.98)
            plt.title(f"Forecast Error 2025 per City (Predicted - Actual)", fontsize=12, pad=10)
            plt.ylabel("Absolute Error (Units)", fontsize=12)
            plt.xlabel("City", fontsize=12)
            plt.xticks(rotation=45)
            plt.grid(axis='y', linestyle=':', alpha=0.6)
            
            # Add labels
            for bar, pct in zip(bars, city_summary["Pct_Error"]):
                yval = bar.get_height()
                va = 'bottom' if yval >= 0 else 'top'
                plt.text(bar.get_x() + bar.get_width()/2, yval, f"{pct}%", ha='center', va=va, fontsize=10, fontweight='bold')

            plt.tight_layout()
            
            # Save plot
            plot_filename = filename.replace(".xlsx", ".png")
            plot_path = os.path.join(plot_folder, plot_filename)
            plt.savefig(plot_path, dpi=300)
            plt.close()
            print(f"Generated plot: {plot_path}")

if __name__ == "__main__":
    generate_plots()

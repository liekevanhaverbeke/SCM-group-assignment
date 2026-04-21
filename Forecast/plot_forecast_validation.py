import pandas as pd
import matplotlib.pyplot as plt
import os

# Settings
forecast_folder = "Forecast"
files = {
    "Holts Method": "forecast_holts_method_2026.xlsx",
    "Simple Exponential Smoothing": "forecast_simple_exp_smoothing_2026.xlsx",
    "Linear Regression": "forecast_linear_regression_2026.xlsx",
    "Moving Average": "forecast_moving_average_2026.xlsx"
}

def generate_plots():
    for method_name, filename in files.items():
        file_path = os.path.join(forecast_folder, filename)
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
        
        # Load the validation data
        df_val = pd.read_excel(file_path, sheet_name="Validatie_2025")
        
        # Aggregate per city
        city_summary = df_val.groupby("stad")[["actual_2025", "predicted_2025"]].sum()
        
        # Calculate Error (Predicted - Actual)
        # Positive = Over-forecasted (Above)
        # Negative = Under-forecasted (Below)
        city_summary["Error"] = city_summary["predicted_2025"] - city_summary["actual_2025"]
        city_summary["Pct_Error"] = (city_summary["Error"] / city_summary["actual_2025"] * 100).round(1)
        
        # Plotting
        plt.figure(figsize=(12, 7))
        
        # Colors: Blue for above, Red for below
        colors = ['#2ecc71' if x >= 0 else '#e74c3c' for x in city_summary["Error"]]
        
        bars = plt.bar(city_summary.index, city_summary["Error"], color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        plt.axhline(0, color='black', linewidth=1, linestyle='--')
        plt.title(f"Forecast Error 2025 per City - {method_name}\n(Predicted - Actual)", fontsize=14, fontweight='bold', pad=20)
        plt.ylabel("Absolute Error (Units)", fontsize=12)
        plt.xlabel("City", fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle=':', alpha=0.6)
        
        # Add labels on top/bottom of bars
        for bar, pct in zip(bars, city_summary["Pct_Error"]):
            yval = bar.get_height()
            va = 'bottom' if yval >= 0 else 'top'
            plt.text(bar.get_x() + bar.get_width()/2, yval, f"{pct}%", ha='center', va=va, fontsize=10, fontweight='bold')

        plt.tight_layout()
        
        # Save plot
        plot_filename = f"forecast_error_2025_{method_name.lower().replace(' ', '_')}.png"
        plot_path = os.path.join(forecast_folder, plot_filename)
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Generated plot: {plot_path}")

if __name__ == "__main__":
    generate_plots()

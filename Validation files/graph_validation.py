import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Load data ────────────────────────────────────────────────────────────────
df = pd.read_excel("Validation files/metrics.xlsx", sheet_name="Sheet1", header=None)
df.columns = ["col0", "MAE", "MSE", "MAPE", "MPE"]

# ── Model name + row indices of the 6 data rows (2020-2025, excl. header/average) ──
models = {
    "Hierarchical Hybrid Model":                    (2,  7),
    "Baseline Model":                               (11, 16),
    "Top-Down Linear Regression Model":             (20, 25),
    "Top-Down Linear Reconciliation Model (MinT)":  (29, 34),
    "Product-City Level SES Model":                 (38, 43),
}

seasons = [2020, 2021, 2022, 2023, 2024, 2025]

data = {}
for model, (start, end) in models.items():
    mae_values = pd.to_numeric(df.loc[start:end, "MAE"], errors="coerce").dropna().tolist()
    data[model] = mae_values

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5.5))

colors     = ["#1f4e79", "#c00000", "#ed7d31", "#ffc000", "#70ad47"]
markers    = ["o", "s", "^", "D", "v"]
linewidths = [2.5, 1.5, 1.5, 1.5, 1.5]
linestyles = ["-", "--", "--", "--", "--"]

for (model, mae), color, marker, lw, ls in zip(
    data.items(), colors, markers, linewidths, linestyles
):
    ax.plot(
        seasons, mae,
        label=model,
        color=color,
        marker=marker,
        linewidth=lw,
        linestyle=ls,
        markersize=6,
    )

# ── Formatting ───────────────────────────────────────────────────────────────
ax.set_title(
    "Mean Absolute Error (MAE) per Year by Forecast Model",
    fontsize=13, fontweight="bold", pad=14
)
ax.set_xlabel("Season", fontsize=11)
ax.set_ylabel("MAE (units)", fontsize=11)
ax.set_xticks(seasons)
ax.set_ylim(4, 11)
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
ax.grid(axis="y", which="minor", linestyle=":", linewidth=0.4, alpha=0.4)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(fontsize=9, frameon=False, loc="upper left")

plt.tight_layout()
plt.savefig("mae_over_time.png", dpi=180, bbox_inches="tight")
print("Saved: mae_over_time.png")
plt.show()
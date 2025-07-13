import numpy as np
import pandas as pd

# Parameters
investment_amount = 1000  # Invest $1000 every 2 weeks
period_days = 14

# Load simulated price paths
df = pd.read_csv("Investment Strategy Analasis/Bitcoin Analysis/monte_carlo_paths.csv")
paths = df.values  # shape: (n_paths, n_steps)
n_paths, n_steps = paths.shape

# Indices for every 2 weeks
dca_indices = np.arange(0, n_steps, period_days)

# Time horizons in years
horizons = [1, 3, 5, 10]
steps_per_year = 365

summary_lines = []

for horizon in horizons:
    steps = int(horizon * steps_per_year)
    dca_idx = dca_indices[dca_indices < steps]
    final_values = []
    total_invested = []
    total_btc = []
    for i in range(n_paths):
        prices = paths[i][:steps]
        btc = 0
        invested = 0
        for idx in dca_idx:
            price = prices[int(idx)]
            if price <= 0 or np.isnan(price):
                continue  # skip invalid prices
            btc += investment_amount / price
            invested += investment_amount
        final_value = btc * prices[-1]
        final_values.append(final_value)
        total_invested.append(invested)
        total_btc.append(btc)
    final_values = np.array(final_values)
    total_invested = np.array(total_invested)
    roi = (final_values - total_invested) / total_invested * 100
    # Annualized return (CAGR)
    cagr = (final_values / total_invested) ** (1/horizon) - 1
    summary_lines.append(f"\n==== {horizon}-Year DCA Results ====")
    summary_lines.append(f"  Mean final value: ${final_values.mean():,.2f}")
    summary_lines.append(f"  Median final value: ${np.median(final_values):,.2f}")
    summary_lines.append(f"  10th percentile: ${np.percentile(final_values, 10):,.2f}")
    summary_lines.append(f"  90th percentile: ${np.percentile(final_values, 90):,.2f}")
    summary_lines.append(f"  Mean ROI: {roi.mean():.2f}%")
    summary_lines.append(f"  Median ROI: {np.median(roi):.2f}%")
    summary_lines.append(f"  Mean CAGR: {cagr.mean():.2%}")
    summary_lines.append(f"  Median CAGR: {np.median(cagr):.2%}")

summary_path = "Investment Strategy Analasis/Bitcoin Analysis/bitcoin_dca_horizon_summary.txt"
with open(summary_path, "w") as f:
    for line in summary_lines:
        f.write(line + "\n")
print(f"Multi-horizon summary statistics saved to {summary_path}") 
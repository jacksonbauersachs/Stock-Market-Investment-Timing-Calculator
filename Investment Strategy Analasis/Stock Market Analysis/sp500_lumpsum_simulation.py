import numpy as np
import pandas as pd

# Parameters
lump_sum = 100000  # $100,000 invested at t=0
dividend_yield = 0.015  # 1.5% annual dividend yield
period_days = 14
periods_per_year = 365 // period_days

# Load simulated price paths
df = pd.read_csv("Investment Strategy Analasis/Stock Market Analysis/sp500_monte_carlo_paths.csv")
paths = df.values  # shape: (n_paths, n_steps)
n_paths, n_steps = paths.shape

# Time horizons in years
horizons = [1, 3, 5, 10]
steps_per_year = 365

summary_lines = []

for horizon in horizons:
    steps = int(horizon * steps_per_year)
    final_values = []
    for i in range(n_paths):
        prices = paths[i][:steps]
        if prices[0] <= 0 or np.isnan(prices[0]):
            continue
        shares = lump_sum / prices[0]
        # Reinvest dividends every period_days (every 2 weeks)
        for t in range(0, steps, period_days):
            shares += shares * (dividend_yield / periods_per_year)
        final_value = shares * prices[-1]
        final_values.append(final_value)
    final_values = np.array(final_values)
    roi = (final_values - lump_sum) / lump_sum * 100
    cagr = (final_values / lump_sum) ** (1/horizon) - 1
    summary_lines.append(f"\n==== {horizon}-Year Lump Sum Results ====")
    summary_lines.append(f"  Mean final value: ${final_values.mean():,.2f}")
    summary_lines.append(f"  Median final value: ${np.median(final_values):,.2f}")
    summary_lines.append(f"  10th percentile: ${np.percentile(final_values, 10):,.2f}")
    summary_lines.append(f"  90th percentile: ${np.percentile(final_values, 90):,.2f}")
    summary_lines.append(f"  Mean ROI: {roi.mean():.2f}%")
    summary_lines.append(f"  Median ROI: {np.median(roi):.2f}%")
    summary_lines.append(f"  Mean CAGR: {cagr.mean():.2%}")
    summary_lines.append(f"  Median CAGR: {np.median(cagr):.2%}")

summary_path = "Investment Strategy Analasis/Stock Market Analysis/sp500_lumpsum_horizon_summary.txt"
with open(summary_path, "w") as f:
    for line in summary_lines:
        f.write(line + "\n")
print(f"Lump sum summary statistics saved to {summary_path}") 
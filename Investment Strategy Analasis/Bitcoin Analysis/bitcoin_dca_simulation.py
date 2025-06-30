import numpy as np
import pandas as pd

# Parameters
investment_per_period = 1000  # $1,000 every 2 weeks
period_days = 14
initial_cash = 0  # No lump sum, just DCA

# Load simulated price paths
df = pd.read_csv("Investment Strategy Analasis/Bitcoin Analysis/monte_carlo_paths.csv")
paths = df.values  # shape: (n_paths, n_steps)
n_paths, n_steps = paths.shape

# Time grid (assume daily steps as in simulation)
steps_per_year = 365
years = n_steps / steps_per_year

# Indices for every 2 weeks
dca_indices = np.arange(0, n_steps, period_days)

# Results storage
final_values = []
total_invested = []
total_btc = []

for i in range(n_paths):
    prices = paths[i]
    btc = 0
    invested = 0
    for idx in dca_indices:
        price = prices[int(idx)]
        if price <= 0 or np.isnan(price):
            continue  # skip invalid prices
        btc += investment_per_period / price
        invested += investment_per_period
    final_value = btc * prices[-1]
    final_values.append(final_value)
    total_invested.append(invested)
    total_btc.append(btc)

# Summary statistics
final_values = np.array(final_values)
total_invested = np.array(total_invested)
roi = (final_values - total_invested) / total_invested * 100

print("\nAll-In DCA Strategy Results (across all Monte Carlo paths):")
print(f"  Mean final value: ${final_values.mean():,.2f}")
print(f"  Median final value: ${np.median(final_values):,.2f}")
print(f"  Mean ROI: {roi.mean():.2f}%")
print(f"  Median ROI: {np.median(roi):.2f}%")
print(f"  10th percentile final value: ${np.percentile(final_values, 10):,.2f}")
print(f"  90th percentile final value: ${np.percentile(final_values, 90):,.2f}")

# Frequency of outcomes in ROI bands
bands = [(-100, 0), (0, 50), (50, 100), (100, 200), (200, 500), (500, 1000), (1000, 10000)]
print("\nFrequency of outcomes by ROI band:")
for low, high in bands:
    count = np.sum((roi >= low) & (roi < high))
    print(f"  {low:>4}% to {high:>5}%: {count} paths ({count/n_paths:.1%})")
count = np.sum(roi >= 10000)
print(f"  >=10000%: {count} paths ({count/n_paths:.1%})")

# Save results for further analysis
out_path = "Investment Strategy Analasis/Bitcoin Analysis/dca_simulation_results.csv"
pd.DataFrame({
    'final_value': final_values,
    'total_invested': total_invested,
    'total_btc': total_btc,
    'roi_percent': roi
}).to_csv(out_path, index=False)
print(f"DCA simulation results saved to {out_path}")

# Save summary to file
summary_lines = []
summary_lines.append("All-In DCA Strategy Results (across all Monte Carlo paths):")
summary_lines.append(f"  Mean final value: ${final_values.mean():,.2f}")
summary_lines.append(f"  Median final value: ${np.median(final_values):,.2f}")
summary_lines.append(f"  Mean ROI: {roi.mean():.2f}%")
summary_lines.append(f"  Median ROI: {np.median(roi):.2f}%")
summary_lines.append(f"  10th percentile final value: ${np.percentile(final_values, 10):,.2f}")
summary_lines.append(f"  90th percentile final value: ${np.percentile(final_values, 90):,.2f}")

bands = [(-100, 0), (0, 50), (50, 100), (100, 200), (200, 500), (500, 1000), (1000, 10000)]
summary_lines.append("\nFrequency of outcomes by ROI band:")
for low, high in bands:
    count = np.sum((roi >= low) & (roi < high))
    summary_lines.append(f"  {low:>4}% to {high:>5}%: {count} paths ({count/n_paths:.1%})")
count = np.sum(roi >= 10000)
summary_lines.append(f"  >=10000%: {count} paths ({count/n_paths:.1%})")

summary_path = "Investment Strategy Analasis/Bitcoin Analysis/bitcoin_dca_summary.txt"
with open(summary_path, "w") as f:
    for line in summary_lines:
        f.write(line + "\n")
print(f"Summary statistics saved to {summary_path}") 
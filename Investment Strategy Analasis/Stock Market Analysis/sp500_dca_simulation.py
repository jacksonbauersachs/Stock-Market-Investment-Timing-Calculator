import numpy as np
import pandas as pd

# Parameters
investment_per_period = 1000  # $1,000 every 2 weeks
period_days = 14
dividend_yield = 0.015  # 1.5% annual dividend yield
periods_per_year = 365 // period_days

# Load simulated price paths
df = pd.read_csv("Investment Strategy Analasis/Stock Market Analysis/sp500_monte_carlo_paths.csv")
paths = df.values  # shape: (n_paths, n_steps)
n_paths, n_steps = paths.shape

# Indices for every 2 weeks
dca_indices = np.arange(0, n_steps, period_days)

# Results storage
final_values = []
total_invested = []
total_shares = []

for i in range(n_paths):
    prices = paths[i]
    shares = 0
    invested = 0
    for idx in dca_indices:
        price = prices[int(idx)]
        if price <= 0 or np.isnan(price):
            continue  # skip invalid prices
        shares += investment_per_period / price
        invested += investment_per_period
        # Add dividend shares (reinvested)
        shares += shares * (dividend_yield / periods_per_year)
    final_value = shares * prices[-1]
    final_values.append(final_value)
    total_invested.append(invested)
    total_shares.append(shares)

# Summary statistics
final_values = np.array(final_values)
total_invested = np.array(total_invested)
roi = (final_values - total_invested) / total_invested * 100

print("\nAll-In DCA Strategy Results (S&P 500, Monte Carlo):")
print(f"  Mean final value: ${final_values.mean():,.2f}")
print(f"  Median final value: ${np.median(final_values):,.2f}")
print(f"  Std final value: ${final_values.std():,.2f}")
print(f"  Min final value: ${final_values.min():,.2f}")
print(f"  Max final value: ${final_values.max():,.2f}")
print(f"  10th percentile: ${np.percentile(final_values, 10):,.2f}")
print(f"  90th percentile: ${np.percentile(final_values, 90):,.2f}")
print(f"  Mean ROI: {roi.mean():.2f}%")
print(f"  Median ROI: {np.median(roi):.2f}%")
print(f"  Std ROI: {roi.std():.2f}%")
print(f"  Min ROI: {roi.min():.2f}%")
print(f"  Max ROI: {roi.max():.2f}%")
print(f"  10th percentile ROI: {np.percentile(roi, 10):.2f}%")
print(f"  90th percentile ROI: {np.percentile(roi, 90):.2f}%")

# Frequency of outcomes in ROI bands
bands = [(-100, 0), (0, 50), (50, 100), (100, 200), (200, 500), (500, 1000), (1000, 10000)]
print("\nFrequency of outcomes by ROI band:")
for low, high in bands:
    count = np.sum((roi >= low) & (roi < high))
    print(f"  {low:>4}% to {high:>5}%: {count} paths ({count/n_paths:.1%})")
count = np.sum(roi >= 10000)
print(f"  >=10000%: {count} paths ({count/n_paths:.1%})")

# Save results for further analysis
out_path = "Investment Strategy Analasis/Stock Market Analysis/sp500_dca_simulation_results.csv"
pd.DataFrame({
    'final_value': final_values,
    'total_invested': total_invested,
    'total_shares': total_shares,
    'roi_percent': roi
}).to_csv(out_path, index=False)
print(f"DCA simulation results saved to {out_path}")

summary_lines = []
summary_lines.append("All-In DCA Strategy Results (S&P 500, Monte Carlo):")
summary_lines.append(f"  Mean final value: ${final_values.mean():,.2f}")
summary_lines.append(f"  Median final value: ${np.median(final_values):,.2f}")
summary_lines.append(f"  Std final value: ${final_values.std():,.2f}")
summary_lines.append(f"  Min final value: ${final_values.min():,.2f}")
summary_lines.append(f"  Max final value: ${final_values.max():,.2f}")
summary_lines.append(f"  10th percentile: ${np.percentile(final_values, 10):,.2f}")
summary_lines.append(f"  90th percentile: ${np.percentile(final_values, 90):,.2f}")
summary_lines.append(f"  Mean ROI: {roi.mean():.2f}%")
summary_lines.append(f"  Median ROI: {np.median(roi):.2f}%")
summary_lines.append(f"  Std ROI: {roi.std():.2f}%")
summary_lines.append(f"  Min ROI: {roi.min():.2f}%")
summary_lines.append(f"  Max ROI: {roi.max():.2f}%")
summary_lines.append(f"  10th percentile ROI: {np.percentile(roi, 10):.2f}%")
summary_lines.append(f"  90th percentile ROI: {np.percentile(roi, 90):.2f}%")

bands = [(-100, 0), (0, 50), (50, 100), (100, 200), (200, 500), (500, 1000), (1000, 10000)]
summary_lines.append("\nFrequency of outcomes by ROI band:")
for low, high in bands:
    count = np.sum((roi >= low) & (roi < high))
    summary_lines.append(f"  {low:>4}% to {high:>5}%: {count} paths ({count/n_paths:.1%})")
count = np.sum(roi >= 10000)
summary_lines.append(f"  >=10000%: {count} paths ({count/n_paths:.1%})")

# Save summary to file
summary_path = "Investment Strategy Analasis/Stock Market Analysis/sp500_dca_summary.txt"
with open(summary_path, "w") as f:
    for line in summary_lines:
        f.write(line + "\n")
print(f"Summary statistics saved to {summary_path}") 
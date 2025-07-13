import numpy as np
import pandas as pd

# Parameters
investment_per_period = 1000  # $1,000 every 2 weeks
period_days = 14
reserve_ratio = 0.2  # 20% to reserve
regular_ratio = 0.8  # 80% regular DCA
reserve_dip_threshold = 0.10  # 10% below ATH triggers reserve deployment

# Load simulated price paths
df = pd.read_csv("Investment Strategy Analasis/Bitcoin Analysis/monte_carlo_paths.csv")
paths = df.values  # shape: (n_paths, n_steps)
n_paths, n_steps = paths.shape

dca_indices = np.arange(0, n_steps, period_days)

final_values = []
total_invested = []
total_btc = []
reserve_deployments = []

for i in range(n_paths):
    prices = paths[i]
    btc = 0
    invested = 0
    reserve = 0
    ath = prices[0]
    deploy_count = 0
    for idx in dca_indices:
        price = prices[int(idx)]
        if price <= 0 or np.isnan(price):
            continue
        # Update ATH
        if price > ath:
            ath = price
        # Regular DCA
        regular_investment = investment_per_period * regular_ratio
        reserve += investment_per_period * reserve_ratio
        invested += investment_per_period
        btc += regular_investment / price
        # Deploy reserve if price is below threshold
        dip = (ath - price) / ath
        if dip >= reserve_dip_threshold and reserve > 0:
            btc += reserve / price
            reserve = 0
            deploy_count += 1
    final_value = btc * prices[-1]
    final_values.append(final_value)
    total_invested.append(invested)
    total_btc.append(btc)
    reserve_deployments.append(deploy_count)

final_values = np.array(final_values)
total_invested = np.array(total_invested)
roi = (final_values - total_invested) / total_invested * 100
reserve_deployments = np.array(reserve_deployments)

print("\nReserve DCA Strategy Results (Bitcoin, Monte Carlo):")
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
print(f"  Mean reserve deployments: {reserve_deployments.mean():.2f}")

bands = [(-100, 0), (0, 50), (50, 100), (100, 200), (200, 500), (500, 1000), (1000, 10000)]
print("\nFrequency of outcomes by ROI band:")
for low, high in bands:
    count = np.sum((roi >= low) & (roi < high))
    print(f"  {low:>4}% to {high:>5}%: {count} paths ({count/n_paths:.1%})")
count = np.sum(roi >= 10000)
print(f"  >=10000%: {count} paths ({count/n_paths:.1%})")

# Save results for further analysis
out_path = "Investment Strategy Analasis/Bitcoin Analysis/bitcoin_reserve_dca_simulation_results.csv"
pd.DataFrame({
    'final_value': final_values,
    'total_invested': total_invested,
    'total_btc': total_btc,
    'roi_percent': roi,
    'reserve_deployments': reserve_deployments
}).to_csv(out_path, index=False)
print(f"Reserve DCA simulation results saved to {out_path}")

# Save summary to file
summary_lines = []
summary_lines.append("Reserve DCA Strategy Results (Bitcoin, Monte Carlo):")
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
summary_lines.append(f"  Mean reserve deployments: {reserve_deployments.mean():.2f}")

bands = [(-100, 0), (0, 50), (50, 100), (100, 200), (200, 500), (500, 1000), (1000, 10000)]
summary_lines.append("\nFrequency of outcomes by ROI band:")
for low, high in bands:
    count = np.sum((roi >= low) & (roi < high))
    summary_lines.append(f"  {low:>4}% to {high:>5}%: {count} paths ({count/n_paths:.1%})")
count = np.sum(roi >= 10000)
summary_lines.append(f"  >=10000%: {count} paths ({count/n_paths:.1%})")

summary_path = "Investment Strategy Analasis/Bitcoin Analysis/bitcoin_reserve_dca_summary.txt"
with open(summary_path, "w") as f:
    for line in summary_lines:
        f.write(line + "\n")
print(f"Summary statistics saved to {summary_path}") 
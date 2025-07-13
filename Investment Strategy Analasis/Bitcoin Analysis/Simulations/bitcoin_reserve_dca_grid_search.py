import numpy as np
import pandas as pd

# Parameters
investment_per_period = 1000  # $1,000 every 2 weeks
period_days = 14
n_paths = 500  # Use only 500 Monte Carlo paths for speed

# Reserve ratios and tier structures to try
reserve_ratios = [0.1, 0.2, 0.3]
tier_structures = [
    # (thresholds, deploy_ratios)
    ([0.05, 0.10, 0.15], [0.2, 0.3, 0.5]),
    ([0.10, 0.20, 0.30], [0.2, 0.3, 0.5]),
    ([0.05, 0.15, 0.25], [0.25, 0.35, 0.40]),
]

df = pd.read_csv("Investment Strategy Analasis/Bitcoin Analysis/monte_carlo_paths.csv")
paths = df.values[:n_paths]  # shape: (n_paths, n_steps)
n_paths, n_steps = paths.shape

dca_indices = np.arange(0, n_steps, period_days)

results = []

total_combos = len(reserve_ratios) * len(tier_structures)
combo_num = 0
for reserve_ratio in reserve_ratios:
    regular_ratio = 1 - reserve_ratio
    for thresholds, deploy_ratios in tier_structures:
        combo_num += 1
        print(f"\nRunning combo {combo_num}/{total_combos}: reserve_ratio={reserve_ratio}, thresholds={thresholds}, deploy_ratios={deploy_ratios}")
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
                if price > ath:
                    ath = price
                # Regular DCA
                regular_investment = investment_per_period * regular_ratio
                reserve += investment_per_period * reserve_ratio
                invested += investment_per_period
                btc += regular_investment / price
                # Tiered reserve deployment
                dip = (ath - price) / ath
                remaining_reserve = reserve
                for t, r in zip(thresholds, deploy_ratios):
                    if dip >= t and remaining_reserve > 0:
                        deploy_amt = min(remaining_reserve, reserve * r)
                        btc += deploy_amt / price
                        reserve -= deploy_amt
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
        median_final = np.median(final_values)
        median_roi = np.median(roi)
        mean_final = np.mean(final_values)
        mean_roi = np.mean(roi)
        results.append({
            'reserve_ratio': reserve_ratio,
            'thresholds': thresholds,
            'deploy_ratios': deploy_ratios,
            'median_final': median_final,
            'median_roi': median_roi,
            'mean_final': mean_final,
            'mean_roi': mean_roi,
            'mean_deployments': reserve_deployments.mean(),
        })
        print(f"  Median final value: ${median_final:,.2f}, Median ROI: {median_roi:.2f}%, Mean deployments: {reserve_deployments.mean():.2f}")

# Sort and save summary
results = sorted(results, key=lambda x: x['median_final'], reverse=True)
summary_path = "Investment Strategy Analasis/Bitcoin Analysis/bitcoin_reserve_dca_grid_summary.txt"
with open(summary_path, "w") as f:
    f.write("Reserve DCA Grid Search Results (Bitcoin, Monte Carlo)\n")
    f.write("Sorted by median final value (desc)\n\n")
    for r in results:
        f.write(f"Reserve ratio: {r['reserve_ratio']}, thresholds: {r['thresholds']}, deploy_ratios: {r['deploy_ratios']}\n")
        f.write(f"  Median final value: ${r['median_final']:,.2f}, Median ROI: {r['median_roi']:.2f}%, Mean deployments: {r['mean_deployments']:.2f}\n")
        f.write(f"  Mean final value: ${r['mean_final']:,.2f}, Mean ROI: {r['mean_roi']:.2f}%\n\n")
print(f"Grid search summary saved to {summary_path}") 
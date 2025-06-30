import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === USER PARAMETERS ===
# Growth model: price = 125.86 * exp(0.0818 * years)
growth_a = 125.86
growth_b = 0.0818

# Volatility model (30d, 3rd order polynomial)
# Vol_30d: Polynomial 3rd, params=[-4.46905154e-06, 2.59644502e-04, -3.00132646e-03, 1.80174615e-01]
vol_p3 = [-4.46905154e-06, 2.59644502e-04, -3.00132646e-03, 1.80174615e-01]

def growth_model(years):
    return growth_a * np.exp(growth_b * years)

def volatility_model(years):
    # 3rd order polynomial
    return (
        vol_p3[0] * years**3 +
        vol_p3[1] * years**2 +
        vol_p3[2] * years +
        vol_p3[3]
    )

# Simulation settings
years = 10
n_paths = 1000
steps_per_year = 365  # daily steps

# Get initial price from latest S&P 500 data
sp500_df = pd.read_csv("Data Sets/S&P 500 Data Sets/S&P 500 Total Data Cleaned 2.csv")
initial_price = sp500_df['Price'].iloc[-1]

# Time grid
total_steps = years * steps_per_year
all_years = np.linspace(1e-6, years, int(total_steps))  # avoid log(0)
dt = 1 / steps_per_year

# Precompute expected price and volatility for each step
expected_prices = growth_model(all_years)
vols = volatility_model(all_years)

# Simulate paths
print(f"Simulating {n_paths} S&P 500 paths for {years} years...")
paths = np.zeros((n_paths, len(all_years)))
paths[:, 0] = initial_price

for i in range(n_paths):
    if i % max(1, n_paths // 10) == 0:
        print(f"  Path {i+1}/{n_paths}")
    for t in range(1, len(all_years)):
        # Annualized volatility to per-step stddev
        step_vol = vols[t] * np.sqrt(dt)
        # Simulate log-return
        rand_return = np.random.normal(0, step_vol)
        paths[i, t] = paths[i, t-1] * np.exp(rand_return)

# Save to CSV
out_path = "Investment Strategy Analasis/Stock Market Analysis/sp500_monte_carlo_paths.csv"
pd.DataFrame(paths, columns=[f"Year_{y:.2f}" for y in all_years]).to_csv(out_path, index=False)
print(f"Simulated S&P 500 paths saved to {out_path}")

# Plot a sample of the paths
plt.figure(figsize=(12, 6))
for i in range(min(50, n_paths)):
    plt.plot(all_years, paths[i], alpha=0.2, color='blue')
plt.plot(all_years, expected_prices * initial_price / expected_prices[0], color='red', linewidth=2, label='Growth Model')
plt.xlabel('Years')
plt.ylabel('S&P 500 Price')
plt.title('Monte Carlo Simulated S&P 500 Price Paths')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Print summary statistics
final_prices = paths[:, -1]
print(f"\nSummary after {years} years:")
print(f"  Mean final price: ${final_prices.mean():,.2f}")
print(f"  Median final price: ${np.median(final_prices):,.2f}")
print(f"  10th percentile: ${np.percentile(final_prices, 10):,.2f}")
print(f"  90th percentile: ${np.percentile(final_prices, 90):,.2f}") 
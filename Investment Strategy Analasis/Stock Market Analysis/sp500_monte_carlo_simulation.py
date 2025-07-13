import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

# === USER PARAMETERS ===
# Load S&P 500 data
sp500_df = pd.read_csv("Data Sets/S&P 500 Data Sets/S&P 500 Total Data Cleaned 2.csv")
sp500_df['Date'] = pd.to_datetime(sp500_df['Date'])
sp500_df = sp500_df.sort_values('Date')
sp500_df['Days'] = (sp500_df['Date'] - sp500_df['Date'].min()).dt.days
sp500_df['Years'] = sp500_df['Days'] / 365.25
sp500_df['Price'] = pd.to_numeric(sp500_df['Price'], errors='coerce')

# Fit log(price) = a + b * years
log_prices = np.log(sp500_df['Price'])
years = sp500_df['Years']
mask = np.isfinite(log_prices)
reg = linregress(years[mask], log_prices[mask])
log_a, log_b = reg.intercept, reg.slope
print(f"Fitted log(price) = {log_a:.5f} + {log_b:.5f} * years (R^2={reg.rvalue**2:.4f})")

# Volatility model (30d, 3rd order polynomial)
vol_p3 = [-4.46905154e-06, 2.59644502e-04, -3.00132646e-03, 1.80174615e-01]

def growth_model(years):
    return np.exp(log_a + log_b * years)

def volatility_model(years):
    p = vol_p3
    return p[0]*years**3 + p[1]*years**2 + p[2]*years + p[3]

# Simulation settings
years_sim = 10
n_paths = 1000
steps_per_year = 365  # daily steps

initial_price = sp500_df['Price'].iloc[-1]

total_steps = years_sim * steps_per_year
all_years = np.linspace(1e-6, years_sim, int(total_steps))  # avoid log(0)
dt = 1 / steps_per_year

expected_prices = growth_model(all_years)
vols = volatility_model(all_years)

# Simulate paths
print(f"Simulating {n_paths} S&P 500 paths for {years_sim} years...")
paths = np.zeros((n_paths, len(all_years)))
paths[:, 0] = initial_price

for i in range(n_paths):
    if i % max(1, n_paths // 10) == 0:
        print(f"  Path {i+1}/{n_paths}")
    for t in range(1, len(all_years)):
        # Use log_b as drift (log-CAGR)
        step_vol = vols[t] * np.sqrt(dt)
        drift = (log_b - 0.5 * vols[t]**2) * dt
        rand_return = drift + step_vol * np.random.normal()
        paths[i, t] = max(paths[i, t-1] * np.exp(rand_return), 1e-8)

out_path = "Investment Strategy Analasis/Stock Market Analysis/sp500_monte_carlo_paths.csv"
pd.DataFrame(paths, columns=[f"Year_{y:.2f}" for y in all_years]).to_csv(out_path, index=False)
print(f"Simulated S&P 500 paths saved to {out_path}")

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
print(f"\nSummary after {years_sim} years:")
print(f"  Mean final price: ${final_prices.mean():,.2f}")
print(f"  Median final price: ${np.median(final_prices):,.2f}")
print(f"  10th percentile: ${np.percentile(final_prices, 10):,.2f}")
print(f"  90th percentile: ${np.percentile(final_prices, 90):,.2f}") 
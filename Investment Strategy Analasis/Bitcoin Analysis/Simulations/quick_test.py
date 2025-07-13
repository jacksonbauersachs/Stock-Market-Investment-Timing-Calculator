import numpy as np
from bitcoin_monte_carlo_fixed import run_bitcoin_lump_sum_monte_carlo

# Bitcoin Growth Model coefficients
a = 1.6329135221917355
b = -9.328646304661454

def bitcoin_growth_model(days):
    return 10**(a * np.log(days) + b)

def calculate_growth_rate(start_day, end_day):
    start_price = bitcoin_growth_model(start_day)
    end_price = bitcoin_growth_model(end_day)
    return end_price / start_price

# Test 1-year case
print("QUICK TEST: 1-Year Monte Carlo vs Pure Growth")
print("=" * 50)

initial_investment = 100000
current_day = 5439
years = 1

# Pure Growth Model
days_in_horizon = int(years * 365.25)
growth_multiple = calculate_growth_rate(current_day, current_day + days_in_horizon)
pure_growth_value = initial_investment * growth_multiple

print(f"Pure Growth Model: ${pure_growth_value:,.0f}")

# Monte Carlo (reduced paths for speed)
print("Running Monte Carlo...")
results, _ = run_bitcoin_lump_sum_monte_carlo(
    initial_investment=initial_investment,
    years=years,
    n_paths=1000
)

mc_mean = results['mean_final_value']
mc_median = results['median_final_value']

mean_diff = (mc_mean - pure_growth_value) / pure_growth_value * 100
median_diff = (mc_median - pure_growth_value) / pure_growth_value * 100

print(f"Monte Carlo Mean:  ${mc_mean:,.0f} (diff: {mean_diff:+.2f}%)")
print(f"Monte Carlo Median: ${mc_median:,.0f} (diff: {median_diff:+.2f}%)")

if abs(mean_diff) < 2.0:
    print("✓ SUCCESS: Means match within 2%")
else:
    print("✗ ISSUE: Means differ by more than 2%") 
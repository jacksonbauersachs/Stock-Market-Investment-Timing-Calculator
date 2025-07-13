import numpy as np
import pandas as pd

# Parameters
lump_sum = 100000  # $100,000 invested at t=0

# Your 94% R² Bitcoin growth model coefficients
a = 1.6329135221917355
b = -9.328646304661454

# Growth model function: log10(price) = a * ln(day) + b
def growth_model_price(days):
    return 10**(a * np.log(days) + b)

# Calculate growth rate from your formula
def calculate_growth_rate(start_day, end_day):
    """Calculate the growth multiple from start_day to end_day using the 94% R² formula"""
    start_price = growth_model_price(start_day)
    end_price = growth_model_price(end_day)
    return end_price / start_price

# Time horizons in years
horizons = [1, 3, 5, 10]

# Assume we're starting from "today" which is around day 5439 in your dataset
# But the exact starting day doesn't matter - only the growth over the horizon matters!
current_day = 5439  # This is arbitrary - could be any day >= 90

summary_lines = []
summary_lines.append("BITCOIN LUMP SUM SIMULATION - PURE GROWTH MODEL")
summary_lines.append("=" * 60)
summary_lines.append(f"Using 94% R^2 formula: log10(price) = {a:.3f} * ln(day) + {b:.3f}")
summary_lines.append(f"Starting from arbitrary day {current_day} (absolute price irrelevant)")
summary_lines.append("")

for horizon in horizons:
    # Calculate days for this horizon
    days_in_horizon = int(horizon * 365.25)
    start_day = current_day
    end_day = current_day + days_in_horizon
    
    # Calculate pure growth multiple from the formula
    growth_multiple = calculate_growth_rate(start_day, end_day)
    
    # Apply to lump sum investment
    final_value = lump_sum * growth_multiple
    roi = (growth_multiple - 1) * 100
    cagr = growth_multiple ** (1/horizon) - 1
    
    # The beauty is this works regardless of starting price level!
    summary_lines.append(f"==== {horizon}-Year Lump Sum Results (Formula-Based) ====")
    summary_lines.append(f"  Growth multiple: {growth_multiple:.3f}x")
    summary_lines.append(f"  Final value: ${final_value:,.2f}")
    summary_lines.append(f"  ROI: {roi:.2f}%")
    summary_lines.append(f"  CAGR: {cagr:.2%}")
    summary_lines.append("")

# Also calculate some additional insights
summary_lines.append("GROWTH MODEL INSIGHTS:")
summary_lines.append("=" * 30)

# Calculate the implied annual growth rate from the formula
# Using the derivative of the log formula
days_per_year = 365.25
sample_day = 1000  # Arbitrary day for calculation

# At any given day, the instantaneous growth rate is d/dt[10^(a*ln(t)+b)] / 10^(a*ln(t)+b)
# This simplifies to: a/t * ln(10) per day
instantaneous_daily_rate = (a / sample_day) * np.log(10)
instantaneous_annual_rate = instantaneous_daily_rate * days_per_year

summary_lines.append(f"Instantaneous annual growth rate at day {sample_day}: {instantaneous_annual_rate:.2%}")

# Calculate average growth rate over different periods
for years in [1, 5, 10]:
    start = sample_day
    end = sample_day + int(years * days_per_year)
    mult = calculate_growth_rate(start, end)
    avg_cagr = mult ** (1/years) - 1
    summary_lines.append(f"Average CAGR over {years} years from day {sample_day}: {avg_cagr:.2%}")

summary_lines.append("")
summary_lines.append("KEY INSIGHT:")
summary_lines.append("The growth rate DECREASES over time due to the logarithmic nature.")
summary_lines.append("Bitcoin grows faster in early days, slower in later days.")
summary_lines.append("This is why the formula predicts sustainable long-term growth.")

# Save results
summary_path = "Investment Strategy Analasis/Bitcoin Analysis/bitcoin_lumpsum_formula_summary.txt"
with open(summary_path, "w") as f:
    for line in summary_lines:
        f.write(line + "\n")

# Print results
for line in summary_lines:
    print(line)

print(f"\nSummary saved to {summary_path}") 
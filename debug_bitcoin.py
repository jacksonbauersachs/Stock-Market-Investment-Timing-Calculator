import math
from datetime import datetime, timedelta

# Bitcoin Rainbow Chart formula
slope = 1.828876
intercept = -10.888322

# Test different start dates
start_dates = [
    datetime(2010, 7, 18),  # Original start
    datetime(2011, 7, 18),  # Model start (from date range)
]

today = datetime(2025, 8, 6)
july_20_2025 = datetime(2025, 7, 20)

# Test with the hardcoded day from bitcoin_gbm_fair_value_start.py
hardcoded_day = 6041
log_price_hardcoded = slope * math.log(hardcoded_day) + intercept
fair_value_hardcoded = 10 ** log_price_hardcoded

print(f"Testing with hardcoded day 6041 from bitcoin_gbm_fair_value_start.py:")
print(f"Fair value: ${fair_value_hardcoded:,.2f}")

# Calculate what date day 6041 corresponds to
days_from_2010 = hardcoded_day - 1  # Since we start at day 1
date_for_day_6041 = datetime(2010, 7, 18) + timedelta(days=days_from_2010)
print(f"Date for day 6041: {date_for_day_6041.strftime('%Y-%m-%d')}")

print(f"\nTesting different start dates and models:")
print(f"Slope: {slope}")
print(f"Intercept: {intercept}")

for i, start_date in enumerate(start_dates):
    days = (today - start_date).days + 1
    days_july_20 = (july_20_2025 - start_date).days + 1
    
    print(f"\nTest {i+1}: Start date {start_date.strftime('%Y-%m-%d')}")
    print(f"Days since start (today): {days}")
    print(f"Days since start (July 20, 2025): {days_july_20}")
    
    # Calculate fair value for today
    log_price = slope * math.log(days) + intercept
    fair_value = 10 ** log_price
    
    # Calculate fair value for July 20, 2025
    log_price_july_20 = slope * math.log(days_july_20) + intercept
    fair_value_july_20 = 10 ** log_price_july_20
    
    print(f"Fair value (today): ${fair_value:,.2f}")
    print(f"Fair value (July 20, 2025): ${fair_value_july_20:,.2f}")

# Let's also test what day would give us $107k
target_fair_value = 107000
target_log_price = math.log10(target_fair_value)
target_days = math.exp((target_log_price - intercept) / slope)

print(f"\nTo get ${target_fair_value:,.2f}, we need day: {target_days:.0f}")

# Calculate what date that would be
days_from_2010 = target_days - 1  # Since we start at day 1
target_date = datetime(2010, 7, 18) + timedelta(days=days_from_2010)
print(f"Target date: {target_date.strftime('%Y-%m-%d')}")

# Let's also check what the fair value should be for July 20, 2025
print(f"\nExpected fair value for July 20, 2025: ~$107k")
print(f"Actual calculation for July 20, 2025: ${fair_value_july_20:,.2f}")
print(f"Difference: {((107000 - fair_value_july_20) / fair_value_july_20 * 100):+.1f}%") 
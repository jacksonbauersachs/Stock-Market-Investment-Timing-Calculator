import numpy as np

# Growth model parameters
a = 1.827743
b = -10.880943

# Volatility model parameters
a_vol = 2.310879
b_vol = 0.124138
c_vol = 0.077392

print("="*60)
print("DEBUG: GROWTH RATE CALCULATION")
print("="*60)

# Check what the formula predicts for different years
today_day = 5476 - 365  # Adjusted for 365-day offset
print(f"Today's formula day: {today_day}")
print(f"Formula prediction for today: ${10**(a * np.log(today_day) + b):,.2f}")
print()

# Check growth rates for the first few years
dt = 1/365.25

for year in range(1, 6):
    current_day = today_day + int(year * 365.25)
    prev_day = today_day + int((year-1) * 365.25)
    
    current_price = 10**(a * np.log(current_day) + b)
    prev_price = 10**(a * np.log(prev_day) + b)
    
    # Calculate growth rate
    mu = np.log(current_price / prev_price) / dt
    
    print(f"Year {year}:")
    print(f"  Day {prev_day} -> Day {current_day}")
    print(f"  Price: ${prev_price:,.2f} -> ${current_price:,.2f}")
    print(f"  Growth rate: {mu:.4f} ({mu*100:.2f}%)")
    print()

print("="*60)
print("DEBUG: VOLATILITY CALCULATION")
print("="*60)

# Check volatility for different years
for year in range(6):
    volatility = a_vol * np.exp(-b_vol * year) + c_vol
    print(f"Year {year}: Volatility = {volatility:.4f} ({volatility*100:.2f}%)")

print()
print("="*60)
print("ISSUE ANALYSIS")
print("="*60)

# The problem might be that we're calculating growth rate between consecutive years
# but the formula might not be linear in that way
print("Let's check if the growth rate is consistent:")
for year in range(1, 11):
    current_day = today_day + int(year * 365.25)
    current_price = 10**(a * np.log(current_day) + b)
    
    # Calculate growth from today to this year
    today_price = 10**(a * np.log(today_day) + b)
    total_growth = np.log(current_price / today_price) / year
    
    print(f"Year {year}: Total growth rate = {total_growth:.4f} ({total_growth*100:.2f}%)") 
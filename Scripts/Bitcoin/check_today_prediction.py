import numpy as np

# Growth model parameters
a = 1.827743
b = -10.880943

print("="*60)
print("CHECKING FORMULA PREDICTION FOR TODAY")
print("="*60)

# Today's actual day number (from our data)
actual_today_day = 5476
print(f"Actual today's day number: {actual_today_day}")

# Adjust for 365-day offset in our formula
formula_today_day = actual_today_day - 365
print(f"Formula today's day number (adjusted): {formula_today_day}")

# Calculate what the formula predicts for today
formula_prediction = 10**(a * np.log(formula_today_day) + b)
print(f"Formula prediction for today: ${formula_prediction:,.2f}")

# Check what the formula predicts for the actual day number (without offset)
actual_formula_prediction = 10**(a * np.log(actual_today_day) + b)
print(f"Formula prediction for actual day {actual_today_day}: ${actual_formula_prediction:,.2f}")

# Current actual price
actual_price = 118075
print(f"Actual current price: ${actual_price:,.2f}")

print()
print("="*60)
print("ANALYSIS")
print("="*60)

# Check which prediction makes more sense
if abs(formula_prediction - actual_price) < abs(actual_formula_prediction - actual_price):
    print("✅ Using 365-day offset gives better prediction")
    print(f"Difference with offset: {((actual_price/formula_prediction - 1) * 100):.1f}%")
    print(f"Difference without offset: {((actual_price/actual_formula_prediction - 1) * 100):.1f}%")
else:
    print("❌ Using actual day number gives better prediction")
    print(f"Difference with offset: {((actual_price/formula_prediction - 1) * 100):.1f}%")
    print(f"Difference without offset: {((actual_price/actual_formula_prediction - 1) * 100):.1f}%")

print()
print("Let's also check what the formula predicts for recent days:")
print("-" * 50)
for day_offset in range(-5, 6):
    check_day = actual_today_day + day_offset
    check_formula_day = check_day - 365
    prediction = 10**(a * np.log(check_formula_day) + b)
    print(f"Day {check_day} (formula day {check_formula_day}): ${prediction:,.2f}") 
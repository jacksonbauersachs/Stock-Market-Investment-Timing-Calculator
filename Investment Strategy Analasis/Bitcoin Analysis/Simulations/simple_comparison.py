import numpy as np

# Bitcoin Growth Model coefficients
a = 1.6329135221917355
b = -9.328646304661454

def bitcoin_growth_model(days):
    return 10**(a * np.log(days) + b)

# Calculate pure growth model result
initial_investment = 100000
current_day = 5439
years = 1
days_in_horizon = int(years * 365.25)

start_price = bitcoin_growth_model(current_day)
end_price = bitcoin_growth_model(current_day + days_in_horizon)
growth_multiple = end_price / start_price
pure_growth_value = initial_investment * growth_multiple

print(f"Pure Growth Model Calculation:")
print(f"Start price (day {current_day}): ${start_price:,.2f}")
print(f"End price (day {current_day + days_in_horizon}): ${end_price:,.2f}")
print(f"Growth multiple: {growth_multiple:.4f}x")
print(f"Final value: ${pure_growth_value:,.0f}")

# This should match our previous result of $127,662
print(f"\nExpected from previous calculation: $127,662")
print(f"Difference: ${pure_growth_value - 127662:,.0f}")

# Now let's check what the Monte Carlo should target
current_actual_price = 105740  # What we use in Monte Carlo
expected_final_price = bitcoin_growth_model(current_day + days_in_horizon)
total_expected_return = np.log(expected_final_price / current_actual_price)
annualized_drift = total_expected_return / years

print(f"\nMonte Carlo Target Calculation:")
print(f"Starting with actual price: ${current_actual_price:,.0f}")
print(f"Target final price: ${expected_final_price:,.2f}")
print(f"Total expected return: {total_expected_return:.4f}")
print(f"Annualized drift: {annualized_drift:.4f}")
print(f"Expected final value: ${current_actual_price * np.exp(total_expected_return):,.0f}")

# The Monte Carlo mean should be close to this value
target_mc_mean = current_actual_price * np.exp(total_expected_return)
print(f"\nMonte Carlo mean should be: ${target_mc_mean:,.0f}")
print(f"Pure growth model gives: ${pure_growth_value:,.0f}")
print(f"Difference: ${target_mc_mean - pure_growth_value:,.0f}") 
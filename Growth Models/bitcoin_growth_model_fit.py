import pandas as pd
import numpy as np
from scipy.stats import linregress

# Load full historical Bitcoin data
df = pd.read_csv('Data Sets/Bitcoin Data/Bitcoin Historical Data Full.csv')
df['Price'] = pd.to_numeric(df['Price'].str.replace(',', ''), errors='coerce')
df['Date'] = pd.to_datetime(df['Date'])
df['Days'] = (df['Date'] - df['Date'].min()).dt.days

df = df.dropna(subset=['Price'])

# Skip the first 90 days (start at day 90 exactly)
df = df[df['Days'] >= 90].copy()

# Excel process: T = LN(day number), U = LOG10(price)
df['T'] = np.log(df['Days'])
df['U'] = np.log10(df['Price'])

# Linear regression: U = a * T + b
slope, intercept, r_value, p_value, std_err = linregress(df['T'], df['U'])

print('Excel-style fit: log10(price) = a * ln(day) + b')
print(f'a = {slope}')
print(f'b = {intercept}')
print(f'R² = {r_value**2}')

# Save to file
with open('bitcoin_growth_formula_coefficients.txt', 'w') as f:
    f.write(f'a = {slope}\n')
    f.write(f'b = {intercept}\n')
    f.write(f'R2 = {r_value**2}\n')
    f.write('# Formula: log10(price) = a * ln(day) + b (day >= 90)\n')

# Test prediction
pred = slope * df['T'].iloc[-1] + intercept
actual = df['U'].iloc[-1]
print('Latest prediction (log10):', pred)
print('Actual (log10):', actual)
print('Error:', abs(pred - actual) / abs(actual) * 100, '%')

# Convert back to price for last day
pred_price = 10**pred
actual_price = df['Price'].iloc[-1]
print('Latest prediction (price):', pred_price)
print('Actual price:', actual_price)
print('Error:', abs(pred_price - actual_price) / actual_price * 100, '%') 
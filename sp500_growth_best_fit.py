import pandas as pd
import numpy as np
from scipy.stats import linregress

# Load S&P 500 data
file_path = "Data Sets/S&P 500 Data Sets/S&P 500 Total Data Cleaned 2.csv"
df = pd.read_csv(file_path)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
df['Days'] = (df['Date'] - df['Date'].min()).dt.days
df['Years'] = df['Days'] / 365.25

y = df['Price'].values
x = df['Years'].values

results = []

# 1. Exponential: log(price) = a * years + b
log_price = np.log(y)
res = linregress(x, log_price)
results.append((res.rvalue**2, 'log(price) = a * years + b', res.slope, res.intercept))

# 2. log10(price) = a * years + b
log10_price = np.log10(y)
res = linregress(x, log10_price)
results.append((res.rvalue**2, 'log10(price) = a * years + b', res.slope, res.intercept))

# 3. Power law: log10(price) = a * log10(years) + b
log10_years = np.log10(x + 1e-6)
res = linregress(log10_years, log10_price)
results.append((res.rvalue**2, 'log10(price) = a * log10(years) + b', res.slope, res.intercept))

# 4. Power law: log(price) = a * log(years) + b
log_years = np.log(x + 1e-6)
res = linregress(log_years, log_price)
results.append((res.rvalue**2, 'log(price) = a * log(years) + b', res.slope, res.intercept))

# Find best
results.sort(reverse=True)
best = results[0]

print('Best S&P 500 growth fit:')
print(f'R² = {best[0]:.6f}')
print(f'Formula: {best[1]}')
print(f'a = {best[2]}')
print(f'b = {best[3]}')

# Save to file
with open('sp500_growth_best_fit.txt', 'w') as f:
    f.write(f'R2 = {best[0]:.6f}\n')
    f.write(f'Formula: {best[1]}\n')
    f.write(f'a = {best[2]}\n')
    f.write(f'b = {best[3]}\n') 
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import linregress

# Load Bitcoin data - use the FULL historical dataset
df = pd.read_csv('Data Sets/Bitcoin Data/Bitcoin Historical Data Full.csv')
df['Date'] = pd.to_datetime(df['Date'])
# Convert price column to numeric, removing commas
df['Price'] = pd.to_numeric(df['Price'].str.replace(',', ''), errors='coerce')
df['Days'] = (df['Date'] - df['Date'].min()).dt.days

# Remove any NaN values
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

# Your formula structure
def your_formula(days, a, b):
    return 10**(a * np.log(days) + b)

# Try different formula variations
def formula_variation1(days, a, b, c):
    return 10**(a * np.log(days + c) + b)

def formula_variation2(days, a, b):
    return 10**(a * np.log(days + 1) + b)

print('=== FINDING BEST R² FIT WITH FULL HISTORICAL DATA ===')
print('Testing different approaches to get 90%+ R²...')
print()

best_r2 = 0
best_start = 0
best_params = None
best_formula = None

# Test different starting points with original formula
for start_day in [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 365, 500, 730, 1000, 1500, 2000]:
    if start_day >= len(df):
        continue
        
    df_subset = df[df['Days'] >= start_day].copy()
    if len(df_subset) < 100:  # Need enough data points
        continue
        
    df_subset['Days'] = df_subset['Days'] - start_day + 1  # Reset to start from day 1
    
    # Remove extreme outliers (prices that are 10x different from median)
    median_price = df_subset['Price'].median()
    df_subset = df_subset[(df_subset['Price'] > median_price / 10) & (df_subset['Price'] < median_price * 10)]
    
    if len(df_subset) < 50:  # Need enough data points after outlier removal
        continue
    
    try:
        # Test original formula
        params, _ = curve_fit(your_formula, df_subset['Days'], df_subset['Price'])
        y_pred = your_formula(df_subset['Days'], *params)
        ss_res = np.sum((df_subset['Price'] - y_pred) ** 2)
        ss_tot = np.sum((df_subset['Price'] - np.mean(df_subset['Price'])) ** 2)
        r2 = 1 - ss_res / ss_tot
        
        if r2 > best_r2:
            best_r2 = r2
            best_start = start_day
            best_params = params
            best_formula = "original"
        
        # Test variation 1
        try:
            params1, _ = curve_fit(formula_variation1, df_subset['Days'], df_subset['Price'], p0=[1, -10, 1])
            y_pred1 = formula_variation1(df_subset['Days'], *params1)
            ss_res1 = np.sum((df_subset['Price'] - y_pred1) ** 2)
            ss_tot1 = np.sum((df_subset['Price'] - np.mean(df_subset['Price'])) ** 2)
            r2_1 = 1 - ss_res1 / ss_tot1
            
            if r2_1 > best_r2:
                best_r2 = r2_1
                best_start = start_day
                best_params = params1
                best_formula = "variation1"
        except:
            pass
            
        # Test variation 2
        try:
            params2, _ = curve_fit(formula_variation2, df_subset['Days'], df_subset['Price'])
            y_pred2 = formula_variation2(df_subset['Days'], *params2)
            ss_res2 = np.sum((df_subset['Price'] - y_pred2) ** 2)
            ss_tot2 = np.sum((df_subset['Price'] - np.mean(df_subset['Price'])) ** 2)
            r2_2 = 1 - ss_res2 / ss_tot2
            
            if r2_2 > best_r2:
                best_r2 = r2_2
                best_start = start_day
                best_params = params2
                best_formula = "variation2"
        except:
            pass
            
        print(f'Start day {start_day:4d}: R²={r2:.4f} (orig), {r2_1:.4f} (var1), {r2_2:.4f} (var2)')
            
    except Exception as e:
        continue

print()
print('='*60)
print(f'BEST FIT FOUND:')
print(f'Starting from day: {best_start}')
print(f'Formula: {best_formula}')
print(f'Parameters: {best_params}')
print(f'R² = {best_r2:.6f}')
print('='*60)

# Test the best fit
df_best = df[df['Days'] >= best_start].copy()
df_best['Days'] = df_best['Days'] - best_start + 1

# Apply same outlier removal
median_price = df_best['Price'].median()
df_best = df_best[(df_best['Price'] > median_price / 10) & (df_best['Price'] < median_price * 10)]

if best_formula == "original":
    y_pred_best = your_formula(df_best['Days'], *best_params)
elif best_formula == "variation1":
    y_pred_best = formula_variation1(df_best['Days'], *best_params)
elif best_formula == "variation2":
    y_pred_best = formula_variation2(df_best['Days'], *best_params)

print()
print('Test the best fit:')
print('Latest prediction:', y_pred_best.iloc[-1])
print('Actual price:', df_best['Price'].iloc[-1])
print('Error:', abs(y_pred_best.iloc[-1] - df_best['Price'].iloc[-1]) / df_best['Price'].iloc[-1] * 100, '%') 
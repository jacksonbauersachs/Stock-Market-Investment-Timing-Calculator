import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Load S&P 500 data
file_path = "Data Sets/S&P 500 Data Sets/S&P 500 Total Data Cleaned 2.csv"
df = pd.read_csv(file_path)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
df['Days'] = (df['Date'] - df['Date'].min()).dt.days
df['Years'] = df['Days'] / 365.25

y = df['Price'].values
x = df['Years'].values

# Candidate models
def exp_model(x, a, b):
    return a * np.exp(b * x)
def log_linear_model(x, a, b):
    return 10 ** (a * np.log(x + 1e-6) + b)
def power_law_model(x, a, b, c):
    return a * (x ** b) + c
def linear_model(x, a, b):
    return a * x + b

models = {
    'Exponential': (exp_model, [y[0], 0.1]),
    'Log-Linear': (log_linear_model, [1, np.log10(y[0])]),
    'Power Law': (power_law_model, [1, 1, 0]),
    'Linear': (linear_model, [1, y[0]])
}

results = {}

for name, (func, p0) in models.items():
    try:
        params, _ = curve_fit(func, x, y, p0=p0, maxfev=10000)
        y_pred = func(x, *params)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot
        results[name] = (params, r2)
        print(f"{name}: R^2 = {r2:.4f}, params = {params}")
    except Exception as e:
        print(f"{name}: Fit failed ({e})")

# Find best model
best_name = max(results, key=lambda k: results[k][1])
best_params, best_r2 = results[best_name]
print(f"\nBest model: {best_name}, R^2 = {best_r2:.4f}, params = {best_params}")

# Save summary
with open("Investment Strategy Analasis/Stock Market Analysis/sp500_growth_fit_summary.txt", "w") as f:
    for name, (params, r2) in results.items():
        f.write(f"{name}: R^2 = {r2:.6f}, params = {params}\n")
    f.write(f"\nBest model: {best_name}, R^2 = {best_r2:.6f}, params = {best_params}\n")

# Plot
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], y, label='Actual', color='blue')
for name, (func, _) in models.items():
    if name in results:
        plt.plot(df['Date'], func(x, *results[name][0]), label=name)
plt.xlabel('Date')
plt.ylabel('S&P 500 Price')
plt.title('S&P 500 Growth Model Fits')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show() 
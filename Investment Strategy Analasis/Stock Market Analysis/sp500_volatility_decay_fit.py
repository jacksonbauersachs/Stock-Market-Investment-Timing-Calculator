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

# Calculate returns
df['Returns'] = df['Price'].pct_change()

# Calculate rolling volatilities (annualized)
windows = [7, 30, 90, 180, 365]
for w in windows:
    df[f'Vol_{w}d'] = df['Returns'].rolling(w).std() * np.sqrt(365)

# Decay models
def linear_decay(x, a, b):
    return a * x + b

def exp_decay(x, a, b, c):
    return a * np.exp(-b * x) + c

def power_law_decay(x, a, b, c):
    return a * (x ** (-b)) + c

def log_decay(x, a, b):
    return a * np.log(x + 1) + b

def inverse_decay(x, a, b, c):
    return a / (1 + b * x) + c

def sqrt_decay(x, a, b, c):
    return a / np.sqrt(1 + b * x) + c

def poly2_decay(x, a, b, c):
    return a * (x ** 2) + b * x + c

def poly3_decay(x, a, b, c, d):
    return a * (x ** 3) + b * (x ** 2) + c * x + d

models = {
    'Linear': (linear_decay, [1, 1]),
    'Exponential': (exp_decay, [1, 0.1, 0]),
    'Power Law': (power_law_decay, [1, 0.5, 0]),
    'Logarithmic': (log_decay, [1, 1]),
    'Inverse': (inverse_decay, [1, 0.1, 0]),
    'Square Root': (sqrt_decay, [1, 0.1, 0]),
    'Polynomial 2nd': (poly2_decay, [1, 1, 1]),
    'Polynomial 3rd': (poly3_decay, [1, 1, 1, 1])
}

summary_lines = []

for w in windows:
    col = f'Vol_{w}d'
    clean = df[['Years', col]].dropna()
    clean = clean[clean[col] > 0]
    print(f"Window {w}d: {len(clean)} data points")
    x = clean['Years'].values
    y = clean[col].values
    if len(x) == 0:
        print(f"Warning: No valid data for window {w}d, skipping.")
        continue
    best_r2 = -np.inf
    best_name = None
    best_params = None
    best_func = None
    model_results = {}
    for name, (func, p0) in models.items():
        try:
            params, _ = curve_fit(func, x, y, p0=p0, maxfev=10000)
            y_pred = func(x, *params)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - ss_res / ss_tot
            model_results[name] = (params, r2)
            if r2 > best_r2:
                best_r2 = r2
                best_name = name
                best_params = params
                best_func = func
        except Exception as e:
            continue
    if best_name:
        line = f"Vol_{w}d: {best_name}, R^2={best_r2:.4f}, params={best_params}"
        print(line)
        summary_lines.append(line)
        # Save plot instead of showing
        plt.figure(figsize=(10, 4))
        plt.scatter(x, y, s=5, alpha=0.5, label='Actual')
        plt.plot(x, best_func(x, *best_params), 'r-', label=f'Best Fit: {best_name}')
        plt.xlabel('Years Since Start')
        plt.ylabel(f'Annualized Volatility ({w}d)')
        plt.title(f'S&P 500 Volatility Decay Fit ({w}d)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = f"Investment Strategy Analasis/Stock Market Analysis/sp500_vol_decay_{w}d.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"Plot saved to {plot_path}")

print("Preparing to write summary file...")
if not summary_lines:
    print("Warning: No summary lines to write! Check if any fits succeeded.")
with open("Investment Strategy Analasis/Stock Market Analysis/sp500_volatility_decay_fits.txt", "w") as f:
    for line in summary_lines:
        f.write(line + '\n')
print("Summary file written to Investment Strategy Analasis/Stock Market Analysis/sp500_volatility_decay_fits.txt") 
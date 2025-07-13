import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Load Bitcoin data
file_path = "Data Sets/Bitcoin Data/Bitcoin_Cleaned_Data.csv"
df = pd.read_csv(file_path)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
df['Days'] = (df['Date'] - df['Date'].min()).dt.days
df['Years'] = df['Days'] / 365.25

y = df['Close/Last'].values
x = df['Days'].values

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
    'Exponential': (exp_model, [y[0], 0.001]),
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

# Manual testing of different coefficients for log-linear model
print(f"\n{'='*60}")
print("MANUAL COEFFICIENT TESTING FOR LOG-LINEAR MODEL")
print(f"{'='*60}")
print("Testing different 'a' and 'b' values in: 10^(a * ln(days) + b)")
print()

# Your original parameters
original_a, original_b = 1.633, -9.32
y_pred_orig = log_linear_model(x, original_a, original_b)
ss_res_orig = np.sum((y - y_pred_orig) ** 2)
ss_tot_orig = np.sum((y - np.mean(y)) ** 2)
r2_orig = 1 - ss_res_orig / ss_tot_orig
print(f"Your original: a={original_a}, b={original_b}, R²={r2_orig:.4f}")

# Test different coefficient combinations
test_coefficients = [
    # Try smaller ranges around your original
    (1.4, -8.0),
    (1.5, -8.5),
    (1.6, -9.0),
    (1.7, -9.5),
    (1.8, -10.0),
    (1.9, -10.5),
    (2.0, -11.0),
    # Try the fitted parameters
    (0.49, 1.09),  # From curve_fit
]

best_manual_r2 = r2_orig
best_manual_params = (original_a, original_b)

for a, b in test_coefficients:
    y_pred = log_linear_model(x, a, b)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot
    print(f"a={a:.2f}, b={b:.2f}: R²={r2:.4f}")
    
    if r2 > best_manual_r2:
        best_manual_r2 = r2
        best_manual_params = (a, b)

print(f"\nBest manual coefficients: a={best_manual_params[0]:.3f}, b={best_manual_params[1]:.3f}, R²={best_manual_r2:.4f}")

# Also test exponential model coefficients
print(f"\n{'='*60}")
print("TESTING EXPONENTIAL MODEL COEFFICIENTS")
print(f"{'='*60}")
print("Formula: price = a * exp(b * days)")

# Fitted exponential parameters
exp_a, exp_b = 10667, 0.000947
y_pred_exp = exp_model(x, exp_a, exp_b)
ss_res_exp = np.sum((y - y_pred_exp) ** 2)
ss_tot_exp = np.sum((y - np.mean(y)) ** 2)
r2_exp = 1 - ss_res_exp / ss_tot_exp
print(f"Fitted exponential: a={exp_a:.0f}, b={exp_b:.6f}, R²={r2_exp:.4f}")

# Test different exponential coefficients
exp_test_coefficients = [
    (10000, 0.0009),
    (11000, 0.00095),
    (12000, 0.0010),
    (13000, 0.00105),
    (14000, 0.0011),
]

best_exp_r2 = r2_exp
best_exp_params = (exp_a, exp_b)

for a, b in exp_test_coefficients:
    y_pred = exp_model(x, a, b)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot
    print(f"a={a:.0f}, b={b:.6f}: R²={r2:.4f}")
    
    if r2 > best_exp_r2:
        best_exp_r2 = r2
        best_exp_params = (a, b)

print(f"\nBest exponential coefficients: a={best_exp_params[0]:.0f}, b={best_exp_params[1]:.6f}, R²={best_exp_r2:.4f}")

# Save summary
with open("Investment Strategy Analasis/Bitcoin Analysis/bitcoin_growth_fit_summary.txt", "w") as f:
    for name, (params, r2) in results.items():
        f.write(f"{name}: R^2 = {r2:.6f}, params = {params}\n")
    f.write(f"\nBest model: {best_name}, R^2 = {best_r2:.6f}, params = {best_params}\n")
    f.write(f"\nManual coefficient testing:\n")
    f.write(f"Original: a={original_a}, b={original_b}, R²={r2_orig:.6f}\n")
    f.write(f"Best manual: a={best_manual_params[0]:.3f}, b={best_manual_params[1]:.3f}, R²={best_manual_r2:.6f}\n")
    f.write(f"\nExponential model testing:\n")
    f.write(f"Fitted exponential: a={exp_a:.0f}, b={exp_b:.6f}, R²={r2_exp:.6f}\n")
    f.write(f"Best exponential: a={best_exp_params[0]:.0f}, b={best_exp_params[1]:.6f}, R²={best_exp_r2:.6f}\n")

# Plot
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], y, label='Actual', color='blue')
for name, (func, _) in models.items():
    if name in results:
        plt.plot(df['Date'], func(x, *results[name][0]), label=name)
plt.xlabel('Date')
plt.ylabel('Bitcoin Price')
plt.title('Bitcoin Growth Model Fits')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show() 
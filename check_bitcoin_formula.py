import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

# Load Bitcoin data
df = pd.read_csv('Data Sets/Bitcoin Data/Bitcoin_Cleaned_Data.csv')
df['Date'] = pd.to_datetime(df['Date'])
df['Days'] = (df['Date'] - df['Date'].min()).dt.days

def growth_model(days, a, b):
    return 10**(a * np.log(days) + b)

# Fit the model
params, _ = curve_fit(growth_model, df['Days'], df['Close/Last'])

print('Fitted parameters:')
print('a =', params[0])
print('b =', params[1])
print()

print('Test the fitted model:')
latest_prediction = growth_model(df['Days'].iloc[-1], *params)
print('Latest day prediction:', latest_prediction)
print('Actual price:', df['Close/Last'].iloc[-1])
print('Error:', abs(latest_prediction - df['Close/Last'].iloc[-1]) / df['Close/Last'].iloc[-1] * 100, '%')

print()
print('Your original parameters:')
print('a = 1.633, b = -9.32')
your_prediction = growth_model(df['Days'].iloc[-1], 1.633, -9.32)
print('Your prediction:', your_prediction)
print('Error:', abs(your_prediction - df['Close/Last'].iloc[-1]) / df['Close/Last'].iloc[-1] * 100, '%') 
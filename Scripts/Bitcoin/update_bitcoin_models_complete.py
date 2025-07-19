import pandas as pd
import numpy as np
from scipy.stats import linregress
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_bitcoin_data():
    """Load the complete Bitcoin dataset"""
    print("Loading complete Bitcoin dataset...")
    df = pd.read_csv('Data Sets/Bitcoin Data/Bitcoin_Final_Complete_Data_20250719.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.dropna(subset=['Price'])
    df = df.sort_values('Date')
    
    print(f"Loaded {len(df):,} days of Bitcoin data")
    print(f"Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"Price range: ${df['Price'].min():.2f} to ${df['Price'].max():,.2f}")
    
    return df

def fit_growth_model(df):
    """Fit the Bitcoin growth model"""
    print("\n" + "="*60)
    print("FITTING BITCOIN GROWTH MODEL")
    print("="*60)
    
    # Calculate days from start (start at 1)
    df['Days'] = (df['Date'] - df['Date'].min()).dt.days + 1
    
    # Skip the first 364 days (start at day 365)
    df_growth = df[df['Days'] >= 365].copy()
    
    print(f"Using {len(df_growth):,} days starting from day 365")
    
    # Model: log10(price) = a * ln(day) + b
    X = np.log(df_growth['Days'])
    Y = np.log10(df_growth['Price'])
    
    # Central model fit
    slope, intercept, r_value, p_value, std_err = linregress(X, Y)
    R2 = r_value ** 2
    
    print(f"Model Formula: log10(price) = {slope:.6f} * ln(day) + {intercept:.6f}")
    print(f"R² = {R2:.6f}")
    print(f"Data points: {len(df_growth)}")
    
    return {
        'slope': slope,
        'intercept': intercept,
        'R2': R2,
        'data_points': len(df_growth),
        'start_date': df_growth['Date'].min(),
        'end_date': df_growth['Date'].max()
    }

def fit_volatility_decay_model(df):
    """Fit the Bitcoin volatility decay model"""
    print("\n" + "="*60)
    print("FITTING BITCOIN VOLATILITY DECAY MODEL")
    print("="*60)
    
    # Calculate time metrics
    df['Days'] = (df['Date'] - df['Date'].min()).dt.days
    df['Years'] = df['Days'] / 365.25
    
    # Calculate returns and volatility
    df['Returns'] = df['Price'].pct_change()
    df['Volatility_30d'] = df['Returns'].rolling(30).std() * np.sqrt(365)
    
    # Remove NaN values
    clean_data = df[['Years', 'Volatility_30d']].dropna()
    
    print(f"Using {len(clean_data):,} days for volatility analysis")
    print(f"Volatility range: {clean_data['Volatility_30d'].min():.4f} to {clean_data['Volatility_30d'].max():.4f}")
    
    # Try different decay models
    models = {
        'Power Law Decay': fit_power_law_decay,
        'Exponential Decay': fit_exponential_decay,
        'Linear Decay': fit_linear_decay,
        'Logarithmic Decay': fit_logarithmic_decay
    }
    
    best_model = None
    best_r_squared = -np.inf
    best_params = None
    
    for model_name, fit_function in models.items():
        try:
            params, r_squared = fit_function(clean_data['Years'], clean_data['Volatility_30d'])
            
            if r_squared > best_r_squared:
                best_r_squared = r_squared
                best_model = model_name
                best_params = params
                
            print(f"{model_name}: R² = {r_squared:.4f}")
            
        except Exception as e:
            print(f"{model_name}: Failed to fit - {e}")
    
    print(f"\nBest Model: {best_model}")
    print(f"R² = {best_r_squared:.4f}")
    print(f"Parameters: {best_params}")
    
    return {
        'model_name': best_model,
        'params': best_params,
        'R2': best_r_squared,
        'data_points': len(clean_data),
        'volatility_range': (clean_data['Volatility_30d'].min(), clean_data['Volatility_30d'].max())
    }

def fit_power_law_decay(x, y):
    """Fit power law decay model: volatility = a * years^b + c"""
    def power_law_model(x, a, b, c):
        return a * np.power(x, b) + c
    
    # Initial guess
    p0 = [np.max(y), -0.5, np.min(y)]
    params, _ = curve_fit(power_law_model, x, y, p0=p0, maxfev=10000)
    predictions = power_law_model(x, *params)
    r_squared = 1 - np.sum((y - predictions) ** 2) / np.sum((y - np.mean(y)) ** 2)
    
    return params, r_squared

def fit_exponential_decay(x, y):
    """Fit exponential decay model: volatility = a * exp(-b * years) + c"""
    def exponential_model(x, a, b, c):
        return a * np.exp(-b * x) + c
    
    # Initial guess
    p0 = [np.max(y), 0.1, np.min(y)]
    params, _ = curve_fit(exponential_model, x, y, p0=p0, maxfev=10000)
    predictions = exponential_model(x, *params)
    r_squared = 1 - np.sum((y - predictions) ** 2) / np.sum((y - np.mean(y)) ** 2)
    
    return params, r_squared

def fit_linear_decay(x, y):
    """Fit linear decay model: volatility = a * years + b"""
    def linear_model(x, a, b):
        return a * x + b
    
    params, _ = curve_fit(linear_model, x, y)
    predictions = linear_model(x, *params)
    r_squared = 1 - np.sum((y - predictions) ** 2) / np.sum((y - np.mean(y)) ** 2)
    
    return params, r_squared

def fit_logarithmic_decay(x, y):
    """Fit logarithmic decay model: volatility = a * ln(years) + b"""
    def logarithmic_model(x, a, b):
        return a * np.log(x) + b
    
    params, _ = curve_fit(logarithmic_model, x, y)
    predictions = logarithmic_model(x, *params)
    r_squared = 1 - np.sum((y - predictions) ** 2) / np.sum((y - np.mean(y)) ** 2)
    
    return params, r_squared

def save_complete_models(growth_model, volatility_model):
    """Save both growth and volatility models to the same file"""
    print("\n" + "="*60)
    print("SAVING COMPLETE BITCOIN MODELS")
    print("="*60)
    
    filename = 'Models/Growth Models/bitcoin_complete_models.txt'
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("BITCOIN COMPLETE MODELS\n")
        f.write("="*60 + "\n")
        f.write(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data range: {growth_model['start_date'].strftime('%Y-%m-%d')} to {growth_model['end_date'].strftime('%Y-%m-%d')}\n")
        f.write(f"Total data points: {growth_model['data_points']:,}\n")
        f.write("\n")
        
        # Growth Model Section
        f.write("-"*40 + "\n")
        f.write("GROWTH MODEL\n")
        f.write("-"*40 + "\n")
        f.write(f"Formula: log10(price) = {growth_model['slope']:.6f} * ln(day) + {growth_model['intercept']:.6f}\n")
        f.write(f"R² = {growth_model['R2']:.6f}\n")
        f.write(f"Data points: {growth_model['data_points']:,}\n")
        f.write(f"Start day: 365 (after first year)\n")
        f.write(f"Date range: {growth_model['start_date'].strftime('%Y-%m-%d')} to {growth_model['end_date'].strftime('%Y-%m-%d')}\n")
        f.write("\n")
        
        # Volatility Model Section
        f.write("-"*40 + "\n")
        f.write("VOLATILITY DECAY MODEL\n")
        f.write("-"*40 + "\n")
        f.write(f"Model type: {volatility_model['model_name']}\n")
        f.write(f"R² = {volatility_model['R2']:.6f}\n")
        f.write(f"Data points: {volatility_model['data_points']:,}\n")
        f.write(f"Volatility range: {volatility_model['volatility_range'][0]:.4f} to {volatility_model['volatility_range'][1]:.4f}\n")
        f.write(f"Parameters: {volatility_model['params']}\n")
        
        # Add specific formula based on model type
        if volatility_model['model_name'] == 'Power Law Decay':
            a, b, c = volatility_model['params']
            f.write(f"Formula: volatility = {a:.6f} * years^{b:.6f} + {c:.6f}\n")
        elif volatility_model['model_name'] == 'Exponential Decay':
            a, b, c = volatility_model['params']
            f.write(f"Formula: volatility = {a:.6f} * exp(-{b:.6f} * years) + {c:.6f}\n")
        elif volatility_model['model_name'] == 'Linear Decay':
            a, b = volatility_model['params']
            f.write(f"Formula: volatility = {a:.6f} * years + {b:.6f}\n")
        elif volatility_model['model_name'] == 'Logarithmic Decay':
            a, b = volatility_model['params']
            f.write(f"Formula: volatility = {a:.6f} * ln(years) + {b:.6f}\n")
        
        f.write("\n")
        
        # Summary Section
        f.write("-"*40 + "\n")
        f.write("MODEL SUMMARY\n")
        f.write("-"*40 + "\n")
        f.write(f"Growth model R²: {growth_model['R2']:.4f}\n")
        f.write(f"Volatility model R²: {volatility_model['R2']:.4f}\n")
        f.write(f"Combined analysis quality: {'Excellent' if min(growth_model['R2'], volatility_model['R2']) > 0.9 else 'Good' if min(growth_model['R2'], volatility_model['R2']) > 0.8 else 'Fair'}\n")
        f.write("\n")
        f.write("="*60 + "\n")
    
    print(f"Complete models saved to: {filename}")
    return filename

def main():
    """Main function to run complete Bitcoin model analysis"""
    print("="*60)
    print("BITCOIN COMPLETE MODEL ANALYSIS")
    print("="*60)
    print("This script will fit both growth and volatility decay models")
    print("and save them to a single comprehensive file.")
    print("="*60)
    
    # Load data
    df = load_bitcoin_data()
    
    # Fit growth model
    growth_model = fit_growth_model(df)
    
    # Fit volatility decay model
    volatility_model = fit_volatility_decay_model(df)
    
    # Save complete models
    filename = save_complete_models(growth_model, volatility_model)
    
    # Display summary
    print("\n" + "="*50)
    print("ANALYSIS SUMMARY")
    print("="*50)
    print(f"Growth Model R²: {growth_model['R2']:.4f}")
    print(f"Volatility Model R²: {volatility_model['R2']:.4f}")
    print(f"Growth Formula: log₁₀(price) = {growth_model['slope']:.3f} × ln(day) + {growth_model['intercept']:.3f}")
    print(f"Volatility Model: {volatility_model['model_name']}")
    print(f"Complete models file: {filename}")
    print("\nBoth models are now ready for rainbow chart and volatility analysis!")

if __name__ == "__main__":
    main() 
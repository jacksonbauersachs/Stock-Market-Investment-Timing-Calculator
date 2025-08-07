import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import linregress
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_bitcoin_data():
    """Load and prepare Bitcoin price data for analysis."""
    
    data_file = "Portfolio/Data/Bitcoin_all_time_price.csv"
    
    if not os.path.exists(data_file):
        print(f"Error: Bitcoin data file not found - {data_file}")
        return None
    
    # Read the data
    df = pd.read_csv(data_file)
    
    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Clean Price column - convert to float
    df['Price'] = df['Price'].astype(float)
    
    # Sort by date
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Create days since start column for modeling
    df['Days'] = (df['Date'] - df['Date'].min()).dt.days
    
    print(f"Bitcoin data loaded: {len(df)} records from {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"Price range: ${df['Price'].min():.2f} to ${df['Price'].max():,.2f}")
    
    return df

def load_gold_data():
    """Load and prepare gold price data for analysis."""
    
    data_file = "Portfolio/Data/Gold_all_time_price.csv"
    
    if not os.path.exists(data_file):
        print(f"Error: Gold data file not found - {data_file}")
        return None
    
    # Read the data
    df = pd.read_csv(data_file)
    
    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Clean Price column - remove quotes and commas, convert to float
    df['Price'] = df['Price'].astype(str).str.replace('"', '').str.replace(',', '').astype(float)
    
    # Sort by date
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Create days since start column for modeling
    df['Days'] = (df['Date'] - df['Date'].min()).dt.days
    
    print(f"Gold data loaded: {len(df)} records from {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"Price range: ${df['Price'].min():.2f} to ${df['Price'].max():.2f}")
    
    return df

def load_silver_data():
    """Load and prepare silver price data for analysis."""
    
    data_file = "Portfolio/Data/Silver_all_time_price.csv"
    
    if not os.path.exists(data_file):
        print(f"Error: Silver data file not found - {data_file}")
        return None
    
    # Read the data
    df = pd.read_csv(data_file)
    
    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Clean Price column - convert to float
    df['Price'] = df['Price'].astype(float)
    
    # Sort by date
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Create days since start column for modeling
    df['Days'] = (df['Date'] - df['Date'].min()).dt.days
    
    print(f"Silver data loaded: {len(df)} records from {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"Price range: ${df['Price'].min():.3f} to ${df['Price'].max():.3f}")
    
    return df

def fit_bitcoin_rainbow_model(df):
    """Fit Bitcoin rainbow chart model (log10(price) = a * ln(day) + b)."""
    
    # Skip the first 364 days (start at day 365)
    df_filtered = df[df['Days'] >= 365].copy()
    
    if len(df_filtered) == 0:
        print("Warning: Not enough data for Bitcoin rainbow model (need at least 365 days)")
        return None
    
    # Model: log10(price) = a * ln(day) + b
    X = np.log(df_filtered['Days'])
    Y = np.log10(df_filtered['Price'])
    
    # Central model fit
    slope, intercept, r_value, p_value, std_err = linregress(X, Y)
    R2 = r_value ** 2
    
    model_results = {
        'asset': 'Bitcoin',
        'model_type': 'Rainbow Chart (Log-Log)',
        'formula': f"log10(price) = {slope:.6f} * ln(day) + {intercept:.6f}",
        'slope': slope,
        'intercept': intercept,
        'r2': R2,
        'data_points': len(df_filtered),
        'date_range': f"{df_filtered['Date'].min().strftime('%Y-%m-%d')} to {df_filtered['Date'].max().strftime('%Y-%m-%d')}",
        'current_price': df_filtered['Price'].iloc[-1],
        'all_time_high': df_filtered['Price'].max()
    }
    
    return model_results

def fit_standard_models(df, asset_name):
    """Fit standard growth models (linear, exponential, polynomial, logarithmic, power)."""
    
    x = df['Days'].values
    y = df['Price'].values
    
    models = {}
    
    # Linear model
    try:
        popt_linear, _ = curve_fit(lambda x, a, b: a * x + b, x, y, p0=[0.1, 100])
        y_pred_linear = popt_linear[0] * x + popt_linear[1]
        r2_linear = r2_score(y, y_pred_linear)
        rmse_linear = np.sqrt(mean_squared_error(y, y_pred_linear))
        
        models['Linear'] = {
            'formula': f"Price = {popt_linear[0]:.6f} * Days + {popt_linear[1]:.2f}",
            'params': popt_linear,
            'r2': r2_linear,
            'rmse': rmse_linear
        }
    except:
        print(f"Warning: Could not fit linear model for {asset_name}")
    
    # Exponential model
    try:
        popt_exp, _ = curve_fit(lambda x, a, b, c: a * np.exp(b * x) + c, x, y, p0=[100, 0.001, 100])
        y_pred_exp = popt_exp[0] * np.exp(popt_exp[1] * x) + popt_exp[2]
        r2_exp = r2_score(y, y_pred_exp)
        rmse_exp = np.sqrt(mean_squared_error(y, y_pred_exp))
        
        models['Exponential'] = {
            'formula': f"Price = {popt_exp[0]:.2f} * exp({popt_exp[1]:.6f} * Days) + {popt_exp[2]:.2f}",
            'params': popt_exp,
            'r2': r2_exp,
            'rmse': rmse_exp
        }
    except:
        print(f"Warning: Could not fit exponential model for {asset_name}")
    
    # Polynomial model (cubic)
    try:
        popt_poly, _ = curve_fit(lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d, x, y, p0=[1e-8, 1e-5, 0.1, 100])
        y_pred_poly = popt_poly[0] * x**3 + popt_poly[1] * x**2 + popt_poly[2] * x + popt_poly[3]
        r2_poly = r2_score(y, y_pred_poly)
        rmse_poly = np.sqrt(mean_squared_error(y, y_pred_poly))
        
        models['Polynomial'] = {
            'formula': f"Price = {popt_poly[0]:.2e}*Days³ + {popt_poly[1]:.2e}*Days² + {popt_poly[2]:.4f}*Days + {popt_poly[3]:.2f}",
            'params': popt_poly,
            'r2': r2_poly,
            'rmse': rmse_poly
        }
    except:
        print(f"Warning: Could not fit polynomial model for {asset_name}")
    
    # Logarithmic model
    try:
        popt_log, _ = curve_fit(lambda x, a, b, c: a * np.log(b * x + 1) + c, x, y, p0=[100, 0.01, 100])
        y_pred_log = popt_log[0] * np.log(popt_log[1] * x + 1) + popt_log[2]
        r2_log = r2_score(y, y_pred_log)
        rmse_log = np.sqrt(mean_squared_error(y, y_pred_log))
        
        models['Logarithmic'] = {
            'formula': f"Price = {popt_log[0]:.2f} * ln({popt_log[1]:.4f} * Days + 1) + {popt_log[2]:.2f}",
            'params': popt_log,
            'r2': r2_log,
            'rmse': rmse_log
        }
    except:
        print(f"Warning: Could not fit logarithmic model for {asset_name}")
    
    # Power model
    try:
        popt_power, _ = curve_fit(lambda x, a, b, c: a * np.power(x, b) + c, x, y, p0=[1, 0.5, 100])
        y_pred_power = popt_power[0] * np.power(x, popt_power[1]) + popt_power[2]
        r2_power = r2_score(y, y_pred_power)
        rmse_power = np.sqrt(mean_squared_error(y, y_pred_power))
        
        models['Power'] = {
            'formula': f"Price = {popt_power[0]:.2f} * Days^{popt_power[1]:.4f} + {popt_power[2]:.2f}",
            'params': popt_power,
            'r2': r2_power,
            'rmse': rmse_power
        }
    except:
        print(f"Warning: Could not fit power model for {asset_name}")
    
    # Find best model
    if models:
        best_model = max(models.items(), key=lambda x: x[1]['r2'])
        models['Best_Model'] = {
            'name': best_model[0],
            'formula': best_model[1]['formula'],
            'r2': best_model[1]['r2'],
            'rmse': best_model[1]['rmse']
        }
    
    return models

def save_comprehensive_results(bitcoin_results, gold_results, silver_results):
    """Save all growth model results to a comprehensive file."""
    
    # Create Portfolio/Models directory if it doesn't exist
    os.makedirs("Portfolio/Models", exist_ok=True)
    
    results_file = "Portfolio/Models/updated_models.txt"
    
    with open(results_file, 'w') as f:
        f.write("COMPREHENSIVE GROWTH MODEL ANALYSIS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data Sources: Updated Portfolio/Data/ files\n\n")
        
        # Bitcoin Results
        f.write("BITCOIN GROWTH MODELS\n")
        f.write("-" * 30 + "\n")
        if bitcoin_results:
            f.write(f"Rainbow Chart Model:\n")
            f.write(f"  Formula: {bitcoin_results['formula']}\n")
            f.write(f"  R² Score: {bitcoin_results['r2']:.6f}\n")
            f.write(f"  Data Points: {bitcoin_results['data_points']:,}\n")
            f.write(f"  Date Range: {bitcoin_results['date_range']}\n")
            f.write(f"  Current Price: ${bitcoin_results['current_price']:,.2f}\n")
            f.write(f"  All-Time High: ${bitcoin_results['all_time_high']:,.2f}\n\n")
        
        # Gold Results
        f.write("GOLD GROWTH MODELS\n")
        f.write("-" * 25 + "\n")
        if gold_results:
            for model_name, model_data in gold_results.items():
                if model_name != 'Best_Model':
                    f.write(f"{model_name} Model:\n")
                    f.write(f"  Formula: {model_data['formula']}\n")
                    f.write(f"  R² Score: {model_data['r2']:.6f}\n")
                    f.write(f"  RMSE: {model_data['rmse']:.2f}\n\n")
            
            if 'Best_Model' in gold_results:
                f.write(f"Best Gold Model: {gold_results['Best_Model']['name']}\n")
                f.write(f"  Formula: {gold_results['Best_Model']['formula']}\n")
                f.write(f"  R² Score: {gold_results['Best_Model']['r2']:.6f}\n")
                f.write(f"  RMSE: {gold_results['Best_Model']['rmse']:.2f}\n\n")
        
        # Silver Results
        f.write("SILVER GROWTH MODELS\n")
        f.write("-" * 27 + "\n")
        if silver_results:
            for model_name, model_data in silver_results.items():
                if model_name != 'Best_Model':
                    f.write(f"{model_name} Model:\n")
                    f.write(f"  Formula: {model_data['formula']}\n")
                    f.write(f"  R² Score: {model_data['r2']:.6f}\n")
                    f.write(f"  RMSE: {model_data['rmse']:.2f}\n\n")
            
            if 'Best_Model' in silver_results:
                f.write(f"Best Silver Model: {silver_results['Best_Model']['name']}\n")
                f.write(f"  Formula: {silver_results['Best_Model']['formula']}\n")
                f.write(f"  R² Score: {silver_results['Best_Model']['r2']:.6f}\n")
                f.write(f"  RMSE: {silver_results['Best_Model']['rmse']:.2f}\n\n")
        
        # Summary
        f.write("SUMMARY\n")
        f.write("-" * 10 + "\n")
        f.write("All growth models have been updated using the latest data from:\n")
        f.write("- Portfolio/Data/Bitcoin_all_time_price.csv\n")
        f.write("- Portfolio/Data/Gold_all_time_price.csv\n")
        f.write("- Portfolio/Data/Silver_all_time_price.csv\n\n")
        f.write("The models can be used for:\n")
        f.write("- Price trend analysis\n")
        f.write("- Future price predictions\n")
        f.write("- Investment timing decisions\n")
        f.write("- Portfolio allocation strategies\n")
    
    print(f"Comprehensive results saved to: {results_file}")
    return results_file

def main():
    """Main function to update all growth models."""
    
    print("Comprehensive Growth Model Update")
    print("=" * 40)
    
    # Load data
    print("\nLoading updated data files...")
    bitcoin_df = load_bitcoin_data()
    gold_df = load_gold_data()
    silver_df = load_silver_data()
    
    if bitcoin_df is None or gold_df is None or silver_df is None:
        print("Error: Could not load all data files.")
        return
    
    # Fit models
    print("\nFitting growth models...")
    
    # Bitcoin rainbow model
    print("Fitting Bitcoin rainbow chart model...")
    bitcoin_results = fit_bitcoin_rainbow_model(bitcoin_df)
    
    # Gold standard models
    print("Fitting Gold growth models...")
    gold_results = fit_standard_models(gold_df, "Gold")
    
    # Silver standard models
    print("Fitting Silver growth models...")
    silver_results = fit_standard_models(silver_df, "Silver")
    
    # Save results
    print("\nSaving comprehensive results...")
    results_file = save_comprehensive_results(bitcoin_results, gold_results, silver_results)
    
    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    
    if bitcoin_results:
        print(f"Bitcoin Rainbow Model: R² = {bitcoin_results['r2']:.6f}")
    
    if gold_results and 'Best_Model' in gold_results:
        print(f"Best Gold Model: {gold_results['Best_Model']['name']} (R² = {gold_results['Best_Model']['r2']:.6f})")
    
    if silver_results and 'Best_Model' in silver_results:
        print(f"Best Silver Model: {silver_results['Best_Model']['name']} (R² = {silver_results['Best_Model']['r2']:.6f})")
    
    print(f"\nResults saved to: {results_file}")

if __name__ == "__main__":
    main() 
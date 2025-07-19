import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
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

def calculate_365d_volatility(df):
    """Calculate 365-day rolling volatility"""
    print("\nCalculating 365-day rolling volatility...")
    
    # Calculate time metrics
    df['Days'] = (df['Date'] - df['Date'].min()).dt.days
    df['Years'] = df['Days'] / 365.25
    
    # Calculate returns and 365-day volatility
    df['Returns'] = df['Price'].pct_change()
    df['Volatility_365d'] = df['Returns'].rolling(365).std() * np.sqrt(365)
    
    # Remove NaN values
    clean_data = df[['Years', 'Volatility_365d']].dropna()
    
    print(f"Using {len(clean_data):,} days for volatility analysis")
    print(f"Volatility range: {clean_data['Volatility_365d'].min():.4f} to {clean_data['Volatility_365d'].max():.4f}")
    print(f"Mean volatility: {clean_data['Volatility_365d'].mean():.4f}")
    
    return clean_data

def fit_volatility_models(years, volatility):
    """Fit different volatility decay models"""
    print("\n" + "="*60)
    print("FITTING VOLATILITY DECAY MODELS")
    print("="*60)
    
    models = {
        'Linear Decay': fit_linear_decay,
        'Exponential Decay': fit_exponential_decay,
        'Power Law Decay': fit_power_law_decay,
        'Logarithmic Decay': fit_logarithmic_decay,
        'Polynomial Decay': fit_polynomial_decay
    }
    
    best_model = None
    best_r_squared = -np.inf
    best_params = None
    
    for model_name, fit_function in models.items():
        try:
            params, r_squared = fit_function(years, volatility)
            
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
    
    return best_model, best_params, best_r_squared

def fit_linear_decay(x, y):
    """Fit linear decay model: volatility = a * years + b"""
    def linear_model(x, a, b):
        return a * x + b
    
    params, _ = curve_fit(linear_model, x, y)
    predictions = linear_model(x, *params)
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

def fit_logarithmic_decay(x, y):
    """Fit logarithmic decay model: volatility = a * ln(years) + b"""
    def logarithmic_model(x, a, b):
        return a * np.log(x) + b
    
    params, _ = curve_fit(logarithmic_model, x, y)
    predictions = logarithmic_model(x, *params)
    r_squared = 1 - np.sum((y - predictions) ** 2) / np.sum((y - np.mean(y)) ** 2)
    
    return params, r_squared

def fit_polynomial_decay(x, y):
    """Fit polynomial decay model: volatility = a * years^2 + b * years + c"""
    def polynomial_model(x, a, b, c):
        return a * (x ** 2) + b * x + c
    
    params, _ = curve_fit(polynomial_model, x, y)
    predictions = polynomial_model(x, *params)
    r_squared = 1 - np.sum((y - predictions) ** 2) / np.sum((y - np.mean(y)) ** 2)
    
    return params, r_squared

def get_model_function(model_name, params):
    """Get the model function for predictions"""
    if model_name == 'Linear Decay':
        return lambda x: params[0] * x + params[1]
    elif model_name == 'Exponential Decay':
        return lambda x: params[0] * np.exp(-params[1] * x) + params[2]
    elif model_name == 'Power Law Decay':
        return lambda x: params[0] * np.power(x, params[1]) + params[2]
    elif model_name == 'Logarithmic Decay':
        return lambda x: params[0] * np.log(x) + params[1]
    elif model_name == 'Polynomial Decay':
        return lambda x: params[0] * (x ** 2) + params[1] * x + params[2]
    else:
        return None

def create_volatility_graph(years, volatility, model_name, params, r_squared):
    """Create comprehensive volatility analysis graph"""
    print("\n" + "="*60)
    print("CREATING VOLATILITY ANALYSIS GRAPH")
    print("="*60)
    
    # Set up the plot
    plt.style.use('default')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Bitcoin 365-Day Volatility Decay Analysis', fontsize=16, fontweight='bold')
    
    # Get model function
    model_func = get_model_function(model_name, params)
    
    # Generate model predictions
    years_range = np.linspace(years.min(), years.max(), 100)
    model_predictions = model_func(years_range)
    
    # Plot 1: Raw volatility data with model fit
    ax1.scatter(years, volatility, alpha=0.6, s=10, color='blue', label='Actual Volatility')
    ax1.plot(years_range, model_predictions, 'r-', linewidth=2, label=f'Model Fit ({model_name})')
    ax1.set_xlabel('Years Since Start')
    ax1.set_ylabel('365-Day Annualized Volatility')
    ax1.set_title(f'Volatility Decay: {model_name}\nR² = {r_squared:.4f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Volatility distribution
    ax2.hist(volatility, bins=30, alpha=0.7, edgecolor='black', color='skyblue')
    ax2.axvline(volatility.mean(), color='red', linestyle='--', label=f'Mean: {volatility.mean():.3f}')
    ax2.axvline(volatility.median(), color='green', linestyle='--', label=f'Median: {volatility.median():.3f}')
    ax2.set_xlabel('Volatility')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Volatility Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Volatility over time (log scale)
    ax3.scatter(years, volatility, alpha=0.6, s=10, color='blue')
    ax3.plot(years_range, model_predictions, 'r-', linewidth=2)
    ax3.set_yscale('log')
    ax3.set_xlabel('Years Since Start')
    ax3.set_ylabel('Volatility (Log Scale)')
    ax3.set_title('Volatility vs Time (Log Scale)')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Future predictions
    current_years = years.max()
    future_years = np.linspace(current_years, current_years + 5, 50)
    future_predictions = model_func(future_years)
    
    # Ensure predictions are reasonable
    future_predictions = np.maximum(future_predictions, 0.1)
    future_predictions = np.minimum(future_predictions, 2.0)
    
    ax4.scatter(years, volatility, alpha=0.6, s=10, color='blue', label='Historical')
    ax4.plot(years_range, model_predictions, 'r-', linewidth=2, label='Model Fit')
    ax4.plot(future_years, future_predictions, 'g--', linewidth=2, label='Future Predictions')
    ax4.axvline(x=current_years, color='black', linestyle=':', label='Current Time')
    ax4.set_xlabel('Years Since Start')
    ax4.set_ylabel('Volatility')
    ax4.set_title('Volatility with Future Predictions')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    filename = f'Results/Bitcoin/bitcoin_365d_volatility_analysis_{datetime.now().strftime("%Y%m%d")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Graph saved to: {filename}")
    
    plt.show()
    
    return future_years, future_predictions

def save_volatility_results(model_name, params, r_squared, future_years, future_predictions):
    """Save volatility analysis results"""
    print("\n" + "="*60)
    print("SAVING VOLATILITY RESULTS")
    print("="*60)
    
    filename = f'Models/Volatility Models/bitcoin_365d_volatility_results_{datetime.now().strftime("%Y%m%d")}.txt'
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("BITCOIN 365-DAY VOLATILITY DECAY ANALYSIS\n")
        f.write("="*60 + "\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data Source: Bitcoin_Final_Complete_Data_20250719.csv\n")
        f.write("\n")
        
        f.write("-"*40 + "\n")
        f.write("MODEL RESULTS\n")
        f.write("-"*40 + "\n")
        f.write(f"Best Model: {model_name}\n")
        f.write(f"R² = {r_squared:.6f}\n")
        f.write(f"Parameters: {params}\n")
        
        # Add specific formula based on model type
        if model_name == 'Power Law Decay':
            a, b, c = params
            f.write(f"Formula: volatility = {a:.6f} * years^{b:.6f} + {c:.6f}\n")
        elif model_name == 'Exponential Decay':
            a, b, c = params
            f.write(f"Formula: volatility = {a:.6f} * exp(-{b:.6f} * years) + {c:.6f}\n")
        elif model_name == 'Linear Decay':
            a, b = params
            f.write(f"Formula: volatility = {a:.6f} * years + {b:.6f}\n")
        elif model_name == 'Logarithmic Decay':
            a, b = params
            f.write(f"Formula: volatility = {a:.6f} * ln(years) + {b:.6f}\n")
        elif model_name == 'Polynomial Decay':
            a, b, c = params
            f.write(f"Formula: volatility = {a:.6f} * years^2 + {b:.6f} * years + {c:.6f}\n")
        
        f.write("\n")
        
        f.write("-"*40 + "\n")
        f.write("FUTURE PREDICTIONS\n")
        f.write("-"*40 + "\n")
        current_year = future_years[0]
        for i, (year, pred) in enumerate(zip(future_years[::10], future_predictions[::10])):
            years_ahead = year - current_year
            if years_ahead <= 5:
                f.write(f"Year {years_ahead:.1f} from now: {pred:.1%}\n")
        
        f.write("\n")
        f.write("="*60 + "\n")
    
    print(f"Results saved to: {filename}")
    return filename

def main():
    """Main function to run 365-day volatility analysis"""
    print("="*60)
    print("BITCOIN 365-DAY VOLATILITY DECAY ANALYSIS")
    print("="*60)
    print("This script analyzes Bitcoin's 365-day volatility decay pattern")
    print("and finds the best fitting model for Monte Carlo simulations.")
    print("="*60)
    
    # Load data
    df = load_bitcoin_data()
    
    # Calculate 365-day volatility
    clean_data = calculate_365d_volatility(df)
    
    # Fit volatility models
    best_model, best_params, best_r_squared = fit_volatility_models(
        clean_data['Years'], clean_data['Volatility_365d']
    )
    
    # Create graph
    future_years, future_predictions = create_volatility_graph(
        clean_data['Years'], clean_data['Volatility_365d'], 
        best_model, best_params, best_r_squared
    )
    
    # Save results
    filename = save_volatility_results(
        best_model, best_params, best_r_squared, future_years, future_predictions
    )
    
    # Display summary
    print("\n" + "="*50)
    print("ANALYSIS SUMMARY")
    print("="*50)
    print(f"Best Model: {best_model}")
    print(f"R² = {best_r_squared:.4f}")
    print(f"Parameters: {best_params}")
    
    # Show future predictions
    current_year = future_years[0]
    print(f"\nFuture Volatility Predictions:")
    for i, (year, pred) in enumerate(zip(future_years[::10], future_predictions[::10])):
        years_ahead = year - current_year
        if years_ahead <= 5:
            print(f"  Year {years_ahead:.1f} from now: {pred:.1%}")
    
    print(f"\nResults file: {filename}")
    print("\nThis volatility model is ready for Monte Carlo simulations!")

if __name__ == "__main__":
    main() 
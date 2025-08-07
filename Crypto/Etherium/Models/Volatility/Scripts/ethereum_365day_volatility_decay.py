import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from datetime import datetime
import os

def load_ethereum_data():
    """Load Ethereum data and calculate daily returns"""
    print("Loading Ethereum dataset...")
    df = pd.read_csv('Etherium/Data/Ethereum Historical Data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Convert Price to numeric if needed
    if df['Price'].dtype == 'object':
        df['Price'] = pd.to_numeric(df['Price'].str.replace(',', '').str.replace('$', ''), errors='coerce')
    
    # Calculate daily returns
    df['Returns'] = df['Price'].pct_change()
    
    print(f"Loaded {len(df)} days of Ethereum data")
    print(f"Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"Price range: ${df['Price'].min():.2f} to ${df['Price'].max():.2f}")
    
    return df

def calculate_365day_volatility_over_time(df):
    """Calculate 365-day rolling volatility and track how it changes over time"""
    print("\nCalculating 365-day rolling volatility over time...")
    
    # Calculate 365-day rolling volatility
    rolling_vol = df['Returns'].rolling(window=365).std() * np.sqrt(365) * 100
    
    # Create a dataframe with date, age (days since start), and volatility
    volatility_data = pd.DataFrame({
        'Date': df['Date'],
        'Age_Days': range(len(df)),
        'Age_Years': np.array(range(len(df))) / 365.25,
        'Volatility_365d': rolling_vol
    })
    
    # Remove NaN values (first 364 days don't have enough data for 365-day window)
    volatility_data = volatility_data.dropna()
    
    print(f"Volatility data points: {len(volatility_data):,}")
    print(f"Age range: {volatility_data['Age_Years'].min():.1f} to {volatility_data['Age_Years'].max():.1f} years")
    print(f"Volatility range: {volatility_data['Volatility_365d'].min():.1f}% to {volatility_data['Volatility_365d'].max():.1f}%")
    
    return volatility_data

def fit_volatility_decay_models(volatility_data):
    """Fit different models to volatility decay over time"""
    print("\n" + "="*60)
    print("ETHEREUM 365-DAY VOLATILITY DECAY MODEL FITTING")
    print("="*60)
    
    X = volatility_data['Age_Years'].values
    Y = volatility_data['Volatility_365d'].values
    
    # Linear model: volatility = a * age + b
    slope_linear, intercept_linear, r_value_linear, p_value_linear, std_err_linear = linregress(X, Y)
    R2_linear = r_value_linear ** 2
    
    # Exponential model: log(volatility) = a * age + b
    Y_log = np.log(Y)
    slope_exp, intercept_exp, r_value_exp, p_value_exp, std_err_exp = linregress(X, Y_log)
    R2_exp = r_value_exp ** 2
    
    # Power Law model: log(volatility) = a * log(age) + b
    X_log = np.log(X)
    slope_power, intercept_power, r_value_power, p_value_power, std_err_power = linregress(X_log, Y_log)
    R2_power = r_value_power ** 2
    
    # Inverse model: volatility = a / age + b
    X_inv = 1 / X
    slope_inv, intercept_inv, r_value_inv, p_value_inv, std_err_inv = linregress(X_inv, Y)
    R2_inv = r_value_inv ** 2
    
    print(f"Linear Model: volatility = {slope_linear:.6f} * age + {intercept_linear:.6f}")
    print(f"Linear R² = {R2_linear:.6f}")
    
    print(f"Exponential Model: log(volatility) = {slope_exp:.6f} * age + {intercept_exp:.6f}")
    print(f"Exponential R² = {R2_exp:.6f}")
    
    print(f"Power Law Model: log(volatility) = {slope_power:.6f} * log(age) + {intercept_power:.6f}")
    print(f"Power Law R² = {R2_power:.6f}")
    
    print(f"Inverse Model: volatility = {slope_inv:.6f} / age + {intercept_inv:.6f}")
    print(f"Inverse R² = {R2_inv:.6f}")
    
    # Determine best model
    models = [
        ("Linear", R2_linear, slope_linear, intercept_linear),
        ("Exponential", R2_exp, slope_exp, intercept_exp),
        ("Power Law", R2_power, slope_power, intercept_power),
        ("Inverse", R2_inv, slope_inv, intercept_inv)
    ]
    
    best_model = max(models, key=lambda x: x[1])
    print(f"\nBest Model: {best_model[0]} (R² = {best_model[1]:.6f})")
    
    return models, best_model

def save_decay_formulas(models, best_model, volatility_data):
    """Save the volatility decay model formulas"""
    os.makedirs('Etherium/Models/Volatility/Formulas', exist_ok=True)
    
    with open('Etherium/Models/Volatility/Formulas/ethereum_365day_volatility_decay_coefficients.txt', 'w') as f:
        f.write(f'# Ethereum 365-Day Volatility Decay Model Coefficients\n')
        f.write(f'# Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'# Data range: {volatility_data["Date"].min().strftime("%Y-%m-%d")} to {volatility_data["Date"].max().strftime("%Y-%m-%d")}\n')
        f.write(f'# Age range: {volatility_data["Age_Years"].min():.1f} to {volatility_data["Age_Years"].max():.1f} years\n')
        f.write(f'# Total data points: {len(volatility_data):,}\n\n')
        
        for model_name, R2, slope, intercept in models:
            f.write(f'{model_name} Model:\n')
            f.write(f'  R² = {R2:.6f}\n')
            f.write(f'  Slope = {slope:.6f}\n')
            f.write(f'  Intercept = {intercept:.6f}\n')
            
            if model_name == "Linear":
                f.write(f'  Formula: volatility = {slope:.6f} * age + {intercept:.6f}\n')
            elif model_name == "Exponential":
                f.write(f'  Formula: log(volatility) = {slope:.6f} * age + {intercept:.6f}\n')
            elif model_name == "Power Law":
                f.write(f'  Formula: log(volatility) = {slope:.6f} * log(age) + {intercept:.6f}\n')
            elif model_name == "Inverse":
                f.write(f'  Formula: volatility = {slope:.6f} / age + {intercept:.6f}\n')
            
            f.write(f'  Best Model: {"Yes" if model_name == best_model[0] else "No"}\n\n')

def create_decay_visualization(volatility_data, models, best_model):
    """Create visualization of the volatility decay over time"""
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    X = volatility_data['Age_Years'].values
    Y = volatility_data['Volatility_365d'].values
    
    # Plot 1: Linear scale
    ax1.scatter(X, Y, color='white', s=20, alpha=0.6, label='Actual 365-day Volatility')
    
    # Plot model predictions
    X_plot = np.linspace(X.min(), X.max(), 1000)
    
    for model_name, R2, slope, intercept in models:
        if model_name == "Linear":
            Y_pred = slope * X_plot + intercept
            ax1.plot(X_plot, Y_pred, label=f'{model_name} (R²={R2:.3f})', linewidth=2)
        elif model_name == "Exponential":
            Y_pred = np.exp(slope * X_plot + intercept)
            ax1.plot(X_plot, Y_pred, label=f'{model_name} (R²={R2:.3f})', linewidth=2)
        elif model_name == "Power Law":
            Y_pred = np.exp(slope * np.log(X_plot) + intercept)
            ax1.plot(X_plot, Y_pred, label=f'{model_name} (R²={R2:.3f})', linewidth=2)
        elif model_name == "Inverse":
            Y_pred = slope / X_plot + intercept
            ax1.plot(X_plot, Y_pred, label=f'{model_name} (R²={R2:.3f})', linewidth=2)
    
    ax1.set_xlabel('Age (Years)', fontsize=12)
    ax1.set_ylabel('365-Day Rolling Volatility (%)', fontsize=12)
    ax1.set_title('Ethereum 365-Day Volatility Decay Over Time (Linear Scale)', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Plot 2: Log-log scale
    ax2.scatter(X, Y, color='white', s=20, alpha=0.6, label='Actual 365-day Volatility')
    
    for model_name, R2, slope, intercept in models:
        if model_name == "Linear":
            Y_pred = slope * X_plot + intercept
            ax2.plot(X_plot, Y_pred, label=f'{model_name} (R²={R2:.3f})', linewidth=2)
        elif model_name == "Exponential":
            Y_pred = np.exp(slope * X_plot + intercept)
            ax2.plot(X_plot, Y_pred, label=f'{model_name} (R²={R2:.3f})', linewidth=2)
        elif model_name == "Power Law":
            Y_pred = np.exp(slope * np.log(X_plot) + intercept)
            ax2.plot(X_plot, Y_pred, label=f'{model_name} (R²={R2:.3f})', linewidth=2)
        elif model_name == "Inverse":
            Y_pred = slope / X_plot + intercept
            ax2.plot(X_plot, Y_pred, label=f'{model_name} (R²={R2:.3f})', linewidth=2)
    
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Age (Years)', fontsize=12)
    ax2.set_ylabel('365-Day Rolling Volatility (%)', fontsize=12)
    ax2.set_title('Ethereum 365-Day Volatility Decay Over Time (Log-Log Scale)', fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    # Add best model info
    best_text = f'Best Model: {best_model[0]}\nR² = {best_model[1]:.4f}'
    ax1.text(0.02, 0.98, best_text, transform=ax1.transAxes, fontsize=12, color='white', 
             bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.5'),
             verticalalignment='top')
    
    # Add data info
    data_text = f'Data: {volatility_data["Date"].min().strftime("%b %Y")} - {volatility_data["Date"].max().strftime("%b %Y")}\nPoints: {len(volatility_data):,}'
    ax2.text(0.98, 0.02, data_text, transform=ax2.transAxes, fontsize=10, color='white', 
             bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.5'),
             horizontalalignment='right', verticalalignment='bottom')
    
    plt.tight_layout()
    
    # Save the plot
    save_path = 'Etherium/Visualizations/ethereum_365day_volatility_decay.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {save_path}")
    plt.show()

def main():
    """Main function to run the 365-day volatility decay analysis"""
    # Load data
    df = load_ethereum_data()
    
    # Calculate 365-day volatility over time
    volatility_data = calculate_365day_volatility_over_time(df)
    
    # Fit decay models
    models, best_model = fit_volatility_decay_models(volatility_data)
    
    # Save formulas
    save_decay_formulas(models, best_model, volatility_data)
    
    # Create visualization
    create_decay_visualization(volatility_data, models, best_model)
    
    print(f"\nModel coefficients saved to: Etherium/Models/Volatility/Formulas/ethereum_365day_volatility_decay_coefficients.txt")
    print(f"Best model: {best_model[0]} with R² = {best_model[1]:.6f}")
    
    # Show current volatility
    current_vol = volatility_data['Volatility_365d'].iloc[-1]
    current_age = volatility_data['Age_Years'].iloc[-1]
    print(f"\nCurrent 365-day volatility: {current_vol:.1f}% (at age {current_age:.1f} years)")

if __name__ == "__main__":
    main() 
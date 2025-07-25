import pandas as pd
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.ticker as ticker
import os

def load_bitcoin_data():
    """Load Bitcoin data and filter to 2020-10-20, starting from day 365"""
    try:
        btc_data = pd.read_csv('Bitcoin/Data/2010_2025_Daily_Data_(BTC).csv')
        btc_data['Date'] = pd.to_datetime(btc_data['Date'])
        
        # Convert Price column to numeric if needed
        if btc_data['Price'].dtype == 'object':
            btc_data['Price'] = pd.to_numeric(btc_data['Price'].str.replace(',', '').str.replace('$', ''), errors='coerce')
        else:
            btc_data['Price'] = pd.to_numeric(btc_data['Price'], errors='coerce')
        
        # Filter to 2020-10-20
        btc_data = btc_data[btc_data['Date'] <= '2020-10-20']
        btc_data = btc_data.dropna(subset=['Price'])
        btc_data = btc_data.sort_values('Date')
        
        # Calculate days from start
        btc_data['Days'] = (btc_data['Date'] - btc_data['Date'].min()).dt.days + 1
        
        # Skip the first 364 days (start at day 365)
        btc_data = btc_data[btc_data['Days'] >= 365].copy()
        
        return btc_data
    except Exception as e:
        print(f"Error loading Bitcoin data: {e}")
        return None

def load_ethereum_data():
    """Load Ethereum data starting from day 365"""
    try:
        eth_data = pd.read_csv('Etherium/Data/Ethereum Historical Data.csv')
        eth_data['Date'] = pd.to_datetime(eth_data['Date'])
        
        # Convert Price column to numeric if needed
        if eth_data['Price'].dtype == 'object':
            eth_data['Price'] = pd.to_numeric(eth_data['Price'].str.replace(',', '').str.replace('$', ''), errors='coerce')
        else:
            eth_data['Price'] = pd.to_numeric(eth_data['Price'], errors='coerce')
        
        eth_data = eth_data.dropna(subset=['Price'])
        eth_data = eth_data.sort_values('Date')
        
        # Calculate days from start
        eth_data['Days'] = (eth_data['Date'] - eth_data['Date'].min()).dt.days + 1
        
        # Skip the first 364 days (start at day 365)
        eth_data = eth_data[eth_data['Days'] >= 365].copy()
        
        return eth_data
    except Exception as e:
        print(f"Error loading Ethereum data: {e}")
        return None

def create_rainbow_chart(data, title, ax, color):
    """Create a rainbow chart for the given data"""
    if data is None or len(data) == 0:
        return
    
    # Fit the model: log10(price) = slope * ln(day) + intercept
    X = np.log(data['Days'])
    Y = np.log10(data['Price'])
    
    slope, intercept, r_value, p_value, std_err = linregress(X, Y)
    R2 = r_value ** 2
    
    # Model prediction
    model_price = 10**(slope * np.log(data['Days']) + intercept)
    log_model = np.log10(model_price)
    log_price = np.log10(data['Price'])
    log_dev = log_price - log_model
    
    # Calculate bands based on data from day 365 onwards (more stable growth period)
    data_bands = data[data['Days'] >= 365].copy()
    if len(data_bands) > 0:
        data_bands['log_dev_bands'] = np.log10(data_bands['Price']) - (slope * np.log(data_bands['Days']) + intercept)
        
        # For each year in the bands data, find the max and min log deviation
        data_bands['Year'] = data_bands['Date'].dt.year
        annual = data_bands.groupby('Year').agg({'Days': 'median', 'log_dev_bands': ['max', 'min']})
        annual.columns = ['Days', 'log_dev_max', 'log_dev_min']
        annual = annual.reset_index()
        
        # Expand envelopes by 10% of the band range
        band_range = annual['log_dev_max'].max() - annual['log_dev_min'].min()
        expand = 0.10 * band_range
        annual['log_dev_max'] += expand
        annual['log_dev_min'] -= expand
        
        # Fit log-linear models to the annual max and min log deviations
        X_env = np.log(annual['Days'])
        log_model_annual = slope * np.log(annual['Days']) + intercept
        Y_max = log_model_annual + annual['log_dev_max']
        Y_min = log_model_annual + annual['log_dev_min']
        
        # Fit upper and lower envelopes
        slope_upper, intercept_upper, *_ = linregress(X_env, Y_max)
        slope_lower, intercept_lower, *_ = linregress(X_env, Y_min)
        
        # Prepare band colors and names
        band_colors = ['blue', 'green', 'yellow', 'orange', 'red']
        band_names = ['Firesale!', 'Buy', 'Hold', 'Sell', 'Danger!']
        num_fills = len(band_colors)
        num_bounds = num_fills + 1
        
        # Interpolate slopes and intercepts for each boundary
        slopes = np.linspace(slope_lower, slope_upper, num_bounds)
        intercepts = np.linspace(intercept_lower, intercept_upper, num_bounds)
        
        # Plot rainbow bands
        bound_curves = [10**(slopes[i] * np.log(data['Days']) + intercepts[i]) for i in range(num_bounds)]
        for i in range(num_fills):
            ax.fill_between(data['Date'], bound_curves[i], bound_curves[i+1], color=band_colors[i], alpha=0.7)
    
    # Plot actual price and model
    ax.semilogy(data['Date'], data['Price'], color='white', linewidth=2, label='Actual Price')
    ax.semilogy(data['Date'], model_price, color='black', linestyle='--', linewidth=2, label='Model')
    
    # Formatting
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Price (USD)', fontsize=12)
    ax.grid(True, alpha=0.3, which='both')
    # Set y-axis limits relative to each model's price range
    min_price = max(data['Price'].min() * 0.5, 1.0)
    max_price = max(data['Price'].max(), model_price.max()) * 2
    ax.set_ylim(min_price, max_price)
    
    # Custom y-axis formatting
    def price_formatter(x, pos):
        if x >= 1000:
            return f'${x/1000:.0f}K'
        else:
            return f'${x:.0f}'
    
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(price_formatter))
    
    # Custom legend
    from matplotlib.patches import Patch
    legend_patches = [Patch(color=band_colors[i], label=band_names[i]) for i in reversed(range(num_fills))]
    legend_patches.append(Patch(color='white', label='Actual Price'))
    legend_patches.append(Patch(color='black', label='Model'))
    ax.legend(handles=legend_patches, fontsize=10, loc='upper left')
    
    # Add model formula and R²
    formula_text = f'Model: log₁₀(price) = {slope:.3f} × ln(day) + {intercept:.3f}\nR² = {R2:.4f}'
    ax.text(0.02, 0.08, formula_text, transform=ax.transAxes, fontsize=11, color='white', 
            bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.5'))
    
    # Add date range and current price
    date_range_text = f'Data: {data["Date"].min().strftime("%b %Y")} - {data["Date"].max().strftime("%b %Y")}\nCurrent: ${data["Price"].iloc[-1]:,.0f}'
    ax.text(0.98, 0.08, date_range_text, transform=ax.transAxes, fontsize=10, color='white', 
            bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.5'), 
            horizontalalignment='right', verticalalignment='bottom')

def create_comparison_plot():
    """Create the comparison plot"""
    # Load data
    btc_data = load_bitcoin_data()
    eth_data = load_ethereum_data()
    
    if btc_data is None or eth_data is None:
        print("Failed to load data. Please check file paths.")
        return
    
    print(f"Bitcoin data: {len(btc_data):,} days from {btc_data['Date'].min().strftime('%Y-%m-%d')} to {btc_data['Date'].max().strftime('%Y-%m-%d')}")
    print(f"Ethereum data: {len(eth_data):,} days from {eth_data['Date'].min().strftime('%Y-%m-%d')} to {eth_data['Date'].max().strftime('%Y-%m-%d')}")
    
    # Set up the plot
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Create rainbow charts
    create_rainbow_chart(btc_data, 'Bitcoin (BTC) Rainbow Chart: 2011-2020 (Start Day 365)', ax1, '#F7931A')
    create_rainbow_chart(eth_data, 'Ethereum (ETH) Rainbow Chart: 2017-2025 (Start Day 365)', ax2, '#627EEA')
    
    # Set x-axis labels
    ax1.set_xlabel('Date', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    
    plt.tight_layout()
    
    # Save the plot
    save_path = 'Etherium/Visualizations/Images/rainbow_chart_comparison.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved as: {save_path}")
    plt.show()
    
    return fig

if __name__ == "__main__":
    create_comparison_plot() 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np
import os

def load_ethereum_data():
    """Load Ethereum historical data"""
    try:
        # Load Ethereum data
        eth_data = pd.read_csv('Etherium/Data/Ethereum Historical Data.csv')
        print(f"Ethereum data loaded successfully. Shape: {eth_data.shape}")
        print(f"Ethereum columns: {list(eth_data.columns)}")
        
        # Convert date column to datetime
        eth_data['Date'] = pd.to_datetime(eth_data['Date'])
        
        # Convert Price column to numeric (Ethereum data might have formatting)
        if eth_data['Price'].dtype == 'object':
            eth_data['Price'] = pd.to_numeric(eth_data['Price'].str.replace(',', '').str.replace('$', ''), errors='coerce')
        else:
            eth_data['Price'] = pd.to_numeric(eth_data['Price'], errors='coerce')
        
        # Sort by date
        eth_data = eth_data.sort_values('Date')
        
        return eth_data
    except Exception as e:
        print(f"Error loading Ethereum data: {e}")
        return None

def load_bitcoin_data():
    """Load Bitcoin historical data from 2010 to 2019"""
    try:
        # Load Bitcoin data
        btc_data = pd.read_csv('Bitcoin/Data/2010_2025_Daily_Data_(BTC).csv')
        print(f"Bitcoin data loaded successfully. Shape: {btc_data.shape}")
        print(f"Bitcoin columns: {list(btc_data.columns)}")
        
        # Convert date column to datetime
        btc_data['Date'] = pd.to_datetime(btc_data['Date'])
        
        # Convert Price column to numeric (Bitcoin data might already be numeric)
        if btc_data['Price'].dtype == 'object':
            btc_data['Price'] = pd.to_numeric(btc_data['Price'].str.replace(',', '').str.replace('$', ''), errors='coerce')
        else:
            btc_data['Price'] = pd.to_numeric(btc_data['Price'], errors='coerce')
        
        # Filter to 2019 and earlier
        btc_data = btc_data[btc_data['Date'] <= '2020-10-20']
        print(f"Bitcoin data after 2019 filter: {len(btc_data)} rows")
        
        # Sort by date
        btc_data = btc_data.sort_values('Date')
        
        return btc_data
    except Exception as e:
        print(f"Error loading Bitcoin data: {e}")
        return None

def create_comparison_plot():
    """Create the comparison plot with two subplots"""
    
    # Load data
    eth_data = load_ethereum_data()
    btc_data = load_bitcoin_data()
    
    if eth_data is None or btc_data is None:
        print("Failed to load data. Please check file paths.")
        return
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Plot 1: Ethereum Full Price History
    ax1.plot(eth_data['Date'], eth_data['Price'], color='#627EEA', linewidth=1.5, alpha=0.8)
    ax1.set_title('Ethereum (ETH) Full Price History', fontsize=16, fontweight='bold', pad=20)
    ax1.set_ylabel('Price (USD)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor('#f8f9fa')
    
    # Format x-axis for Ethereum plot
    ax1.xaxis.set_major_locator(mdates.YearLocator(2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.tick_params(axis='x', rotation=45)
    
    # Add Ethereum logo color styling
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Plot 2: Bitcoin Price from Start to 2019
    ax2.plot(btc_data['Date'], btc_data['Price'], color='#F7931A', linewidth=1.5, alpha=0.8)
    ax2.set_title('Bitcoin (BTC) Price History: 2010-2019', fontsize=16, fontweight='bold', pad=20)
    ax2.set_ylabel('Price (USD)', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor('#f8f9fa')
    
    # Format x-axis for Bitcoin plot
    ax2.xaxis.set_major_locator(mdates.YearLocator(2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax2.tick_params(axis='x', rotation=45)
    
    # Add Bitcoin logo color styling
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Add statistics text boxes
    eth_start_price = eth_data['Price'].iloc[0]
    eth_end_price = eth_data['Price'].iloc[-1]
    eth_total_return = ((eth_end_price - eth_start_price) / eth_start_price) * 100
    
    btc_start_price = btc_data['Price'].iloc[0]
    btc_end_price = btc_data['Price'].iloc[-1]
    btc_total_return = ((btc_end_price - btc_start_price) / btc_start_price) * 100
    
    # Ethereum stats
    eth_stats_text = f'Ethereum Stats:\nStart: ${eth_start_price:.2f}\nEnd: ${eth_end_price:.2f}\nTotal Return: {eth_total_return:.1f}%'
    ax1.text(0.02, 0.98, eth_stats_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Bitcoin stats
    btc_stats_text = f'Bitcoin Stats (2010-2019):\nStart: ${btc_start_price:.2f}\nEnd: ${btc_end_price:.2f}\nTotal Return: {btc_total_return:.1f}%'
    ax2.text(0.02, 0.98, btc_stats_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot to the Images folder, always overwriting
    filename = 'Etherium/Visualizations/Images/ethereum_bitcoin_comparison.png'
    
    # Ensure the Images directory exists
    os.makedirs('Etherium/Visualizations/Images', exist_ok=True)
    
    # Save the plot (this will overwrite if file exists)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {filename}")
    print("Previous image has been overwritten.")
    
    # Show the plot
    plt.show()
    
    return fig

def main():
    """Main function to run the comparison"""
    print("Loading Ethereum and Bitcoin data...")
    print("Creating comparison plot...")
    
    fig = create_comparison_plot()
    
    if fig:
        print("Comparison plot created successfully!")
    else:
        print("Failed to create comparison plot.")

if __name__ == "__main__":
    main() 
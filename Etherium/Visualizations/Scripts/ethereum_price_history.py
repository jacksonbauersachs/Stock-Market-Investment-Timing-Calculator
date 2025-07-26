import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def load_ethereum_data():
    """Load Ethereum data for entire history"""
    print("Loading Ethereum dataset...")
    df = pd.read_csv('Etherium/Data/Ethereum Historical Data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Convert Price to numeric if needed
    if df['Price'].dtype == 'object':
        df['Price'] = pd.to_numeric(df['Price'].str.replace(',', '').str.replace('$', ''), errors='coerce')
    
    print(f"Loaded {len(df)} days of Ethereum data")
    print(f"Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"Price range: ${df['Price'].min():.2f} to ${df['Price'].max():.2f}")
    
    return df

def create_ethereum_chart(df):
    """Create Ethereum price history chart"""
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # Set background to pure white
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Plot Ethereum price with enhanced styling
    ax.plot(df['Date'], df['Price'], color='#627EEA', linewidth=2.5, label='Ethereum (ETH)', alpha=0.9)
    
    # Customize the plot with professional styling
    ax.set_xlabel('Date', fontsize=14, fontweight='bold', color='#333333')
    ax.set_ylabel('Price (USD)', fontsize=14, fontweight='bold', color='#333333')
    ax.set_title('Ethereum Price History', fontsize=20, fontweight='bold', color='#333333', pad=20)
    
    # Enhanced grid
    ax.grid(True, alpha=0.2, color='#CCCCCC', linewidth=0.5)
    ax.set_axisbelow(True)  # Put grid behind data
    
    # Legend styling
    ax.legend(fontsize=12, framealpha=0.9, fancybox=True, shadow=True, loc='upper left')
    
    # Format y-axis to show prices nicely
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Style the axes
    ax.tick_params(axis='both', which='major', labelsize=11, color='#666666')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Add subtle grid lines for key price levels (no labels)
    key_prices = [100, 1000, 5000, 10000, 20000, 50000]
    for price in key_prices:
        if price <= df['Price'].max():
            ax.axhline(y=price, color='lightgray', alpha=0.2, linestyle='-', linewidth=0.5)
    
    # Remove data info box for cleaner look
    
    plt.tight_layout()
    
    # Save the plot
    save_path = 'Etherium/Visualizations/Images/ethereum_price_history_blue.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Chart saved to: {save_path}")
    plt.show()

def main():
    """Main function"""
    print("Ethereum Price History Chart")
    print("="*50)
    
    # Load data
    df = load_ethereum_data()
    
    # Create chart
    create_ethereum_chart(df)
    
    print("\n✅ Chart creation completed!")

if __name__ == "__main__":
    main() 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import os
import warnings

# Suppress ALL matplotlib warnings aggressively
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', module='matplotlib')
warnings.filterwarnings('ignore', module='matplotlib.animation')

def load_bitcoin_data():
    """Load Bitcoin data filtered to 2014-2020"""
    print("Loading Bitcoin dataset (2014-2020)...")
    df = pd.read_csv('Bitcoin/Data/2010_2025_Daily_Data_(BTC).csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Filter to 2014-2020
    start_date = '2014-01-01'
    end_date = '2020-10-20'
    df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()
    
    print(f"Loaded {len(df)} days of Bitcoin data")
    print(f"Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"Price range: ${df['Price'].min():.2f} to ${df['Price'].max():.2f}")
    
    return df

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

def prepare_data_for_morphing(btc_df, eth_df):
    """Prepare both datasets for morphing with different time periods"""
    print("Preparing data for morphing...")
    
    # Keep full datasets with their original time periods
    btc_prepared = btc_df.copy()
    eth_prepared = eth_df.copy()
    
    # Get the dates for each dataset
    btc_dates = btc_prepared['Date']
    eth_dates = eth_prepared['Date']
    
    # Keep actual prices
    btc_prices = btc_prepared['Price'].values
    eth_prices = eth_prepared['Price'].values
    
    print(f"Bitcoin: {len(btc_prices)} data points from {btc_dates.min().strftime('%Y-%m-%d')} to {btc_dates.max().strftime('%Y-%m-%d')}")
    print(f"Ethereum: {len(eth_prices)} data points from {eth_dates.min().strftime('%Y-%m-%d')} to {eth_dates.max().strftime('%Y-%m-%d')}")
    print(f"Bitcoin price range: ${btc_prices.min():.2f} to ${btc_prices.max():.2f}")
    print(f"Ethereum price range: ${eth_prices.min():.2f} to ${eth_prices.max():.2f}")
    
    return btc_dates, eth_dates, btc_prices, eth_prices, btc_prepared, eth_prepared

def create_morphing_animation(btc_dates, eth_dates, btc_prices, eth_prices, btc_df, eth_df):
    """Create morphing animation from Bitcoin to Ethereum with different time periods"""
    print("Creating morphing animation...")
    
    # Setup the plot
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # Set background to pure white
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Enhanced grid
    ax.grid(True, alpha=0.2, color='#CCCCCC', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Style the axes
    ax.tick_params(axis='both', which='major', labelsize=11, color='#666666')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')
    
    # Set labels - remove for clean look
    ax.set_xlabel('')  # Remove x-axis label
    ax.set_ylabel('')  # Remove y-axis label
    
    # Format y-axis to show prices nicely
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Rotate x-axis labels
    plt.xticks(rotation=45)
    
    # Add subtle grid lines for key price levels (no labels) - match zoom animation
    key_prices = [100, 1000, 10000, 20000, 50000, 100000, 150000]
    price_lines = []
    for price in key_prices:
        if price <= max(btc_prices.max(), eth_prices.max()):
            line_h = ax.axhline(y=price, color='lightgray', alpha=0.2, linestyle='-', linewidth=0.5)
            price_lines.append(line_h)
    
    plt.tight_layout()
    
    # Animation parameters
    frames = 120  # 6 seconds at 20 FPS
    interval = 50  # 50ms between frames
    
    def update(frame):
        ax.clear()
        
        # Set background and grid again
        ax.set_facecolor('white')
        ax.grid(True, alpha=0.2, color='#CCCCCC', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Style the axes - match zoom animation exactly
        ax.tick_params(axis='both', which='major', labelsize=11, color='#666666')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#CCCCCC')
        ax.spines['bottom'].set_color('#CCCCCC')
        
        # Format y-axis to show prices nicely
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Rotate x-axis labels
        plt.xticks(rotation=45)
        
        # Calculate morphing progress (0 to 1)
        progress = frame / (frames - 1)
        
        # Better easing function for more proportional morphing
        # This creates a smoother, more proportional transition
        ease_progress = progress ** 2 * (3 - 2 * progress)  # Smoothstep function
        
        # Interpolate between Bitcoin and Ethereum prices
        # We need to handle different array lengths by resampling to same length
        max_len = max(len(btc_prices), len(eth_prices))
        
        # Create normalized indices for interpolation
        btc_indices = np.linspace(0, 1, len(btc_prices))
        eth_indices = np.linspace(0, 1, len(eth_prices))
        morphed_indices = np.linspace(0, 1, max_len)
        
        # Interpolate prices to same length
        from scipy.interpolate import interp1d
        btc_interp = interp1d(btc_indices, btc_prices, kind='linear', fill_value='extrapolate')
        eth_interp = interp1d(eth_indices, eth_prices, kind='linear', fill_value='extrapolate')
        
        btc_resampled = btc_interp(morphed_indices)
        eth_resampled = eth_interp(morphed_indices)
        
        # Now interpolate between the resampled prices
        morphed_prices = (1 - ease_progress) * btc_resampled + ease_progress * eth_resampled
        
        # Interpolate between price ranges for y-axis scaling with slower morphing
        btc_min, btc_max = btc_prices.min(), btc_prices.max()
        eth_min, eth_max = eth_prices.min(), eth_prices.max()
        
        # Use a slower easing for y-axis to prevent rapid changes
        y_ease_progress = ease_progress ** 1.5  # Slower morphing for y-axis
        morphed_min = (1 - y_ease_progress) * btc_min + y_ease_progress * eth_min
        morphed_max = (1 - y_ease_progress) * btc_max + y_ease_progress * eth_max
        
        # Determine color based on progress - change at 75% (delayed switch)
        if ease_progress < 0.75:
            # Bitcoin (orange)
            color = (0.969, 0.580, 0.102, 1)  # Bitcoin orange
        else:
            # Ethereum (blue)
            color = (0.384, 0.494, 0.918, 1)  # Ethereum blue
        
        # Create morphed dates that interpolate between the two time periods
        btc_date_nums = (btc_dates - btc_dates.min()).dt.total_seconds()
        eth_date_nums = (eth_dates - eth_dates.min()).dt.total_seconds()
        
        # Normalize date ranges to 0-1
        btc_date_norm = btc_date_nums / btc_date_nums.max()
        eth_date_norm = eth_date_nums / eth_date_nums.max()
        
        # Interpolate dates to same length
        btc_date_interp = interp1d(btc_indices, btc_date_norm, kind='linear', fill_value='extrapolate')
        eth_date_interp = interp1d(eth_indices, eth_date_norm, kind='linear', fill_value='extrapolate')
        
        btc_dates_resampled = btc_date_interp(morphed_indices)
        eth_dates_resampled = eth_date_interp(morphed_indices)
        
        # Interpolate between the date ranges
        morphed_dates_norm = (1 - ease_progress) * btc_dates_resampled + ease_progress * eth_dates_resampled
        
        # Convert back to actual dates
        morphed_dates = btc_dates.min() + pd.to_timedelta(morphed_dates_norm * btc_date_nums.max(), unit='s')
        
        # Plot the morphed line
        line, = ax.plot(morphed_dates, morphed_prices, color=color, linewidth=3, alpha=0.9)
        
        # Update title with smooth fade transition using consistent text positioning
        if ease_progress < 0.75:
            # Bitcoin title (full opacity)
            btc_title = "Bitcoin (2014-2020)"
            ax.text(0.5, 1.02, btc_title, transform=ax.transAxes, fontsize=16, fontweight='normal', 
                   color='#333333', ha='center', va='bottom', alpha=1.0)
        elif ease_progress < 0.85:
            # Fade transition period (75% to 85%)
            fade_progress = (ease_progress - 0.75) / 0.1  # 0 to 1 over the fade period
            
            # Fade out Bitcoin title
            btc_alpha = 1.0 - fade_progress
            btc_title = "Bitcoin (2014-2020)"
            ax.text(0.5, 1.02, btc_title, transform=ax.transAxes, fontsize=16, fontweight='normal', 
                   color='#333333', ha='center', va='bottom', alpha=btc_alpha)
            
            # Fade in Ethereum title (same positioning)
            eth_alpha = fade_progress
            eth_title = "Ethereum (2016-2025)"
            ax.text(0.5, 1.02, eth_title, transform=ax.transAxes, fontsize=16, fontweight='normal', 
                   color='#333333', ha='center', va='bottom', alpha=eth_alpha)
        else:
            # Ethereum title (full opacity)
            eth_title = "Ethereum (2016-2025)"
            ax.text(0.5, 1.02, eth_title, transform=ax.transAxes, fontsize=16, fontweight='normal', 
                   color='#333333', ha='center', va='bottom', alpha=1.0)
        
        # Set axis limits that morph between the two time periods and price ranges
        morphed_date_min = morphed_dates.min()
        morphed_date_max = morphed_dates.max()
        ax.set_xlim(morphed_date_min, morphed_date_max)
        ax.set_ylim(morphed_min * 0.8, morphed_max * 1.2)
        
        # Keep x-axis completely fixed - no morphing
        # Use Bitcoin's time period as the fixed range
        ax.set_xlim(btc_dates.min(), btc_dates.max())
        
        # Plot the morphed line
        line, = ax.plot(morphed_dates, morphed_prices, color=color, linewidth=3, alpha=0.9)
        
        # Manually control x-axis labels to match the static charts exactly
        if ease_progress < 0.75:
            # Bitcoin period labels - match the Bitcoin chart exactly
            # 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021
            years = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]
        else:
            # Ethereum period labels - match the Ethereum chart exactly  
            # 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026
            years = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]
        
        # Set custom x-axis labels with proper spacing
        num_ticks = len(years)
        tick_positions = np.linspace(0, 1, num_ticks)
        
        # Convert tick positions to actual dates within our fixed range
        date_range = btc_dates.max() - btc_dates.min()
        actual_tick_positions = btc_dates.min() + tick_positions * date_range
        
        ax.set_xticks(actual_tick_positions)
        ax.set_xticklabels([str(year) for year in years], rotation=45)
        
        # Simple x-axis label - remove for clean look
        ax.set_xlabel('')
        
        # Format y-axis to show prices nicely
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Simple x-axis formatting - no custom labels to avoid warnings
        ax.tick_params(axis='x', rotation=45)
        
        # Add legend
        if ease_progress < 0.3:
            ax.legend(['Bitcoin (BTC)'], fontsize=12, framealpha=0.9, fancybox=True, shadow=True, loc='upper left')
        elif ease_progress > 0.7:
            ax.legend(['Ethereum (ETH)'], fontsize=12, framealpha=0.9, fancybox=True, shadow=True, loc='upper left')
        else:
            ax.legend(['Bitcoin → Ethereum'], fontsize=12, framealpha=0.9, fancybox=True, shadow=True, loc='upper left')
        
        return [line] + price_lines
    
    # Create animation
    print(f"Creating animation with {frames} frames...")
    ani = FuncAnimation(fig, update, frames=frames, interval=interval, blit=True, repeat=False)
    
    # Save the animation
    print("Saving animation...")
    save_path = 'Bitcoin/Visualizations/Images/btc_to_eth_morph.mp4'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    try:
        # Try to save as MP4 (requires ffmpeg)
        ani.save(save_path, writer='ffmpeg', fps=20, dpi=100)
        print(f"Animation saved as MP4: {save_path}")
    except:
        try:
            # Try to save as GIF (requires pillow)
            gif_path = save_path.replace('.mp4', '.gif')
            ani.save(gif_path, writer='pillow', fps=20)
            print(f"Animation saved as GIF: {gif_path}")
        except Exception as e:
            print(f"Could not save animation: {e}")
            print("Showing animation in window instead...")
            plt.show()
    
    return ani

def main():
    """Main function"""
    print("Bitcoin to Ethereum Morphing Animation")
    print("="*50)
    
    # Load data
    btc_df = load_bitcoin_data()
    eth_df = load_ethereum_data()
    
    # Prepare data for morphing
    btc_dates, eth_dates, btc_prices, eth_prices, btc_prepared, eth_prepared = prepare_data_for_morphing(btc_df, eth_df)
    
    # Create morphing animation
    ani = create_morphing_animation(btc_dates, eth_dates, btc_prices, eth_prices, btc_prepared, eth_prepared)
    
    print("\n✅ Morphing animation completed!")

if __name__ == "__main__":
    main() 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

def load_bitcoin_data():
    """Load Bitcoin data for entire history"""
    print("Loading Bitcoin dataset...")
    df = pd.read_csv('Bitcoin/Data/2010_2025_Daily_Data_(BTC).csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Convert Price to numeric if needed
    if df['Price'].dtype == 'object':
        df['Price'] = pd.to_numeric(df['Price'].str.replace(',', '').str.replace('$', ''), errors='coerce')
    
    print(f"Loaded {len(df)} days of Bitcoin data")
    print(f"Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    
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
    
    return df

def create_combined_animation(btc_df, eth_df):
    """Create combined zoom + morph animation"""
    print("Creating combined zoom + morph animation...")
    
    # Setup the plot
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # Set background to pure white
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Plot the Bitcoin price line from 2014 onwards
    df_filtered = btc_df[btc_df['Date'] >= '2014-01-01'].copy()
    line, = ax.plot(df_filtered['Date'], df_filtered['Price'], color='#F7931A', linewidth=2.5, label='Bitcoin (BTC)', alpha=0.9)
    
    # Customize the plot with professional styling
    ax.set_xlabel('')  # Remove x-axis label
    ax.set_ylabel('')  # Remove y-axis label
    ax.set_title('Bitcoin (2014-2025)', fontsize=16, fontweight='normal', color='#333333', pad=15)
    
    # Enhanced grid
    ax.grid(True, alpha=0.2, color='#CCCCCC', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Style the axes
    ax.tick_params(axis='both', which='major', labelsize=11, color='#666666')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')
    
    # Legend styling
    legend = ax.legend(fontsize=12, framealpha=0.9, fancybox=True, shadow=True, loc='upper left')
    
    # Format y-axis to show prices nicely
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Rotate x-axis labels
    plt.xticks(rotation=45)
    
    # Add subtle grid lines for key price levels
    key_prices = [100, 1000, 10000, 20000, 50000, 100000, 150000]
    price_lines = []
    for price in key_prices:
        if price <= btc_df['Price'].max():
            line_h = ax.axhline(y=price, color='lightgray', alpha=0.2, linestyle='-', linewidth=0.5)
            price_lines.append(line_h)
    
    plt.tight_layout()
    
    # Animation parameters
    total_frames = 200  # Total frames for both animations
    zoom_frames = 80    # First 80 frames for zoom
    morph_frames = 120  # Last 120 frames for morph
    
    # Zoom parameters
    x_start = pd.to_datetime('2014-01-01')
    x_end = btc_df['Date'].max()
    y_start = btc_df['Price'].min() * 0.8
    y_end = btc_df['Price'].max() * 1.2
    
    # Target zoom area (2014-2020)
    target_x_start = pd.to_datetime('2014-01-01')
    target_x_end = pd.to_datetime('2020-10-20')
    target_y_start = btc_df[btc_df['Date'] <= '2020-10-20']['Price'].min() * 0.8
    target_y_end = btc_df[btc_df['Date'] <= '2020-10-20']['Price'].max() * 1.2
    
    # Prepare Ethereum data for morphing
    eth_dates = eth_df['Date']
    eth_prices = eth_df['Price'].values
    
    # Bitcoin data for morphing (2014-2020)
    btc_morph_df = btc_df[(btc_df['Date'] >= '2014-01-01') & (btc_df['Date'] <= '2020-10-20')].copy()
    btc_morph_dates = btc_morph_df['Date']
    btc_morph_prices = btc_morph_df['Price'].values
    
    def update(frame):
        ax.clear()
        
        # Set background and grid again
        ax.set_facecolor('white')
        ax.grid(True, alpha=0.2, color='#CCCCCC', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Apply the same clean styling in each frame
        ax.tick_params(axis='both', which='major', labelsize=11, color='#666666')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#CCCCCC')
        ax.spines['bottom'].set_color('#CCCCCC')
        
        # Format y-axis to show prices nicely
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Rotate x-axis labels
        plt.xticks(rotation=45)
        
        if frame < zoom_frames:
            # ZOOM PHASE (frames 0-79)
            zoom_progress = frame / zoom_frames
            ease_progress = 0.5 * (1 - np.cos(zoom_progress * np.pi))
            
            # Interpolate between initial and target view
            x_start_new = x_start + (target_x_start - x_start) * ease_progress
            x_end_new = x_end + (target_x_end - x_end) * ease_progress
            y_start_new = y_start + (target_y_start - y_start) * ease_progress
            y_end_new = y_end + (target_y_end - y_end) * ease_progress
            
            # Update plot limits
            ax.set_xlim(x_start_new, x_end_new)
            ax.set_ylim(y_start_new, y_end_new)
            
            # Plot Bitcoin data
            df_visible = df_filtered[(df_filtered['Date'] >= x_start_new) & (df_filtered['Date'] <= x_end_new)]
            line, = ax.plot(df_visible['Date'], df_visible['Price'], color='#F7931A', linewidth=2.5, alpha=0.9)
            
            # Update title with consistent positioning
            if zoom_progress < 0.75:
                ax.text(0.5, 1.02, 'Bitcoin (2014-2025)', transform=ax.transAxes, fontsize=16, fontweight='normal', 
                       color='#333333', ha='center', va='bottom', alpha=1.0)
            else:
                ax.text(0.5, 1.02, 'Bitcoin (2014-2020)', transform=ax.transAxes, fontsize=16, fontweight='normal', 
                       color='#333333', ha='center', va='bottom', alpha=1.0)
            
            # Set custom x-axis labels for zoom phase (2014-2020 only)
            years = [2014, 2015, 2016, 2017, 2018, 2019, 2020]
            num_ticks = len(years)
            tick_positions = np.linspace(0, 1, num_ticks)
            date_range = x_end_new - x_start_new
            actual_tick_positions = x_start_new + tick_positions * date_range
            
            ax.set_xticks(actual_tick_positions)
            ax.set_xticklabels([str(year) for year in years], rotation=45)
            
            # Add legend
            ax.legend(['Bitcoin (BTC)'], fontsize=12, framealpha=0.9, fancybox=True, shadow=True, loc='upper left')
            
            return [line] + price_lines + [legend]
            
        else:
            # MORPH PHASE (frames 80-199)
            morph_progress = (frame - zoom_frames) / morph_frames
            ease_progress = morph_progress ** 2 * (3 - 2 * morph_progress)  # Smoothstep function
            
            # Interpolate between Bitcoin and Ethereum prices
            max_len = max(len(btc_morph_prices), len(eth_prices))
            
            # Create normalized indices for interpolation
            btc_indices = np.linspace(0, 1, len(btc_morph_prices))
            eth_indices = np.linspace(0, 1, len(eth_prices))
            morphed_indices = np.linspace(0, 1, max_len)
            
            # Interpolate prices to same length
            from scipy.interpolate import interp1d
            btc_interp = interp1d(btc_indices, btc_morph_prices, kind='linear', fill_value='extrapolate')
            eth_interp = interp1d(eth_indices, eth_prices, kind='linear', fill_value='extrapolate')
            
            btc_resampled = btc_interp(morphed_indices)
            eth_resampled = eth_interp(morphed_indices)
            
            # Now interpolate between the resampled prices
            morphed_prices = (1 - ease_progress) * btc_resampled + ease_progress * eth_resampled
            
            # Interpolate between price ranges for y-axis scaling
            btc_min, btc_max = btc_morph_prices.min(), btc_morph_prices.max()
            eth_min, eth_max = eth_prices.min(), eth_prices.max()
            
            # Use a slower easing for y-axis
            y_ease_progress = ease_progress ** 1.5
            morphed_min = (1 - y_ease_progress) * btc_min + y_ease_progress * eth_min
            morphed_max = (1 - y_ease_progress) * btc_max + y_ease_progress * eth_max
            
            # Determine color based on progress
            if ease_progress < 0.75:
                color = (0.969, 0.580, 0.102, 1)  # Bitcoin orange
            else:
                color = (0.384, 0.494, 0.918, 1)  # Ethereum blue
            
            # Create morphed dates
            btc_date_nums = (btc_morph_dates - btc_morph_dates.min()).dt.total_seconds()
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
            morphed_dates = btc_morph_dates.min() + pd.to_timedelta(morphed_dates_norm * btc_date_nums.max(), unit='s')
            
            # Keep x-axis fixed to Bitcoin's time period
            ax.set_xlim(btc_morph_dates.min(), btc_morph_dates.max())
            ax.set_ylim(morphed_min * 0.8, morphed_max * 1.2)
            
            # Plot the morphed line
            line, = ax.plot(morphed_dates, morphed_prices, color=color, linewidth=3, alpha=0.9)
            
            # Update title with smooth fade transition
            if ease_progress < 0.75:
                btc_title = "Bitcoin (2014-2020)"
                ax.text(0.5, 1.02, btc_title, transform=ax.transAxes, fontsize=16, fontweight='normal', 
                       color='#333333', ha='center', va='bottom', alpha=1.0)
            elif ease_progress < 0.85:
                # Fade transition period
                fade_progress = (ease_progress - 0.75) / 0.1
                
                # Fade out Bitcoin title
                btc_alpha = 1.0 - fade_progress
                btc_title = "Bitcoin (2014-2020)"
                ax.text(0.5, 1.02, btc_title, transform=ax.transAxes, fontsize=16, fontweight='normal', 
                       color='#333333', ha='center', va='bottom', alpha=btc_alpha)
                
                # Fade in Ethereum title
                eth_alpha = fade_progress
                eth_title = "Ethereum (2016-2025)"
                ax.text(0.5, 1.02, eth_title, transform=ax.transAxes, fontsize=16, fontweight='normal', 
                       color='#333333', ha='center', va='bottom', alpha=eth_alpha)
            else:
                # Ethereum title (full opacity)
                eth_title = "Ethereum (2016-2025)"
                ax.text(0.5, 1.02, eth_title, transform=ax.transAxes, fontsize=16, fontweight='normal', 
                       color='#333333', ha='center', va='bottom', alpha=1.0)
            
            # Manually control x-axis labels
            if ease_progress < 0.75:
                years = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]
            else:
                years = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]
            
            # Set custom x-axis labels
            num_ticks = len(years)
            tick_positions = np.linspace(0, 1, num_ticks)
            date_range = btc_morph_dates.max() - btc_morph_dates.min()
            actual_tick_positions = btc_morph_dates.min() + tick_positions * date_range
            
            ax.set_xticks(actual_tick_positions)
            ax.set_xticklabels([str(year) for year in years], rotation=45)
            
            # Add legend
            if ease_progress < 0.3:
                ax.legend(['Bitcoin (BTC)'], fontsize=12, framealpha=0.9, fancybox=True, shadow=True, loc='upper left')
            elif ease_progress > 0.7:
                ax.legend(['Ethereum (ETH)'], fontsize=12, framealpha=0.9, fancybox=True, shadow=True, loc='upper left')
            else:
                ax.legend(['Bitcoin → Ethereum'], fontsize=12, framealpha=0.9, fancybox=True, shadow=True, loc='upper left')
            
            return [line] + price_lines
    
    # Create animation
    print(f"Creating combined animation with {total_frames} frames...")
    ani = FuncAnimation(fig, update, frames=total_frames, interval=50, blit=True, repeat=False)
    
    # Save the animation
    print("Saving animation...")
    save_path = 'Bitcoin/Visualizations/Images/bitcoin_zoom_to_ethereum_morph.mp4'
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
    print("Bitcoin Zoom to Ethereum Morph Animation")
    print("="*50)
    
    # Load data
    btc_df = load_bitcoin_data()
    eth_df = load_ethereum_data()
    
    # Create combined animation
    ani = create_combined_animation(btc_df, eth_df)
    
    print("\n✅ Combined animation completed!")

if __name__ == "__main__":
    main() 
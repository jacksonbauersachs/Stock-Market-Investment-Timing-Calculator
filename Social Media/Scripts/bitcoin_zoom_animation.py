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

def create_zoom_animation(df):
    """Create zoom animation of Bitcoin price chart"""
    print("Creating zoom animation...")
    
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # Set background to pure white
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Plot the Bitcoin price line from 2014 onwards with enhanced styling
    df_filtered = df[df['Date'] >= '2014-01-01'].copy()
    line, = ax.plot(df_filtered['Date'], df_filtered['Price'], color='#F7931A', linewidth=2.5, label='Bitcoin (BTC)', alpha=0.9)
    
    # Customize the plot with professional styling - match the morph animation exactly
    ax.set_xlabel('')  # Remove x-axis label
    ax.set_ylabel('')  # Remove y-axis label
    ax.set_title('Bitcoin (2014-2025)', fontsize=16, fontweight='normal', color='#333333', pad=15)
    
    # Enhanced grid
    ax.grid(True, alpha=0.2, color='#CCCCCC', linewidth=0.5)
    ax.set_axisbelow(True)  # Put grid behind data
    
    # Style the axes - match morph animation exactly
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
    
    # Add subtle grid lines for key price levels (no labels) - match static chart
    key_prices = [100, 1000, 10000, 20000, 50000, 100000, 150000]
    price_lines = []
    for price in key_prices:
        if price <= df['Price'].max():
            line_h = ax.axhline(y=price, color='lightgray', alpha=0.2, linestyle='-', linewidth=0.5)
            price_lines.append(line_h)
    
    # Remove data info box for cleaner look - match static chart
    
    plt.tight_layout()
    
    # Define zoom parameters
    frames = 80  # Number of animation frames (80 ÷ 20 = 4 seconds)
    zoom_factor = 0.95  # How much to zoom per frame (closer to 1 = slower)
    
    # Initial view (2014-2025 range - matches the static chart)
    x_start = pd.to_datetime('2014-01-01')
    x_end = df['Date'].max()  # End at today
    y_start = df['Price'].min() * 0.8  # Start with some padding
    y_end = df['Price'].max() * 1.2
    
    # Target zoom area (focus on 2014-2020) - match morph animation exactly
    target_x_start = pd.to_datetime('2014-01-01')
    target_x_end = pd.to_datetime('2020-10-20')
    # Use the exact same price range as the morph animation
    target_y_start = df[df['Date'] <= '2020-10-20']['Price'].min() * 0.8
    target_y_end = df[df['Date'] <= '2020-10-20']['Price'].max() * 1.2
    
    def update(frame):
        nonlocal x_start, x_end, y_start, y_end
        
        # Calculate progress (0 to 1)
        progress = frame / frames
        
        # Smooth easing function (ease-in-out)
        ease_progress = 0.5 * (1 - np.cos(progress * np.pi))
        
        # Apply the same clean styling in each frame
        ax.tick_params(axis='both', which='major', labelsize=11, color='#666666')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#CCCCCC')
        ax.spines['bottom'].set_color('#CCCCCC')
        
        # Interpolate between initial and target view
        x_start_new = x_start + (target_x_start - x_start) * ease_progress
        x_end_new = x_end + (target_x_end - x_end) * ease_progress
        y_start_new = y_start + (target_y_start - y_start) * ease_progress
        y_end_new = y_end + (target_y_end - y_end) * ease_progress
        
        # Update plot limits
        ax.set_xlim(x_start_new, x_end_new)
        ax.set_ylim(y_start_new, y_end_new)
        
        # Update title to show current zoom level - match morph animation exactly
        if progress < 0.75:
            ax.set_title(f'Bitcoin (2014-2025)', fontsize=16, fontweight='normal', color='#333333')
        else:
            ax.set_title(f'Bitcoin (2014-2020)', fontsize=16, fontweight='normal', color='#333333')
        
        return [line] + price_lines + [legend]
    
    # Create animation
    print(f"Creating animation with {frames} frames...")
    ani = FuncAnimation(fig, update, frames=frames, interval=50, blit=True, repeat=False)
    
    # Save the animation
    print("Saving animation...")
    save_path = 'Bitcoin/Visualizations/Images/bitcoin_zoom_animation.mp4'
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
    print("Bitcoin Price History Zoom Animation")
    print("="*50)
    
    # Load data
    df = load_bitcoin_data()
    
    # Create animation
    ani = create_zoom_animation(df)
    
    print("\n✅ Animation creation completed!")
    print("\nNote: If the animation didn't save, it will play in the window.")
    print("To save animations, you may need to install:")
    print("  - ffmpeg: for MP4 files")
    print("  - pillow: for GIF files")

if __name__ == "__main__":
    main() 
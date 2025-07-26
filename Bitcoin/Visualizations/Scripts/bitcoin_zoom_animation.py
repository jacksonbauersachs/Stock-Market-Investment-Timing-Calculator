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
    
    # Plot the full Bitcoin price line with enhanced styling
    line, = ax.plot(df['Date'], df['Price'], color='#F7931A', linewidth=2.5, label='Bitcoin (BTC)', alpha=0.9)
    
    # Customize the plot with professional styling
    ax.set_xlabel('Date', fontsize=14, fontweight='bold', color='#333333')
    ax.set_ylabel('Price (USD)', fontsize=14, fontweight='bold', color='#333333')
    ax.set_title('Bitcoin Price History (Zoom Animation)', fontsize=20, fontweight='bold', color='#333333', pad=20)
    
    # Enhanced grid
    ax.grid(True, alpha=0.2, color='#CCCCCC', linewidth=0.5)
    ax.set_axisbelow(True)  # Put grid behind data
    
    # Legend styling
    ax.legend(fontsize=12, framealpha=0.9, fancybox=True, shadow=True)
    
    # Format y-axis to show prices nicely
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Rotate x-axis labels
    plt.xticks(rotation=45)
    
    # Add key price levels
    key_prices = [100, 1000, 10000, 20000, 50000, 100000]
    price_lines = []
    price_texts = []
    for price in key_prices:
        if price <= df['Price'].max():
            line_h = ax.axhline(y=price, color='gray', alpha=0.3, linestyle='--', linewidth=1)
            text_h = ax.text(df['Date'].iloc[-1], price, f'${price:,}', 
                           verticalalignment='bottom', horizontalalignment='right', 
                           color='black', alpha=0.7, fontsize=10)
            price_lines.append(line_h)
            price_texts.append(text_h)
    
    # Add data info
    data_text = f'Data: {df["Date"].min().strftime("%b %Y")} - {df["Date"].max().strftime("%b %Y")}\nPoints: {len(df):,}'
    info_text = ax.text(0.98, 0.02, data_text, transform=ax.transAxes, fontsize=10, color='black', 
                       bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'),
                       horizontalalignment='right', verticalalignment='bottom')
    
    plt.tight_layout()
    
    # Define zoom parameters
    frames = 80  # Number of animation frames (80 ÷ 20 = 4 seconds)
    zoom_factor = 0.95  # How much to zoom per frame (closer to 1 = slower)
    
    # Initial view (full range)
    x_start = df['Date'].min()
    x_end = df['Date'].max()
    y_start = df['Price'].min() * 0.8  # Start with some padding
    y_end = df['Price'].max() * 1.2
    
    # Target zoom area (focus on 2010-2020)
    target_x_start = pd.to_datetime('2010-07-18')
    target_x_end = pd.to_datetime('2020-10-20')
    target_y_start = df['Price'].min() * 0.8
    target_y_end = df[df['Date'] <= '2020-10-20']['Price'].max() * 1.2
    
    def update(frame):
        nonlocal x_start, x_end, y_start, y_end
        
        # Calculate progress (0 to 1)
        progress = frame / frames
        
        # Smooth easing function (ease-in-out)
        ease_progress = 0.5 * (1 - np.cos(progress * np.pi))
        
        # Interpolate between initial and target view
        x_start_new = x_start + (target_x_start - x_start) * ease_progress
        x_end_new = x_end + (target_x_end - x_end) * ease_progress
        y_start_new = y_start + (target_y_start - y_start) * ease_progress
        y_end_new = y_end + (target_y_end - y_end) * ease_progress
        
        # Update plot limits
        ax.set_xlim(x_start_new, x_end_new)
        ax.set_ylim(y_start_new, y_end_new)
        
        # Update title to show current zoom level
        if progress < 0.95:
            ax.set_title(f'Bitcoin Price History', fontsize=16, fontweight='bold')
        else:
            ax.set_title(f'Bitcoin Price History: 2010-2020', fontsize=16, fontweight='bold')
        
        # Update price level text positions
        for i, text in enumerate(price_texts):
            if key_prices[i] >= y_start_new and key_prices[i] <= y_end_new:
                text.set_position((x_end_new, key_prices[i]))
                text.set_alpha(0.7)
            else:
                text.set_alpha(0.0)
        
        return [line] + price_lines + price_texts + [info_text]
    
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
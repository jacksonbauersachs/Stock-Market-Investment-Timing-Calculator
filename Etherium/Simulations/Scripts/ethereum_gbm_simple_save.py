import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

def load_ethereum_data():
    """Load Ethereum historical data"""
    print("Loading Ethereum historical data...")
    df = pd.read_csv('Etherium/Data/Ethereum Historical Data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Convert Price to numeric if needed
    if df['Price'].dtype == 'object':
        df['Price'] = pd.to_numeric(df['Price'].str.replace(',', '').str.replace('$', ''), errors='coerce')
    
    print(f"Loaded {len(df)} days of Ethereum data")
    print(f"Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"Current price: ${df['Price'].iloc[-1]:,.2f}")
    return df

def dynamic_gbm_monte_carlo_with_save(S0, T, num_paths=1000, num_steps=3653):
    """Run Dynamic GBM simulation and save paths immediately"""
    print(f"\nRunning Dynamic GBM Monte Carlo Simulation with Immediate Save")
    print(f"="*60)
    print(f"Initial price: ${S0:,.2f}")
    print(f"Time horizon: {T} years")
    print(f"Number of paths: {num_paths:,}")
    print(f"Time steps: {num_steps:,} (daily)")
    
    # Load growth model coefficients (day 365 start)
    with open('Etherium/Models/Growth/Formulas/ethereum_growth_model_coefficients_day365.txt', 'r') as f:
        lines = f.readlines()
        slope = None
        intercept = None
        for line in lines:
            if line.startswith('a ='):
                slope = float(line.split('=')[1].strip())
            elif line.startswith('b ='):
                intercept = float(line.split('=')[1].strip())
                break
        
        if slope is None or intercept is None:
            raise ValueError("Could not read growth model coefficients")
    
    # Load volatility decay coefficients
    with open('Etherium/Models/Volatility/Formulas/ethereum_365day_volatility_decay_coefficients.txt', 'r') as f:
        lines = f.readlines()
        vol_slope = None
        vol_intercept = None
        in_exponential = False
        for line in lines:
            if 'Exponential Model:' in line:
                in_exponential = True
            elif in_exponential and line.strip().startswith('Slope ='):
                vol_slope = float(line.split('=')[1].strip())
            elif in_exponential and line.strip().startswith('Intercept ='):
                vol_intercept = float(line.split('=')[1].strip())
                break
        
        if vol_slope is None or vol_intercept is None:
            raise ValueError("Could not read volatility decay coefficients")
    
    # Ethereum's current age (since 2015-07-30)
    ethereum_current_age = 10.0  # Approximately 10 years old
    today_day = 3650  # Starting from day 365 for growth model
    
    # Time setup
    dt = T / num_steps
    time_steps = np.linspace(0, T, num_steps + 1)
    
    # Create output file immediately
    paths_filename = f'Etherium/Simulations/Results/ethereum_gbm_paths_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    print(f"Creating output file: {paths_filename}")
    
    # Write header
    with open(paths_filename, 'w') as f:
        f.write('Years')
        for i in range(num_paths):
            f.write(f',Path_{i+1}')
        f.write('\n')
    
    # Initialize price paths (only keep current step in memory)
    current_prices = np.full(num_paths, S0)
    
    print(f"Dynamic Parameters:")
    print(f"  Growth formula: log10(price) = {slope:.6f} * ln(day) + {intercept:.6f}")
    print(f"  Volatility formula: log(volatility) = {vol_slope:.6f} * age + {vol_intercept:.6f}")
    print(f"  Ethereum current age: {ethereum_current_age:.1f} years")
    
    # Calculate expected growth over 10 years
    current_fair_value = 10**(slope * np.log(today_day) + intercept)
    future_day_10y = today_day + int(10 * 365.25)
    future_fair_value_10y = 10**(slope * np.log(future_day_10y) + intercept)
    expected_growth_10y = (future_fair_value_10y / current_fair_value - 1) * 100
    
    # Calculate volatility at different ages
    current_vol = np.exp(vol_slope * ethereum_current_age + vol_intercept)
    vol_5y = np.exp(vol_slope * (ethereum_current_age + 5) + vol_intercept)
    vol_10y = np.exp(vol_slope * (ethereum_current_age + 10) + vol_intercept)
    
    print(f"\nExpected Growth (10 years):")
    print(f"  Current fair value: ${current_fair_value:,.2f}")
    print(f"  Future fair value (10y): ${future_fair_value_10y:,.2f}")
    print(f"  Expected growth: {expected_growth_10y:.1f}%")
    print(f"  Annualized growth: {(future_fair_value_10y / current_fair_value) ** (1/10) - 1:.1%}")
    
    print(f"\nVolatility Decay (10 years):")
    print(f"  Current volatility: {current_vol:.1f}%")
    print(f"  Volatility in 5 years: {vol_5y:.1f}%")
    print(f"  Volatility in 10 years: {vol_10y:.1f}%")
    
    print(f"\nDebug Info:")
    print(f"  today_day: {today_day}")
    print(f"  future_day_10y: {future_day_10y}")
    print(f"  vol_slope: {vol_slope}")
    print(f"  vol_intercept: {vol_intercept}")
    print(f"  ethereum_current_age: {ethereum_current_age}")
    
    # Run simulation with immediate saving
    np.random.seed(42)
    
    print(f"\nRunning simulation and saving paths...")
    for j in range(num_steps + 1):
        current_time = time_steps[j]
        
        # Save current prices to file
        with open(paths_filename, 'a') as f:
            f.write(f'{current_time:.6f}')
            for price in current_prices:
                f.write(f',{price:.2f}')
            f.write('\n')
        
        # Progress indicator
        if j % 365 == 0:
            print(f"  Year {j//365}: ${np.mean(current_prices):,.0f}")
            if j == 365:  # Debug first year
                print(f"    Debug - mu: {mu:.6f}, sigma: {sigma:.6f}, drift: {drift:.6f}, volatility: {volatility:.6f}")
        
        # Calculate next step if not at the end
        if j < num_steps:
            # Calculate dynamic parameters
            future_age = ethereum_current_age + current_time
            future_day = today_day + int(current_time * 365.25)
            
            # Dynamic growth rate
            current_fair_value = 10**(slope * np.log(today_day) + intercept)
            future_fair_value = 10**(slope * np.log(future_day) + intercept)
            mu = (future_fair_value / current_fair_value) ** (1/current_time) - 1 if current_time > 0 else 0
            
            # Dynamic volatility (exponential decay) - convert from percentage to decimal
            sigma = np.exp(vol_slope * future_age + vol_intercept) / 100
            
            # GBM parameters
            drift = (mu - 0.5 * sigma**2) * dt
            volatility = sigma * np.sqrt(dt)
            
            # Update prices
            random_shocks = np.random.normal(0, 1, num_paths)
            current_prices = current_prices * np.exp(drift + volatility * random_shocks)
    
    print(f"✅ Simulation complete! Paths saved to: {paths_filename}")
    return paths_filename

def analyze_gbm_results_from_file(paths_filename):
    """Analyze GBM results from saved file"""
    print(f"\nAnalyzing GBM results from {paths_filename}...")
    
    # Load the saved paths
    df = pd.read_csv(paths_filename, index_col=0)
    
    # Calculate statistics at different time points
    time_points = [0, 0.25, 0.5, 1, 2, 3, 5, 10]  # Years
    results = {}
    
    for year_point in time_points:
        if year_point <= 10:
            # Find closest time step
            time_steps = df.index.astype(float)
            step_index = np.argmin(np.abs(time_steps - year_point))
            
            prices_at_time = df.iloc[step_index, :].values
            
            mean_price = np.mean(prices_at_time)
            median_price = np.median(prices_at_time)
            std_price = np.std(prices_at_time)
            
            # Percentiles
            p5 = np.percentile(prices_at_time, 5)
            p25 = np.percentile(prices_at_time, 25)
            p75 = np.percentile(prices_at_time, 75)
            p95 = np.percentile(prices_at_time, 95)
            
            results[year_point] = {
                'mean': mean_price,
                'median': median_price,
                'std': std_price,
                'p5': p5,
                'p25': p25,
                'p75': p75,
                'p95': p95
            }
            
            print(f"Year {year_point}: ${mean_price:,.0f} (${p5:,.0f} - ${p95:,.0f})")
    
    return results

def save_summary_statistics(results, paths_filename):
    """Save summary statistics"""
    summary_filename = paths_filename.replace('_paths_', '_summary_')
    
    summary_data = []
    for year in sorted(results.keys()):
        r = results[year]
        summary_data.append({
            'Year': year,
            'Mean_Price': r['mean'],
            'Median_Price': r['median'],
            'Std_Dev': r['std'],
            'P5_Price': r['p5'],
            'P25_Price': r['p25'],
            'P75_Price': r['p75'],
            'P95_Price': r['p95']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(summary_filename, index=False)
    print(f"Summary statistics saved to: {summary_filename}")
    
    return summary_filename

def create_gbm_visualization(paths_filename, results):
    """Create visualization of GBM results"""
    print(f"\nCreating GBM visualization...")
    
    # Load the paths data
    df = pd.read_csv(paths_filename, index_col=0)
    
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Plot 1: Price paths
    time_steps = df.index.astype(float)
    
    # Plot all paths (transparent)
    for col in df.columns:
        ax1.plot(time_steps, df[col], alpha=0.1, color='cyan', linewidth=0.5)
    
    # Plot mean path
    mean_path = df.mean(axis=1)
    ax1.plot(time_steps, mean_path, color='white', linewidth=3, label='Mean Path')
    
    # Plot percentiles
    p5_path = df.quantile(0.05, axis=1)
    p25_path = df.quantile(0.25, axis=1)
    p75_path = df.quantile(0.75, axis=1)
    p95_path = df.quantile(0.95, axis=1)
    
    ax1.fill_between(time_steps, p5_path, p95_path, alpha=0.3, color='red', label='90% Confidence')
    ax1.fill_between(time_steps, p25_path, p75_path, alpha=0.5, color='orange', label='50% Confidence')
    
    ax1.set_xlabel('Years', fontsize=12)
    ax1.set_ylabel('Ethereum Price ($)', fontsize=12)
    ax1.set_title('Ethereum GBM Simulation - Price Paths', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_yscale('log')
    
    # Plot 2: Statistics over time
    years = sorted(results.keys())
    means = [results[year]['mean'] for year in years]
    p5s = [results[year]['p5'] for year in years]
    p95s = [results[year]['p95'] for year in years]
    
    ax2.plot(years, means, 'o-', color='white', linewidth=3, markersize=8, label='Mean Price')
    ax2.fill_between(years, p5s, p95s, alpha=0.3, color='red', label='90% Confidence')
    
    ax2.set_xlabel('Years', fontsize=12)
    ax2.set_ylabel('Ethereum Price ($)', fontsize=12)
    ax2.set_title('Ethereum GBM Simulation - Price Statistics', fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_yscale('log')
    
    # Add simulation info
    info_text = f'Paths: {len(df.columns):,}\nTime Steps: {len(df):,}\nSimulation: 10 years'
    ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes, fontsize=10, color='white', 
             bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.5'),
             verticalalignment='top')
    
    plt.tight_layout()
    
    # Save the plot
    viz_filename = paths_filename.replace('_paths_', '_visualization_').replace('.csv', '.png')
    viz_filename = viz_filename.replace('Simulations/Results/', 'Visualizations/')
    os.makedirs(os.path.dirname(viz_filename), exist_ok=True)
    plt.savefig(viz_filename, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {viz_filename}")
    plt.show()

def main():
    """Main function"""
    print("Ethereum Dynamic GBM with Immediate Save and Analysis")
    print("="*60)
    
    # Load data
    df = load_ethereum_data()
    current_price = df['Price'].iloc[-1]
    
    # Load growth model coefficients to calculate fair value
    with open('Etherium/Models/Growth/Formulas/ethereum_growth_model_coefficients_day365.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('a ='):
                slope = float(line.split('=')[1].strip())
            elif line.startswith('b ='):
                intercept = float(line.split('=')[1].strip())
                break
    
    today_day = 3650  # Starting from day 365 for growth model
    S0 = 10**(slope * np.log(today_day) + intercept)
    print(f"Fair value (formula): ${S0:,.2f}")
    print(f"Current price: ${current_price:,.2f}")
    
    # For quick exploratory runs, use fewer paths (e.g., 100 instead of 1000)
    paths_file = dynamic_gbm_monte_carlo_with_save(S0, T=10, num_paths=100, num_steps=3653)
    
    # Analyze results
    results = analyze_gbm_results_from_file(paths_file)
    
    # Save summary statistics
    summary_file = save_summary_statistics(results, paths_file)
    
    # Create visualization
    create_gbm_visualization(paths_file, results)
    
    print(f"\n" + "="*50)
    print("DYNAMIC GBM SIMULATION SUMMARY")
    print("="*50)
    print(f"Starting price (fair value): ${S0:,.2f}")
    print(f"Simulation period: 10 years")
    print(f"Number of paths: 100 (quick test)")
    print(f"Parameter update frequency: Every day (365 times per year)")
    print(f"Growth rate: Dynamic based on formula fair value")
    print(f"Volatility: Dynamic based on exponential decay formula")
    
    if 1 in results:
        print(f"\nAfter 1 year:")
        print(f"  Expected price: ${results[1]['mean']:,.2f}")
        print(f"  90% confidence: ${results[1]['p5']:,.2f} - ${results[1]['p95']:,.2f}")
    
    if 5 in results:
        print(f"\nAfter 5 years:")
        print(f"  Expected price: ${results[5]['mean']:,.2f}")
        print(f"  90% confidence: ${results[5]['p5']:,.2f} - ${results[5]['p95']:,.2f}")
    
    if 10 in results:
        print(f"\nAfter 10 years:")
        print(f"  Expected price: ${results[10]['mean']:,.2f}")
        print(f"  90% confidence: ${results[10]['p5']:,.2f} - ${results[10]['p95']:,.2f}")
    
    print(f"\nCSV Files Created:")
    print(f"  Price paths: {paths_file}")
    print(f"  Summary stats: {summary_file}")
    print(f"  File size: {os.path.getsize(paths_file) / (1024*1024):.1f} MB")
    print("\n✅ Dynamic GBM simulation completed successfully!")
    print("Note: Growth and volatility parameters updated every day during simulation")

if __name__ == "__main__":
    main() 
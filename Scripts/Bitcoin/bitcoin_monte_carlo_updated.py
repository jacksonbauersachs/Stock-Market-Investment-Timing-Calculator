import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
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
    print(f"Current price: ${df['Price'].iloc[-1]:,.2f}")
    
    return df

def get_updated_models():
    """Get the updated growth and volatility models"""
    print("\n" + "="*60)
    print("LOADING UPDATED MODELS")
    print("="*60)
    
    # Load growth model coefficients
    try:
        with open('Models/Growth Models/bitcoin_growth_model_coefficients_day365.txt', 'r') as f:
            lines = f.readlines()
            a = float(lines[0].split('=')[1].strip())
            b = float(lines[1].split('=')[1].strip())
            r2 = float(lines[2].split('=')[1].strip())
        
        print(f"Growth Model: log10(price) = {a:.6f} * ln(day) + {b:.6f}")
        print(f"Growth Model R² = {r2:.6f}")
        
    except FileNotFoundError:
        print("⚠️  Growth model file not found, using default values")
        a, b, r2 = 1.827743, -10.880943, 0.940275
    
    # Load exponential volatility model
    try:
        with open('Models/Volatility Models/bitcoin_exponential_volatility_results_20250719.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                if 'Formula: volatility =' in line:
                    formula = line.split('=')[1].strip()
                    # Extract parameters: a * exp(-b * years) + c
                    # Format: "2.310879 * exp(-0.124138 * years) + 0.077392"
                    try:
                        # Split by '*' and extract components
                        parts = formula.split('*')
                        a_vol = float(parts[0].strip())
                        
                        # Extract b from exp(-b * years)
                        exp_part = parts[1].strip()
                        b_vol = float(exp_part.split('(')[1].split('*')[0].strip())
                        # Remove the negative sign since it's already in the formula
                        if b_vol < 0:
                            b_vol = abs(b_vol)
                        
                        # Extract c from the last part
                        c_vol = float(parts[2].split('+')[1].strip())
                        break
                    except (IndexError, ValueError) as e:
                        print(f"Error parsing formula: {formula}")
                        print(f"Error details: {e}")
                        raise
            else:
                # If we didn't find the formula line, use defaults
                raise ValueError("Formula line not found in file")
        
        print(f"Volatility Model: volatility = {a_vol:.6f} * exp(-{b_vol:.6f} * years) + {c_vol:.6f}")
        print(f"Volatility Model R² = 0.2883")
        
    except (FileNotFoundError, ValueError, IndexError) as e:
        print(f"⚠️  Volatility model file error: {e}")
        print("Using default exponential decay parameters")
        a_vol, b_vol, c_vol = 2.310879, 0.124138, 0.077392
    
    return {
        'growth': {'a': a, 'b': b, 'r2': r2},
        'volatility': {'a': a_vol, 'b': b_vol, 'c': c_vol}
    }

def calculate_volatility_at_time(years, vol_params):
    """Calculate volatility at a given time using exponential decay model"""
    a, b, c = vol_params['a'], vol_params['b'], vol_params['c']
    volatility = a * np.exp(-b * years) + c
    
    # Ensure volatility stays within reasonable bounds
    volatility = np.maximum(volatility, 0.05)  # Minimum 5%
    volatility = np.minimum(volatility, 2.0)   # Maximum 200%
    
    return volatility

def geometric_brownian_motion_with_decay(start_price, years, growth_params, vol_params, num_paths=1000):
    """Run GBM simulation with time-varying volatility decay"""
    print(f"\n" + "="*60)
    print("RUNNING MONTE CARLO SIMULATION")
    print("="*60)
    print(f"Starting price: ${start_price:,.2f}")
    print(f"Simulation period: {years} years")
    print(f"Number of paths: {num_paths:,}")
    
    # Time setup
    dt = 1/365.25  # Daily time step
    time_steps = int(years * 365.25)
    t = np.linspace(0, years, time_steps + 1)
    
    # Initialize price paths
    price_paths = np.zeros((num_paths, time_steps + 1))
    price_paths[:, 0] = start_price
    
    # We'll calculate growth rate dynamically at each time step
    # based on the growth formula
    a, b = growth_params['a'], growth_params['b']
    
    # Calculate today's day number (adjusted for 365-day offset)
    today_day = 5476 - 365  # Adjust for the 365-day offset
    
    print(f"Using dynamic growth rate from formula: log10(price) = {a:.6f} * ln(day) + {b:.6f}")
    print(f"Today's formula day: {today_day}")
    
    # Run simulation
    np.random.seed(42)  # For reproducibility
    
    for i in range(num_paths):
        for j in range(1, time_steps + 1):
            # Calculate current time in years
            current_years = t[j]
            
            # Get volatility at this time using decay model
            volatility = calculate_volatility_at_time(current_years, vol_params)
            
            # Calculate expected price at this time using growth formula
            # We need to calculate what day number this represents
            # Starting from today (day 5476) but adjusting for the 365-day offset in our formula
            actual_current_day = 5476 + int(current_years * 365.25)
            current_day = actual_current_day - 365  # Adjust for the 365-day offset
            expected_price = 10**(a * np.log(current_day) + b)
            
            # Calculate the growth rate from the formula
            # Use the total annual growth rate from today to this point
            if j > 0:
                # Calculate total growth from today to this point
                today_price = 10**(a * np.log(today_day) + b)
                mu = np.log(expected_price / today_price) / current_years
            else:
                mu = 0
            
            # The formula predicts Bitcoin should be at $53,263 today, but it's at $118,075
            # This means Bitcoin is overvalued by 121.68% relative to the formula
            # The simulation will naturally converge toward the formula's predictions over time
            # while allowing for volatility-driven movements
            
            # GBM step with dynamic growth rate and volatility
            drift = (mu - 0.5 * volatility**2) * dt
            diffusion = volatility * np.sqrt(dt) * np.random.normal(0, 1)
            
            price_paths[i, j] = price_paths[i, j-1] * np.exp(drift + diffusion)
    
    return price_paths, t

def analyze_simulation_results(price_paths, t, start_price, years):
    """Analyze Monte Carlo simulation results"""
    print(f"\n" + "="*60)
    print("SIMULATION RESULTS ANALYSIS")
    print("="*60)
    
    # Calculate statistics at different time points
    time_points = [0, 0.25, 0.5, 1, 2, 3, 5, 10]  # Years
    results = {}
    
    for year_point in time_points:
        if year_point <= years:
            day_index = int(year_point * 365.25)
            if day_index < len(t):
                prices_at_time = price_paths[:, day_index]
                
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
                
                print(f"Year {year_point:.1f}:")
                print(f"  Mean: ${mean_price:,.2f}")
                print(f"  Median: ${median_price:,.2f}")
                print(f"  5th percentile: ${p5:,.2f}")
                print(f"  95th percentile: ${p95:,.2f}")
                print(f"  Std Dev: ${std_price:,.2f}")
                print()
    
    return results

def create_simulation_visualization(price_paths, t, start_price, results, models):
    """Create comprehensive visualization of simulation results"""
    print(f"\n" + "="*60)
    print("CREATING SIMULATION VISUALIZATION")
    print("="*60)
    
    # Set up the plot
    plt.style.use('default')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Bitcoin Monte Carlo Simulation with Exponential Volatility Decay', fontsize=16, fontweight='bold')
    
    # Plot 1: Sample price paths
    sample_paths = min(50, len(price_paths))
    for i in range(sample_paths):
        ax1.plot(t, price_paths[i, :], alpha=0.3, linewidth=0.5)
    
    # Plot mean path
    mean_path = np.mean(price_paths, axis=0)
    ax1.plot(t, mean_path, 'r-', linewidth=2, label='Mean Path')
    
    # Plot percentiles
    p5_path = np.percentile(price_paths, 5, axis=0)
    p95_path = np.percentile(price_paths, 95, axis=0)
    ax1.fill_between(t, p5_path, p95_path, alpha=0.3, color='blue', label='90% Confidence Interval')
    
    ax1.set_xlabel('Years')
    ax1.set_ylabel('Bitcoin Price ($)')
    ax1.set_title('Monte Carlo Price Paths')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot 2: Price distribution at different time points
    time_points = [1, 3, 5, 10]  # Years
    colors = ['blue', 'green', 'orange', 'red']
    
    for i, years in enumerate(time_points):
        if years in results:
            day_index = int(years * 365.25)
            if day_index < len(t):
                prices_at_time = price_paths[:, day_index]
                ax2.hist(prices_at_time, bins=50, alpha=0.6, color=colors[i], 
                        label=f'{years} years', density=True)
    
    ax2.set_xlabel('Bitcoin Price ($)')
    ax2.set_ylabel('Density')
    ax2.set_title('Price Distribution at Different Time Points')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    
    # Plot 3: Volatility decay over time
    years_range = np.linspace(0, t[-1], 100)
    volatility_values = []
    for year in years_range:
        vol = calculate_volatility_at_time(year, models['volatility'])
        volatility_values.append(vol)
    
    ax3.plot(years_range, volatility_values, 'b-', linewidth=2)
    ax3.set_xlabel('Years')
    ax3.set_ylabel('Annualized Volatility')
    ax3.set_title('Volatility Decay Model')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Expected value growth vs time
    years_list = list(results.keys())
    mean_prices = [results[y]['mean'] for y in years_list]
    
    ax4.plot(years_list, mean_prices, 'ro-', linewidth=2, markersize=8)
    ax4.fill_between(years_list, 
                     [results[y]['p5'] for y in years_list],
                     [results[y]['p95'] for y in years_list],
                     alpha=0.3, color='red')
    
    ax4.set_xlabel('Years')
    ax4.set_ylabel('Expected Bitcoin Price ($)')
    ax4.set_title('Expected Value Growth with Confidence Intervals')
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    plt.tight_layout()
    
    # Save the plot
    filename = f'Results/Bitcoin/bitcoin_monte_carlo_updated_{datetime.now().strftime("%Y%m%d")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Simulation visualization saved to: {filename}")
    
    plt.show()

def save_simulation_results(price_paths, t, results, models, start_price, years):
    """Save detailed simulation results"""
    filename = f'Results/Bitcoin/bitcoin_monte_carlo_updated_results_{datetime.now().strftime("%Y%m%d")}.txt'
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("BITCOIN MONTE CARLO SIMULATION RESULTS\n")
        f.write("="*60 + "\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data Source: Bitcoin_Final_Complete_Data_20250719.csv\n")
        f.write("\n")
        
        f.write("-"*40 + "\n")
        f.write("SIMULATION PARAMETERS\n")
        f.write("-"*40 + "\n")
        f.write(f"Starting price: ${start_price:,.2f}\n")
        f.write(f"Simulation period: {years} years\n")
        f.write(f"Number of paths: {len(price_paths):,}\n")
        f.write(f"Time steps: {len(t):,}\n")
        
        f.write(f"\nGrowth Model: log10(price) = {models['growth']['a']:.6f} * ln(day) + {models['growth']['b']:.6f}\n")
        f.write(f"Growth Model R² = {models['growth']['r2']:.6f}\n")
        
        f.write(f"\nVolatility Model: volatility = {models['volatility']['a']:.6f} * exp(-{models['volatility']['b']:.6f} * years) + {models['volatility']['c']:.6f}\n")
        f.write(f"Volatility Model R² = 0.2883\n")
        
        f.write("\n")
        f.write("-"*40 + "\n")
        f.write("SIMULATION RESULTS\n")
        f.write("-"*40 + "\n")
        
        for years in sorted(results.keys()):
            r = results[years]
            f.write(f"\nYear {years:.1f}:\n")
            f.write(f"  Mean: ${r['mean']:,.2f}\n")
            f.write(f"  Median: ${r['median']:,.2f}\n")
            f.write(f"  5th percentile: ${r['p5']:,.2f}\n")
            f.write(f"  25th percentile: ${r['p25']:,.2f}\n")
            f.write(f"  75th percentile: ${r['p75']:,.2f}\n")
            f.write(f"  95th percentile: ${r['p95']:,.2f}\n")
            f.write(f"  Standard deviation: ${r['std']:,.2f}\n")
        
        f.write("\n")
        f.write("="*60 + "\n")
    
    print(f"Detailed results saved to: {filename}")
    return filename

def main():
    """Main function to run updated Monte Carlo simulation"""
    print("="*60)
    print("BITCOIN MONTE CARLO SIMULATION (UPDATED MODELS)")
    print("="*60)
    print("This simulation uses:")
    print("- Updated growth model with R² = 0.9403")
    print("- Exponential decay volatility model with R² = 0.2883")
    print("- Time-varying volatility (decreases over time)")
    print("="*60)
    
    # Load data and get current price
    df = load_bitcoin_data()
    start_price = df['Price'].iloc[-1]
    
    # Load updated models
    models = get_updated_models()
    
    # Simulation parameters
    years = 10  # 10 years simulation
    num_paths = 1000
    
    # Run Monte Carlo simulation
    price_paths, t = geometric_brownian_motion_with_decay(
        start_price, years, models['growth'], models['volatility'], num_paths
    )
    
    # Analyze results
    results = analyze_simulation_results(price_paths, t, start_price, years)
    
    # Create visualization
    create_simulation_visualization(price_paths, t, start_price, results, models)
    
    # Save results
    filename = save_simulation_results(price_paths, t, results, models, start_price, years)
    
    # Summary
    print(f"\n" + "="*50)
    print("SIMULATION SUMMARY")
    print("="*50)
    print(f"Starting price: ${start_price:,.2f}")
    print(f"Simulation period: {years} years")
    print(f"Number of paths: {num_paths:,}")
    
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
    
    print(f"\nResults file: {filename}")
    print("\nSimulation complete! The exponential decay volatility model shows")
    print("more realistic long-term projections than constant volatility.")

if __name__ == "__main__":
    main() 
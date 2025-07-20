"""
Bitcoin GBM Monte Carlo Simulation
==================================

This script implements a proper Geometric Brownian Motion (GBM) Monte Carlo simulation
for Bitcoin price forecasting using standard financial modeling techniques.

GBM Formula: dS = μSdt + σSdW
Where:
- S = price
- μ = drift (expected annual return)
- σ = volatility (annual standard deviation)
- dW = Wiener process (random walk)

This approach is used by hedge funds and financial institutions for asset pricing.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def load_bitcoin_data():
    """Load Bitcoin historical data to estimate parameters"""
    print("Loading Bitcoin historical data for parameter estimation...")
    df = pd.read_csv('Data Sets/Bitcoin Data/Bitcoin_Final_Complete_Data_20250719.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.dropna(subset=['Price'])
    df = df.sort_values('Date')
    
    print(f"Loaded {len(df):,} days of Bitcoin data")
    print(f"Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"Current price: ${df['Price'].iloc[-1]:,.2f}")
    
    return df

def estimate_gbm_parameters(df, lookback_years=5):
    """
    Estimate GBM parameters using our formulas for both growth and volatility
    
    Args:
        df: Bitcoin price data
        lookback_years: Not used (for compatibility)
    
    Returns:
        mu: Annualized drift (expected return) from growth formula
        sigma: Annualized volatility from volatility decay formula
    """
    print(f"\nEstimating GBM parameters using our formulas...")
    
    # Get growth rate from our formula
    with open('Models/Growth Models/bitcoin_growth_model_coefficients_day365.txt', 'r') as f:
        lines = f.readlines()
        a = float(lines[0].split('=')[1].strip())
        b = float(lines[1].split('=')[1].strip())
    
    today_day = 6041
    current_fair_value = 10**(a * np.log(today_day) + b)
    
    # Calculate formula predictions for next 10 years
    formula_prices = []
    for year in range(1, 11):
        future_day = today_day + int(year * 365.25)
        formula_price = 10**(a * np.log(future_day) + b)
        formula_prices.append(formula_price)
    
    # Calculate annual growth rate from formula (using fair value as baseline)
    formula_growth_rates = []
    for i, future_price in enumerate(formula_prices):
        years = i + 1
        annual_growth_rate = (future_price / current_fair_value) ** (1/years) - 1
        formula_growth_rates.append(annual_growth_rate)
    
    # Use average growth rate over next 10 years
    mu = np.mean(formula_growth_rates)
    
    print(f"Formula-based growth estimates (from fair value):")
    print(f"  Current fair value: ${current_fair_value:,.0f}")
    for i, (price, rate) in enumerate(zip(formula_prices, formula_growth_rates)):
        print(f"  Year {i+1}: ${price:,.0f} ({(rate*100):.1f}% annual)")
    print(f"  Average annual growth rate (μ): {mu*100:.1f}%")
    
    # Get volatility from our volatility decay formula
    print(f"\nEstimating volatility using our volatility decay formula...")
    
    # Load volatility model parameters
    with open('Models/Volatility Models/bitcoin_exponential_volatility_results_20250719.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'Formula: volatility =' in line:
                formula = line.split('=')[1].strip()
                # Extract parameters: a * exp(-b * years) + c
                parts = formula.split('*')
                a_vol = float(parts[0].strip())
                
                exp_part = parts[1].strip()
                b_vol = float(exp_part.split('(')[1].split('*')[0].strip())
                if b_vol < 0:
                    b_vol = abs(b_vol)
                
                c_vol = float(parts[2].split('+')[1].strip())
                break
    
    # Bitcoin's current age (years since 2010-07-18)
    bitcoin_current_age = 15.0  # 2025-07-19 minus 2010-07-18
    
    # Calculate current volatility (year 0 from now)
    sigma = a_vol * np.exp(-b_vol * bitcoin_current_age) + c_vol
    
    print(f"Volatility decay formula: volatility = {a_vol:.6f} * exp(-{b_vol:.6f} * years) + {c_vol:.6f}")
    print(f"Bitcoin's current age: {bitcoin_current_age:.1f} years")
    print(f"Current volatility (σ): {sigma*100:.1f}%")
    
    # Show volatility predictions for next 10 years
    print(f"\nVolatility predictions:")
    for year in range(11):
        future_age = bitcoin_current_age + year
        future_vol = a_vol * np.exp(-b_vol * future_age) + c_vol
        print(f"  Year {year}: {future_vol*100:.1f}%")
    
    return mu, sigma

def dynamic_gbm_monte_carlo_simulation(S0, T, num_paths=1000, num_steps=3653):
    """
    Run Dynamic GBM Monte Carlo simulation with time-varying parameters
    
    Args:
        S0: Initial price
        T: Time horizon in years
        num_paths: Number of simulation paths
        num_steps: Number of time steps (daily = 365.25 * T)
    
    Returns:
        price_paths: Array of price paths
        time_steps: Array of time points
    """
    print(f"\nRunning Dynamic GBM Monte Carlo Simulation")
    print(f"="*50)
    print(f"Initial price: ${S0:,.2f}")
    print(f"Time horizon: {T} years")
    print(f"Number of paths: {num_paths:,}")
    print(f"Time steps: {num_steps:,} (daily)")
    
    # Load formula parameters
    with open('Models/Growth Models/bitcoin_growth_model_coefficients_day365.txt', 'r') as f:
        lines = f.readlines()
        a = float(lines[0].split('=')[1].strip())
        b = float(lines[1].split('=')[1].strip())
    
    with open('Models/Volatility Models/bitcoin_exponential_volatility_results_20250719.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'Formula: volatility =' in line:
                formula = line.split('=')[1].strip()
                parts = formula.split('*')
                a_vol = float(parts[0].strip())
                exp_part = parts[1].strip()
                b_vol = float(exp_part.split('(')[1].split('*')[0].strip())
                if b_vol < 0:
                    b_vol = abs(b_vol)
                c_vol = float(parts[2].split('+')[1].strip())
                break
    
    # Bitcoin's current age
    bitcoin_current_age = 15.0
    today_day = 6041
    
    # Time setup
    dt = T / num_steps  # Time step size
    time_steps = np.linspace(0, T, num_steps + 1)
    
    # Initialize price paths
    price_paths = np.zeros((num_paths, num_steps + 1))
    price_paths[:, 0] = S0  # All paths start at initial price
    
    print(f"Dynamic Parameters:")
    print(f"  Growth formula: log10(price) = {a:.6f} * ln(day) + {b:.6f}")
    print(f"  Volatility formula: {a_vol:.6f} * exp(-{b_vol:.6f} * years) + {c_vol:.6f}")
    print(f"  Bitcoin current age: {bitcoin_current_age:.1f} years")
    
    # Run simulation with dynamic parameters (updated every day)
    np.random.seed(42)  # For reproducibility
    
    # Track parameter evolution for debugging
    param_tracker = {
        'years': [],
        'mu': [],
        'sigma': [],
        'fair_value': []
    }
    
    for i in range(num_paths):
        for j in range(1, num_steps + 1):
            # Calculate current time in years
            current_time = time_steps[j]
            
            # Calculate dynamic parameters at this time (updated every day)
            future_age = bitcoin_current_age + current_time
            future_day = today_day + int(current_time * 365.25)
            
            # Dynamic growth rate (μ) from formula - changes every day
            current_fair_value = 10**(a * np.log(today_day) + b)
            future_fair_value = 10**(a * np.log(future_day) + b)
            mu = (future_fair_value / current_fair_value) ** (1/current_time) - 1 if current_time > 0 else 0
            
            # Dynamic volatility (σ) from decay formula - changes every day
            sigma = a_vol * np.exp(-b_vol * future_age) + c_vol
            
            # Track parameters for first path only (for debugging)
            if i == 0 and j % 365 == 0:  # Log yearly
                param_tracker['years'].append(current_time)
                param_tracker['mu'].append(mu)
                param_tracker['sigma'].append(sigma)
                param_tracker['fair_value'].append(future_fair_value)
            
            # GBM parameters for this time step
            drift = (mu - 0.5 * sigma**2) * dt
            volatility = sigma * np.sqrt(dt)
            
            # GBM formula: S(t+dt) = S(t) * exp(drift + volatility * random_normal)
            random_shock = np.random.normal(0, 1)
            price_paths[i, j] = price_paths[i, j-1] * np.exp(drift + volatility * random_shock)
    
    # Print parameter evolution
    print(f"\nParameter Evolution (Yearly):")
    print(f"{'Year':<6} {'Growth Rate':<12} {'Volatility':<12} {'Fair Value':<15}")
    print("-" * 50)
    for i in range(len(param_tracker['years'])):
        year = param_tracker['years'][i]
        mu = param_tracker['mu'][i]
        sigma = param_tracker['sigma'][i]
        fair_val = param_tracker['fair_value'][i]
        print(f"{year:<6.1f} {mu*100:<11.1f}% {sigma*100:<11.1f}% ${fair_val:<14,.0f}")
    
    print(f"\nNote: Parameters update every day (365 times per year) during simulation")
    
    return price_paths, time_steps

def analyze_gbm_results(price_paths, time_steps, S0, T):
    """Analyze GBM simulation results"""
    print(f"\nGBM Simulation Results Analysis")
    print(f"="*50)
    
    # Calculate statistics at different time points
    time_points = [0, 0.25, 0.5, 1, 2, 3, 5, 10]  # Years
    results = {}
    
    for year_point in time_points:
        if year_point <= T:
            step_index = int(year_point * len(time_steps) / T)
            if step_index < len(time_steps):
                prices_at_time = price_paths[:, step_index]
                
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

def create_gbm_visualization(price_paths, time_steps, S0, results, mu, sigma):
    """Create visualization for GBM simulation"""
    print(f"\nCreating GBM visualization...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Bitcoin GBM Monte Carlo Simulation', fontsize=16, fontweight='bold')
    
    # 1. Sample price paths
    sample_paths = min(50, len(price_paths))  # Show first 50 paths
    for i in range(sample_paths):
        ax1.plot(time_steps, price_paths[i, :], alpha=0.3, linewidth=0.5)
    
    ax1.axhline(y=S0, color='red', linestyle='--', linewidth=2, label=f'Start: ${S0:,.0f}')
    ax1.set_title('Sample Price Paths (First 50)')
    ax1.set_ylabel('Bitcoin Price ($)')
    ax1.set_xlabel('Years')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # 2. Price distribution at different years
    years_to_plot = [1, 5, 10]
    colors = ['blue', 'green', 'red']
    
    for i, year in enumerate(years_to_plot):
        if year in results:
            step_index = int(year * len(time_steps) / 10)  # Assuming 10-year simulation
            if step_index < len(time_steps):
                prices_at_year = price_paths[:, step_index]
                ax2.hist(prices_at_year, bins=50, alpha=0.6, color=colors[i], 
                        label=f'{year} year(s)')
    
    ax2.set_title('Price Distribution at Different Years')
    ax2.set_xlabel('Bitcoin Price ($)')
    ax2.set_ylabel('Number of Paths')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    
    # 3. Expected value growth
    years_list = sorted(results.keys())
    mean_prices = [results[y]['mean'] for y in years_list]
    
    ax3.plot(years_list, mean_prices, 'ro-', linewidth=2, markersize=8, label='Simulation Mean')
    ax3.fill_between(years_list, 
                     [results[y]['p5'] for y in years_list],
                     [results[y]['p95'] for y in years_list],
                     alpha=0.3, color='red')
    
    # Add theoretical GBM expectation
    theoretical_prices = [S0 * np.exp(mu * year) for year in years_list]
    ax3.plot(years_list, theoretical_prices, 'go-', linewidth=2, markersize=8, label='Theoretical E[S(t)]')
    
    ax3.set_xlabel('Years')
    ax3.set_ylabel('Expected Bitcoin Price ($)')
    ax3.set_title('Expected Value Growth vs Theoretical')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # 4. Volatility over time
    volatilities = []
    for year in years_list:
        if year > 0:
            step_index = int(year * len(time_steps) / 10)
            if step_index < len(time_steps):
                prices_at_year = price_paths[:, step_index]
                cv = prices_at_year.std() / prices_at_year.mean() * 100
                volatilities.append(cv)
    
    ax4.plot([y for y in years_list if y > 0], volatilities, 'bo-', linewidth=2, markersize=8)
    ax4.set_title('Price Volatility Over Time')
    ax4.set_xlabel('Years')
    ax4.set_ylabel('Coefficient of Variation (%)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    filename = f'Results/Bitcoin/bitcoin_gbm_simulation_{datetime.now().strftime("%Y%m%d")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"GBM visualization saved to: {filename}")
    
    plt.show()

def save_gbm_results(price_paths, time_steps, results, mu, sigma, S0, T):
    """Save GBM simulation results to CSV files"""
    print(f"\nSaving GBM simulation results...")
    
    # Save price paths with memory-efficient approach
    paths_filename = f'Results/Bitcoin/bitcoin_gbm_paths_{datetime.now().strftime("%Y%m%d")}.csv'
    
    print(f"Creating price paths DataFrame...")
    print(f"  Shape: {price_paths.shape[0]} paths × {price_paths.shape[1]} time steps")
    print(f"  Total data points: {price_paths.shape[0] * price_paths.shape[1]:,}")
    
    try:
        # Create DataFrame with time as index and paths as columns
        paths_df = pd.DataFrame(price_paths.T, index=time_steps, columns=[f'Path_{i+1}' for i in range(len(price_paths))])
        paths_df.index.name = 'Years'
        
        print(f"Saving to {paths_filename}...")
        paths_df.to_csv(paths_filename)
        print(f"✅ Price paths saved to: {paths_filename}")
        
    except MemoryError:
        print(f"❌ Memory error while saving. Trying chunked approach...")
        # Fallback: save in chunks
        chunk_size = 100  # Save 100 paths at a time
        for chunk_start in range(0, len(price_paths), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(price_paths))
            chunk_filename = f'Results/Bitcoin/bitcoin_gbm_paths_chunk_{chunk_start//chunk_size + 1}_{datetime.now().strftime("%Y%m%d")}.csv'
            
            chunk_df = pd.DataFrame(price_paths[chunk_start:chunk_end].T, 
                                  index=time_steps, 
                                  columns=[f'Path_{i+1}' for i in range(chunk_start, chunk_end)])
            chunk_df.index.name = 'Years'
            chunk_df.to_csv(chunk_filename)
            print(f"  Chunk {chunk_start//chunk_size + 1} saved: {chunk_filename}")
        
        paths_filename = f'Results/Bitcoin/bitcoin_gbm_paths_chunked_{datetime.now().strftime("%Y%m%d")}'
        print(f"✅ Price paths saved in chunks to: {paths_filename}_chunk_*.csv")
    
    except Exception as e:
        print(f"❌ Error saving price paths: {e}")
        paths_filename = None
    
    # Save summary statistics
    summary_filename = f'Results/Bitcoin/bitcoin_gbm_summary_{datetime.now().strftime("%Y%m%d")}.csv'
    
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
    
    # Save GBM parameters
    params_filename = f'Results/Bitcoin/bitcoin_gbm_parameters_{datetime.now().strftime("%Y%m%d")}.csv'
    
    params_data = [{
        'Parameter': 'Initial_Price',
        'Value': S0,
        'Description': 'Starting Bitcoin price'
    }, {
        'Parameter': 'Annual_Drift',
        'Value': mu,
        'Description': 'Expected annual return (μ)'
    }, {
        'Parameter': 'Annual_Volatility',
        'Value': sigma,
        'Description': 'Annual volatility (σ)'
    }, {
        'Parameter': 'Time_Horizon',
        'Value': T,
        'Description': 'Simulation time horizon in years'
    }, {
        'Parameter': 'Num_Paths',
        'Value': len(price_paths),
        'Description': 'Number of simulation paths'
    }]
    
    params_df = pd.DataFrame(params_data)
    params_df.to_csv(params_filename, index=False)
    print(f"GBM parameters saved to: {params_filename}")
    
    return paths_filename, summary_filename, params_filename

def main():
    """Main function to run Dynamic GBM Monte Carlo simulation"""
    print("="*60)
    print("BITCOIN DYNAMIC GBM MONTE CARLO SIMULATION")
    print("="*60)
    print("This simulation uses Dynamic Geometric Brownian Motion (GBM)")
    print("with time-varying parameters: μ(t) and σ(t)")
    print("Parameters update every day based on growth and volatility formulas")
    print("="*60)
    
    # Load data to get current price
    df = load_bitcoin_data()
    S0 = df['Price'].iloc[-1]  # Current Bitcoin price
    
    # Simulation parameters
    T = 10  # 10 years simulation
    num_paths = 1000
    num_steps = int(T * 365.25)  # Daily time steps
    
    # Run Dynamic GBM simulation (parameters update every day)
    price_paths, time_steps = dynamic_gbm_monte_carlo_simulation(
        S0, T, num_paths, num_steps
    )
    
    # Analyze results
    results = analyze_gbm_results(price_paths, time_steps, S0, T)
    
    # Create visualization (using placeholder values since parameters are dynamic)
    create_gbm_visualization(price_paths, time_steps, S0, results, 0.15, 0.80)
    
    # Save results (using placeholder values since parameters are dynamic)
    paths_file, summary_file, params_file = save_gbm_results(
        price_paths, time_steps, results, 0.15, 0.80, S0, T
    )
    
    # Summary
    print(f"\n" + "="*50)
    print("DYNAMIC GBM SIMULATION SUMMARY")
    print("="*50)
    print(f"Starting price: ${S0:,.2f}")
    print(f"Simulation period: {T} years")
    print(f"Number of paths: {num_paths:,}")
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
    print(f"  GBM parameters: {params_file}")
    print("\n✅ Dynamic GBM simulation completed successfully!")
    print("Note: Growth and volatility parameters updated every day during simulation")

if __name__ == "__main__":
    main() 
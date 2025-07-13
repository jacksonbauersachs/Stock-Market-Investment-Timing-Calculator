import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def bitcoin_growth_model(days):
    """Bitcoin growth model: log10(price) = a * ln(day) + b"""
    a = 1.6329135221917355
    b = -9.328646304661454
    return 10**(a * np.log(days) + b)

def bitcoin_volatility_model(years):
    """
    Better volatility model: σ(t) = 0.4 + 0.6 × e^(-0.15t)
    NO FLOOR NEEDED - model naturally approaches 40% asymptote
    """
    initial_vol = 1.0  # 100% starting volatility
    mature_vol = 0.4   # 40% mature volatility (natural asymptote)
    decay_rate = 0.15  # moderate exponential decay
    return mature_vol + (initial_vol - mature_vol) * np.exp(-decay_rate * years)

def generate_bitcoin_monte_carlo_paths(n_paths=1000, years=10, steps_per_year=365):
    """
    Generate Bitcoin price paths using GBM with time-varying volatility
    Starting from current Bitcoin age (15 years) and current actual price
    """
    print("GENERATING BITCOIN MONTE CARLO PRICE PATHS")
    print("=" * 50)
    print(f"Number of paths: {n_paths}")
    print(f"Time horizon: {years} years")
    print(f"Steps per year: {steps_per_year}")
    print(f"Using volatility model: σ(t) = 0.4 + 0.6 × e^(-0.15t)")
    print("NO VOLATILITY FLOOR - model naturally approaches 40%")
    print()
    
    # Current Bitcoin parameters
    current_actual_price = 105740  # Current actual Bitcoin price
    current_bitcoin_age = 15  # Bitcoin is ~15 years old
    current_day = 5439  # Current day in dataset
    
    # Simulation parameters
    total_steps = int(years * steps_per_year)
    dt = 1 / steps_per_year  # Time step in years
    
    # Initialize price paths
    paths = np.zeros((n_paths, total_steps + 1))
    paths[:, 0] = current_actual_price
    
    # Pre-calculate time points and volatilities
    time_points = np.linspace(current_bitcoin_age, current_bitcoin_age + years, total_steps + 1)
    volatilities = np.array([bitcoin_volatility_model(t) for t in time_points])
    
    print(f"Starting volatility: {volatilities[0]:.1%}")
    print(f"Ending volatility: {volatilities[-1]:.1%}")
    print(f"Volatility decay over {years} years: {volatilities[0] - volatilities[-1]:.1%}")
    print()
    
    # Generate paths
    print("Generating price paths...")
    
    for step in range(total_steps):
        # Calculate drift from growth model
        current_model_price = bitcoin_growth_model(current_day + step)
        next_model_price = bitcoin_growth_model(current_day + step + 1)
        drift = np.log(next_model_price / current_model_price) / dt
        
        # Get volatility for this time step
        vol = volatilities[step]
        
        # Generate random shocks
        random_shocks = np.random.normal(0, 1, n_paths)
        
        # GBM formula: S(t+dt) = S(t) * exp((μ - 0.5σ²)dt + σ√(dt) * Z)
        log_returns = (drift - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * random_shocks
        
        # Update prices
        paths[:, step + 1] = paths[:, step] * np.exp(log_returns)
    
    # Create time grid for output
    time_grid = np.linspace(0, years, total_steps + 1)
    
    # Create DataFrame
    columns = [f'Year_{t:.3f}' for t in time_grid]
    df = pd.DataFrame(paths.T, columns=[f'Path_{i+1}' for i in range(n_paths)])
    df.index = time_grid
    df.index.name = 'Years'
    
    # Save to CSV
    output_file = 'Investment Strategy Analasis/Bitcoin Analysis/Results/bitcoin_monte_carlo_paths_updated.csv'
    df.to_csv(output_file)
    
    print(f"Price paths saved to: {output_file}")
    print(f"File size: {df.shape[0]} time steps × {df.shape[1]} paths")
    
    # Calculate some statistics
    final_prices = paths[:, -1]
    price_multiples = final_prices / current_actual_price
    
    print()
    print("FINAL PRICE STATISTICS:")
    print("-" * 30)
    print(f"Starting price: ${current_actual_price:,.0f}")
    print(f"Mean final price: ${np.mean(final_prices):,.0f}")
    print(f"Median final price: ${np.median(final_prices):,.0f}")
    print(f"Min final price: ${np.min(final_prices):,.0f}")
    print(f"Max final price: ${np.max(final_prices):,.0f}")
    print()
    print(f"Mean price multiple: {np.mean(price_multiples):.2f}x")
    print(f"Median price multiple: {np.median(price_multiples):.2f}x")
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Sample price paths
    plt.subplot(2, 2, 1)
    sample_paths = paths[:min(100, n_paths), :]
    for i in range(len(sample_paths)):
        plt.plot(time_grid, sample_paths[i], alpha=0.3, linewidth=0.5)
    plt.title('Sample Bitcoin Price Paths (100 paths)')
    plt.xlabel('Years from now')
    plt.ylabel('Bitcoin Price ($)')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Plot 2: Final price distribution
    plt.subplot(2, 2, 2)
    plt.hist(final_prices, bins=50, alpha=0.7, density=True)
    plt.axvline(np.mean(final_prices), color='red', linestyle='--', label=f'Mean: ${np.mean(final_prices):,.0f}')
    plt.axvline(np.median(final_prices), color='green', linestyle='--', label=f'Median: ${np.median(final_prices):,.0f}')
    plt.title('Distribution of Final Prices')
    plt.xlabel('Final Price ($)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Volatility decay
    plt.subplot(2, 2, 3)
    vol_display_time = np.linspace(0, years, 100)
    vol_display_values = [bitcoin_volatility_model(current_bitcoin_age + t) for t in vol_display_time]
    plt.plot(vol_display_time, vol_display_values, 'b-', linewidth=2)
    plt.title('Bitcoin Volatility Decay (No Floor Needed)')
    plt.xlabel('Years from now')
    plt.ylabel('Annualized Volatility')
    plt.grid(True, alpha=0.3)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    # Plot 4: Price multiple distribution
    plt.subplot(2, 2, 4)
    plt.hist(price_multiples, bins=50, alpha=0.7, density=True)
    plt.axvline(np.mean(price_multiples), color='red', linestyle='--', label=f'Mean: {np.mean(price_multiples):.1f}x')
    plt.axvline(np.median(price_multiples), color='green', linestyle='--', label=f'Median: {np.median(price_multiples):.1f}x')
    plt.title('Distribution of Price Multiples')
    plt.xlabel('Price Multiple')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bitcoin_monte_carlo_paths_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df, paths

if __name__ == "__main__":
    # Generate the Monte Carlo paths
    df, paths = generate_bitcoin_monte_carlo_paths(
        n_paths=1000,
        years=10,
        steps_per_year=365
    ) 
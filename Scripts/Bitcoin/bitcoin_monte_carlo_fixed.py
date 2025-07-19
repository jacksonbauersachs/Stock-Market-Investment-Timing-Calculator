import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Bitcoin Growth Model (94% R² formula)
def bitcoin_growth_model(days):
    """Bitcoin growth model: log10(price) = a * ln(day) + b"""
    a = 1.6329135221917355
    b = -9.328646304661454
    return 10**(a * np.log(days) + b)

def bitcoin_volatility_model(years):
    """Actual volatility model from data analysis: Polynomial Decay (Volatility_365d)"""
    # Parameters from our analysis: [0.00839752, -0.24824989, 2.35663838]
    # Formula: volatility = 0.008398 * years^2 + (-0.248250) * years + 2.356638
    a = 0.00839752
    b = -0.24824989
    c = 2.35663838
    
    # Calculate volatility using polynomial decay
    volatility = a * (years**2) + b * years + c
    
    # Ensure volatility is reasonable (not negative, not too high)
    volatility = max(0.1, min(2.0, volatility))  # Between 10% and 200%
    
    return volatility

def generate_bitcoin_price_paths(initial_price, years, n_paths=1000, steps_per_year=252):
    """
    Generate Bitcoin price paths using proper Geometric Brownian Motion
    with time-varying volatility based on the better model
    """
    total_steps = int(years * steps_per_year)
    dt = 1 / steps_per_year  # Time step in years
    
    # Initialize price paths
    paths = np.zeros((n_paths, total_steps + 1))
    paths[:, 0] = initial_price
    
    # Pre-calculate volatility for each time step
    # Start from current Bitcoin age (15 years) and project forward
    current_bitcoin_age = 15  # Bitcoin is ~15 years old
    time_points = np.linspace(current_bitcoin_age, current_bitcoin_age + years, total_steps + 1)
    volatilities = np.array([bitcoin_volatility_model(t) for t in time_points])
    
    # Calculate the total expected growth from the growth model
    current_day = 5439  # Starting from current day in dataset
    final_day = current_day + int(years * 365.25)
    
    # Calculate the expected final price from the growth model
    expected_final_price = bitcoin_growth_model(final_day)
    
    # Calculate the annualized drift that will give us this expected final price
    # Using: E[S(T)] = S(0) * exp(μ * T), so μ = ln(E[S(T)]/S(0)) / T
    total_expected_return = np.log(expected_final_price / initial_price)
    annualized_drift = total_expected_return / years
    
    # Convert to per-step drift
    drift_per_step = annualized_drift * dt
    
    for step in range(total_steps):
        # Get volatility for this time step
        vol = volatilities[step]
        
        # Generate random shocks
        random_shocks = np.random.normal(0, 1, n_paths)
        
        # GBM formula: S(t+dt) = S(t) * exp((μ - 0.5σ²)dt + σ√(dt) * Z)
        # Use constant drift to ensure mean matches growth model
        log_returns = (drift_per_step - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * random_shocks
        
        # Update prices
        paths[:, step + 1] = paths[:, step] * np.exp(log_returns)
    
    return paths, time_points

def run_bitcoin_lump_sum_monte_carlo(initial_investment=100000, years=10, n_paths=1000):
    """
    Run Bitcoin lump sum Monte Carlo simulation with proper volatility model
    """
    print("BITCOIN LUMP SUM MONTE CARLO SIMULATION")
    print("=" * 50)
    print(f"Initial Investment: ${initial_investment:,}")
    print(f"Time Horizon: {years} years")
    print(f"Number of Paths: {n_paths:,}")
    print(f"Using Actual Volatility Model: σ(t) = 0.008398 × t² + (-0.248250) × t + 2.356638")
    print()
    
    # Use the growth model's predicted price for "today" to match pure growth model
    current_day = 5439  # Starting day in dataset
    model_current_price = bitcoin_growth_model(current_day)
    
    # Generate price paths starting from model price (not actual price)
    print("Generating price paths...")
    price_paths, time_points = generate_bitcoin_price_paths(
        initial_price=model_current_price,
        years=years,
        n_paths=n_paths,
        steps_per_year=252
    )
    
    # Calculate investment outcomes
    # Since we're using model price, the investment buys: initial_investment / model_current_price BTC
    initial_btc_amount = initial_investment / model_current_price
    final_prices = price_paths[:, -1]
    final_values = initial_btc_amount * final_prices
    
    # Calculate returns
    total_returns = (final_values - initial_investment) / initial_investment
    annualized_returns = (final_values / initial_investment) ** (1/years) - 1
    
    # Calculate statistics
    results = {
        'mean_final_value': np.mean(final_values),
        'median_final_value': np.median(final_values),
        'std_final_value': np.std(final_values),
        'percentile_10': np.percentile(final_values, 10),
        'percentile_25': np.percentile(final_values, 25),
        'percentile_75': np.percentile(final_values, 75),
        'percentile_90': np.percentile(final_values, 90),
        'mean_total_return': np.mean(total_returns),
        'median_total_return': np.median(total_returns),
        'mean_annualized_return': np.mean(annualized_returns),
        'median_annualized_return': np.median(annualized_returns),
        'probability_positive': np.mean(final_values > initial_investment),
        'probability_double': np.mean(final_values > 2 * initial_investment),
        'probability_loss_50': np.mean(final_values < 0.5 * initial_investment)
    }
    
    # Print results
    print("SIMULATION RESULTS:")
    print("-" * 30)
    print(f"Mean Final Value: ${results['mean_final_value']:,.0f}")
    print(f"Median Final Value: ${results['median_final_value']:,.0f}")
    print(f"Standard Deviation: ${results['std_final_value']:,.0f}")
    print()
    print("PERCENTILE ANALYSIS:")
    print(f"10th Percentile: ${results['percentile_10']:,.0f}")
    print(f"25th Percentile: ${results['percentile_25']:,.0f}")
    print(f"75th Percentile: ${results['percentile_75']:,.0f}")
    print(f"90th Percentile: ${results['percentile_90']:,.0f}")
    print()
    print("RETURN ANALYSIS:")
    print(f"Mean Total Return: {results['mean_total_return']:.1%}")
    print(f"Median Total Return: {results['median_total_return']:.1%}")
    print(f"Mean Annualized Return: {results['mean_annualized_return']:.1%}")
    print(f"Median Annualized Return: {results['median_annualized_return']:.1%}")
    print()
    print("PROBABILITY ANALYSIS:")
    print(f"Probability of Positive Return: {results['probability_positive']:.1%}")
    print(f"Probability of Doubling: {results['probability_double']:.1%}")
    print(f"Probability of 50%+ Loss: {results['probability_loss_50']:.1%}")
    
    # Save results
    results_df = pd.DataFrame({
        'Final_Value': final_values,
        'Total_Return': total_returns,
        'Annualized_Return': annualized_returns
    })
    
    results_df.to_csv('../Results/bitcoin_monte_carlo_lump_sum_results.csv', index=False)
    print(f"\nDetailed results saved to: bitcoin_monte_carlo_lump_sum_results.csv")
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Sample price paths
    sample_paths = price_paths[:min(100, n_paths), :]
    time_years = np.linspace(0, years, len(time_points))  # Display as years from now, not Bitcoin age
    
    for i in range(len(sample_paths)):
        ax1.plot(time_years, sample_paths[i], alpha=0.3, linewidth=0.5)
    
    ax1.set_title('Sample Bitcoin Price Paths')
    ax1.set_xlabel('Years')
    ax1.set_ylabel('Bitcoin Price ($)')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Final value distribution
    ax2.hist(final_values, bins=50, alpha=0.7, density=True)
    ax2.axvline(results['mean_final_value'], color='red', linestyle='--', label=f'Mean: ${results["mean_final_value"]:,.0f}')
    ax2.axvline(results['median_final_value'], color='green', linestyle='--', label=f'Median: ${results["median_final_value"]:,.0f}')
    ax2.set_title('Distribution of Final Investment Values')
    ax2.set_xlabel('Final Value ($)')
    ax2.set_ylabel('Density')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Annualized returns distribution
    ax3.hist(annualized_returns, bins=50, alpha=0.7, density=True)
    ax3.axvline(results['mean_annualized_return'], color='red', linestyle='--', label=f'Mean: {results["mean_annualized_return"]:.1%}')
    ax3.axvline(results['median_annualized_return'], color='green', linestyle='--', label=f'Median: {results["median_annualized_return"]:.1%}')
    ax3.set_title('Distribution of Annualized Returns')
    ax3.set_xlabel('Annualized Return')
    ax3.set_ylabel('Density')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    
    # Plot 4: Volatility decay over time
    vol_time = np.linspace(0, years, 100)
    vol_values = [bitcoin_volatility_model(15 + t) for t in vol_time]  # Start from current Bitcoin age
    
    ax4.plot(vol_time, vol_values, 'b-', linewidth=2, label='Volatility Model')
    ax4.set_title('Bitcoin Volatility Decay Over Time')
    ax4.set_xlabel('Years')
    ax4.set_ylabel('Annualized Volatility')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    plt.tight_layout()
    plt.savefig('bitcoin_monte_carlo_lump_sum_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results, price_paths

if __name__ == "__main__":
    # Run comprehensive lump sum analysis for multiple horizons
    horizons = [1, 3, 5, 10]
    initial_investment = 100000
    n_paths = 1000
    
    all_results = []
    
    for years in horizons:
        print(f"\n{'='*60}")
        print(f"RUNNING {years}-YEAR LUMP SUM SIMULATION")
        print(f"{'='*60}")
        
        results, paths = run_bitcoin_lump_sum_monte_carlo(
            initial_investment=initial_investment,
            years=years,
            n_paths=n_paths
        )
        
        # Add horizon info to results
        results['horizon_years'] = years
        results['initial_investment'] = initial_investment
        results['n_paths'] = n_paths
        all_results.append(results)
    
    # Save comprehensive results to CSV
    import pandas as pd
    df_results = pd.DataFrame(all_results)
    
    # Reorder columns for better readability
    column_order = ['horizon_years', 'initial_investment', 'n_paths', 
                   'mean_final_value', 'median_final_value', 'std_final_value',
                   'percentile_10', 'percentile_25', 'percentile_75', 'percentile_90',
                   'mean_total_return', 'median_total_return', 
                   'mean_annualized_return', 'median_annualized_return',
                   'probability_positive', 'probability_double', 'probability_loss_50']
    
    df_results = df_results[column_order]
    
    results_path = "../Results/bitcoin_monte_carlo_lump_sum_results.csv"
    df_results.to_csv(results_path, index=False)
    
    print(f"\n{'='*60}")
    print("COMPREHENSIVE RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Results saved to: {results_path}")
    print("\nSUMMARY TABLE:")
    print("-" * 80)
    print(f"{'Horizon':<8} {'Median Value':<15} {'Median CAGR':<12} {'P(Positive)':<12} {'P(Double)':<10}")
    print("-" * 80)
    
    for _, row in df_results.iterrows():
        print(f"{row['horizon_years']:<8} ${row['median_final_value']:<14,.0f} {row['median_annualized_return']:<11.1%} {row['probability_positive']:<11.1%} {row['probability_double']:<9.1%}") 
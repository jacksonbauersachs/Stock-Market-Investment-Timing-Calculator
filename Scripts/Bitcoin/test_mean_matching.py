import numpy as np
from bitcoin_monte_carlo_fixed import run_bitcoin_lump_sum_monte_carlo

# Bitcoin Growth Model coefficients
a = 1.6329135221917355
b = -9.328646304661454

def bitcoin_growth_model(days):
    """Bitcoin growth model: log10(price) = a * ln(day) + b"""
    return 10**(a * np.log(days) + b)

def calculate_growth_rate(start_day, end_day):
    """Calculate the growth multiple from start_day to end_day"""
    start_price = bitcoin_growth_model(start_day)
    end_price = bitcoin_growth_model(end_day)
    return end_price / start_price

def test_mean_matching():
    print("TESTING MONTE CARLO MEAN vs PURE GROWTH MODEL")
    print("=" * 60)
    
    initial_investment = 100000
    current_day = 5439
    
    test_horizons = [1, 3, 5, 10]
    
    for years in test_horizons:
        print(f"\n--- {years}-Year Test ---")
        
        # Pure Growth Model prediction
        days_in_horizon = int(years * 365.25)
        growth_multiple = calculate_growth_rate(current_day, current_day + days_in_horizon)
        pure_growth_value = initial_investment * growth_multiple
        
        # Monte Carlo simulation
        print(f"Running Monte Carlo simulation for {years} years...")
        results, _ = run_bitcoin_lump_sum_monte_carlo(
            initial_investment=initial_investment,
            years=years,
            n_paths=5000  # More paths for better accuracy
        )
        
        mc_mean = results['mean_final_value']
        mc_median = results['median_final_value']
        
        # Calculate differences
        mean_diff = (mc_mean - pure_growth_value) / pure_growth_value * 100
        median_diff = (mc_median - pure_growth_value) / pure_growth_value * 100
        
        print(f"Pure Growth Model: ${pure_growth_value:,.0f}")
        print(f"Monte Carlo Mean:  ${mc_mean:,.0f} (diff: {mean_diff:+.2f}%)")
        print(f"Monte Carlo Median: ${mc_median:,.0f} (diff: {median_diff:+.2f}%)")
        
        # Check if means are close (within 1%)
        if abs(mean_diff) < 1.0:
            print("✓ PASS: Means match within 1%")
        else:
            print("✗ FAIL: Means differ by more than 1%")

if __name__ == "__main__":
    test_mean_matching() 
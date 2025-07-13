import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from bitcoin_monte_carlo_fixed import run_bitcoin_lump_sum_monte_carlo

def main():
    print("BITCOIN LUMP SUM INVESTMENT ANALYSIS")
    print("=" * 60)
    print("Growth Model: log10(price) = 1.633 * ln(day) - 9.329 (R² = 94%)")
    print("Volatility Model: σ(t) = 0.4 + 0.6 × e^(-0.15t)")
    print("Starting Investment: $100,000")
    print("=" * 60)
    
    horizons = [1, 3, 5, 10]
    summary_results = []
    
    for years in horizons:
        print(f"\n{'='*50}")
        print(f"  {years}-YEAR LUMP SUM ANALYSIS")
        print(f"{'='*50}")
        
        try:
            results, _ = run_bitcoin_lump_sum_monte_carlo(
                initial_investment=100000,
                years=years,
                n_paths=1000
            )
            
            summary_results.append({
                'years': years,
                'median_value': results['median_final_value'],
                'mean_value': results['mean_final_value'],
                'median_cagr': results['median_annualized_return'],
                'prob_positive': results['probability_positive'],
                'prob_double': results['probability_double'],
                'prob_loss_50': results['probability_loss_50'],
                'percentile_10': results['percentile_10'],
                'percentile_90': results['percentile_90']
            })
            
            print(f"✓ {years}-year simulation completed")
            
        except Exception as e:
            print(f"✗ Error in {years}-year simulation: {e}")
    
    # Print comprehensive summary
    print(f"\n{'='*80}")
    print("COMPREHENSIVE BITCOIN LUMP SUM SUMMARY")
    print(f"{'='*80}")
    print(f"{'Horizon':<8} {'Median Value':<15} {'Median CAGR':<12} {'P(Positive)':<12} {'P(Double)':<12} {'P(50% Loss)':<12}")
    print("-" * 80)
    
    for result in summary_results:
        print(f"{result['years']:<8} ${result['median_value']:<14,.0f} {result['median_cagr']:<11.1%} {result['prob_positive']:<11.1%} {result['prob_double']:<11.1%} {result['prob_loss_50']:<11.1%}")
    
    print(f"\n{'='*80}")
    print("KEY INSIGHTS:")
    print(f"{'='*80}")
    
    for result in summary_results:
        years = result['years']
        median_val = result['median_value']
        median_cagr = result['median_cagr']
        prob_pos = result['prob_positive']
        
        print(f"\n{years}-Year Investment:")
        print(f"  • Median outcome: ${median_val:,.0f} ({median_cagr:.1%} annual return)")
        print(f"  • Probability of profit: {prob_pos:.1%}")
        print(f"  • 10th-90th percentile range: ${result['percentile_10']:,.0f} - ${result['percentile_90']:,.0f}")
    
    # Compare with traditional growth model
    print(f"\n{'='*80}")
    print("COMPARISON WITH PURE GROWTH MODEL:")
    print(f"{'='*80}")
    
    # Pure growth model predictions
    a = 1.6329135221917355
    b = -9.328646304661454
    
    def growth_model_price(days):
        import numpy as np
        return 10**(a * np.log(days) + b)
    
    def calculate_growth_rate(start_day, end_day):
        start_price = growth_model_price(start_day)
        end_price = growth_model_price(end_day)
        return end_price / start_price
    
    current_day = 5439
    
    print(f"{'Horizon':<8} {'Monte Carlo':<15} {'Growth Model':<15} {'Difference':<12}")
    print("-" * 60)
    
    for result in summary_results:
        years = result['years']
        days_in_horizon = int(years * 365.25)
        growth_multiple = calculate_growth_rate(current_day, current_day + days_in_horizon)
        growth_value = 100000 * growth_multiple
        mc_value = result['median_value']
        diff = (mc_value - growth_value) / growth_value
        
        print(f"{years:<8} ${mc_value:<14,.0f} ${growth_value:<14,.0f} {diff:<11.1%}")
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")

if __name__ == "__main__":
    main() 
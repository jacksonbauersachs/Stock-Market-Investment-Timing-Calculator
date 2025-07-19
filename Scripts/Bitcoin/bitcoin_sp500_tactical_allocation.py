import numpy as np
import pandas as pd
from itertools import product
import matplotlib.pyplot as plt
from tqdm import tqdm

class BitcoinSP500TacticalAllocation:
    def __init__(self):
        # Bitcoin Growth Model (94% R² formula)
        self.btc_a = 1.6329135221917355
        self.btc_b = -9.328646304661454
        
        # S&P 500 Growth Model (93.97% R² exponential)
        self.sp500_a = 1.25856063e+02  # 125.86
        self.sp500_b = 8.18333315e-02  # 8.18% annual growth
        
        # Strategy parameters
        self.total_investment = 100000  # $100k total to invest
        self.years = 5  # 5-year investment horizon
        self.monthly_amount = 1000  # $1k per month available
        self.n_simulations = 500  # Monte Carlo simulations per strategy
        
        print("BITCOIN vs S&P 500 TACTICAL ALLOCATION STRATEGY")
        print("=" * 70)
        print(f"Bitcoin Growth Model: log10(price) = {self.btc_a:.3f} * ln(day) + {self.btc_b:.3f}")
        print(f"S&P 500 Growth Model: price = {self.sp500_a:.2f} * exp({self.sp500_b:.4f} * years)")
        print(f"Bitcoin Volatility: log10(vol) = -0.364 * log10(years) + 0.103")
        print(f"S&P 500 Volatility: 3rd order polynomial (30d model)")
        print(f"Total Investment Budget: ${self.total_investment:,}")
        print(f"Monthly Investment: ${self.monthly_amount:,}")
        print(f"Investment Horizon: {self.years} years")
        print(f"Simulations per strategy: {self.n_simulations}")
        print()
    
    def bitcoin_growth_model(self, days):
        """Bitcoin growth model: log10(price) = a * ln(day) + b"""
        return 10**(self.btc_a * np.log(days) + self.btc_b)
    
    def bitcoin_volatility_model(self, years):
        """Bitcoin volatility model: log10(vol) = a * log10(years) + b"""
        a = -0.36442287700521414
        b = 0.1028262655650134
        years = max(years, 0.01)
        log_vol = a * np.log10(years) + b
        volatility = 10 ** log_vol
        return volatility
    
    def sp500_growth_model(self, years):
        """S&P 500 exponential growth model: price = a * exp(b * years)"""
        return self.sp500_a * np.exp(self.sp500_b * years)
    
    def sp500_volatility_model(self, years):
        """S&P 500 volatility model: 3rd order polynomial (30d model)"""
        # Vol_30d: Polynomial 3rd, params=[-4.46905154e-06  2.59644502e-04 -3.00132646e-03  1.80174615e-01]
        coeffs = [-4.46905154e-06, 2.59644502e-04, -3.00132646e-03, 1.80174615e-01]
        years = max(years, 0.01)
        volatility = coeffs[0] * years**3 + coeffs[1] * years**2 + coeffs[2] * years + coeffs[3]
        return max(volatility, 0.05)  # Minimum 5% volatility
    
    def generate_price_paths(self, years, steps_per_year=12):
        """Generate correlated Bitcoin and S&P 500 price paths"""
        total_steps = int(years * steps_per_year)
        dt = 1 / steps_per_year
        
        # Bitcoin initialization
        btc_current_day = 5439  # Starting day
        btc_final_day = btc_current_day + int(years * 365.25)
        btc_initial_price = self.bitcoin_growth_model(btc_current_day)
        btc_expected_final_price = self.bitcoin_growth_model(btc_final_day)
        btc_total_expected_return = np.log(btc_expected_final_price / btc_initial_price)
        btc_annualized_drift = btc_total_expected_return / years
        btc_drift_per_step = btc_annualized_drift * dt
        
        # S&P 500 initialization
        sp500_initial_price = self.sp500_growth_model(0)
        sp500_expected_final_price = self.sp500_growth_model(years)
        sp500_total_expected_return = np.log(sp500_expected_final_price / sp500_initial_price)
        sp500_annualized_drift = sp500_total_expected_return / years
        sp500_drift_per_step = sp500_annualized_drift * dt
        
        # Generate paths
        btc_prices = np.zeros(total_steps + 1)
        btc_prices[0] = btc_initial_price
        btc_trend_prices = np.zeros(total_steps + 1)
        btc_trend_prices[0] = btc_initial_price
        
        sp500_prices = np.zeros(total_steps + 1)
        sp500_prices[0] = sp500_initial_price
        sp500_trend_prices = np.zeros(total_steps + 1)
        sp500_trend_prices[0] = sp500_initial_price
        
        current_bitcoin_age = 15  # Bitcoin is ~15 years old
        correlation = 0.3  # Moderate correlation between Bitcoin and S&P 500
        
        for step in range(total_steps):
            # Current time
            current_time = step * dt
            
            # Volatilities
            btc_vol = self.bitcoin_volatility_model(current_bitcoin_age + current_time)
            sp500_vol = self.sp500_volatility_model(current_time + 1)  # S&P 500 years start from 1
            
            # Correlated random shocks
            z1 = np.random.normal(0, 1)
            z2 = np.random.normal(0, 1)
            btc_shock = z1
            sp500_shock = correlation * z1 + np.sqrt(1 - correlation**2) * z2
            
            # Update Bitcoin prices
            btc_log_return = (btc_drift_per_step - 0.5 * btc_vol**2) * dt + btc_vol * np.sqrt(dt) * btc_shock
            btc_prices[step + 1] = btc_prices[step] * np.exp(btc_log_return)
            btc_trend_prices[step + 1] = btc_trend_prices[step] * np.exp(btc_drift_per_step * dt)
            
            # Update S&P 500 prices
            sp500_log_return = (sp500_drift_per_step - 0.5 * sp500_vol**2) * dt + sp500_vol * np.sqrt(dt) * sp500_shock
            sp500_prices[step + 1] = sp500_prices[step] * np.exp(sp500_log_return)
            sp500_trend_prices[step + 1] = sp500_trend_prices[step] * np.exp(sp500_drift_per_step * dt)
        
        return btc_prices, btc_trend_prices, sp500_prices, sp500_trend_prices
    
    def simulate_balanced_strategy(self, btc_allocation, years):
        """Simulate fixed allocation strategy (baseline)"""
        steps_per_year = 12
        total_steps = int(years * steps_per_year)
        
        final_values = []
        
        for sim in range(self.n_simulations):
            btc_prices, _, sp500_prices, _ = self.generate_price_paths(years, steps_per_year)
            
            total_btc_shares = 0
            total_sp500_shares = 0
            total_spent = 0
            
            for step in range(1, min(total_steps + 1, len(btc_prices))):
                if total_spent >= self.total_investment:
                    break
                
                btc_price = btc_prices[step]
                sp500_price = sp500_prices[step]
                monthly_cash = min(self.monthly_amount, self.total_investment - total_spent)
                
                if monthly_cash <= 0:
                    break
                
                # Fixed allocation
                btc_investment = monthly_cash * (btc_allocation / 100)
                sp500_investment = monthly_cash * (1 - btc_allocation / 100)
                
                # Buy shares
                total_btc_shares += btc_investment / btc_price
                total_sp500_shares += sp500_investment / sp500_price
                total_spent += monthly_cash
            
            # Final value
            final_btc_value = total_btc_shares * btc_prices[-1]
            final_sp500_value = total_sp500_shares * sp500_prices[-1]
            final_value = final_btc_value + final_sp500_value
            final_values.append(final_value)
        
        return np.array(final_values)
    
    def simulate_tactical_allocation_strategy(self, base_btc_allocation, buy_tiers, sell_tiers, years):
        """
        Simulate tactical allocation strategy
        
        Parameters:
        - base_btc_allocation: Base % allocation to Bitcoin (rest goes to S&P 500)
        - buy_tiers: List of (threshold_pct, rebalance_pct) for buying more Bitcoin
        - sell_tiers: List of (threshold_pct, rebalance_pct) for selling Bitcoin
        """
        steps_per_year = 12
        total_steps = int(years * steps_per_year)
        
        final_values = []
        
        for sim in range(self.n_simulations):
            btc_prices, btc_trend_prices, sp500_prices, sp500_trend_prices = self.generate_price_paths(years, steps_per_year)
            
            total_btc_shares = 0
            total_sp500_shares = 0
            total_spent = 0
            
            for step in range(1, min(total_steps + 1, len(btc_prices))):
                if total_spent >= self.total_investment:
                    break
                
                btc_price = btc_prices[step]
                btc_trend_price = btc_trend_prices[step]
                sp500_price = sp500_prices[step]
                monthly_cash = min(self.monthly_amount, self.total_investment - total_spent)
                
                if monthly_cash <= 0:
                    break
                
                # Calculate Bitcoin deviation from trend
                btc_vs_trend = (btc_price - btc_trend_price) / btc_trend_price
                
                # Determine tactical allocation based on Bitcoin trend deviation
                tactical_btc_allocation = base_btc_allocation
                
                # Check buy tiers (Bitcoin below trend - increase allocation)
                if btc_vs_trend < 0:
                    for threshold_pct, rebalance_pct in buy_tiers:
                        if btc_vs_trend < -threshold_pct / 100:
                            tactical_btc_allocation = min(100, base_btc_allocation + rebalance_pct)
                            break
                
                # Check sell tiers (Bitcoin above trend - decrease allocation)
                elif btc_vs_trend > 0:
                    for threshold_pct, rebalance_pct in sell_tiers:
                        if btc_vs_trend > threshold_pct / 100:
                            tactical_btc_allocation = max(0, base_btc_allocation - rebalance_pct)
                            break
                
                # Invest based on tactical allocation
                btc_investment = monthly_cash * (tactical_btc_allocation / 100)
                sp500_investment = monthly_cash * (1 - tactical_btc_allocation / 100)
                
                # Buy shares
                total_btc_shares += btc_investment / btc_price
                total_sp500_shares += sp500_investment / sp500_price
                total_spent += monthly_cash
            
            # Final value
            final_btc_value = total_btc_shares * btc_prices[-1]
            final_sp500_value = total_sp500_shares * sp500_prices[-1]
            final_value = final_btc_value + final_sp500_value
            final_values.append(final_value)
        
        return np.array(final_values)
    
    def run_strategy_search(self):
        """Run search for best tactical allocation strategies"""
        print("RUNNING TACTICAL ALLOCATION STRATEGY SEARCH...")
        print("=" * 60)
        
        # Define parameter grids
        base_btc_allocations = [20, 30, 40, 50, 60, 70, 80]  # Base Bitcoin allocation %
        
        # Define buy tier configurations (increase Bitcoin allocation when below trend)
        buy_tier_configs = [
            [(10, 20), (20, 30), (30, 40)],  # Conservative rebalancing
            [(15, 25), (25, 35), (35, 45)],  # Moderate rebalancing
            [(10, 30), (20, 40), (30, 50)],  # Aggressive rebalancing
            [(5, 15), (15, 25), (25, 35)],   # Fine-grained rebalancing
            [(20, 40), (40, 60)],            # Simple 2-tier
        ]
        
        # Define sell tier configurations (decrease Bitcoin allocation when above trend)
        sell_tier_configs = [
            [(20, 15), (40, 25), (60, 35)],  # Conservative rebalancing
            [(30, 20), (50, 30), (70, 40)],  # Moderate rebalancing
            [(20, 25), (40, 35), (60, 45)],  # Aggressive rebalancing
            [(15, 10), (30, 20), (45, 30)],  # Fine-grained rebalancing
            [(25, 30), (50, 50)],            # Simple 2-tier
        ]
        
        total_combinations = len(base_btc_allocations) * len(buy_tier_configs) * len(sell_tier_configs)
        print(f"Testing {total_combinations} tactical allocation combinations...")
        
        results = []
        
        # Test baseline strategies
        print("\nTesting baseline fixed allocation strategies...")
        baseline_results = {}
        for btc_alloc in [20, 40, 60, 80]:
            baseline_res = self.simulate_balanced_strategy(btc_alloc, self.years)
            baseline_stats = self.calculate_detailed_stats(baseline_res)
            baseline_results[btc_alloc] = baseline_stats
            print(f"  {btc_alloc}% Bitcoin / {100-btc_alloc}% S&P 500: ${baseline_stats['median']:,.0f} median")
        
        # Use 60/40 as primary baseline (common portfolio allocation)
        baseline_stats = baseline_results[60]
        
        print(f"\nBaseline (60% Bitcoin / 40% S&P 500):")
        print(f"  Median: ${baseline_stats['median']:,.0f}")
        print(f"  Mean: ${baseline_stats['mean']:,.0f}")
        print(f"  CAGR: {baseline_stats['cagr']:.2f}%")
        print()
        
        # Grid search with progress bar
        pbar = tqdm(total=total_combinations, desc="Tactical Search")
        
        for base_btc_alloc in base_btc_allocations:
            for buy_config in buy_tier_configs:
                for sell_config in sell_tier_configs:
                    strategy_results = self.simulate_tactical_allocation_strategy(
                        base_btc_alloc, buy_config, sell_config, self.years
                    )
                    
                    stats = self.calculate_detailed_stats(strategy_results)
                    
                    # Calculate performance vs baseline
                    median_improvement = (stats['median'] - baseline_stats['median']) / baseline_stats['median'] * 100
                    mean_improvement = (stats['mean'] - baseline_stats['mean']) / baseline_stats['mean'] * 100
                    cagr_improvement = stats['cagr'] - baseline_stats['cagr']
                    
                    # Calculate Sharpe-like ratio
                    sharpe_ratio = (stats['mean'] - baseline_stats['mean']) / stats['std'] if stats['std'] > 0 else 0
                    
                    results.append({
                        'base_btc_allocation': base_btc_alloc,
                        'buy_tiers': str(buy_config),
                        'sell_tiers': str(sell_config),
                        'median_value': stats['median'],
                        'mean_value': stats['mean'],
                        'std_value': stats['std'],
                        'cagr': stats['cagr'],
                        'max_value': stats['max'],
                        'min_value': stats['min'],
                        'percentile_25': stats['p25'],
                        'percentile_75': stats['p75'],
                        'median_improvement': median_improvement,
                        'mean_improvement': mean_improvement,
                        'cagr_improvement': cagr_improvement,
                        'sharpe_ratio': sharpe_ratio,
                        'win_rate': np.mean(strategy_results > baseline_stats['median']) * 100,
                        'beat_baseline_rate': np.mean(strategy_results > baseline_results[60]['median']) * 100
                    })
                    
                    pbar.update(1)
        
        pbar.close()
        
        # Convert to DataFrame and sort by performance
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('median_improvement', ascending=False)
        
        return results_df, baseline_stats, baseline_results
    
    def calculate_detailed_stats(self, results):
        """Calculate comprehensive statistics"""
        final_investment = min(self.total_investment, self.monthly_amount * 12 * self.years)
        
        stats = {
            'median': np.median(results),
            'mean': np.mean(results),
            'std': np.std(results),
            'min': np.min(results),
            'max': np.max(results),
            'p25': np.percentile(results, 25),
            'p75': np.percentile(results, 75),
            'cagr': ((np.median(results) / final_investment) ** (1/self.years) - 1) * 100
        }
        
        return stats
    
    def analyze_results(self, results_df, baseline_stats, baseline_results):
        """Analyze and display comprehensive results"""
        print("\nTOP 10 TACTICAL ALLOCATION STRATEGIES (by median improvement):")
        print("=" * 80)
        
        top_10 = results_df.head(10)
        
        for i, row in top_10.iterrows():
            print(f"\n#{len(top_10) - len(top_10) + list(top_10.index).index(i) + 1}. Base: {row['base_btc_allocation']:.0f}% Bitcoin")
            print(f"    Buy Tiers: {row['buy_tiers']}")
            print(f"    Sell Tiers: {row['sell_tiers']}")
            print(f"    Median: ${row['median_value']:,.0f} ({row['median_improvement']:+.1f}% vs 60/40)")
            print(f"    Mean: ${row['mean_value']:,.0f} ({row['mean_improvement']:+.1f}% vs 60/40)")
            print(f"    CAGR: {row['cagr']:.2f}% ({row['cagr_improvement']:+.2f}% vs 60/40)")
            print(f"    Win Rate: {row['win_rate']:.1f}% | Beat 60/40: {row['beat_baseline_rate']:.1f}%")
            print(f"    Sharpe Ratio: {row['sharpe_ratio']:.3f}")
        
        # Best strategy detailed analysis
        best_strategy = results_df.iloc[0]
        
        print(f"\n🏆 BEST TACTICAL ALLOCATION STRATEGY:")
        print("=" * 60)
        print(f"Base Bitcoin Allocation: {best_strategy['base_btc_allocation']:.0f}%")
        print(f"Base S&P 500 Allocation: {100 - best_strategy['base_btc_allocation']:.0f}%")
        print(f"Buy Tiers: {best_strategy['buy_tiers']}")
        print(f"Sell Tiers: {best_strategy['sell_tiers']}")
        print(f"Median improvement: {best_strategy['median_improvement']:+.1f}%")
        print(f"CAGR improvement: {best_strategy['cagr_improvement']:+.2f}%")
        print(f"Win rate: {best_strategy['win_rate']:.1f}%")
        
        # Comprehensive comparison
        print(f"\n📊 COMPREHENSIVE PERFORMANCE COMPARISON:")
        print("=" * 60)
        
        # Baseline Strategy
        print(f"📈 BASELINE (60% Bitcoin / 40% S&P 500):")
        print(f"   Median Return: ${baseline_stats['median']:,.0f}")
        print(f"   Mean Return: ${baseline_stats['mean']:,.0f}")
        print(f"   CAGR: {baseline_stats['cagr']:.2f}%")
        print(f"   Standard Deviation: ${baseline_stats['std']:,.0f}")
        print(f"   25th Percentile: ${baseline_stats['p25']:,.0f}")
        print(f"   75th Percentile: ${baseline_stats['p75']:,.0f}")
        print(f"   Best Case: ${baseline_stats['max']:,.0f}")
        print(f"   Worst Case: ${baseline_stats['min']:,.0f}")
        
        # Best Tactical Strategy
        print(f"\n🚀 BEST TACTICAL ALLOCATION STRATEGY:")
        print(f"   Median Return: ${best_strategy['median_value']:,.0f}")
        print(f"   Mean Return: ${best_strategy['mean_value']:,.0f}")
        print(f"   CAGR: {best_strategy['cagr']:.2f}%")
        print(f"   Standard Deviation: ${best_strategy['std_value']:,.0f}")
        print(f"   25th Percentile: ${best_strategy['percentile_25']:,.0f}")
        print(f"   75th Percentile: ${best_strategy['percentile_75']:,.0f}")
        print(f"   Best Case: ${best_strategy['max_value']:,.0f}")
        print(f"   Worst Case: ${best_strategy['min_value']:,.0f}")
        
        # Performance Differences
        print(f"\n🔥 PERFORMANCE IMPROVEMENTS:")
        print(f"   Median Improvement: {best_strategy['median_improvement']:+.1f}%")
        print(f"   Mean Improvement: {best_strategy['mean_improvement']:+.1f}%")
        print(f"   CAGR Improvement: {best_strategy['cagr_improvement']:+.2f}%")
        print(f"   Additional Return: ${best_strategy['median_value'] - baseline_stats['median']:,.0f}")
        print(f"   Beat 60/40 Rate: {best_strategy['beat_baseline_rate']:.1f}%")
        
        # Show other baseline comparisons
        print(f"\n📈 OTHER BASELINE COMPARISONS:")
        print("=" * 50)
        for btc_alloc in [20, 40, 80]:
            if btc_alloc in baseline_results:
                baseline = baseline_results[btc_alloc]
                improvement = (best_strategy['median_value'] - baseline['median']) / baseline['median'] * 100
                print(f"{btc_alloc}% Bitcoin / {100-btc_alloc}% S&P 500: ${baseline['median']:,.0f} → {improvement:+.1f}%")
        
        # Save results
        results_df.to_csv('Results/Bitcoin/bitcoin_sp500_tactical_allocation_results.csv', index=False)
        
        # Summary statistics
        print(f"\n📋 SUMMARY STATISTICS:")
        print(f"   Strategies tested: {len(results_df)}")
        print(f"   Strategies beating 60/40: {len(results_df[results_df['median_improvement'] > 0])}")
        print(f"   Best improvement: {results_df['median_improvement'].max():+.1f}%")
        print(f"   Worst performance: {results_df['median_improvement'].min():+.1f}%")
        print(f"   Average improvement: {results_df['median_improvement'].mean():+.1f}%")
        
        return best_strategy

def main():
    strategy = BitcoinSP500TacticalAllocation()
    results_df, baseline_stats, baseline_results = strategy.run_strategy_search()
    best_strategy = strategy.analyze_results(results_df, baseline_stats, baseline_results)
    
    print(f"\n📁 Results saved to: Results/Bitcoin/bitcoin_sp500_tactical_allocation_results.csv")
    
    return best_strategy

if __name__ == "__main__":
    best_strategy = main() 
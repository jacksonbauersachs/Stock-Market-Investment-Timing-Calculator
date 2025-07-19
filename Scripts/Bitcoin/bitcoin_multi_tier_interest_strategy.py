import numpy as np
import pandas as pd
from itertools import product
import matplotlib.pyplot as plt
from tqdm import tqdm

class BitcoinMultiTierInterestStrategy:
    def __init__(self):
        # Bitcoin Growth Model (94% R² formula)
        self.a = 1.6329135221917355
        self.b = -9.328646304661454
        
        # Strategy parameters
        self.total_investment = 100000  # $100k total to invest
        self.years = 5  # 5-year investment horizon
        self.monthly_amount = 1000  # $1k per month available
        self.n_simulations = 500  # Monte Carlo simulations per strategy
        self.interest_rate = 0.05  # 5% annual interest on cash reserves
        
        print("BITCOIN MULTI-TIER STRATEGY WITH 5% INTEREST RESERVES")
        print("=" * 70)
        print(f"Using Bitcoin Growth Model: log10(price) = {self.a:.3f} * ln(day) + {self.b:.3f}")
        print(f"Using Bitcoin Volatility Model: log10(vol) = -0.364 * log10(years) + 0.103")
        print(f"Cash Reserve Interest Rate: {self.interest_rate:.1%} annually")
        print(f"Total Investment Budget: ${self.total_investment:,}")
        print(f"Monthly Investment: ${self.monthly_amount:,}")
        print(f"Investment Horizon: {self.years} years")
        print(f"Simulations per strategy: {self.n_simulations}")
        print()
    
    def bitcoin_growth_model(self, days):
        """Bitcoin growth model: log10(price) = a * ln(day) + b"""
        return 10**(self.a * np.log(days) + self.b)
    
    def bitcoin_volatility_model(self, years):
        """Bitcoin volatility model: log10(vol) = a * log10(years) + b"""
        a = -0.36442287700521414
        b = 0.1028262655650134
        years = max(years, 0.01)
        log_vol = a * np.log10(years) + b
        volatility = 10 ** log_vol
        return volatility
    
    def generate_price_path(self, years, steps_per_year=12):
        """Generate single Bitcoin price path"""
        total_steps = int(years * steps_per_year)
        dt = 1 / steps_per_year
        
        # Initialize
        current_day = 5439  # Starting day
        final_day = current_day + int(years * 365.25)
        
        # Calculate drift to match growth model
        initial_price = self.bitcoin_growth_model(current_day)
        expected_final_price = self.bitcoin_growth_model(final_day)
        total_expected_return = np.log(expected_final_price / initial_price)
        annualized_drift = total_expected_return / years
        drift_per_step = annualized_drift * dt
        
        # Generate path
        prices = np.zeros(total_steps + 1)
        prices[0] = initial_price
        trend_prices = np.zeros(total_steps + 1)
        trend_prices[0] = initial_price
        
        current_bitcoin_age = 15  # Bitcoin is ~15 years old
        
        for step in range(total_steps):
            # Current volatility
            current_time = current_bitcoin_age + (step * dt)
            vol = self.bitcoin_volatility_model(current_time)
            
            # Random shock
            shock = np.random.normal(0, 1)
            
            # Update actual price (with volatility)
            log_return = (drift_per_step - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * shock
            prices[step + 1] = prices[step] * np.exp(log_return)
            
            # Update trend price (smooth growth model)
            trend_prices[step + 1] = trend_prices[step] * np.exp(drift_per_step * dt)
        
        return prices, trend_prices
    
    def simulate_dca_strategy(self, monthly_amount, years):
        """Simulate traditional DCA strategy"""
        steps_per_year = 12
        total_steps = int(years * steps_per_year)
        
        final_values = []
        
        for sim in range(self.n_simulations):
            prices, _ = self.generate_price_path(years, steps_per_year)
            
            total_btc = 0
            total_spent = 0
            
            for step in range(1, min(total_steps + 1, len(prices))):
                if total_spent >= self.total_investment:
                    break
                
                current_price = prices[step]
                monthly_cash = min(monthly_amount, self.total_investment - total_spent)
                
                if monthly_cash <= 0:
                    break
                
                # DCA purchase
                btc_bought = monthly_cash / current_price
                total_btc += btc_bought
                total_spent += monthly_cash
            
            final_value = total_btc * prices[-1]
            final_values.append(final_value)
        
        return np.array(final_values)
    
    def simulate_multi_tier_interest_strategy(self, reserve_pct, buy_tiers, sell_tiers, years):
        """
        Simulate multi-tier strategy with interest-earning reserves
        
        Parameters:
        - reserve_pct: % of monthly amount to keep as reserve
        - buy_tiers: List of (threshold_pct, allocation_pct) tuples for buying
        - sell_tiers: List of (threshold_pct, allocation_pct) tuples for selling
        """
        steps_per_year = 12
        total_steps = int(years * steps_per_year)
        dt = 1 / steps_per_year
        monthly_interest_rate = self.interest_rate / 12  # Convert to monthly
        
        final_values = []
        
        for sim in range(self.n_simulations):
            prices, trend_prices = self.generate_price_path(years, steps_per_year)
            
            total_btc = 0
            cash_reserve = 0
            total_spent = 0
            
            for step in range(1, min(total_steps + 1, len(prices))):
                if total_spent >= self.total_investment:
                    break
                
                current_price = prices[step]
                trend_price = trend_prices[step]
                
                # Apply interest to existing cash reserves
                cash_reserve *= (1 + monthly_interest_rate)
                
                # Calculate how far price is from trend
                price_vs_trend = (current_price - trend_price) / trend_price
                
                # Monthly cash flow
                monthly_cash = min(self.monthly_amount, self.total_investment - total_spent)
                if monthly_cash <= 0:
                    break
                
                # Reserve allocation
                reserve_amount = monthly_cash * (reserve_pct / 100)
                dca_amount = monthly_cash - reserve_amount
                cash_reserve += reserve_amount
                
                # DCA purchase (always happens)
                if dca_amount > 0:
                    btc_bought = dca_amount / current_price
                    total_btc += btc_bought
                    total_spent += dca_amount
                
                # Multi-tier buying (when price is below trend)
                if price_vs_trend < 0:
                    for threshold_pct, allocation_pct in buy_tiers:
                        if price_vs_trend < -threshold_pct / 100:
                            # Calculate buy amount based on allocation percentage
                            buy_amount = cash_reserve * (allocation_pct / 100)
                            if buy_amount > 0:
                                btc_bought = buy_amount / current_price
                                total_btc += btc_bought
                                cash_reserve -= buy_amount
                                total_spent += buy_amount
                                break  # Only trigger the first (most aggressive) tier
                
                # Multi-tier selling (when price is above trend)
                elif price_vs_trend > 0:
                    for threshold_pct, allocation_pct in sell_tiers:
                        if price_vs_trend > threshold_pct / 100:
                            # Calculate sell amount based on allocation percentage
                            sell_amount = total_btc * (allocation_pct / 100)
                            if sell_amount > 0:
                                total_btc -= sell_amount
                                cash_reserve += sell_amount * current_price
                                break  # Only trigger the first (most aggressive) tier
            
            final_value = total_btc * prices[-1] + cash_reserve
            final_values.append(final_value)
        
        return np.array(final_values)
    
    def run_strategy_search(self):
        """Run search for best multi-tier strategies with interest"""
        print("RUNNING MULTI-TIER STRATEGY SEARCH WITH 5% INTEREST...")
        print("=" * 60)
        
        # Define parameter grids - now testing higher reserve percentages
        reserve_pcts = [10, 20, 30, 40, 50, 60, 70, 80]  # Extended range
        
        # Define different buy tier configurations
        buy_tier_configs = [
            [(10, 20), (20, 40), (30, 60)],  # Conservative: small buys
            [(15, 30), (25, 50), (35, 70)],  # Moderate
            [(10, 40), (20, 60), (30, 80)],  # Aggressive: bigger buys
            [(5, 15), (15, 30), (25, 45), (35, 60)],  # Fine-grained
            [(20, 50), (40, 100)],  # Simple 2-tier
        ]
        
        # Define different sell tier configurations
        sell_tier_configs = [
            [(20, 10), (40, 20), (60, 30)],  # Conservative: small sells
            [(30, 15), (50, 25), (70, 35)],  # Moderate
            [(20, 20), (40, 40), (60, 60)],  # Aggressive: bigger sells
            [(15, 10), (30, 20), (45, 30), (60, 40)],  # Fine-grained
            [(25, 25), (50, 50)],  # Simple 2-tier
        ]
        
        total_combinations = len(reserve_pcts) * len(buy_tier_configs) * len(sell_tier_configs)
        print(f"Testing {total_combinations} strategy combinations...")
        
        results = []
        
        # First, test traditional DCA as baseline
        print("\nTesting baseline DCA strategy...")
        dca_results = self.simulate_dca_strategy(self.monthly_amount, self.years)
        dca_stats = self.calculate_detailed_stats(dca_results)
        
        print(f"DCA Baseline Results:")
        print(f"  Median: ${dca_stats['median']:,.0f}")
        print(f"  Mean: ${dca_stats['mean']:,.0f}")
        print(f"  CAGR: {dca_stats['cagr']:.2f}%")
        print()
        
        # Grid search with progress bar
        pbar = tqdm(total=total_combinations, desc="Strategy Search")
        
        for reserve_pct in reserve_pcts:
            for buy_config in buy_tier_configs:
                for sell_config in sell_tier_configs:
                    strategy_results = self.simulate_multi_tier_interest_strategy(
                        reserve_pct, buy_config, sell_config, self.years
                    )
                    
                    stats = self.calculate_detailed_stats(strategy_results)
                    
                    # Calculate performance vs DCA
                    median_improvement = (stats['median'] - dca_stats['median']) / dca_stats['median'] * 100
                    mean_improvement = (stats['mean'] - dca_stats['mean']) / dca_stats['mean'] * 100
                    cagr_improvement = stats['cagr'] - dca_stats['cagr']
                    
                    # Calculate Sharpe-like ratio
                    sharpe_ratio = (stats['mean'] - dca_stats['mean']) / stats['std'] if stats['std'] > 0 else 0
                    
                    results.append({
                        'reserve_pct': reserve_pct,
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
                        'win_rate': np.mean(strategy_results > dca_stats['median']) * 100,
                        'beat_dca_rate': np.mean(strategy_results > dca_results) * 100
                    })
                    
                    pbar.update(1)
        
        pbar.close()
        
        # Convert to DataFrame and sort by performance
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('median_improvement', ascending=False)
        
        return results_df, dca_stats, dca_results
    
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
    
    def analyze_results(self, results_df, dca_stats, dca_results):
        """Analyze and display comprehensive results"""
        print("\nTOP 10 MULTI-TIER STRATEGIES WITH 5% INTEREST (by median improvement):")
        print("=" * 80)
        
        top_10 = results_df.head(10)
        
        for i, row in top_10.iterrows():
            print(f"\n#{len(top_10) - len(top_10) + list(top_10.index).index(i) + 1}. Reserve: {row['reserve_pct']:.0f}%")
            print(f"    Buy Tiers: {row['buy_tiers']}")
            print(f"    Sell Tiers: {row['sell_tiers']}")
            print(f"    Median: ${row['median_value']:,.0f} ({row['median_improvement']:+.1f}% vs DCA)")
            print(f"    Mean: ${row['mean_value']:,.0f} ({row['mean_improvement']:+.1f}% vs DCA)")
            print(f"    CAGR: {row['cagr']:.2f}% ({row['cagr_improvement']:+.2f}% vs DCA)")
            print(f"    Win Rate: {row['win_rate']:.1f}% | Beat DCA: {row['beat_dca_rate']:.1f}%")
            print(f"    Sharpe Ratio: {row['sharpe_ratio']:.3f}")
        
        # Best strategy detailed analysis
        best_strategy = results_df.iloc[0]
        
        print(f"\n🏆 BEST MULTI-TIER STRATEGY WITH 5% INTEREST:")
        print("=" * 60)
        print(f"Reserve: {best_strategy['reserve_pct']:.0f}% of monthly amount")
        print(f"Buy Tiers: {best_strategy['buy_tiers']}")
        print(f"Sell Tiers: {best_strategy['sell_tiers']}")
        print(f"Median improvement: {best_strategy['median_improvement']:+.1f}%")
        print(f"CAGR improvement: {best_strategy['cagr_improvement']:+.2f}%")
        print(f"Win rate: {best_strategy['win_rate']:.1f}%")
        
        # Compare 0% vs 5% interest impact
        print(f"\n💰 IMPACT OF 5% INTEREST ON RESERVES:")
        print("=" * 50)
        
        # Calculate theoretical benefit of 5% interest
        monthly_reserve = self.monthly_amount * (best_strategy['reserve_pct'] / 100)
        months = self.years * 12
        
        # Compound interest calculation for reserves
        total_interest_earned = 0
        for month in range(1, months + 1):
            # Interest on accumulated reserves
            accumulated_reserves = monthly_reserve * month
            monthly_interest = accumulated_reserves * (self.interest_rate / 12)
            total_interest_earned += monthly_interest
        
        print(f"Monthly reserve amount: ${monthly_reserve:,.0f}")
        print(f"Theoretical interest earned over {self.years} years: ${total_interest_earned:,.0f}")
        print(f"Interest as % of total investment: {total_interest_earned/self.total_investment*100:.1f}%")
        
        # Comprehensive comparison
        print(f"\n📊 COMPREHENSIVE PERFORMANCE COMPARISON:")
        print("=" * 60)
        
        # DCA Strategy
        print(f"📈 TRADITIONAL DCA STRATEGY:")
        print(f"   Median Return: ${dca_stats['median']:,.0f}")
        print(f"   Mean Return: ${dca_stats['mean']:,.0f}")
        print(f"   CAGR: {dca_stats['cagr']:.2f}%")
        print(f"   Standard Deviation: ${dca_stats['std']:,.0f}")
        print(f"   25th Percentile: ${dca_stats['p25']:,.0f}")
        print(f"   75th Percentile: ${dca_stats['p75']:,.0f}")
        print(f"   Best Case: ${dca_stats['max']:,.0f}")
        print(f"   Worst Case: ${dca_stats['min']:,.0f}")
        
        # Best Multi-Tier Strategy with Interest
        print(f"\n🚀 BEST MULTI-TIER STRATEGY WITH 5% INTEREST:")
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
        print(f"   Additional Return: ${best_strategy['median_value'] - dca_stats['median']:,.0f}")
        print(f"   Beat DCA Rate: {best_strategy['beat_dca_rate']:.1f}%")
        
        # Analyze reserve percentage distribution
        print(f"\n📈 RESERVE PERCENTAGE ANALYSIS WITH 5% INTEREST:")
        print("=" * 60)
        
        reserve_analysis = results_df.groupby('reserve_pct')['median_improvement'].agg(['mean', 'max', 'count']).round(2)
        print(reserve_analysis)
        
        # Save results
        results_df.to_csv('Results/Bitcoin/bitcoin_multi_tier_interest_strategy_results.csv', index=False)
        
        # Summary statistics
        print(f"\n📋 SUMMARY STATISTICS:")
        print(f"   Strategies tested: {len(results_df)}")
        print(f"   Strategies beating DCA: {len(results_df[results_df['median_improvement'] > 0])}")
        print(f"   Best improvement: {results_df['median_improvement'].max():+.1f}%")
        print(f"   Worst performance: {results_df['median_improvement'].min():+.1f}%")
        print(f"   Average improvement: {results_df['median_improvement'].mean():+.1f}%")
        
        return best_strategy

def main():
    strategy = BitcoinMultiTierInterestStrategy()
    results_df, dca_stats, dca_results = strategy.run_strategy_search()
    best_strategy = strategy.analyze_results(results_df, dca_stats, dca_results)
    
    print(f"\n📁 Results saved to: Results/Bitcoin/bitcoin_multi_tier_interest_strategy_results.csv")
    
    return best_strategy

if __name__ == "__main__":
    best_strategy = main() 
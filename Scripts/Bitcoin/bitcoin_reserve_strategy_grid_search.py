import numpy as np
import pandas as pd
from itertools import product
import matplotlib.pyplot as plt
from tqdm import tqdm

class BitcoinReserveStrategyGridSearch:
    def __init__(self):
        # Bitcoin Growth Model (94% R² formula)
        self.a = 1.6329135221917355
        self.b = -9.328646304661454
        
        # Strategy parameters
        self.total_investment = 100000  # $100k total to invest
        self.years = 5  # 5-year investment horizon
        self.monthly_amount = 1000  # $1k per month available
        self.n_simulations = 500  # Monte Carlo simulations per strategy
        
        print("BITCOIN RESERVE STRATEGY GRID SEARCH")
        print("=" * 60)
        print(f"Using Bitcoin Growth Model: log10(price) = {self.a:.3f} * ln(day) + {self.b:.3f}")
        print(f"Using Bitcoin Volatility Model: log10(vol) = -0.364 * log10(years) + 0.103")
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
        # Actual fitted coefficients from our volatility model
        a = -0.36442287700521414
        b = 0.1028262655650134
        
        # Ensure years > 0 to avoid log(0)
        years = max(years, 0.01)
        
        # Calculate volatility using our fitted formula
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
                if total_spent + monthly_amount <= self.total_investment:
                    btc_bought = monthly_amount / prices[step]
                    total_btc += btc_bought
                    total_spent += monthly_amount
                else:
                    break
            
            final_value = total_btc * prices[-1]
            final_values.append(final_value)
        
        return np.array(final_values)
    
    def simulate_reserve_strategy(self, reserve_pct, buy_threshold, sell_threshold, years):
        """
        Simulate reserve-based strategy
        
        Parameters:
        - reserve_pct: % of monthly amount to keep as reserve (0-50%)
        - buy_threshold: % below trend to trigger buying (5-30%)
        - sell_threshold: % above trend to trigger selling (20-100%)
        """
        steps_per_year = 12
        total_steps = int(years * steps_per_year)
        
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
                
                # Reserve strategy decisions
                if price_vs_trend < -buy_threshold / 100:  # Price below trend - BUY
                    if cash_reserve > 0:
                        # Use reserve to buy more
                        buy_amount = min(cash_reserve, monthly_cash)  # Limit to monthly amount
                        btc_bought = buy_amount / current_price
                        total_btc += btc_bought
                        cash_reserve -= buy_amount
                        total_spent += buy_amount
                
                elif price_vs_trend > sell_threshold / 100:  # Price above trend - SELL
                    if total_btc > 0:
                        # Sell some BTC to rebuild reserve
                        sell_amount = min(total_btc * 0.1, monthly_cash / current_price)  # Sell max 10% or monthly equivalent
                        btc_sold = sell_amount
                        total_btc -= btc_sold
                        cash_reserve += btc_sold * current_price
            
            final_value = total_btc * prices[-1] + cash_reserve
            final_values.append(final_value)
        
        return np.array(final_values)
    
    def run_grid_search(self):
        """Run comprehensive grid search"""
        print("RUNNING GRID SEARCH...")
        print("=" * 40)
        
        # Define parameter grids
        reserve_pcts = [0, 10, 20, 30, 40, 50]  # % of monthly amount to reserve
        buy_thresholds = [5, 10, 15, 20, 25, 30]  # % below trend to buy
        sell_thresholds = [20, 40, 60, 80, 100]  # % above trend to sell
        
        total_combinations = len(reserve_pcts) * len(buy_thresholds) * len(sell_thresholds)
        print(f"Testing {total_combinations} strategy combinations...")
        
        results = []
        
        # First, test traditional DCA as baseline
        print("\nTesting baseline DCA strategy...")
        dca_results = self.simulate_dca_strategy(self.monthly_amount, self.years)
        dca_median = np.median(dca_results)
        dca_mean = np.mean(dca_results)
        dca_std = np.std(dca_results)
        
        print(f"DCA Baseline Results:")
        print(f"  Median: ${dca_median:,.0f}")
        print(f"  Mean: ${dca_mean:,.0f}")
        print(f"  Std Dev: ${dca_std:,.0f}")
        print()
        
        # Grid search with progress bar
        pbar = tqdm(total=total_combinations, desc="Grid Search Progress")
        
        for reserve_pct, buy_threshold, sell_threshold in product(reserve_pcts, buy_thresholds, sell_thresholds):
            if reserve_pct == 0 and buy_threshold == 5 and sell_threshold == 20:
                # Skip redundant case (this is basically DCA)
                pbar.update(1)
                continue
            
            strategy_results = self.simulate_reserve_strategy(
                reserve_pct, buy_threshold, sell_threshold, self.years
            )
            
            median_result = np.median(strategy_results)
            mean_result = np.mean(strategy_results)
            std_result = np.std(strategy_results)
            
            # Calculate performance vs DCA
            median_improvement = (median_result - dca_median) / dca_median * 100
            mean_improvement = (mean_result - dca_mean) / dca_mean * 100
            
            # Calculate Sharpe-like ratio (excess return / volatility)
            sharpe_ratio = (mean_result - dca_mean) / std_result if std_result > 0 else 0
            
            results.append({
                'reserve_pct': reserve_pct,
                'buy_threshold': buy_threshold,
                'sell_threshold': sell_threshold,
                'median_value': median_result,
                'mean_value': mean_result,
                'std_value': std_result,
                'median_improvement': median_improvement,
                'mean_improvement': mean_improvement,
                'sharpe_ratio': sharpe_ratio,
                'win_rate': np.mean(strategy_results > dca_median) * 100
            })
            
            pbar.update(1)
        
        pbar.close()
        
        # Convert to DataFrame and sort by performance
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('median_improvement', ascending=False)
        
        return results_df, dca_median, dca_mean, dca_std
    
    def analyze_results(self, results_df, dca_median, dca_mean, dca_std):
        """Analyze and display results"""
        print("\nTOP 10 STRATEGIES (by median improvement):")
        print("=" * 80)
        
        top_10 = results_df.head(10)
        
        for i, row in top_10.iterrows():
            print(f"\n#{len(top_10) - len(top_10) + list(top_10.index).index(i) + 1}. Reserve: {row['reserve_pct']:.0f}%, Buy: -{row['buy_threshold']:.0f}%, Sell: +{row['sell_threshold']:.0f}%")
            print(f"    Median Value: ${row['median_value']:,.0f} ({row['median_improvement']:+.1f}% vs DCA)")
            print(f"    Mean Value: ${row['mean_value']:,.0f} ({row['mean_improvement']:+.1f}% vs DCA)")
            print(f"    Win Rate: {row['win_rate']:.1f}%")
            print(f"    Sharpe Ratio: {row['sharpe_ratio']:.3f}")
        
        # Best strategy analysis
        best_strategy = results_df.iloc[0]
        print(f"\n🏆 BEST STRATEGY:")
        print(f"   Reserve: {best_strategy['reserve_pct']:.0f}% of monthly amount")
        print(f"   Buy when: {best_strategy['buy_threshold']:.0f}% below trend")
        print(f"   Sell when: {best_strategy['sell_threshold']:.0f}% above trend")
        print(f"   Median improvement: {best_strategy['median_improvement']:+.1f}%")
        print(f"   Win rate: {best_strategy['win_rate']:.1f}%")
        
        # Save results
        results_df.to_csv('Results/Bitcoin/bitcoin_reserve_strategy_grid_results.csv', index=False)
        
        # Summary statistics
        print(f"\n📊 SUMMARY STATISTICS:")
        print(f"   Strategies tested: {len(results_df)}")
        print(f"   Strategies beating DCA: {len(results_df[results_df['median_improvement'] > 0])}")
        print(f"   Best improvement: {results_df['median_improvement'].max():+.1f}%")
        print(f"   Worst performance: {results_df['median_improvement'].min():+.1f}%")
        print(f"   Average improvement: {results_df['median_improvement'].mean():+.1f}%")
        
        return best_strategy

def main():
    grid_search = BitcoinReserveStrategyGridSearch()
    results_df, dca_median, dca_mean, dca_std = grid_search.run_grid_search()
    best_strategy = grid_search.analyze_results(results_df, dca_median, dca_mean, dca_std)
    
    print(f"\n📁 Results saved to: Results/Bitcoin/bitcoin_reserve_strategy_grid_results.csv")
    
    return best_strategy

if __name__ == "__main__":
    best_strategy = main() 
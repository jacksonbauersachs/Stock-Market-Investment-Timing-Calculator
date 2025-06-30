import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# === USER PARAMETERS ===
# Growth model: price = 10**(a * ln(years) + b)
growth_a = 1.633
growth_b = -9.32

# Volatility model (30d window): volatility = a / (1 + b * years) + c
vol_a = 31.78
vol_b = 22.19
vol_c = 0.308

# Simulation settings
years = 10
n_paths = 1000
steps_per_year = 365  # daily steps
initial_price = 67000  # or set to latest BTC price

# =======================

def growth_model(years):
    return 10 ** (growth_a * np.log(years) + growth_b)

def volatility_model(years):
    return vol_a / (1 + vol_b * years) + vol_c

# Time grid
total_steps = years * steps_per_year
all_years = np.linspace(1e-6, years, int(total_steps))  # avoid log(0)
dt = 1 / steps_per_year

# Precompute expected price and volatility for each step
expected_prices = growth_model(all_years)
vols = volatility_model(all_years)

# Simulate paths
print(f"Simulating {n_paths} paths for {years} years...")
paths = np.zeros((n_paths, len(all_years)))
paths[:, 0] = initial_price

for i in range(n_paths):
    if i % max(1, n_paths // 10) == 0:
        print(f"  Path {i+1}/{n_paths}")
    for t in range(1, len(all_years)):
        # Annualized volatility to per-step stddev
        step_vol = vols[t] * np.sqrt(dt)
        # Simulate log-return
        rand_return = np.random.normal(0, step_vol)
        paths[i, t] = paths[i, t-1] * np.exp(rand_return)

# Save to CSV
out_path = "Investment Strategy Analasis/Bitcoin Analysis/monte_carlo_paths.csv"
pd.DataFrame(paths, columns=[f"Year_{y:.2f}" for y in all_years]).to_csv(out_path, index=False)
print(f"Simulated paths saved to {out_path}")

# Plot a sample of the paths
plt.figure(figsize=(12, 6))
for i in range(min(50, n_paths)):
    plt.plot(all_years, paths[i], alpha=0.2, color='blue')
plt.plot(all_years, expected_prices * initial_price / expected_prices[0], color='red', linewidth=2, label='Growth Model')
plt.xlabel('Years')
plt.ylabel('BTC Price')
plt.title('Monte Carlo Simulated Bitcoin Price Paths')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Print summary statistics
final_prices = paths[:, -1]
print(f"\nSummary after {years} years:")
print(f"  Mean final price: ${final_prices.mean():,.2f}")
print(f"  Median final price: ${np.median(final_prices):,.2f}")
print(f"  10th percentile: ${np.percentile(final_prices, 10):,.2f}")
print(f"  90th percentile: ${np.percentile(final_prices, 90):,.2f}")

class BitcoinMonteCarloSimulator:
    def __init__(self, data_path, growth_model_params):
        """Initialize Monte Carlo simulator with Bitcoin data and growth model"""
        self.data = pd.read_csv(data_path)
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        
        # Handle price column format
        if pd.api.types.is_string_dtype(self.data['Close/Last']):
            self.data['Close/Last'] = pd.to_numeric(self.data['Close/Last'].str.replace(',', ''), errors='coerce')
        else:
            self.data['Close/Last'] = pd.to_numeric(self.data['Close/Last'], errors='coerce')
        
        self.data = self.data.sort_values('Date').reset_index(drop=True)
        
        # Growth model parameters
        self.a = growth_model_params['a']
        self.b = growth_model_params['b']
        
        # Calculate historical statistics for simulation
        self._calculate_historical_stats()
        
    def _calculate_historical_stats(self):
        """Calculate historical statistics for Monte Carlo simulation"""
        # Calculate returns and volatility
        self.data['Returns'] = self.data['Close/Last'].pct_change()
        
        # Historical volatility (annualized)
        self.historical_volatility = self.data['Returns'].std() * np.sqrt(365)
        
        # Historical mean return (annualized)
        self.historical_mean_return = self.data['Returns'].mean() * 365
        
        # Calculate deviation from growth model
        self.data['Days'] = (self.data['Date'] - self.data['Date'].min()).dt.days
        self.data['Predicted_Value'] = 10**(self.a * np.log(self.data['Days'] + 1) + self.b)
        self.data['Deviation'] = (self.data['Close/Last'] - self.data['Predicted_Value']) / self.data['Predicted_Value']
        
        # Deviation statistics
        self.deviation_mean = self.data['Deviation'].mean()
        self.deviation_std = self.data['Deviation'].std()
        
        # Autocorrelation of returns (for mean reversion)
        self.return_autocorr = self.data['Returns'].autocorr()
        
        print(f"Historical Statistics:")
        print(f"  Annual Volatility: {self.historical_volatility:.2%}")
        print(f"  Annual Mean Return: {self.historical_mean_return:.2%}")
        print(f"  Deviation Mean: {self.deviation_mean:.2%}")
        print(f"  Deviation Std: {self.deviation_std:.2%}")
        print(f"  Return Autocorrelation: {self.return_autocorr:.3f}")
    
    def generate_price_scenarios(self, n_scenarios=1000, years_ahead=5):
        """Generate Monte Carlo price scenarios"""
        print(f"Generating {n_scenarios} price scenarios for {years_ahead} years...")
        
        # Daily frequency
        days_ahead = years_ahead * 365
        
        # Start from the last known price
        start_price = self.data['Close/Last'].iloc[-1]
        start_date = self.data['Date'].iloc[-1]
        start_days = self.data['Days'].iloc[-1]
        
        scenarios = []
        
        for scenario in range(n_scenarios):
            # Initialize price path
            prices = [start_price]
            dates = [start_date]
            days = [start_days]
            
            for day in range(1, days_ahead + 1):
                # Calculate expected price from growth model
                current_days = start_days + day
                expected_price = 10**(self.a * np.log(current_days + 1) + self.b)
                
                # Generate random deviation
                deviation = np.random.normal(self.deviation_mean, self.deviation_std)
                
                # Apply mean reversion to deviation
                if day > 1:
                    prev_deviation = (prices[-1] - 10**(self.a * np.log(days[-1] + 1) + self.b)) / 10**(self.a * np.log(days[-1] + 1) + self.b)
                    deviation = deviation * 0.9 + prev_deviation * 0.1  # Mean reversion
                
                # Calculate new price
                new_price = expected_price * (1 + deviation)
                
                # Add some random noise
                daily_return = np.random.normal(self.historical_mean_return/365, self.historical_volatility/np.sqrt(365))
                new_price *= (1 + daily_return)
                
                # Ensure price stays positive
                new_price = max(new_price, 100)  # Minimum $100
                
                prices.append(new_price)
                dates.append(start_date + pd.Timedelta(days=day))
                days.append(current_days)
            
            scenarios.append({
                'prices': prices,
                'dates': dates,
                'days': days
            })
        
        return scenarios
    
    def backtest_strategy_on_scenario(self, scenario, strategy_params):
        """Backtest trading strategy on a single price scenario"""
        prices = scenario['prices']
        dates = scenario['dates']
        days = scenario['days']
        
        # Initialize portfolio
        cash_reserve = 0
        btc_owned = 0
        total_invested = 0
        cash_from_sales = 0
        
        # Strategy parameters
        reserve_pct = strategy_params['reserve_pct']
        buy_threshold = strategy_params['buy_threshold']
        sell_threshold = strategy_params['sell_threshold']
        
        for i in range(len(prices)):
            current_price = prices[i]
            current_days = days[i]
            
            # Calculate predicted value and deviation
            predicted_value = 10**(self.a * np.log(current_days + 1) + self.b)
            deviation = (current_price - predicted_value) / predicted_value
            
            # SELL LOGIC
            if deviation > sell_threshold and btc_owned > 0:
                sell_pct = min(0.3, (deviation - sell_threshold) * 1.5)
                sell_amount = btc_owned * sell_pct
                cash_from_sales += sell_amount * current_price
                btc_owned -= sell_amount
            
            # BUY LOGIC
            if deviation < buy_threshold and cash_reserve > 0:
                buy_amount = min(cash_reserve, cash_reserve * 0.5)
                btc_bought = buy_amount / current_price
                btc_owned += btc_bought
                cash_reserve -= buy_amount
            
            # Regular investment (monthly)
            if i % 30 == 0:  # Monthly investment
                regular_investment = 1000 * (1 - reserve_pct)
                btc_owned += regular_investment / current_price
                total_invested += regular_investment
                cash_reserve += 1000 * reserve_pct
        
        # Final portfolio value
        final_value = (btc_owned * prices[-1]) + cash_from_sales + cash_reserve
        
        return {
            'final_value': final_value,
            'total_invested': total_invested,
            'profit': final_value - total_invested,
            'roi': (final_value - total_invested) / total_invested if total_invested > 0 else 0,
            'btc_owned': btc_owned,
            'cash_reserve': cash_reserve,
            'cash_from_sales': cash_from_sales
        }
    
    def run_monte_carlo_analysis(self, strategy_params, n_scenarios=1000, years_ahead=5):
        """Run comprehensive Monte Carlo analysis"""
        print(f"Running Monte Carlo analysis with {n_scenarios} scenarios...")
        
        # Generate price scenarios
        scenarios = self.generate_price_scenarios(n_scenarios, years_ahead)
        
        # Backtest strategy on each scenario
        results = []
        for i, scenario in enumerate(scenarios):
            if i % 100 == 0:
                print(f"  Processing scenario {i}/{n_scenarios}")
            
            result = self.backtest_strategy_on_scenario(scenario, strategy_params)
            results.append(result)
        
        # Calculate statistics
        rois = [r['roi'] for r in results]
        profits = [r['profit'] for r in results]
        final_values = [r['final_value'] for r in results]
        
        analysis = {
            'mean_roi': np.mean(rois),
            'median_roi': np.median(rois),
            'std_roi': np.std(rois),
            'min_roi': np.min(rois),
            'max_roi': np.max(rois),
            'roi_5th_percentile': np.percentile(rois, 5),
            'roi_95th_percentile': np.percentile(rois, 95),
            'mean_profit': np.mean(profits),
            'median_profit': np.median(profits),
            'std_profit': np.std(profits),
            'min_profit': np.min(profits),
            'max_profit': np.max(profits),
            'profit_5th_percentile': np.percentile(profits, 5),
            'profit_95th_percentile': np.percentile(profits, 95),
            'success_rate': sum(1 for roi in rois if roi > 0) / len(rois),
            'scenarios': scenarios,
            'results': results
        }
        
        return analysis
    
    def plot_monte_carlo_results(self, analysis, strategy_params):
        """Plot Monte Carlo analysis results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        rois = [r['roi'] for r in analysis['results']]
        profits = [r['profit'] for r in analysis['results']]
        
        # Plot 1: ROI Distribution
        axes[0, 0].hist(rois, bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].axvline(analysis['mean_roi'], color='red', linestyle='--', label=f'Mean: {analysis["mean_roi"]:.2%}')
        axes[0, 0].axvline(analysis['median_roi'], color='green', linestyle='--', label=f'Median: {analysis["median_roi"]:.2%}')
        axes[0, 0].set_title('ROI Distribution')
        axes[0, 0].set_xlabel('ROI')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Profit Distribution
        axes[0, 1].hist(profits, bins=50, alpha=0.7, color='green', edgecolor='black')
        axes[0, 1].axvline(analysis['mean_profit'], color='red', linestyle='--', label=f'Mean: ${analysis["mean_profit"]:,.0f}')
        axes[0, 1].axvline(analysis['median_profit'], color='blue', linestyle='--', label=f'Median: ${analysis["median_profit"]:,.0f}')
        axes[0, 1].set_title('Profit Distribution')
        axes[0, 1].set_xlabel('Profit ($)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Sample Price Scenarios
        axes[1, 0].plot(self.data['Date'], self.data['Close/Last'], label='Historical', color='black', linewidth=2)
        
        # Plot 10 random scenarios
        for i in np.random.choice(len(analysis['scenarios']), 10, replace=False):
            scenario = analysis['scenarios'][i]
            axes[1, 0].plot(scenario['dates'], scenario['prices'], alpha=0.3, color='blue')
        
        axes[1, 0].set_title('Sample Price Scenarios')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Price ($)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Strategy Performance Summary
        summary_text = f"""
Strategy Performance Summary:
---------------------------
Mean ROI: {analysis['mean_roi']:.2%}
Median ROI: {analysis['median_roi']:.2%}
ROI Std Dev: {analysis['std_roi']:.2%}
5th Percentile ROI: {analysis['roi_5th_percentile']:.2%}
95th Percentile ROI: {analysis['roi_95th_percentile']:.2%}
Success Rate: {analysis['success_rate']:.1%}

Mean Profit: ${analysis['mean_profit']:,.0f}
Median Profit: ${analysis['median_profit']:,.0f}
Min Profit: ${analysis['min_profit']:,.0f}
Max Profit: ${analysis['max_profit']:,.0f}
        """
        
        axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes, 
                       fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[1, 1].set_title('Strategy Statistics')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def compare_strategies(self, strategies, n_scenarios=500, years_ahead=5):
        """Compare multiple trading strategies using Monte Carlo"""
        print("Comparing multiple strategies...")
        
        comparison_results = {}
        
        for strategy_name, strategy_params in strategies.items():
            print(f"Testing strategy: {strategy_name}")
            analysis = self.run_monte_carlo_analysis(strategy_params, n_scenarios, years_ahead)
            comparison_results[strategy_name] = analysis
        
        # Create comparison plot
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # ROI comparison
        strategy_names = list(comparison_results.keys())
        mean_rois = [comparison_results[name]['mean_roi'] for name in strategy_names]
        roi_stds = [comparison_results[name]['std_roi'] for name in strategy_names]
        
        x_pos = np.arange(len(strategy_names))
        axes[0].bar(x_pos, mean_rois, yerr=roi_stds, capsize=5, alpha=0.7)
        axes[0].set_title('Strategy ROI Comparison')
        axes[0].set_xlabel('Strategy')
        axes[0].set_ylabel('Mean ROI')
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(strategy_names, rotation=45)
        axes[0].grid(True, alpha=0.3)
        
        # Success rate comparison
        success_rates = [comparison_results[name]['success_rate'] for name in strategy_names]
        axes[1].bar(x_pos, success_rates, alpha=0.7, color='green')
        axes[1].set_title('Strategy Success Rate')
        axes[1].set_xlabel('Strategy')
        axes[1].set_ylabel('Success Rate')
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(strategy_names, rotation=45)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return comparison_results


def main():
    """Main execution function"""
    # Initialize simulator
    data_path = "../../Data Sets/Bitcoin Data/Bitcoin_Cleaned_Data.csv"
    growth_params = {'a': 1.633, 'b': -9.32}
    
    simulator = BitcoinMonteCarloSimulator(data_path, growth_params)
    
    # Define strategies to test
    strategies = {
        'Conservative': {
            'reserve_pct': 0.3,
            'buy_threshold': -0.2,
            'sell_threshold': 0.2
        },
        'Moderate': {
            'reserve_pct': 0.2,
            'buy_threshold': -0.15,
            'sell_threshold': 0.15
        },
        'Aggressive': {
            'reserve_pct': 0.1,
            'buy_threshold': -0.1,
            'sell_threshold': 0.1
        },
        'Buy and Hold': {
            'reserve_pct': 0.0,
            'buy_threshold': -1.0,  # Never buy from reserves
            'sell_threshold': 2.0   # Never sell
        }
    }
    
    # Run comparison
    comparison_results = simulator.compare_strategies(strategies, n_scenarios=500, years_ahead=5)
    
    # Print detailed results
    print("\n=== Strategy Comparison Results ===")
    for strategy_name, results in comparison_results.items():
        print(f"\n{strategy_name}:")
        print(f"  Mean ROI: {results['mean_roi']:.2%}")
        print(f"  Median ROI: {results['median_roi']:.2%}")
        print(f"  ROI Std Dev: {results['std_roi']:.2%}")
        print(f"  Success Rate: {results['success_rate']:.1%}")
        print(f"  Mean Profit: ${results['mean_profit']:,.0f}")
        print(f"  5th Percentile ROI: {results['roi_5th_percentile']:.2%}")
        print(f"  95th Percentile ROI: {results['roi_95th_percentile']:.2%}")


if __name__ == "__main__":
    main() 
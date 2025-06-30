import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class MaturityAdjustedBitcoinTrader:
    def __init__(self, data_path):
        """Initialize with Bitcoin data and maturity-adjusted volatility modeling"""
        self.data = pd.read_csv(data_path)
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        
        # Handle price column format
        if pd.api.types.is_string_dtype(self.data['Close/Last']):
            self.data['Close/Last'] = pd.to_numeric(self.data['Close/Last'].str.replace(',', ''), errors='coerce')
        else:
            self.data['Close/Last'] = pd.to_numeric(self.data['Close/Last'], errors='coerce')
        
        self.data = self.data.sort_values('Date').reset_index(drop=True)
        
        # Growth model parameters (from user's analysis)
        self.a = 1.633
        self.b = -9.32
        
        # Calculate days since first data point
        self.data['Days'] = (self.data['Date'] - self.data['Date'].min()).dt.days
        
        # Calculate predicted fair value using growth model
        self.data['Predicted_Value'] = 10**(self.a * np.log(self.data['Days'] + 1) + self.b)
        
        # Calculate maturity-adjusted variance metrics
        self._calculate_maturity_adjusted_metrics()
        
    def _calculate_maturity_adjusted_metrics(self):
        """Calculate variance metrics that account for Bitcoin's maturing volatility"""
        # Calculate returns
        self.data['Returns'] = self.data['Close/Last'].pct_change()
        
        # Calculate rolling volatility with different windows to show maturity effect
        self.data['Volatility_30d'] = self.data['Returns'].rolling(30).std() * np.sqrt(365)
        self.data['Volatility_90d'] = self.data['Returns'].rolling(90).std() * np.sqrt(365)
        self.data['Volatility_365d'] = self.data['Returns'].rolling(365).std() * np.sqrt(365)
        
        # Calculate maturity-adjusted volatility (decreasing over time)
        self.data['Years_Since_Start'] = self.data['Days'] / 365.25
        
        # Model volatility decay: volatility decreases as Bitcoin matures
        # Early Bitcoin (2010-2015): ~150% annual volatility
        # Recent Bitcoin (2020-2025): ~60% annual volatility
        # We model this as: volatility = base_volatility * (1 + maturity_decay * years)
        
        base_volatility = 0.60  # Recent volatility level
        maturity_decay = -0.05  # Volatility decreases by ~5% per year of maturity
        
        self.data['Maturity_Adjusted_Volatility'] = base_volatility * (1 + maturity_decay * self.data['Years_Since_Start'])
        
        # Ensure volatility doesn't go below a reasonable minimum
        self.data['Maturity_Adjusted_Volatility'] = np.maximum(self.data['Maturity_Adjusted_Volatility'], 0.30)
        
        # Price deviation from growth model
        self.data['Deviation_From_Model'] = (self.data['Close/Last'] - self.data['Predicted_Value']) / self.data['Predicted_Value']
        
        # Cap deviations to reasonable range
        self.data['Deviation_From_Model'] = np.clip(self.data['Deviation_From_Model'], -0.5, 1.0)
        
        # Calculate deviation statistics using recent data (last 2 years) for more relevant thresholds
        recent_data = self.data.tail(730)  # Last 2 years
        self.recent_deviation_mean = recent_data['Deviation_From_Model'].mean()
        self.recent_deviation_std = recent_data['Deviation_From_Model'].std()
        
        # Rolling deviation statistics (using recent data for relevance)
        self.data['Deviation_MA'] = self.data['Deviation_From_Model'].rolling(60).mean()
        self.data['Deviation_Std'] = self.data['Deviation_From_Model'].rolling(60).std()
        
        # ATH and ATL tracking
        self.data['ATH'] = self.data['Close/Last'].cummax()
        self.data['ATL'] = self.data['Close/Last'].cummin()
        
        # Print maturity analysis
        print("=== Bitcoin Maturity Analysis ===")
        print(f"Data spans {self.data['Years_Since_Start'].max():.1f} years")
        print(f"Early volatility (first year): {self.data['Volatility_30d'].iloc[365]:.1%}")
        print(f"Recent volatility (last year): {self.data['Volatility_30d'].iloc[-365:].mean():.1%}")
        print(f"Modeled future volatility: {self.data['Maturity_Adjusted_Volatility'].iloc[-1]:.1%}")
        print(f"Recent deviation mean: {self.recent_deviation_mean:.2%}")
        print(f"Recent deviation std: {self.recent_deviation_std:.2%}")
        
    def calculate_maturity_adjusted_thresholds(self):
        """Calculate thresholds based on recent, more relevant data"""
        # Use recent data (last 2 years) for threshold calculation
        recent_data = self.data.tail(730)['Deviation_From_Model'].dropna()
        
        buy_thresholds = {
            'conservative': np.percentile(recent_data, 25),
            'moderate': np.percentile(recent_data, 15),
            'aggressive': np.percentile(recent_data, 5)
        }
        
        sell_thresholds = {
            'conservative': np.percentile(recent_data, 75),
            'moderate': np.percentile(recent_data, 85),
            'aggressive': np.percentile(recent_data, 95)
        }
        
        return buy_thresholds, sell_thresholds
    
    def calculate_maturity_adjusted_position_size(self, current_deviation, current_volatility, base_investment=1000):
        """Calculate position size considering Bitcoin's maturity"""
        # Base position size
        position_size = base_investment
        
        # Adjust based on deviation from model
        if current_deviation < 0:  # Below model
            deviation_multiplier = 1 + abs(current_deviation) * 2
        else:
            deviation_multiplier = 1 - abs(current_deviation) * 0.5
        
        # Adjust based on maturity-adjusted volatility
        # Lower volatility (more mature Bitcoin) = larger positions
        volatility_multiplier = 1 / (1 + current_volatility)
        
        # Additional maturity bonus: more confident in mature Bitcoin
        maturity_bonus = 1.2  # 20% larger positions due to maturity
        
        final_position = position_size * deviation_multiplier * volatility_multiplier * maturity_bonus
        return max(0, final_position)
    
    def backtest_maturity_adjusted_strategy(self, strategy_params):
        """Backtest strategy with maturity-adjusted parameters"""
        # Reset trading variables
        cash_reserve = 0
        btc_owned = 0
        total_invested = 0
        cash_from_sales = 0
        trade_history = []
        
        # Strategy parameters
        reserve_pct = strategy_params['reserve_pct']
        buy_threshold = strategy_params['buy_threshold']
        sell_threshold = strategy_params['sell_threshold']
        
        for i, row in self.data.iterrows():
            if pd.isna(row['Deviation_From_Model']) or pd.isna(row['Maturity_Adjusted_Volatility']):
                continue
                
            current_price = row['Close/Last']
            current_deviation = row['Deviation_From_Model']
            current_volatility = row['Maturity_Adjusted_Volatility']
            
            # SELL LOGIC
            if current_deviation > sell_threshold and btc_owned > 0:
                # More conservative selling for mature Bitcoin
                sell_pct = min(0.25, (current_deviation - sell_threshold) * 1.2)
                sell_amount = btc_owned * sell_pct
                cash_from_sales += sell_amount * current_price
                btc_owned -= sell_amount
                
                trade_history.append({
                    'date': row['Date'],
                    'action': 'SELL',
                    'price': current_price,
                    'amount': sell_amount,
                    'deviation': current_deviation,
                    'volatility': current_volatility
                })
            
            # BUY LOGIC
            if current_deviation < buy_threshold and cash_reserve > 0:
                # Calculate position size with maturity adjustment
                position_size = self.calculate_maturity_adjusted_position_size(
                    current_deviation, 
                    current_volatility,
                    base_investment=cash_reserve
                )
                
                if position_size > 0:
                    btc_bought = position_size / current_price
                    btc_owned += btc_bought
                    cash_reserve -= position_size
                    
                    trade_history.append({
                        'date': row['Date'],
                        'action': 'BUY',
                        'price': current_price,
                        'amount': btc_bought,
                        'deviation': current_deviation,
                        'volatility': current_volatility
                    })
            
            # Regular investment (monthly)
            if i % 30 == 0:
                regular_investment = 1000 * (1 - reserve_pct)
                btc_owned += regular_investment / current_price
                total_invested += regular_investment
                cash_reserve += 1000 * reserve_pct
        
        # Calculate final portfolio value
        final_price = self.data.iloc[-1]['Close/Last']
        final_value = (btc_owned * final_price) + cash_from_sales + cash_reserve
        
        return {
            'final_value': final_value,
            'total_invested': total_invested,
            'profit': final_value - total_invested,
            'roi': (final_value - total_invested) / total_invested if total_invested > 0 else 0,
            'btc_owned': btc_owned,
            'cash_reserve': cash_reserve,
            'cash_from_sales': cash_from_sales,
            'trade_count': len(trade_history),
            'trade_history': trade_history
        }
    
    def generate_maturity_adjusted_scenarios(self, n_scenarios=500, years_ahead=5):
        """Generate Monte Carlo scenarios with maturity-adjusted volatility"""
        print(f"Generating {n_scenarios} maturity-adjusted scenarios...")
        
        # Start from current state
        start_price = self.data['Close/Last'].iloc[-1]
        start_date = self.data['Date'].iloc[-1]
        start_days = self.data['Days'].iloc[-1]
        start_years = start_days / 365.25
        
        scenarios = []
        
        for scenario in range(n_scenarios):
            prices = [start_price]
            dates = [start_date]
            days = [start_days]
            
            for day in range(1, years_ahead * 365 + 1):
                current_days = start_days + day
                current_years = current_days / 365.25
                
                # Calculate expected price from growth model
                expected_price = 10**(self.a * np.log(current_days + 1) + self.b)
                
                # Calculate maturity-adjusted volatility for this future date
                base_volatility = 0.60
                maturity_decay = -0.05
                future_volatility = base_volatility * (1 + maturity_decay * current_years)
                future_volatility = max(future_volatility, 0.30)  # Minimum 30%
                
                # Generate deviation with mean reversion
                if day == 1:
                    deviation = np.random.normal(self.recent_deviation_mean, self.recent_deviation_std)
                else:
                    prev_deviation = (prices[-1] - 10**(self.a * np.log(days[-1] + 1) + self.b)) / 10**(self.a * np.log(days[-1] + 1) + self.b)
                    deviation = prev_deviation * 0.9 + np.random.normal(self.recent_deviation_mean, self.recent_deviation_std) * 0.1
                
                # Calculate price with maturity-adjusted volatility
                model_price = expected_price * (1 + deviation)
                
                # Add daily noise with maturity-adjusted volatility
                daily_volatility = future_volatility / np.sqrt(365)
                daily_return = np.random.normal(0, daily_volatility)
                new_price = model_price * (1 + daily_return)
                
                # Ensure price stays positive
                new_price = max(new_price, 100)
                
                prices.append(new_price)
                dates.append(start_date + pd.Timedelta(days=day))
                days.append(current_days)
            
            scenarios.append({
                'prices': prices,
                'dates': dates,
                'days': days
            })
        
        return scenarios
    
    def run_maturity_adjusted_analysis(self, strategy_params, n_scenarios=500, years_ahead=5):
        """Run analysis with maturity-adjusted parameters"""
        print("Running maturity-adjusted Monte Carlo analysis...")
        
        # Generate scenarios
        scenarios = self.generate_maturity_adjusted_scenarios(n_scenarios, years_ahead)
        
        # Backtest on each scenario
        results = []
        for i, scenario in enumerate(scenarios):
            if i % 100 == 0:
                print(f"  Processing scenario {i}/{n_scenarios}")
            
            result = self.backtest_strategy_on_scenario(scenario, strategy_params)
            results.append(result)
        
        # Calculate statistics
        rois = [r['roi'] for r in results]
        profits = [r['profit'] for r in results]
        
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
    
    def backtest_strategy_on_scenario(self, scenario, strategy_params):
        """Backtest strategy on a single scenario"""
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
                sell_pct = min(0.25, (deviation - sell_threshold) * 1.2)
                sell_amount = btc_owned * sell_pct
                cash_from_sales += sell_amount * current_price
                btc_owned -= sell_amount
            
            # BUY LOGIC
            if deviation < buy_threshold and cash_reserve > 0:
                buy_amount = min(cash_reserve, cash_reserve * 0.6)  # More aggressive for mature Bitcoin
                btc_bought = buy_amount / current_price
                btc_owned += btc_bought
                cash_reserve -= buy_amount
            
            # Regular investment (monthly)
            if i % 30 == 0:
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
    
    def plot_maturity_analysis(self):
        """Plot Bitcoin's volatility maturation over time"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Historical vs Modeled Volatility
        axes[0, 0].plot(self.data['Date'], self.data['Volatility_30d'], label='30-day Historical', alpha=0.7)
        axes[0, 0].plot(self.data['Date'], self.data['Volatility_365d'], label='365-day Historical', alpha=0.7)
        axes[0, 0].plot(self.data['Date'], self.data['Maturity_Adjusted_Volatility'], label='Maturity-Adjusted Model', linewidth=2)
        axes[0, 0].set_title('Bitcoin Volatility Maturation')
        axes[0, 0].set_ylabel('Annualized Volatility')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Volatility by Year
        yearly_vol = self.data.groupby(self.data['Date'].dt.year)['Volatility_30d'].mean()
        axes[0, 1].bar(yearly_vol.index, yearly_vol.values, alpha=0.7)
        axes[0, 1].set_title('Average Annual Volatility by Year')
        axes[0, 1].set_xlabel('Year')
        axes[0, 1].set_ylabel('Average Volatility')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Price vs Growth Model
        axes[1, 0].plot(self.data['Date'], self.data['Close/Last'], label='Actual Price', alpha=0.7)
        axes[1, 0].plot(self.data['Date'], self.data['Predicted_Value'], label='Growth Model', linestyle='--')
        axes[1, 0].set_title('Bitcoin Price vs Growth Model')
        axes[1, 0].set_ylabel('Price (USD)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Deviation from Model (Recent Data)
        recent_data = self.data.tail(730)  # Last 2 years
        axes[1, 1].plot(recent_data['Date'], recent_data['Deviation_From_Model'], color='orange')
        axes[1, 1].axhline(y=self.recent_deviation_mean, color='red', linestyle='--', label=f'Mean: {self.recent_deviation_mean:.2%}')
        axes[1, 1].axhline(y=self.recent_deviation_mean + self.recent_deviation_std, color='gray', linestyle=':', label=f'+1 Std: {(self.recent_deviation_mean + self.recent_deviation_std):.2%}')
        axes[1, 1].axhline(y=self.recent_deviation_mean - self.recent_deviation_std, color='gray', linestyle=':', label=f'-1 Std: {(self.recent_deviation_mean - self.recent_deviation_std):.2%}')
        axes[1, 1].set_title('Recent Deviation from Growth Model (Last 2 Years)')
        axes[1, 1].set_ylabel('Deviation')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def main():
    """Main execution function"""
    # Initialize trader
    data_path = "../../Data Sets/Bitcoin Data/Bitcoin_Cleaned_Data.csv"
    trader = MaturityAdjustedBitcoinTrader(data_path)
    
    print("\n=== Maturity-Adjusted Bitcoin Trading Strategy ===")
    
    # Calculate maturity-adjusted thresholds
    buy_thresholds, sell_thresholds = trader.calculate_maturity_adjusted_thresholds()
    print(f"\nMaturity-Adjusted Thresholds (based on recent 2 years):")
    print(f"Buy thresholds: {buy_thresholds}")
    print(f"Sell thresholds: {sell_thresholds}")
    
    # Test strategy with moderate parameters
    strategy_params = {
        'reserve_pct': 0.3,
        'buy_threshold': buy_thresholds['moderate'],
        'sell_threshold': sell_thresholds['moderate']
    }
    
    # Run backtest
    result = trader.backtest_maturity_adjusted_strategy(strategy_params)
    
    print(f"\n=== Maturity-Adjusted Strategy Results ===")
    print(f"Reserve Percentage: {strategy_params['reserve_pct']:.0%}")
    print(f"Buy Threshold: {strategy_params['buy_threshold']:.1%}")
    print(f"Sell Threshold: {strategy_params['sell_threshold']:.1%}")
    print(f"Total Invested: ${result['total_invested']:,.2f}")
    print(f"Final Value: ${result['final_value']:,.2f}")
    print(f"Profit: ${result['profit']:,.2f}")
    print(f"ROI: {result['roi']:.2%}")
    print(f"Number of Trades: {result['trade_count']}")
    
    # Run Monte Carlo analysis
    print(f"\nRunning maturity-adjusted Monte Carlo analysis...")
    mc_analysis = trader.run_maturity_adjusted_analysis(strategy_params, n_scenarios=300, years_ahead=5)
    
    print(f"\n=== Monte Carlo Results (Maturity-Adjusted) ===")
    print(f"Mean ROI: {mc_analysis['mean_roi']:.2%}")
    print(f"Median ROI: {mc_analysis['median_roi']:.2%}")
    print(f"ROI Std Dev: {mc_analysis['std_roi']:.2%}")
    print(f"Success Rate: {mc_analysis['success_rate']:.1%}")
    print(f"5th Percentile ROI: {mc_analysis['roi_5th_percentile']:.2%}")
    print(f"95th Percentile ROI: {mc_analysis['roi_95th_percentile']:.2%}")
    
    # Plot maturity analysis
    trader.plot_maturity_analysis()


if __name__ == "__main__":
    main() 
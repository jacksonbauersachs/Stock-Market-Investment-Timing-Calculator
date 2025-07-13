import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class BitcoinVarianceTrader:
    def __init__(self, data_path):
        """Initialize with Bitcoin data and growth model parameters"""
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
        
        # Calculate variance metrics
        self._calculate_variance_metrics()
        
        # Initialize trading variables
        self.cash_reserve = 0
        self.btc_owned = 0
        self.total_invested = 0
        self.cash_from_sales = 0
        self.trade_history = []
        
    def _calculate_variance_metrics(self):
        """Calculate various variance and statistical metrics"""
        # Rolling volatility (30-day window)
        self.data['Returns'] = self.data['Close/Last'].pct_change()
        self.data['Volatility_30d'] = self.data['Returns'].rolling(30).std() * np.sqrt(365)
        
        # Price deviation from growth model
        self.data['Deviation_From_Model'] = (self.data['Close/Last'] - self.data['Predicted_Value']) / self.data['Predicted_Value']
        
        # Rolling deviation statistics
        self.data['Deviation_MA'] = self.data['Deviation_From_Model'].rolling(60).mean()
        self.data['Deviation_Std'] = self.data['Deviation_From_Model'].rolling(60).std()
        
        # Z-score of current deviation
        self.data['Deviation_ZScore'] = (self.data['Deviation_From_Model'] - self.data['Deviation_MA']) / self.data['Deviation_Std']
        
        # ATH and ATL tracking
        self.data['ATH'] = self.data['Close/Last'].cummax()
        self.data['ATL'] = self.data['Close/Last'].cummin()
        
        # Dip and peak percentages
        self.data['Dip_From_ATH'] = (self.data['ATH'] - self.data['Close/Last']) / self.data['ATH']
        self.data['Peak_From_ATH'] = (self.data['Close/Last'] - self.data['ATH']) / self.data['ATH']
        
    def calculate_optimal_thresholds(self):
        """Calculate optimal buy/sell thresholds based on variance statistics"""
        # Analyze historical variance patterns
        deviation_data = self.data['Deviation_From_Model'].dropna()
        
        # Find percentiles for buy/sell decisions
        buy_thresholds = {
            'conservative': np.percentile(deviation_data, 25),  # Buy when 25% below model
            'moderate': np.percentile(deviation_data, 15),      # Buy when 15% below model
            'aggressive': np.percentile(deviation_data, 5)      # Buy when 5% below model
        }
        
        sell_thresholds = {
            'conservative': np.percentile(deviation_data, 75),  # Sell when 25% above model
            'moderate': np.percentile(deviation_data, 85),      # Sell when 15% above model
            'aggressive': np.percentile(deviation_data, 95)     # Sell when 5% above model
        }
        
        return buy_thresholds, sell_thresholds
    
    def calculate_position_size(self, current_deviation, volatility, base_investment=1000):
        """Calculate position size based on deviation and volatility"""
        # Base position size
        position_size = base_investment
        
        # Adjust based on deviation from model (larger positions when further below model)
        if current_deviation < 0:  # Below model
            deviation_multiplier = 1 + abs(current_deviation) * 2  # Up to 3x for large deviations
        else:
            deviation_multiplier = 1 - abs(current_deviation) * 0.5  # Reduce for overvalued periods
        
        # Adjust based on volatility (larger positions in low volatility, smaller in high)
        volatility_multiplier = 1 / (1 + volatility)  # Inverse relationship
        
        final_position = position_size * deviation_multiplier * volatility_multiplier
        return max(0, final_position)  # Ensure non-negative
    
    def backtest_variance_strategy(self, strategy_params):
        """Backtest the variance-based trading strategy"""
        # Reset trading variables
        self.cash_reserve = 0
        self.btc_owned = 0
        self.total_invested = 0
        self.cash_from_sales = 0
        self.trade_history = []
        
        # Strategy parameters
        reserve_pct = strategy_params['reserve_pct']
        buy_threshold = strategy_params['buy_threshold']
        sell_threshold = strategy_params['sell_threshold']
        volatility_weight = strategy_params['volatility_weight']
        
        for i, row in self.data.iterrows():
            if pd.isna(row['Deviation_From_Model']) or pd.isna(row['Volatility_30d']):
                continue
                
            current_price = row['Close/Last']
            current_deviation = row['Deviation_From_Model']
            current_volatility = row['Volatility_30d']
            
            # SELL LOGIC
            if current_deviation > sell_threshold and self.btc_owned > 0:
                # Calculate sell amount based on deviation magnitude
                sell_pct = min(0.5, (current_deviation - sell_threshold) * 2)  # Up to 50% sell
                sell_amount = self.btc_owned * sell_pct
                self.cash_from_sales += sell_amount * current_price
                self.btc_owned -= sell_amount
                
                self.trade_history.append({
                    'date': row['Date'],
                    'action': 'SELL',
                    'price': current_price,
                    'amount': sell_amount,
                    'deviation': current_deviation,
                    'volatility': current_volatility
                })
            
            # BUY LOGIC
            if current_deviation < buy_threshold and self.cash_reserve > 0:
                # Calculate position size based on deviation and volatility
                position_size = self.calculate_position_size(
                    current_deviation, 
                    current_volatility * volatility_weight,
                    base_investment=self.cash_reserve
                )
                
                if position_size > 0:
                    btc_bought = position_size / current_price
                    self.btc_owned += btc_bought
                    self.cash_reserve -= position_size
                    
                    self.trade_history.append({
                        'date': row['Date'],
                        'action': 'BUY',
                        'price': current_price,
                        'amount': btc_bought,
                        'deviation': current_deviation,
                        'volatility': current_volatility
                    })
            
            # Regular investment
            regular_investment = 1000 * (1 - reserve_pct)
            self.btc_owned += regular_investment / current_price
            self.total_invested += regular_investment
            self.cash_reserve += 1000 * reserve_pct
        
        # Calculate final portfolio value
        final_price = self.data.iloc[-1]['Close/Last']
        final_value = (self.btc_owned * final_price) + self.cash_from_sales + self.cash_reserve
        
        return {
            'final_value': final_value,
            'total_invested': self.total_invested,
            'profit': final_value - self.total_invested,
            'roi': (final_value - self.total_invested) / self.total_invested,
            'btc_owned': self.btc_owned,
            'cash_reserve': self.cash_reserve,
            'cash_from_sales': self.cash_from_sales,
            'trade_count': len(self.trade_history)
        }
    
    def optimize_strategy(self):
        """Optimize strategy parameters using the growth model"""
        print("Optimizing variance trading strategy...")
        
        # Parameter ranges
        reserve_pcts = np.linspace(0.1, 0.7, 7)  # 10% to 70% reserves
        buy_thresholds = np.linspace(-0.3, -0.05, 6)  # -30% to -5% deviation
        sell_thresholds = np.linspace(0.05, 0.3, 6)   # +5% to +30% deviation
        volatility_weights = np.linspace(0.5, 2.0, 4)  # Volatility sensitivity
        
        best_result = {'roi': -np.inf}
        total_combinations = len(reserve_pcts) * len(buy_thresholds) * len(sell_thresholds) * len(volatility_weights)
        tested = 0
        
        for reserve_pct in reserve_pcts:
            for buy_thresh in buy_thresholds:
                for sell_thresh in sell_thresholds:
                    for vol_weight in volatility_weights:
                        if sell_thresh <= buy_thresh:  # Skip invalid combinations
                            continue
                            
                        strategy_params = {
                            'reserve_pct': reserve_pct,
                            'buy_threshold': buy_thresh,
                            'sell_threshold': sell_thresh,
                            'volatility_weight': vol_weight
                        }
                        
                        result = self.backtest_variance_strategy(strategy_params)
                        tested += 1
                        
                        if result['roi'] > best_result['roi']:
                            best_result = {**result, **strategy_params}
                            print(f"\nNew best strategy ({tested}/{total_combinations}):")
                            print(f" ROI: {result['roi']:.2%}")
                            print(f" Reserve: {reserve_pct:.0%}")
                            print(f" Buy threshold: {buy_thresh:.1%}")
                            print(f" Sell threshold: {sell_thresh:.1%}")
                            print(f" Volatility weight: {vol_weight:.1f}")
        
        return best_result
    
    def plot_strategy_analysis(self, strategy_params):
        """Plot comprehensive strategy analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Price vs Growth Model
        axes[0, 0].plot(self.data['Date'], self.data['Close/Last'], label='Actual Price', alpha=0.7)
        axes[0, 0].plot(self.data['Date'], self.data['Predicted_Value'], label='Growth Model', linestyle='--')
        axes[0, 0].set_title('Bitcoin Price vs Growth Model Prediction')
        axes[0, 0].set_ylabel('Price (USD)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Deviation from Model
        axes[0, 1].plot(self.data['Date'], self.data['Deviation_From_Model'], label='Deviation', color='orange')
        axes[0, 1].axhline(y=strategy_params['buy_threshold'], color='green', linestyle='--', label='Buy Threshold')
        axes[0, 1].axhline(y=strategy_params['sell_threshold'], color='red', linestyle='--', label='Sell Threshold')
        axes[0, 1].set_title('Deviation from Growth Model')
        axes[0, 1].set_ylabel('Deviation (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Volatility
        axes[1, 0].plot(self.data['Date'], self.data['Volatility_30d'], label='30-day Volatility', color='purple')
        axes[1, 0].set_title('Bitcoin Volatility (30-day rolling)')
        axes[1, 0].set_ylabel('Annualized Volatility')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Portfolio Value Over Time (if we have trade history)
        if self.trade_history:
            # This would require tracking portfolio value over time
            axes[1, 1].text(0.5, 0.5, 'Portfolio tracking\nwould be implemented\nin full backtest', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
        else:
            axes[1, 1].text(0.5, 0.5, 'Run backtest to see\nportfolio performance', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Portfolio Performance')
        
        plt.tight_layout()
        plt.show()
    
    def generate_trading_signals(self, current_date=None):
        """Generate current trading signals based on the strategy"""
        if current_date is None:
            current_date = self.data['Date'].max()
        
        current_data = self.data[self.data['Date'] <= current_date].iloc[-1]
        
        signals = {
            'current_price': current_data['Close/Last'],
            'predicted_value': current_data['Predicted_Value'],
            'deviation': current_data['Deviation_From_Model'],
            'volatility': current_data['Volatility_30d'],
            'recommendation': 'HOLD',
            'confidence': 0.5,
            'position_size': 0
        }
        
        # Calculate optimal thresholds
        buy_thresholds, sell_thresholds = self.calculate_optimal_thresholds()
        
        # Generate recommendation
        if current_data['Deviation_From_Model'] < buy_thresholds['moderate']:
            signals['recommendation'] = 'BUY'
            signals['confidence'] = abs(current_data['Deviation_From_Model']) / abs(buy_thresholds['aggressive'])
            signals['position_size'] = self.calculate_position_size(
                current_data['Deviation_From_Model'],
                current_data['Volatility_30d']
            )
        elif current_data['Deviation_From_Model'] > sell_thresholds['moderate']:
            signals['recommendation'] = 'SELL'
            signals['confidence'] = current_data['Deviation_From_Model'] / sell_thresholds['aggressive']
        
        return signals


def main():
    """Main execution function"""
    # Initialize trader
    data_path = "../../Data Sets/Bitcoin Data/Bitcoin_Cleaned_Data.csv"
    trader = BitcoinVarianceTrader(data_path)
    
    print("=== Bitcoin Variance Trading Strategy ===")
    print(f"Growth Model: 10^({trader.a}*ln(x)+{trader.b})")
    print(f"Data Range: {trader.data['Date'].min().date()} to {trader.data['Date'].max().date()}")
    print(f"Total Data Points: {len(trader.data)}")
    
    # Calculate and display variance statistics
    buy_thresholds, sell_thresholds = trader.calculate_optimal_thresholds()
    print(f"\nOptimal Thresholds (based on historical variance):")
    print(f"Buy thresholds: {buy_thresholds}")
    print(f"Sell thresholds: {sell_thresholds}")
    
    # Generate current trading signals
    current_signals = trader.generate_trading_signals()
    print(f"\nCurrent Trading Signals:")
    print(f"Current Price: ${current_signals['current_price']:,.2f}")
    print(f"Predicted Value: ${current_signals['predicted_value']:,.2f}")
    print(f"Deviation: {current_signals['deviation']:.2%}")
    print(f"Volatility: {current_signals['volatility']:.2%}")
    print(f"Recommendation: {current_signals['recommendation']}")
    print(f"Confidence: {current_signals['confidence']:.2%}")
    
    # Run optimization
    print(f"\nRunning strategy optimization...")
    optimal_result = trader.optimize_strategy()
    
    print(f"\n=== Optimal Strategy Results ===")
    print(f"Reserve Percentage: {optimal_result['reserve_pct']:.0%}")
    print(f"Buy Threshold: {optimal_result['buy_threshold']:.1%}")
    print(f"Sell Threshold: {optimal_result['sell_threshold']:.1%}")
    print(f"Volatility Weight: {optimal_result['volatility_weight']:.1f}")
    print(f"Total Invested: ${optimal_result['total_invested']:,.2f}")
    print(f"Final Value: ${optimal_result['final_value']:,.2f}")
    print(f"Profit: ${optimal_result['profit']:,.2f}")
    print(f"ROI: {optimal_result['roi']:.2%}")
    print(f"Number of Trades: {optimal_result['trade_count']}")
    
    # Plot analysis
    trader.plot_strategy_analysis(optimal_result)


if __name__ == "__main__":
    main() 
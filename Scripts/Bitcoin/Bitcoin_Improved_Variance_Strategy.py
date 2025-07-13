import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class ImprovedBitcoinVarianceTrader:
    def __init__(self, data_path):
        """Initialize with Bitcoin data and improved growth model"""
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
        
    def _calculate_variance_metrics(self):
        """Calculate various variance and statistical metrics"""
        # Rolling volatility (30-day window)
        self.data['Returns'] = self.data['Close/Last'].pct_change()
        self.data['Volatility_30d'] = self.data['Returns'].rolling(30).std() * np.sqrt(365)
        
        # Price deviation from growth model (capped to reasonable range)
        self.data['Deviation_From_Model'] = (self.data['Close/Last'] - self.data['Predicted_Value']) / self.data['Predicted_Value']
        
        # Cap deviations to reasonable range (-50% to +100%)
        self.data['Deviation_From_Model'] = np.clip(self.data['Deviation_From_Model'], -0.5, 1.0)
        
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
    
    def backtest_variance_strategy(self, strategy_params):
        """Backtest the variance-based trading strategy"""
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
            if pd.isna(row['Deviation_From_Model']) or pd.isna(row['Volatility_30d']):
                continue
                
            current_price = row['Close/Last']
            current_deviation = row['Deviation_From_Model']
            current_volatility = row['Volatility_30d']
            
            # SELL LOGIC
            if current_deviation > sell_threshold and btc_owned > 0:
                # Calculate sell amount based on deviation magnitude
                sell_pct = min(0.3, (current_deviation - sell_threshold) * 1.5)
                sell_amount = btc_owned * sell_pct
                cash_from_sales += sell_amount * current_price
                btc_owned -= sell_amount
                
                trade_history.append({
                    'date': row['Date'],
                    'action': 'SELL',
                    'price': current_price,
                    'amount': sell_amount,
                    'deviation': current_deviation
                })
            
            # BUY LOGIC
            if current_deviation < buy_threshold and cash_reserve > 0:
                # Calculate position size based on deviation magnitude
                position_size = min(cash_reserve, cash_reserve * 0.5)
                btc_bought = position_size / current_price
                btc_owned += btc_bought
                cash_reserve -= position_size
                
                trade_history.append({
                    'date': row['Date'],
                    'action': 'BUY',
                    'price': current_price,
                    'amount': btc_bought,
                    'deviation': current_deviation
                })
            
            # Regular investment (monthly)
            if i % 30 == 0:  # Monthly investment
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
    
    def optimize_strategy(self):
        """Optimize strategy parameters"""
        print("Optimizing variance trading strategy...")
        
        # Parameter ranges
        reserve_pcts = np.linspace(0.1, 0.7, 7)  # 10% to 70% reserves
        buy_thresholds = np.linspace(-0.3, -0.05, 6)  # -30% to -5% deviation
        sell_thresholds = np.linspace(0.05, 0.3, 6)   # +5% to +30% deviation
        
        best_result = {'roi': -np.inf}
        total_combinations = len(reserve_pcts) * len(buy_thresholds) * len(sell_thresholds)
        tested = 0
        
        for reserve_pct in reserve_pcts:
            for buy_thresh in buy_thresholds:
                for sell_thresh in sell_thresholds:
                    if sell_thresh <= buy_thresh:  # Skip invalid combinations
                        continue
                        
                    strategy_params = {
                        'reserve_pct': reserve_pct,
                        'buy_threshold': buy_thresh,
                        'sell_threshold': sell_thresh
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
        
        # Plot 4: Trading Signals
        if 'trade_history' in strategy_params:
            trade_history = strategy_params['trade_history']
            buy_trades = [t for t in trade_history if t['action'] == 'BUY']
            sell_trades = [t for t in trade_history if t['action'] == 'SELL']
            
            if buy_trades:
                buy_dates = [t['date'] for t in buy_trades]
                buy_prices = [t['price'] for t in buy_trades]
                axes[1, 1].scatter(buy_dates, buy_prices, color='green', marker='^', s=50, label='Buy Signals')
            
            if sell_trades:
                sell_dates = [t['date'] for t in sell_trades]
                sell_prices = [t['price'] for t in sell_trades]
                axes[1, 1].scatter(sell_dates, sell_prices, color='red', marker='v', s=50, label='Sell Signals')
            
            axes[1, 1].plot(self.data['Date'], self.data['Close/Last'], alpha=0.7, color='blue')
            axes[1, 1].set_title('Trading Signals')
            axes[1, 1].set_ylabel('Price (USD)')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'Run backtest to see\ntrading signals', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Trading Signals')
        
        plt.tight_layout()
        plt.show()
    
    def compare_with_buy_and_hold(self, strategy_params):
        """Compare strategy with simple buy and hold"""
        # Strategy results
        strategy_result = self.backtest_variance_strategy(strategy_params)
        
        # Buy and hold results
        buy_hold_invested = 0
        buy_hold_btc = 0
        
        for i, row in self.data.iterrows():
            if i % 30 == 0:  # Monthly investment
                buy_hold_invested += 1000
                buy_hold_btc += 1000 / row['Close/Last']
        
        buy_hold_final_value = buy_hold_btc * self.data.iloc[-1]['Close/Last']
        buy_hold_profit = buy_hold_final_value - buy_hold_invested
        buy_hold_roi = buy_hold_profit / buy_hold_invested
        
        comparison = {
            'strategy': strategy_result,
            'buy_hold': {
                'final_value': buy_hold_final_value,
                'total_invested': buy_hold_invested,
                'profit': buy_hold_profit,
                'roi': buy_hold_roi,
                'btc_owned': buy_hold_btc
            }
        }
        
        print("\n=== Strategy vs Buy & Hold Comparison ===")
        print(f"Strategy ROI: {strategy_result['roi']:.2%}")
        print(f"Buy & Hold ROI: {buy_hold_roi:.2%}")
        print(f"Strategy Profit: ${strategy_result['profit']:,.2f}")
        print(f"Buy & Hold Profit: ${buy_hold_profit:,.2f}")
        print(f"Strategy Trades: {strategy_result['trade_count']}")
        print(f"Strategy BTC Owned: {strategy_result['btc_owned']:.4f}")
        print(f"Buy & Hold BTC Owned: {buy_hold_btc:.4f}")
        
        return comparison


def main():
    """Main execution function"""
    # Initialize trader
    data_path = "../../Data Sets/Bitcoin Data/Bitcoin_Cleaned_Data.csv"
    trader = ImprovedBitcoinVarianceTrader(data_path)
    
    print("=== Improved Bitcoin Variance Trading Strategy ===")
    print(f"Growth Model: 10^({trader.a}*ln(x)+{trader.b})")
    print(f"Data Range: {trader.data['Date'].min().date()} to {trader.data['Date'].max().date()}")
    print(f"Total Data Points: {len(trader.data)}")
    
    # Calculate and display variance statistics
    buy_thresholds, sell_thresholds = trader.calculate_optimal_thresholds()
    print(f"\nOptimal Thresholds (based on historical variance):")
    print(f"Buy thresholds: {buy_thresholds}")
    print(f"Sell thresholds: {sell_thresholds}")
    
    # Run optimization
    print(f"\nRunning strategy optimization...")
    optimal_result = trader.optimize_strategy()
    
    print(f"\n=== Optimal Strategy Results ===")
    print(f"Reserve Percentage: {optimal_result['reserve_pct']:.0%}")
    print(f"Buy Threshold: {optimal_result['buy_threshold']:.1%}")
    print(f"Sell Threshold: {optimal_result['sell_threshold']:.1%}")
    print(f"Total Invested: ${optimal_result['total_invested']:,.2f}")
    print(f"Final Value: ${optimal_result['final_value']:,.2f}")
    print(f"Profit: ${optimal_result['profit']:,.2f}")
    print(f"ROI: {optimal_result['roi']:.2%}")
    print(f"Number of Trades: {optimal_result['trade_count']}")
    
    # Compare with buy and hold
    comparison = trader.compare_with_buy_and_hold(optimal_result)
    
    # Plot analysis
    trader.plot_strategy_analysis(optimal_result)


if __name__ == "__main__":
    main() 
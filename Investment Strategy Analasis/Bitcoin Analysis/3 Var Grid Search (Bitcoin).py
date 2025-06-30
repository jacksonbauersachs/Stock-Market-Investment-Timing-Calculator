import pandas as pd
import numpy as np
from itertools import product
import os
import time


class EnhancedBitcoinTrader:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.data['Date'] = pd.to_datetime(self.data['Date'])

        # Handle price column format
        if pd.api.types.is_string_dtype(self.data['Close/Last']):
            self.data['Close/Last'] = pd.to_numeric(self.data['Close/Last'].str.replace(',', ''), errors='coerce')
        else:
            self.data['Close/Last'] = pd.to_numeric(self.data['Close/Last'], errors='coerce')

        self.data = self.data.sort_values('Date')
        self.data['ATH'] = self.data['Close/Last'].cummax()
        self.data['ATL'] = self.data['Close/Last'].cummin()

    def backtest_strategy(self, reserve_pct, buy_dips, buy_ratios, sell_peaks, sell_pcts):
        cash_reserve = 0
        btc_owned = 0
        total_invested = 0
        cash_from_sales = 0

        for _, row in self.data.iterrows():
            current_price = row['Close/Last']
            ath_ratio = current_price / row['ATH']

            # SELL LOGIC (multi-tier)
            if sell_peaks and btc_owned > 0:
                for i, (peak, pct) in enumerate(zip(sell_peaks, sell_pcts)):
                    if ath_ratio >= peak:
                        sell_amount = btc_owned * pct
                        cash_from_sales += sell_amount * current_price
                        btc_owned -= sell_amount
                        # Reset ATH after selling
                        row['ATH'] = current_price
                        break

            # BUY LOGIC (multi-tier)
            current_dip = (row['ATH'] - current_price) / row['ATH']
            reserve_deployed = 0

            for dip, ratio in zip(buy_dips, buy_ratios):
                if current_dip >= dip and cash_reserve > 0:
                    reserve_deployed = min(cash_reserve, cash_reserve * ratio)
                    cash_reserve -= reserve_deployed
                    break

            # Make investments
            regular_investment = 1000 * (1 - reserve_pct)
            total_investment = regular_investment + reserve_deployed
            btc_owned += total_investment / current_price
            total_invested += total_investment
            cash_reserve += 1000 * reserve_pct

        final_value = (btc_owned * self.data.iloc[-1]['Close/Last']) + cash_from_sales
        return final_value, total_invested

    def optimize_strategy(self):
        """Expanded parameter ranges with multi-tier selling"""
        # Wider buy parameters
        reserves = np.linspace(0.2, 0.8, 4)  # 20% to 80% reserves
        buy_dip_sets = [
            [0.1, 0.2, 0.3],  # Conservative
            [0.15, 0.25, 0.4],  # Moderate
            [0.2, 0.35, 0.5],  # Aggressive
            [0.25, 0.4, 0.6]  # Extreme
        ]
        buy_ratio_sets = [
            [0.2, 0.3, 0.5],  # Back-loaded
            [0.3, 0.3, 0.4],  # Balanced
            [0.4, 0.3, 0.3],  # Front-loaded
            [0.5, 0.25, 0.25]  # Very front-loaded
        ]

        # Multi-tier sell parameters
        sell_peak_sets = [
            [1.1, 1.3, 1.5],  # 10%/30%/50% above ATH
            [1.15, 1.4, 1.6],  # 15%/40%/60% above
            [1.2, 1.5, 1.8]  # 20%/50%/80% above
        ]
        sell_pct_sets = [
            [0.1, 0.15, 0.2],  # Sell 10%/15%/20% at each tier
            [0.15, 0.2, 0.25],  # Sell 15%/20%/25%
            [0.2, 0.25, 0.3]  # Sell 20%/25%/30%
        ]

        best = {'profit': -np.inf}
        start_time = time.time()
        total_tests = len(reserves) * len(buy_dip_sets) * len(buy_ratio_sets) * len(sell_peak_sets) * len(sell_pct_sets)
        completed = 0

        for reserve in reserves:
            for b_dips in buy_dip_sets:
                for b_ratios in buy_ratio_sets:
                    for s_peaks in sell_peak_sets:
                        for s_pcts in sell_pct_sets:
                            final_value, invested = self.backtest_strategy(
                                reserve, b_dips, b_ratios, s_peaks, s_pcts
                            )
                            profit = final_value - invested
                            completed += 1

                            if profit > best['profit']:
                                best = {
                                    'reserve': reserve,
                                    'buy_dips': b_dips,
                                    'buy_ratios': b_ratios,
                                    'sell_peaks': s_peaks,
                                    'sell_pcts': s_pcts,
                                    'value': final_value,
                                    'invested': invested,
                                    'profit': profit
                                }
                                elapsed = time.time() - start_time
                                print(f"\nNew best after {elapsed:.1f}s ({completed}/{total_tests}):")
                                print(f" Reserve: {reserve:.0%}")
                                print(f" Buy Dips: {[f'{d:.0%}' for d in b_dips]}")
                                print(f" Buy Ratios: {[f'{r:.0%}' for r in b_ratios]}")
                                print(f" Sell Peaks: {[f'{s:.0%}' for s in s_peaks]}")
                                print(f" Sell Amounts: {[f'{s:.0%}' for s in s_pcts]}")
                                print(f" Profit: ${profit:,.2f} (+${profit - best['profit']:,.2f})")

        return best


if __name__ == "__main__":
    data_path = "../../Data Sets/Bitcoin Data/2022-2025 Data (Cleaned Bitcoin)"
    if os.path.exists(data_path):
        print("Running enhanced optimization (expect 2-5 minutes)...")
        trader = EnhancedBitcoinTrader(data_path)
        result = trader.optimize_strategy()

        print("\n=== Enhanced Optimal Strategy ===")
        print(f"Reserve Percentage: {result['reserve']:.0%}")
        print(f"Buy At Dips: {[f'{d:.0%}' for d in result['buy_dips']]}")
        print(f"Buy Ratios: {[f'{r:.0%}' for r in result['buy_ratios']]}")
        print(f"Sell At Peaks: {[f'{s:.0%}' for s in result['sell_peaks']]}")
        print(f"Sell Percentages: {[f'{s:.0%}' for s in result['sell_pcts']]}")
        print(f"Total Invested: ${result['invested']:,.2f}")
        print(f"Final Value: ${result['value']:,.2f}")
        print(f"Profit: ${result['profit']:,.2f}")
    else:
        print(f"Error: Data file not found at {os.path.abspath(data_path)}")



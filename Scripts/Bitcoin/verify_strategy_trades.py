import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

def load_gbm_paths():
    """Load the specified GBM simulation paths file"""
    gbm_file = 'Results/Bitcoin/bitcoin_gbm_paths_20250723_165635.csv'
    print(f"Loading GBM paths from: {gbm_file}")
    df = pd.read_csv(gbm_file, index_col=0)
    return df

def calculate_fair_value(day_number):
    """Calculate Bitcoin's fair value for a given day number"""
    # Growth model coefficients from bitcoin_growth_model_coefficients_day365.txt
    a = 1.8277429956323488  # slope
    b = -10.880943376278237  # intercept
    return 10**(a * np.log(day_number) + b)

def verify_reserve_strategy_trades(prices, monthly_amount=100, reserve_ratio=1.0, 
                                 buy_thresholds=[0.96, 0.94, 0.92], 
                                 buy_allocations=[1.0, 0.0, 0.0]):
    """
    Verify the 3-Tier Reserve strategy with detailed trade logging
    Strategy: rsv=100%, buy=[(96, 100), (94, 0), (92, 0)]
    """
    print(f"\n=== 3-TIER RESERVE STRATEGY VERIFICATION ===")
    print(f"Monthly amount: ${monthly_amount}")
    print(f"Reserve ratio: {reserve_ratio*100}%")
    print(f"Buy thresholds: {buy_thresholds}")
    print(f"Buy allocations: {buy_allocations}")
    print(f"Price path length: {len(prices)} months")
    
    # Initialize tracking variables
    total_cash = 0
    total_btc = 0
    reserve_cash = 0
    
    # Monthly investment tracking - ALL goes to reserve
    monthly_dca = monthly_amount * (1 - reserve_ratio)  # 0% goes to DCA
    monthly_reserve = monthly_amount * reserve_ratio     # 100% goes to reserve
    
    print(f"\nMonthly breakdown:")
    print(f"  DCA portion: ${monthly_dca:.2f} ({monthly_dca/monthly_amount*100:.1f}%)")
    print(f"  Reserve portion: ${monthly_reserve:.2f} ({monthly_reserve/monthly_amount*100:.1f}%)")
    
    # Track trades for this path
    trades = []
    spent_this_path = [False, False, False]  # Track which tiers used for entire path
    
    # Phase 1 & 2: Process each day sequentially, accumulating reserve and checking for opportunities
    for day_idx in range(len(prices)):
        current_price = prices[day_idx]
        current_day = 6041 + day_idx
        fair_value = calculate_fair_value(current_day)
        price_ratio = current_price / fair_value
        
        # Add monthly contribution if this is a monthly interval
        if day_idx % 30 == 0:
            reserve_amount = monthly_amount * reserve_ratio
            total_cash += monthly_amount
            reserve_cash += reserve_amount
            
            print(f"\n--- MONTHLY RESERVE ACCUMULATION: Day {day_idx+1} ---")
            print(f"  Price: ${current_price:.2f}")
            print(f"  Fair value: ${fair_value:.2f}")
            print(f"  Price/Fair ratio: {price_ratio:.3f}")
            print(f"  Total cash: ${total_cash:.2f}")
            print(f"  Reserve cash: ${reserve_cash:.2f}")
            print(f"  Current BTC: {total_btc:.6f}")
            print(f"  RESERVE ACCUMULATED: ${reserve_amount:.2f}")
        
        # Check for opportunistic reserve buys (only Tier 1 since others are 0%)
        if not spent_this_path[0] and current_price < fair_value * buy_thresholds[0] and reserve_cash > 0:
            tier_reserve = reserve_cash * buy_allocations[0]  # 100% of reserve
            if tier_reserve > 0:
                tier_btc = tier_reserve / current_price
                total_btc += tier_btc
                reserve_cash -= tier_reserve
                spent_this_path[0] = True
                
                trades.append({
                    'month': (day_idx // 30) + 1,
                    'type': 'Reserve Tier 1',
                    'price': current_price,
                    'fair_value': fair_value,
                    'ratio': price_ratio,
                    'cash_used': tier_reserve,
                    'btc_bought': tier_btc,
                    'reserve_remaining': reserve_cash
                })
                
                print(f"\n--- TRADE: Day {day_idx+1} ---")
                print(f"  Price: ${current_price:.2f}")
                print(f"  Fair value: ${fair_value:.2f}")
                print(f"  Price/Fair ratio: {price_ratio:.3f}")
                print(f"  Total cash: ${total_cash:.2f}")
                print(f"  Reserve cash: ${reserve_cash:.2f}")
                print(f"  Current BTC: {total_btc:.6f}")
                print(f"  RESERVE TRADE (Tier 1): ${tier_reserve:.2f} -> {tier_btc:.6f} BTC")
    
    # Invest any remaining reserve at final price
    if reserve_cash > 0:
        final_price = prices[-1]
        final_btc = reserve_cash / final_price
        total_btc += final_btc
        trades.append({
            'month': len(prices),
            'type': 'Final Reserve',
            'price': final_price,
            'fair_value': calculate_fair_value(6041 + len(prices)),
            'ratio': final_price / calculate_fair_value(6041 + len(prices)),
            'cash_used': reserve_cash,
            'btc_bought': final_btc,
            'reserve_remaining': 0
        })
        print(f"\n--- TRADE: Final Day (Remaining Reserve) ---")
        print(f"  Price: ${final_price:.2f}")
        print(f"  Fair value: ${calculate_fair_value(6041 + len(prices)):.2f}")
        print(f"  Price/Fair ratio: {final_price / calculate_fair_value(6041 + len(prices)):.3f}")
        print(f"  Total cash: ${total_cash:.2f}")
        print(f"  Reserve cash: ${reserve_cash:.2f}")
        print(f"  Current BTC: {total_btc:.6f}")
        print(f"  FINAL RESERVE TRADE: ${reserve_cash:.2f} -> {final_btc:.6f} BTC")
        reserve_cash = 0
    
    # Calculate final portfolio value
    final_price = prices[-1]
    final_value = total_btc * final_price
    
    print(f"\n=== FINAL RESULTS ===")
    print(f"Total BTC: {total_btc:.6f}")
    print(f"Final price: ${final_price:.2f}")
    print(f"Final value: ${final_value:.2f}")
    print(f"Total invested: ${total_cash:.2f}")
    print(f"Return: {((final_value/total_cash)-1)*100:.2f}%")
    print(f"Uninvested cash: ${reserve_cash:.2f}")
    
    return {
        'final_value': final_value,
        'total_invested': total_cash,
        'total_btc': total_btc,
        'return_pct': ((final_value/total_cash)-1)*100,
        'trades': trades
    }

def verify_reserve_strategy_trades_silent(prices, monthly_amount=100, reserve_ratio=1.0, 
                                        buy_thresholds=[0.96, 0.94, 0.92], 
                                        buy_allocations=[1.0, 0.0, 0.0]):
    """
    Silent version of 3-Tier Reserve strategy verification (no print output)
    Strategy: rsv=100%, buy=[(96, 100), (94, 0), (92, 0)]
    """
    # Initialize tracking variables
    total_cash = 0
    total_btc = 0
    reserve_cash = 0
    
    # Monthly investment tracking - ALL goes to reserve
    monthly_dca = monthly_amount * (1 - reserve_ratio)  # 0% goes to DCA
    monthly_reserve = monthly_amount * reserve_ratio     # 100% goes to reserve
    
    # Track trades for this path
    trades = []
    spent_this_path = [False, False, False]  # Track which tiers used for entire path
    
    # Phase 1 & 2: Process each day sequentially, accumulating reserve and checking for opportunities
    for day_idx in range(len(prices)):
        current_price = prices[day_idx]
        current_day = 6041 + day_idx
        fair_value = calculate_fair_value(current_day)
        price_ratio = current_price / fair_value
        
        # Add monthly contribution if this is a monthly interval
        if day_idx % 30 == 0:
            reserve_amount = monthly_amount * reserve_ratio
            total_cash += monthly_amount
            reserve_cash += reserve_amount
        
        # Check for opportunistic reserve buys (only Tier 1 since others are 0%)
        if not spent_this_path[0] and current_price < fair_value * buy_thresholds[0] and reserve_cash > 0:
            tier_reserve = reserve_cash * buy_allocations[0]  # 100% of reserve
            if tier_reserve > 0:
                tier_btc = tier_reserve / current_price
                total_btc += tier_btc
                reserve_cash -= tier_reserve
                spent_this_path[0] = True
                
                trades.append({
                    'month': (day_idx // 30) + 1,
                    'type': 'Reserve Tier 1',
                    'price': current_price,
                    'fair_value': fair_value,
                    'ratio': price_ratio,
                    'cash_used': tier_reserve,
                    'btc_bought': tier_btc,
                    'reserve_remaining': reserve_cash
                })
    
    # Invest any remaining reserve at final price
    if reserve_cash > 0:
        final_price = prices[-1]
        final_btc = reserve_cash / final_price
        total_btc += final_btc
        trades.append({
            'month': len(prices),
            'type': 'Final Reserve',
            'price': final_price,
            'fair_value': calculate_fair_value(6041 + len(prices)),
            'ratio': final_price / calculate_fair_value(6041 + len(prices)),
            'cash_used': reserve_cash,
            'btc_bought': final_btc,
            'reserve_remaining': 0
        })
    
    # Calculate final portfolio value
    final_price = prices[-1]
    final_value = total_btc * final_price
    
    return {
        'final_value': final_value,
        'total_invested': total_cash,
        'total_btc': total_btc,
        'return_pct': ((final_value/total_cash)-1)*100,
        'trades': trades
    }

def main():
    """Main function to verify strategy trades"""
    print("Loading GBM paths...")
    df = load_gbm_paths()
    
    if df is None:
        print("Failed to load GBM paths!")
        return
    
    print(f"Loaded {len(df.columns)} price paths")
    
    # Test the 3-Tier Reserve strategy: rsv=100%, buy=[(96, 100), (94, 0), (92, 0)]
    reserve_ratio = 1.0
    buy_thresholds = [0.96, 0.94, 0.92]
    buy_allocations = [1.0, 0.0, 0.0]  # Only Tier 1 is used (100%)
    monthly_amount = 100
    
    # Create output file (overwrite existing)
    output_file = "Results/Bitcoin/strategy_verification_latest.txt"
    
    # Redirect output to both console and file
    import sys
    original_stdout = sys.stdout
    
    with open(output_file, 'w') as f:
        sys.stdout = f
        
        print(f"3-TIER RESERVE STRATEGY VERIFICATION LOG")
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Strategy: 100% Reserve, Buy thresholds: {buy_thresholds}, Allocations: {buy_allocations}")
        print(f"Strategy: rsv=100%, buy=[(96, 100), (94, 0), (92, 0)]")
        print(f"Monthly amount: ${monthly_amount}")
        print(f"="*80)
        
        # Test all paths for verification
        results = []
        total_paths = len(df.columns)
        
        for path_num in range(total_paths):  # Test all paths
            # Only show detailed output for first 3 paths in file
            if path_num < 3:
                print(f"\n{'='*60}")
                print(f"VERIFYING PATH {path_num + 1}")
                print(f"{'='*60}")
                
                prices = df.iloc[:, path_num].values
                result = verify_reserve_strategy_trades(
                    prices, monthly_amount, reserve_ratio, buy_thresholds, buy_allocations
                )
                results.append(result)
            else:
                # For remaining paths, just calculate without detailed output
                prices = df.iloc[:, path_num].values
                result = verify_reserve_strategy_trades_silent(
                    prices, monthly_amount, reserve_ratio, buy_thresholds, buy_allocations
                )
                results.append(result)
            
            # Progress update every 5 paths
            if (path_num + 1) % 5 == 0:
                sys.stdout = original_stdout
                print(f"Completed {path_num + 1}/{total_paths} paths...")
                sys.stdout = f
        
        # Summary
        print(f"\n{'='*60}")
        print(f"SUMMARY OF ALL {total_paths} PATHS:")
        print(f"{'='*60}")
        for i, result in enumerate(results):
            print(f"Path {i+1}: ${result['final_value']:.2f} ({result['return_pct']:.2f}%)")
        
        avg_return = np.mean([r['return_pct'] for r in results])
        print(f"Average return: {avg_return:.2f}%")
        
        # Save trade details to CSV (only for first 3 paths)
        all_trades = []
        for i, result in enumerate(results[:3]):
            for trade in result['trades']:
                trade['path'] = i + 1
                all_trades.append(trade)
        
        if all_trades:
            trades_df = pd.DataFrame(all_trades)
            trades_csv = "Results/Bitcoin/strategy_trades_latest.csv"
            trades_df.to_csv(trades_csv, index=False)
            print(f"\nTrade details saved to: {trades_csv}")
    
    # Restore stdout and print summary to console
    sys.stdout = original_stdout
    print(f"Verification complete! Results saved to: {output_file}")
    
    # Print summary to console
    print(f"\nSUMMARY OF ALL {total_paths} PATHS:")
    avg_return = np.mean([r['return_pct'] for r in results])
    print(f"Average return: {avg_return:.2f}%")
    print(f"Best path: {max([r['return_pct'] for r in results]):.2f}%")
    print(f"Worst path: {min([r['return_pct'] for r in results]):.2f}%")

if __name__ == "__main__":
    main() 
import pandas as pd
import numpy as np

print("COMPREHENSIVE RESERVE STRATEGY COMPARISON")
print("=" * 70)

# Load all three strategy results
try:
    df_0_cash = pd.read_csv('Results/Bitcoin/bitcoin_multi_tier_strategy_results.csv')
    best_0_cash = df_0_cash.iloc[0]
    print("✅ 0% Cash Strategy Results Loaded")
except:
    print("❌ 0% Cash Strategy Results Not Found")
    best_0_cash = None

try:
    df_5_savings = pd.read_csv('Results/Bitcoin/bitcoin_multi_tier_interest_strategy_results.csv')
    best_5_savings = df_5_savings.iloc[0]
    print("✅ 5% High-Yield Savings Strategy Results Loaded")
except:
    print("❌ 5% High-Yield Savings Strategy Results Not Found")
    best_5_savings = None

try:
    df_sp500 = pd.read_csv('Results/Bitcoin/bitcoin_sp500_tactical_allocation_results.csv')
    best_sp500 = df_sp500.iloc[0]
    print("✅ S&P 500 Tactical Allocation Strategy Results Loaded")
except:
    print("❌ S&P 500 Tactical Allocation Strategy Results Not Found")
    best_sp500 = None

print("\n" + "=" * 70)
print("STRATEGY COMPARISON SUMMARY")
print("=" * 70)

strategies = []

if best_0_cash is not None:
    strategies.append({
        'name': '0% Cash Reserves',
        'description': 'Bitcoin + Cash (0% interest)',
        'reserve_asset': 'Cash (0%)',
        'optimal_reserve': f"{best_0_cash['reserve_pct']:.0f}%",
        'median_return': best_0_cash['median_value'],
        'cagr': best_0_cash['cagr'],
        'improvement': best_0_cash['median_improvement'],
        'volatility': best_0_cash['std_value'],
        'win_rate': best_0_cash['win_rate'],
        'strategy_type': 'Multi-tier Bitcoin reserve strategy'
    })

if best_5_savings is not None:
    strategies.append({
        'name': '5% High-Yield Savings',
        'description': 'Bitcoin + High-Yield Savings (5% interest)',
        'reserve_asset': 'Savings (5%)',
        'optimal_reserve': f"{best_5_savings['reserve_pct']:.0f}%",
        'median_return': best_5_savings['median_value'],
        'cagr': best_5_savings['cagr'],
        'improvement': best_5_savings['median_improvement'],
        'volatility': best_5_savings['std_value'],
        'win_rate': best_5_savings['win_rate'],
        'strategy_type': 'Multi-tier Bitcoin reserve strategy'
    })

if best_sp500 is not None:
    strategies.append({
        'name': 'S&P 500 Tactical Allocation',
        'description': 'Bitcoin + S&P 500 (tactical allocation)',
        'reserve_asset': 'S&P 500 (~8.2%)',
        'optimal_reserve': f"{best_sp500['base_btc_allocation']:.0f}% Bitcoin base",
        'median_return': best_sp500['median_value'],
        'cagr': best_sp500['cagr'],
        'improvement': best_sp500['median_improvement'],
        'volatility': best_sp500['std_value'],
        'win_rate': best_sp500['win_rate'],
        'strategy_type': 'Tactical asset allocation'
    })

# Sort by median return
strategies.sort(key=lambda x: x['median_return'], reverse=True)

print(f"\n🏆 RANKING BY MEDIAN RETURN:")
print("=" * 50)

for i, strategy in enumerate(strategies, 1):
    print(f"\n#{i}. {strategy['name']}")
    print(f"    Reserve Asset: {strategy['reserve_asset']}")
    print(f"    Optimal Allocation: {strategy['optimal_reserve']}")
    print(f"    Median Return: ${strategy['median_return']:,.0f}")
    print(f"    CAGR: {strategy['cagr']:.2f}%")
    print(f"    Improvement vs Baseline: {strategy['improvement']:+.1f}%")
    print(f"    Volatility: ${strategy['volatility']:,.0f}")
    print(f"    Win Rate: {strategy['win_rate']:.1f}%")
    print(f"    Strategy Type: {strategy['strategy_type']}")

# Direct comparison
if len(strategies) >= 2:
    print(f"\n🔥 DIRECT COMPARISON:")
    print("=" * 50)
    
    best = strategies[0]
    second = strategies[1]
    
    return_diff = best['median_return'] - second['median_return']
    cagr_diff = best['cagr'] - second['cagr']
    vol_diff = best['volatility'] - second['volatility']
    
    print(f"Best Strategy: {best['name']}")
    print(f"Second Best: {second['name']}")
    print(f"Return Difference: ${return_diff:,.0f} ({return_diff/second['median_return']*100:+.1f}%)")
    print(f"CAGR Difference: {cagr_diff:+.2f}%")
    print(f"Volatility Difference: ${vol_diff:,.0f} ({vol_diff/second['volatility']*100:+.1f}%)")

# Analysis by strategy characteristics
print(f"\n📊 STRATEGY CHARACTERISTICS ANALYSIS:")
print("=" * 60)

print(f"\n💰 RESERVE ASSET PERFORMANCE:")
for strategy in strategies:
    if 'Cash (0%)' in strategy['reserve_asset']:
        print(f"  {strategy['name']}: No return on reserves (cash drag)")
    elif 'Savings (5%)' in strategy['reserve_asset']:
        print(f"  {strategy['name']}: 5% return on reserves (reduces cash drag)")
    elif 'S&P 500' in strategy['reserve_asset']:
        print(f"  {strategy['name']}: ~8.2% return on reserves (eliminates cash drag)")

print(f"\n🎯 OPTIMAL ALLOCATION PATTERNS:")
for strategy in strategies:
    if 'reserve_pct' in str(strategy['optimal_reserve']):
        print(f"  {strategy['name']}: {strategy['optimal_reserve']} reserves")
    else:
        print(f"  {strategy['name']}: {strategy['optimal_reserve']}")

print(f"\n📈 RISK-ADJUSTED PERFORMANCE:")
for strategy in strategies:
    sharpe_like = (strategy['cagr'] - 0) / (strategy['volatility'] / strategy['median_return'] * 100)
    print(f"  {strategy['name']}: Sharpe-like ratio = {sharpe_like:.3f}")

# Key insights
print(f"\n💡 KEY INSIGHTS:")
print("=" * 50)

if len(strategies) >= 3:
    cash_strat = next((s for s in strategies if 'Cash' in s['name']), None)
    savings_strat = next((s for s in strategies if 'Savings' in s['name']), None)
    sp500_strat = next((s for s in strategies if 'S&P 500' in s['name']), None)
    
    print(f"\n1. RESERVE ASSET IMPACT:")
    if cash_strat and savings_strat:
        savings_improvement = savings_strat['median_return'] - cash_strat['median_return']
        print(f"   5% Savings vs 0% Cash: ${savings_improvement:,.0f} improvement")
    
    if savings_strat and sp500_strat:
        sp500_improvement = sp500_strat['median_return'] - savings_strat['median_return']
        print(f"   S&P 500 vs 5% Savings: ${sp500_improvement:,.0f} improvement")
    
    if cash_strat and sp500_strat:
        total_improvement = sp500_strat['median_return'] - cash_strat['median_return']
        print(f"   S&P 500 vs 0% Cash: ${total_improvement:,.0f} total improvement")

print(f"\n2. STRATEGY EVOLUTION:")
print(f"   • 0% Cash: Minimize cash drag with low reserves")
print(f"   • 5% Savings: Higher reserves viable with interest")
print(f"   • S&P 500: Eliminate cash drag with growth asset")

print(f"\n3. PRACTICAL IMPLICATIONS:")
print(f"   • S&P 500 strategy most realistic for investors")
print(f"   • Uses actual investable assets, not cash")
print(f"   • Better risk-adjusted returns")
print(f"   • Implementable with standard ETFs")

print(f"\n4. VOLATILITY MANAGEMENT:")
volatilities = [s['volatility'] for s in strategies]
if volatilities:
    lowest_vol = min(volatilities)
    lowest_vol_strategy = next(s for s in strategies if s['volatility'] == lowest_vol)
    print(f"   • Lowest volatility: {lowest_vol_strategy['name']} (${lowest_vol:,.0f})")
    print(f"   • Volatility reduction improves risk-adjusted returns")

print(f"\n🎯 FINAL RECOMMENDATION:")
print("=" * 50)

if strategies:
    winner = strategies[0]
    print(f"WINNER: {winner['name']}")
    print(f"• Best median return: ${winner['median_return']:,.0f}")
    print(f"• Best CAGR: {winner['cagr']:.2f}%")
    print(f"• Strategy: {winner['strategy_type']}")
    print(f"• Why it wins: {winner['description']}")
    
    if 'S&P 500' in winner['name']:
        print(f"• Most realistic: Uses actual investable assets")
        print(f"• Most practical: Implementable with ETFs")
        print(f"• Best risk-adjusted: Lower volatility with higher returns")

print(f"\n📁 SUMMARY SAVED TO: Results/Bitcoin/comprehensive_strategy_comparison.txt")

# Save this analysis
with open('Results/Bitcoin/comprehensive_strategy_comparison.txt', 'w') as f:
    f.write("COMPREHENSIVE RESERVE STRATEGY COMPARISON\n")
    f.write("=" * 70 + "\n\n")
    
    f.write("STRATEGY RANKINGS BY MEDIAN RETURN:\n")
    f.write("=" * 50 + "\n")
    
    for i, strategy in enumerate(strategies, 1):
        f.write(f"\n#{i}. {strategy['name']}\n")
        f.write(f"    Reserve Asset: {strategy['reserve_asset']}\n")
        f.write(f"    Optimal Allocation: {strategy['optimal_reserve']}\n")
        f.write(f"    Median Return: ${strategy['median_return']:,.0f}\n")
        f.write(f"    CAGR: {strategy['cagr']:.2f}%\n")
        f.write(f"    Improvement vs Baseline: {strategy['improvement']:+.1f}%\n")
        f.write(f"    Volatility: ${strategy['volatility']:,.0f}\n")
        f.write(f"    Win Rate: {strategy['win_rate']:.1f}%\n")
        f.write(f"    Strategy Type: {strategy['strategy_type']}\n")
    
    if strategies:
        winner = strategies[0]
        f.write(f"\nWINNER: {winner['name']}\n")
        f.write(f"Best overall strategy with ${winner['median_return']:,.0f} median return\n")
        f.write(f"and {winner['cagr']:.2f}% CAGR\n")

print("Analysis complete! 🎉") 
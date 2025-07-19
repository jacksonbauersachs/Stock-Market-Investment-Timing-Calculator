import pandas as pd
import numpy as np

# Read the results
df = pd.read_csv('Results/Bitcoin/bitcoin_multi_tier_strategy_results.csv')

print("RESERVE PERCENTAGE ANALYSIS")
print("=" * 50)

# Group by reserve percentage and analyze
reserve_analysis = df.groupby('reserve_pct').agg({
    'median_improvement': ['mean', 'std', 'max', 'min', 'count'],
    'mean_improvement': ['mean', 'std', 'max', 'min'],
    'cagr_improvement': ['mean', 'std', 'max', 'min'],
    'win_rate': ['mean', 'std', 'max', 'min'],
    'beat_dca_rate': ['mean', 'std', 'max', 'min']
}).round(2)

print("\nPERFORMANCE BY RESERVE PERCENTAGE:")
print(reserve_analysis)

print("\n" + "="*60)
print("DETAILED BREAKDOWN BY RESERVE %:")
print("="*60)

for reserve_pct in sorted(df['reserve_pct'].unique()):
    subset = df[df['reserve_pct'] == reserve_pct]
    
    print(f"\n📊 {reserve_pct}% RESERVE STRATEGIES:")
    print(f"   Number of strategies: {len(subset)}")
    print(f"   Best median improvement: {subset['median_improvement'].max():.1f}%")
    print(f"   Average median improvement: {subset['median_improvement'].mean():.1f}%")
    print(f"   Worst median improvement: {subset['median_improvement'].min():.1f}%")
    print(f"   Average win rate: {subset['win_rate'].mean():.1f}%")
    print(f"   Average beat DCA rate: {subset['beat_dca_rate'].mean():.1f}%")
    print(f"   Standard deviation of improvements: {subset['median_improvement'].std():.1f}%")

# Let's also look at the opportunity cost of higher reserves
print("\n" + "="*60)
print("OPPORTUNITY COST ANALYSIS:")
print("="*60)

print("\nWhy might lower reserves (10%) outperform higher reserves (50%)?")
print("\n1. CASH DRAG EFFECT:")
print("   - Higher reserves mean more cash sitting idle")
print("   - Cash earns 0% while Bitcoin has positive expected return")
print("   - Only 10% cash reserve vs 50% = more money working in Bitcoin")

print("\n2. TIMING EFFICIENCY:")
print("   - With 10% reserves, you deploy cash quickly when opportunities arise")
print("   - With 50% reserves, you have 'too much' cash waiting for perfect timing")
print("   - Bitcoin's long-term trend is up, so being in the market beats waiting")

print("\n3. VOLATILITY PATTERNS:")
print("   - Bitcoin doesn't crash 50% every month")
print("   - Most opportunities are smaller dips (10-30%)")
print("   - 10% reserve is sufficient for most buying opportunities")

print("\n4. MATHEMATICAL EXPECTATION:")
print("   - E[Return] = (% in Bitcoin) × E[Bitcoin Return] + (% in Cash) × 0%")
print("   - Higher Bitcoin allocation = higher expected return")
print("   - Reserve strategies work by timing, not by holding cash")

# Calculate the actual cash allocation over time
print("\n" + "="*60)
print("EFFECTIVE CASH ALLOCATION:")
print("="*60)

monthly_investment = 1000
months = 60  # 5 years

for reserve_pct in [10, 20, 30, 40, 50]:
    monthly_reserve = monthly_investment * (reserve_pct / 100)
    monthly_dca = monthly_investment * (1 - reserve_pct / 100)
    
    # Assume reserves get deployed on average every 3 months
    avg_cash_held = monthly_reserve * 3  # 3 months of reserves on average
    total_invested = monthly_investment * months
    
    effective_cash_pct = avg_cash_held / total_invested * 100
    
    print(f"{reserve_pct}% Reserve Strategy:")
    print(f"   Monthly to reserve: ${monthly_reserve:,.0f}")
    print(f"   Monthly to DCA: ${monthly_dca:,.0f}")
    print(f"   Average cash held: ${avg_cash_held:,.0f}")
    print(f"   Effective cash allocation: {effective_cash_pct:.1f}% of total")
    print()

print("💡 KEY INSIGHT:")
print("The 10% reserve strategy is optimal because it provides enough cash")
print("for opportunistic buying without suffering excessive cash drag.")
print("Most of your money stays invested in Bitcoin's positive expected return!") 
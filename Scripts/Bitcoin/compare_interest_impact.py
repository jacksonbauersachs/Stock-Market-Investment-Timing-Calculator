import pandas as pd
import numpy as np

# Read both result files
df_0_interest = pd.read_csv('Results/Bitcoin/bitcoin_multi_tier_strategy_results.csv')
df_5_interest = pd.read_csv('Results/Bitcoin/bitcoin_multi_tier_interest_strategy_results.csv')

print("IMPACT OF 5% INTEREST ON CASH RESERVES")
print("=" * 60)

# Compare best strategies
best_0_interest = df_0_interest.iloc[0]
best_5_interest = df_5_interest.iloc[0]

print("\n🏆 BEST STRATEGY COMPARISON:")
print("=" * 40)

print(f"📊 0% INTEREST (Cash Reserves):")
print(f"   Best Reserve %: {best_0_interest['reserve_pct']:.0f}%")
print(f"   Median Return: ${best_0_interest['median_value']:,.0f}")
print(f"   CAGR: {best_0_interest['cagr']:.2f}%")
print(f"   Improvement vs DCA: {best_0_interest['median_improvement']:+.1f}%")
print(f"   Buy Tiers: {best_0_interest['buy_tiers']}")
print(f"   Sell Tiers: {best_0_interest['sell_tiers']}")

print(f"\n💰 5% INTEREST (High-Yield Savings):")
print(f"   Best Reserve %: {best_5_interest['reserve_pct']:.0f}%")
print(f"   Median Return: ${best_5_interest['median_value']:,.0f}")
print(f"   CAGR: {best_5_interest['cagr']:.2f}%")
print(f"   Improvement vs DCA: {best_5_interest['median_improvement']:+.1f}%")
print(f"   Buy Tiers: {best_5_interest['buy_tiers']}")
print(f"   Sell Tiers: {best_5_interest['sell_tiers']}")

# Calculate the difference
return_difference = best_5_interest['median_value'] - best_0_interest['median_value']
improvement_difference = best_5_interest['median_improvement'] - best_0_interest['median_improvement']

print(f"\n🔥 IMPACT OF 5% INTEREST:")
print(f"   Additional Return: ${return_difference:,.0f}")
print(f"   Additional Improvement: {improvement_difference:+.1f}%")
print(f"   Optimal Reserve % Change: {best_0_interest['reserve_pct']:.0f}% → {best_5_interest['reserve_pct']:.0f}%")

# Analyze reserve percentage performance
print(f"\n📈 RESERVE PERCENTAGE PERFORMANCE:")
print("=" * 50)

print("\n0% INTEREST RESERVES:")
reserve_0_analysis = df_0_interest.groupby('reserve_pct')['median_improvement'].agg(['mean', 'max', 'count']).round(2)
print(reserve_0_analysis)

print("\n5% INTEREST RESERVES:")
reserve_5_analysis = df_5_interest.groupby('reserve_pct')['median_improvement'].agg(['mean', 'max', 'count']).round(2)
print(reserve_5_analysis)

# Compare specific reserve percentages
print(f"\n🎯 DETAILED RESERVE % COMPARISON:")
print("=" * 50)

for reserve_pct in [10, 20, 30, 40, 50]:
    if reserve_pct in df_0_interest['reserve_pct'].values and reserve_pct in df_5_interest['reserve_pct'].values:
        avg_0 = df_0_interest[df_0_interest['reserve_pct'] == reserve_pct]['median_improvement'].mean()
        best_0 = df_0_interest[df_0_interest['reserve_pct'] == reserve_pct]['median_improvement'].max()
        
        avg_5 = df_5_interest[df_5_interest['reserve_pct'] == reserve_pct]['median_improvement'].mean()
        best_5 = df_5_interest[df_5_interest['reserve_pct'] == reserve_pct]['median_improvement'].max()
        
        print(f"{reserve_pct}% Reserve:")
        print(f"   0% Interest: Avg {avg_0:.1f}%, Best {best_0:.1f}%")
        print(f"   5% Interest: Avg {avg_5:.1f}%, Best {best_5:.1f}%")
        print(f"   Improvement: {avg_5 - avg_0:+.1f}% avg, {best_5 - best_0:+.1f}% best")
        print()

# Calculate theoretical interest benefit
print(f"\n💡 THEORETICAL INTEREST CALCULATION:")
print("=" * 50)

monthly_investment = 1000
years = 5
months = years * 12
interest_rate = 0.05

for reserve_pct in [10, 20, 30, 40, 50]:
    monthly_reserve = monthly_investment * (reserve_pct / 100)
    
    # Simple compound interest calculation
    # Assuming reserves accumulate and earn interest
    total_interest = 0
    accumulated_reserves = 0
    
    for month in range(1, months + 1):
        accumulated_reserves += monthly_reserve
        monthly_interest = accumulated_reserves * (interest_rate / 12)
        total_interest += monthly_interest
        # Assume some reserves get deployed periodically
        if month % 6 == 0:  # Deploy some reserves every 6 months
            accumulated_reserves *= 0.7  # Keep 70% of reserves
    
    print(f"{reserve_pct}% Reserve Strategy:")
    print(f"   Monthly Reserve: ${monthly_reserve:,.0f}")
    print(f"   Total Interest Earned: ${total_interest:,.0f}")
    print(f"   Interest as % of $100k: {total_interest/100000*100:.1f}%")
    print()

print(f"🎯 KEY INSIGHTS:")
print(f"1. Optimal reserve % increased from {best_0_interest['reserve_pct']:.0f}% to {best_5_interest['reserve_pct']:.0f}%")
print(f"2. 5% interest makes higher reserves more attractive")
print(f"3. Additional return from interest: ${return_difference:,.0f}")
print(f"4. Higher reserves become viable when they earn competitive returns")
print(f"5. Cash drag effect is reduced when cash earns meaningful interest") 
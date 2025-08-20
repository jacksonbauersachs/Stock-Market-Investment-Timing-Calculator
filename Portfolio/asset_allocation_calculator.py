import pandas as pd
import numpy as np
from datetime import datetime
import os

def load_overvaluation_results():
    """Load the most recent overvaluation results"""
    
    # Look for the most recent overvaluation results file
    results_dir = "Portfolio/Overvaluedness"
    results_files = [f for f in os.listdir(results_dir) if f.startswith("overvaluation_results_") and f.endswith(".txt")]
    
    if not results_files:
        print("No overvaluation results found. Please run the overvalued calculator first.")
        return None
    
    # Get the most recent file
    latest_file = max(results_files)
    results_file = os.path.join(results_dir, latest_file)
    
    print(f"Loading results from: {results_file}")
    
    # Parse the results
    overvaluation_data = {}
    
    with open(results_file, 'r') as f:
        content = f.read()
        
    # Extract overvaluation percentages
    lines = content.split('\n')
    current_asset = None
    
    for line in lines:
        # Check for asset headers
        if 'BITCOIN ANALYSIS' in line:
            current_asset = 'Bitcoin'
        elif 'GOLD ANALYSIS' in line:
            current_asset = 'Gold'
        elif 'SILVER ANALYSIS' in line:
            current_asset = 'Silver'
        # Check for overvaluation line
        elif 'Overvaluation:' in line and current_asset:
            pct = float(line.split('Overvaluation:')[1].split('%')[0].strip())
            overvaluation_data[current_asset] = pct
    
    # Debug: print what we found
    print(f"Found overvaluation data: {overvaluation_data}")
    
    return overvaluation_data

def calculate_optimal_allocation(overvaluation_data, base_allocation=None):
    """
    Calculate optimal asset allocation based on overvaluation percentages.
    
    Strategy:
    1. Start with base allocation
    2. Metals are treated as a unified asset class for portfolio rebalancing
    3. Within metals, aggressive rebalancing between Gold and Silver based on relative overvaluation
    4. More overvalued assets get reduced allocation
    5. Less overvalued assets get increased allocation
    6. Roth IRA & HYS are treated as neutral (can absorb rebalancing)
    7. All assets move independently based on their overvaluation vs. average
    8. Conservative adjustment limits prevent extreme allocation swings
    9. Keep total allocation at 100%
    """
    
    # Base allocation (your current target ratios)
    if base_allocation is None:
        base_allocation = {
            'Bitcoin': 30.0,
            'Metals': 36.0,  # Combined Gold + Silver (24% + 12%)
            'Roth IRA': 24.0,
            'High-Yield Savings': 10.0,
        }
    
    print("BASE ALLOCATION:")
    print("-" * 20)
    for asset, allocation in base_allocation.items():
        print(f"{asset}: {allocation:.1f}%")
    
    # Extend overvaluation data to include Roth IRA and HYS
    # These are treated as "safe haven" assets that can absorb some rebalancing
    # but they still participate in the normal overvaluation-based adjustments
    extended_overvaluation = overvaluation_data.copy()
    
    # Calculate metals as a unified asset class (weighted average of Gold and Silver)
    if 'Gold' in overvaluation_data and 'Silver' in overvaluation_data:
        # Weight by base allocation proportions (Gold: 24/36 = 67%, Silver: 12/36 = 33%)
        gold_weight = 24.0 / 36.0
        silver_weight = 12.0 / 36.0
        metals_overval = (overvaluation_data['Gold'] * gold_weight + 
                         overvaluation_data['Silver'] * silver_weight)
        extended_overvaluation['Metals'] = metals_overval
        
        print(f"\nMETALS ANALYSIS:")
        print(f"Gold: {overvaluation_data['Gold']:+.1f}% overvalued (weight: {gold_weight:.1%})")
        print(f"Silver: {overvaluation_data['Silver']:+.1f}% overvalued (weight: {silver_weight:.1%})")
        print(f"Combined Metals: {metals_overval:+.1f}% overvalued")
    
    # For Roth IRA, assign neutral overvaluation
    extended_overvaluation['Roth IRA'] = 0.0  # Neutral - can absorb rebalancing
    
    # For HYS, add stress multiplier based on total market overvaluation
    # This automatically increases defensive positioning during extreme stress
    avg_market_overval = np.mean([overvaluation_data.get(asset, 0) for asset in ['Bitcoin', 'Gold', 'Silver']])
    
    if avg_market_overval > 30:
        hys_stress = -15.0  # 15% "undervalued" during extreme stress
    elif avg_market_overval > 20:
        hys_stress = -10.0  # 10% "undervalued" during high stress
    elif avg_market_overval > 10:
        hys_stress = -5.0   # 5% "undervalued" during moderate stress
    else:
        hys_stress = 0.0    # Neutral during normal conditions
    
    extended_overvaluation['High-Yield Savings'] = hys_stress
    print(f"HYS Stress Response: {hys_stress:+.1f}% (Market avg: {avg_market_overval:+.1f}%)")
    
    print(f"\nOVERVALUATION ANALYSIS:")
    print("-" * 25)
    for asset, overval_pct in extended_overvaluation.items():
        print(f"{asset}: {overval_pct:+.1f}% overvalued")
    
    # Calculate adjustment factors
    # More overvalued = reduce allocation
    # Less overvalued = increase allocation
    adjustment_factors = {}
    
    # Calculate average overvaluation (only for assets that exist in base allocation)
    portfolio_assets = ['Bitcoin', 'Metals', 'Roth IRA', 'High-Yield Savings']
    avg_overval = np.mean([extended_overvaluation[asset] for asset in portfolio_assets])
    
    print(f"\nAverage overvaluation: {avg_overval:+.1f}%")
    
    # Calculate relative adjustments for portfolio-level assets only
    for asset in portfolio_assets:
        overval_pct = extended_overvaluation[asset]
        # Relative to average: if asset is more overvalued than average, reduce allocation
        relative_overval = overval_pct - avg_overval
        adjustment_factors[asset] = -relative_overval / 100  # Convert to decimal
    
    print(f"\nADJUSTMENT FACTORS:")
    print("-" * 20)
    for asset, factor in adjustment_factors.items():
        print(f"{asset}: {factor:+.3f} ({factor*100:+.1f}%)")
    
    # Apply adjustments to all assets
    adjusted_allocation = base_allocation.copy()
    
    # Adjust all assets based on overvaluation
    for asset in portfolio_assets:
        if asset in adjustment_factors:
            # Calculate adjustment based on base allocation (proportional to each asset's size)
            adjustment = adjustment_factors[asset] * base_allocation[asset] * 0.8  # Limit to 80% of base allocation
            
            new_allocation = base_allocation[asset] + adjustment
            
            # Set reasonable bounds for each asset
            if asset == 'Bitcoin':
                new_allocation = max(10.0, min(55.0, new_allocation))  # 10-55%
            elif asset == 'Metals':
                new_allocation = max(15.0, min(50.0, new_allocation))  # 15-50%
            elif asset == 'Roth IRA':
                new_allocation = max(5.0, min(40.0, new_allocation))   # 5-40%
            elif asset == 'High-Yield Savings':
                new_allocation = max(5.0, min(25.0, new_allocation))   # 5-25%
            
            adjusted_allocation[asset] = new_allocation
    
    # Rebalance to ensure total = 100%
    total_allocation = sum(adjusted_allocation.values())
    if abs(total_allocation - 100.0) > 0.1:
        # Scale all assets proportionally
        scale_factor = 100.0 / total_allocation
        
        for asset in adjusted_allocation:
            adjusted_allocation[asset] *= scale_factor
    
    return adjusted_allocation, adjustment_factors

def calculate_metals_allocation(overvaluation_data, total_metals_allocation):
    """
    Calculate the internal allocation between Gold and Silver within the metals allocation.
    This allows for aggressive rebalancing between the two metals based on relative overvaluation.
    """
    
    if 'Gold' not in overvaluation_data or 'Silver' not in overvaluation_data:
        # If we don't have overvaluation data, use base proportions
        gold_allocation = total_metals_allocation * (24.0 / 36.0)
        silver_allocation = total_metals_allocation * (12.0 / 36.0)
        return {'Gold': gold_allocation, 'Silver': silver_allocation}
    
    gold_overval = overvaluation_data['Gold']
    silver_overval = overvaluation_data['Silver']
    
    # Calculate relative overvaluation between metals
    relative_overval = gold_overval - silver_overval
    
    # Base proportions (Gold: 67%, Silver: 33%)
    base_gold_pct = 24.0 / 36.0
    base_silver_pct = 12.0 / 36.0
    
    # Aggressive adjustment factor for metals-to-metals rebalancing
    # This can be more aggressive since we're staying within the same asset class
    adjustment_factor = 0.9  # 90% adjustment factor for aggressive metal rebalancing
    
    # If Gold is more overvalued than Silver, shift allocation toward Silver
    # If Silver is more overvalued than Gold, shift allocation toward Gold
    adjustment = relative_overval / 100 * adjustment_factor
    
    # Apply adjustment (negative because more overvalued = reduce allocation)
    new_gold_pct = base_gold_pct - adjustment
    new_silver_pct = base_silver_pct + adjustment
    
    # Ensure we don't go to extremes (keep reasonable bounds)
    new_gold_pct = max(0.30, min(0.80, new_gold_pct))  # Gold: 30-80% of metals
    new_silver_pct = max(0.20, min(0.70, new_silver_pct))  # Silver: 20-70% of metals
    
    # Normalize to ensure they sum to 1
    total_pct = new_gold_pct + new_silver_pct
    new_gold_pct /= total_pct
    new_silver_pct /= total_pct
    
    # Calculate final allocations
    gold_allocation = total_metals_allocation * new_gold_pct
    silver_allocation = total_metals_allocation * new_silver_pct
    
    print(f"\nMETALS INTERNAL ALLOCATION:")
    print(f"Gold: {gold_allocation:.1f}% ({new_gold_pct:.1%} of metals)")
    print(f"Silver: {silver_allocation:.1f}% ({new_silver_pct:.1%} of metals)")
    print(f"Relative adjustment: {adjustment:+.3f} ({relative_overval:+.1f}% Gold vs Silver)")
    
    return {'Gold': gold_allocation, 'Silver': silver_allocation}

def calculate_allocation_changes(base_allocation, adjusted_allocation):
    """Calculate the changes in allocation"""
    
    changes = {}
    for asset in base_allocation:
        change = adjusted_allocation[asset] - base_allocation[asset]
        changes[asset] = change
    
    return changes

def save_allocation_results(base_allocation, adjusted_allocation, changes, overvaluation_data):
    """Save the allocation results to a file"""
    
    # Use a fixed filename to replace the old results each time
    results_file = "Portfolio/asset_allocation_results_latest.txt"
    
    with open(results_file, 'w') as f:
        f.write("ASSET ALLOCATION CALCULATOR RESULTS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("OVERVALUATION DATA:\n")
        f.write("-" * 20 + "\n")
        for asset, overval_pct in overvaluation_data.items():
            f.write(f"{asset}: {overval_pct:+.1f}% overvalued\n")
        f.write("\n")
        
        f.write("ALLOCATION COMPARISON:\n")
        f.write("-" * 25 + "\n")
        f.write(f"{'Asset':<20} {'Base':<8} {'Adjusted':<10} {'Change':<8}\n")
        f.write("-" * 50 + "\n")
        
        for asset in base_allocation:
            base_pct = base_allocation[asset]
            adj_pct = adjusted_allocation[asset]
            change = changes[asset]
            change_str = f"{change:+.1f}" if change != 0 else "0.0"
            
            f.write(f"{asset:<20} {base_pct:<8.1f} {adj_pct:<10.1f} {change_str:<8}\n")
        
        f.write("\n")
        f.write("RECOMMENDATIONS:\n")
        f.write("-" * 15 + "\n")
        
        # Sort changes by magnitude
        sorted_changes = sorted(changes.items(), key=lambda x: abs(x[1]), reverse=True)
        
        for asset, change in sorted_changes:
            if abs(change) > 0.5:  # Only show significant changes
                if change > 0:
                    f.write(f"• Increase {asset} allocation by {change:.1f}%\n")
                else:
                    f.write(f"• Decrease {asset} allocation by {abs(change):.1f}%\n")
        
        f.write("\n")
        f.write("RATIONALE:\n")
        f.write("-" * 10 + "\n")
        f.write("• More overvalued assets get reduced allocation\n")
        f.write("• Less overvalued assets get increased allocation\n")
        f.write("• Metals are treated as a unified asset class for portfolio rebalancing\n")
        f.write("• Within metals, aggressive rebalancing between Gold and Silver based on relative overvaluation\n")
        f.write("• Roth IRA is treated as neutral and can absorb rebalancing\n")
        f.write("• HYS has dynamic stress response based on total market overvaluation\n")
        f.write("• Conservative bounds prevent extreme allocation swings\n")
    
    print(f"\nResults saved to: {results_file}")

def main():
    """Main function to run the asset allocation calculator"""
    
    print("ASSET ALLOCATION CALCULATOR")
    print("=" * 40)
    
    # Load overvaluation data
    overvaluation_data = load_overvaluation_results()
    if not overvaluation_data:
        return
    
    # Define base allocation (metals as unified asset class)
    base_allocation = {
        'Bitcoin': 30.0,
        'Metals': 36.0,  # Combined Gold + Silver (24% + 12%)
        'Roth IRA': 24.0,
        'High-Yield Savings': 10.0
    }
    
    # Calculate optimal allocation (metals as unified)
    adjusted_allocation, adjustment_factors = calculate_optimal_allocation(overvaluation_data, base_allocation)
    
    # Calculate internal metals allocation
    metals_breakdown = calculate_metals_allocation(overvaluation_data, adjusted_allocation['Metals'])
    
    # Create final allocation with individual metals
    final_allocation = {
        'Bitcoin': adjusted_allocation['Bitcoin'],
        'Gold': metals_breakdown['Gold'],
        'Silver': metals_breakdown['Silver'],
        'Roth IRA': adjusted_allocation['Roth IRA'],
        'High-Yield Savings': adjusted_allocation['High-Yield Savings']
    }
    
    # Calculate changes against the original individual metals allocation
    original_metals_allocation = {
        'Bitcoin': 30.0,
        'Gold': 24.0,
        'Silver': 12.0,
        'Roth IRA': 24.0,
        'High-Yield Savings': 10.0
    }
    
    changes = calculate_allocation_changes(original_metals_allocation, final_allocation)
    
    # Display results
    print(f"\nOPTIMAL ALLOCATION:")
    print("-" * 20)
    for asset, allocation in final_allocation.items():
        change = changes[asset]
        change_str = f"({change:+.1f})" if change != 0 else ""
        print(f"{asset}: {allocation:.1f}% {change_str}")
    
    print(f"\nTOTAL: {sum(final_allocation.values()):.1f}%")
    
    # Show significant changes
    print(f"\nKEY CHANGES:")
    print("-" * 12)
    significant_changes = [(asset, change) for asset, change in changes.items() if abs(change) > 0.5]
    significant_changes.sort(key=lambda x: abs(x[1]), reverse=True)
    
    for asset, change in significant_changes:
        if change > 0:
            print(f"• Increase {asset}: +{change:.1f}%")
        else:
            print(f"• Decrease {asset}: {change:.1f}%")
    
    # Save results
    save_allocation_results(original_metals_allocation, final_allocation, changes, overvaluation_data)
    
    print(f"\n✅ Asset allocation analysis completed!")
    print(f"📄 Results saved to: Portfolio/asset_allocation_results_latest.txt")

if __name__ == "__main__":
    main() 
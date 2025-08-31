"""
ASSET ALLOCATION CALCULATOR - Updated Version

This script calculates optimal asset allocation based on overvaluation percentages.
It uses a two-tier rebalancing system:
1. Portfolio-level: Bitcoin, Metals (combined), and Stocks
2. Internal metals: Aggressive rebalancing between Gold and Silver

CHANGES MADE:
- Updated allocation percentages: Bitcoin 35%, Gold 28%, Silver 12%, Stocks 25%
- Replaced Roth IRA and HYS with Stocks
- Added easy-to-modify configuration section at the top
- Kept combined metals logic for portfolio rebalancing
- Maintained aggressive internal metals rebalancing

USAGE:
- Modify TARGET_ALLOCATION percentages in the configuration section
- Run script to get optimal allocation based on current overvaluation
- Results saved to: Portfolio/asset_allocation_results_latest.txt
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

# =============================================================================
# CONFIGURATION SECTION - EASY TO MODIFY
# =============================================================================
# Change these percentages to adjust your target allocation
# Make sure they sum to 100%!

TARGET_ALLOCATION = {
    'Bitcoin': 35.0,    # Bitcoin allocation percentage
    'Gold': 28.0,       # Gold allocation percentage  
    'Silver': 12.0,     # Silver allocation percentage
    'Stocks': 25.0      # Stocks allocation percentage
}

# Metals configuration (for internal rebalancing)
METALS_TOTAL = TARGET_ALLOCATION['Gold'] + TARGET_ALLOCATION['Silver']  # Should be 40.0%
GOLD_WEIGHT = TARGET_ALLOCATION['Gold'] / METALS_TOTAL  # Should be 0.7 (70%)
SILVER_WEIGHT = TARGET_ALLOCATION['Silver'] / METALS_TOTAL  # Should be 0.3 (30%)

# Portfolio-level bounds (min/max percentages for each asset class)
ALLOCATION_BOUNDS = {
    'Bitcoin': (10.0, 55.0),      # Bitcoin: 10-55%
    'Metals': (15.0, 50.0),       # Combined metals: 15-50%
    'Stocks': (5.0, 40.0),        # Stocks: 5-40%
}

# Internal metals bounds (Gold/Silver as percentage of total metals)
METALS_INTERNAL_BOUNDS = {
    'Gold': (30.0, 80.0),         # Gold: 30-80% of metals
    'Silver': (20.0, 70.0),       # Silver: 20-70% of metals
}

# Adjustment factors (how aggressive rebalancing is)
PORTFOLIO_ADJUSTMENT_FACTOR = 0.8  # 80% adjustment for portfolio-level rebalancing
METALS_ADJUSTMENT_FACTOR = 0.9     # 90% adjustment for internal metals rebalancing

# =============================================================================

def load_overvaluation_results():
    """Load the most recent overvaluation results"""
    
    # Look for the latest overvaluation results file
    results_dir = "Portfolio/Overvaluedness"
    results_file = os.path.join(results_dir, "overvaluation_results_latest.txt")
    
    if not os.path.exists(results_file):
        print(f"No overvaluation results found at: {results_file}")
        print("Please run the overvalued calculator first.")
        return None
    
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
    6. Stocks are treated as neutral (can absorb rebalancing)
    7. All assets move independently based on their overvaluation vs. average
    8. Conservative adjustment limits prevent extreme allocation swings
    9. Keep total allocation at 100%
    """
    
    # Base allocation (your current target ratios)
    if base_allocation is None:
        base_allocation = {
            'Bitcoin': TARGET_ALLOCATION['Bitcoin'],
            'Metals': METALS_TOTAL,  # Combined Gold + Silver (28% + 12% = 40%)
            'Stocks': TARGET_ALLOCATION['Stocks'],
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
        # Weight by base allocation proportions (Gold: 28/40 = 70%, Silver: 12/40 = 30%)
        gold_weight = GOLD_WEIGHT
        silver_weight = SILVER_WEIGHT
        metals_overval = (overvaluation_data['Gold'] * gold_weight + 
                         overvaluation_data['Silver'] * silver_weight)
        extended_overvaluation['Metals'] = metals_overval
        
        print(f"\nMETALS ANALYSIS:")
        print(f"Gold: {overvaluation_data['Gold']:+.1f}% overvalued (weight: {gold_weight:.1%})")
        print(f"Silver: {overvaluation_data['Silver']:+.1f}% overvalued (weight: {silver_weight:.1%})")
        print(f"Combined Metals: {metals_overval:+.1f}% overvalued")
    
    # For Stocks, assign neutral overvaluation
    extended_overvaluation['Stocks'] = 0.0  # Neutral - can absorb rebalancing
    
    print(f"\nOVERVALUATION ANALYSIS:")
    print("-" * 25)
    for asset, overval_pct in extended_overvaluation.items():
        print(f"{asset}: {overval_pct:+.1f}% overvalued")
    
    # Calculate adjustment factors
    # More overvalued = reduce allocation
    # Less overvalued = increase allocation
    adjustment_factors = {}
    
    # Calculate average overvaluation (only for assets that exist in base allocation)
    portfolio_assets = ['Bitcoin', 'Metals', 'Stocks']
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
            adjustment = adjustment_factors[asset] * base_allocation[asset] * PORTFOLIO_ADJUSTMENT_FACTOR
            
            new_allocation = base_allocation[asset] + adjustment
            
            # Set reasonable bounds for each asset using configuration
            if asset == 'Bitcoin':
                new_allocation = max(ALLOCATION_BOUNDS['Bitcoin'][0], min(ALLOCATION_BOUNDS['Bitcoin'][1], new_allocation))
            elif asset == 'Metals':
                new_allocation = max(ALLOCATION_BOUNDS['Metals'][0], min(ALLOCATION_BOUNDS['Metals'][1], new_allocation))
            elif asset == 'Stocks':
                new_allocation = max(ALLOCATION_BOUNDS['Stocks'][0], min(ALLOCATION_BOUNDS['Stocks'][1], new_allocation))
            
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
        gold_allocation = total_metals_allocation * GOLD_WEIGHT
        silver_allocation = total_metals_allocation * SILVER_WEIGHT
        return {'Gold': gold_allocation, 'Silver': silver_allocation}
    
    gold_overval = overvaluation_data['Gold']
    silver_overval = overvaluation_data['Silver']
    
    # Calculate relative overvaluation between metals
    relative_overval = gold_overval - silver_overval
    
    # Base proportions (Gold: 70%, Silver: 30%)
    base_gold_pct = GOLD_WEIGHT
    base_silver_pct = SILVER_WEIGHT
    
    # Aggressive adjustment factor for metals-to-metals rebalancing
    # This can be more aggressive since we're staying within the same asset class
    adjustment_factor = METALS_ADJUSTMENT_FACTOR
    
    # If Gold is more overvalued than Silver, shift allocation toward Silver
    # If Silver is more overvalued than Gold, shift allocation toward Gold
    adjustment = relative_overval / 100 * adjustment_factor
    
    # Apply adjustment (negative because more overvalued = reduce allocation)
    new_gold_pct = base_gold_pct - adjustment
    new_silver_pct = base_silver_pct + adjustment
    
    # Ensure we don't go to extremes (keep reasonable bounds)
    new_gold_pct = max(METALS_INTERNAL_BOUNDS['Gold'][0]/100, min(METALS_INTERNAL_BOUNDS['Gold'][1]/100, new_gold_pct))
    new_silver_pct = max(METALS_INTERNAL_BOUNDS['Silver'][0]/100, min(METALS_INTERNAL_BOUNDS['Silver'][1]/100, new_silver_pct))
    
    # Normalize to ensure they sum to 1
    total_pct = new_gold_pct + new_silver_pct
    new_gold_pct /= total_pct
    new_silver_pct /= total_pct
    
    # Calculate final allocations
    gold_allocation = total_metals_allocation * new_gold_pct
    silver_allocation = total_metals_allocation * new_silver_pct
    
    print(f"\nMETALS INTERNAL ALLOCATION:")
    print(f"Gold: {gold_allocation:.2f}% ({new_gold_pct:.1%} of metals)")
    print(f"Silver: {silver_allocation:.2f}% ({new_silver_pct:.1%} of metals)")
    print(f"Relative adjustment: {adjustment:+.3f} ({relative_overval:+.1f}% Gold vs Silver)")
    
    return {'Gold': gold_allocation, 'Silver': silver_allocation}

def normalize_allocations(allocations):
    """
    Normalize allocations to ensure they sum to exactly 100.00%
    Uses 2 decimal places and adjusts the largest asset to make up any difference
    """
    # Round all allocations to 2 decimal places
    rounded_allocations = {}
    for asset, allocation in allocations.items():
        rounded_allocations[asset] = round(allocation, 2)
    
    # Calculate total and difference from 100%
    total = sum(rounded_allocations.values())
    difference = 100.00 - total
    
    if abs(difference) > 0.01:  # If difference is more than 0.01%
        # Find the largest asset to adjust
        largest_asset = max(rounded_allocations.items(), key=lambda x: x[1])[0]
        
        # Adjust the largest asset to make total exactly 100.00%
        rounded_allocations[largest_asset] = round(rounded_allocations[largest_asset] + difference, 2)
        
        print(f"  Adjusted {largest_asset} by {difference:+.2f}% to ensure total = 100.00%")
    
    return rounded_allocations

def calculate_allocation_changes(base_allocation, adjusted_allocation):
    """Calculate the changes in allocation"""
    
    changes = {}
    for asset in base_allocation:
        change = adjusted_allocation[asset] - base_allocation[asset]
        changes[asset] = round(change, 2)
    
    return changes

def save_allocation_results(base_allocation, adjusted_allocation, changes, overvaluation_data):
    """Save the allocation results to a file"""
    
    # Use a fixed filename to replace the old results each time
    results_file = "Portfolio/Asset_Allocation/asset_allocation_results_latest.txt"
    
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
            change_str = f"{change:+.2f}" if change != 0 else "0.00"
            
            f.write(f"{asset:<20} {base_pct:<8.2f} {adj_pct:<10.2f} {change_str:<8}\n")
        
        # Add total verification
        total_adjusted = sum(adjusted_allocation.values())
        f.write("-" * 50 + "\n")
        f.write(f"{'TOTAL':<20} {'100.00':<8} {total_adjusted:<10.2f} {'0.00':<8}\n")
        
        f.write("\n")
        f.write("RECOMMENDATIONS:\n")
        f.write("-" * 15 + "\n")
        
        # Sort changes by magnitude
        sorted_changes = sorted(changes.items(), key=lambda x: abs(x[1]), reverse=True)
        
        for asset, change in sorted_changes:
            if abs(change) > 0.5:  # Only show significant changes
                if change > 0:
                    f.write(f"• Increase {asset} allocation by {change:.2f}%\n")
                else:
                    f.write(f"• Decrease {asset} allocation by {abs(change):.2f}%\n")
        
        f.write("\n")
        f.write("RATIONALE:\n")
        f.write("-" * 10 + "\n")
        f.write("• More overvalued assets get reduced allocation\n")
        f.write("• Less overvalued assets get increased allocation\n")
        f.write("• Metals are treated as a unified asset class for portfolio rebalancing\n")
        f.write("• Within metals, aggressive rebalancing between Gold and Silver based on relative overvaluation\n")
        f.write("• Stocks are treated as neutral and can absorb rebalancing\n")
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
        'Bitcoin': TARGET_ALLOCATION['Bitcoin'],
        'Metals': METALS_TOTAL,  # Combined Gold + Silver (28% + 12% = 40%)
        'Stocks': TARGET_ALLOCATION['Stocks']
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
        'Stocks': adjusted_allocation['Stocks']
    }
    
    # Normalize allocations to ensure they sum to exactly 100.00%
    print(f"\nNORMALIZING ALLOCATIONS:")
    print("-" * 25)
    final_allocation = normalize_allocations(final_allocation)
    
    # Calculate changes against the original individual metals allocation
    original_metals_allocation = {
        'Bitcoin': TARGET_ALLOCATION['Bitcoin'],
        'Gold': TARGET_ALLOCATION['Gold'],
        'Silver': TARGET_ALLOCATION['Silver'],
        'Stocks': TARGET_ALLOCATION['Stocks']
    }
    
    changes = calculate_allocation_changes(original_metals_allocation, final_allocation)
    
    # Display results with 2 decimal places
    print(f"\nOPTIMAL ALLOCATION:")
    print("-" * 20)
    for asset, allocation in final_allocation.items():
        change = changes[asset]
        change_str = f"({change:+.2f})" if change != 0 else ""
        print(f"{asset}: {allocation:.2f}% {change_str}")
    
    print(f"\nTOTAL: {sum(final_allocation.values()):.2f}%")
    
    # Show significant changes
    print(f"\nKEY CHANGES:")
    print("-" * 12)
    significant_changes = [(asset, change) for asset, change in changes.items() if abs(change) > 0.5]
    significant_changes.sort(key=lambda x: abs(x[1]), reverse=True)
    
    for asset, change in significant_changes:
        if change > 0:
            print(f"• Increase {asset}: +{change:.2f}%")
        else:
            print(f"• Decrease {asset}: {change:.2f}%")
    
    # Save results
    save_allocation_results(original_metals_allocation, final_allocation, changes, overvaluation_data)
    
    print(f"\n✅ Asset allocation analysis completed!")
    print(f"📄 Results saved to: Portfolio/Asset_Allocation/asset_allocation_results_latest.txt")

if __name__ == "__main__":
    main() 
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
    2. Cash stays fixed at 6%
    3. Roth IRA & HYS have dynamic stress indicators based on total overvaluation
    4. All other assets move independently based on overvaluation
    5. More overvalued = reduce allocation
    6. Less overvalued = increase allocation
    7. Drawdown-aware safeguards prevent overshooting into crashing assets
    8. Keep total allocation at 100%
    """
    
    # Base allocation (your current target ratios)
    if base_allocation is None:
        base_allocation = {
            'Bitcoin': 30.0,
            'Gold': 22.0,
            'Silver': 12.0,
            'Roth IRA': 20.0,
            'High-Yield Savings': 10.0,
            'Cash': 6.0
        }
    
    print("BASE ALLOCATION:")
    print("-" * 20)
    for asset, allocation in base_allocation.items():
        print(f"{asset}: {allocation:.1f}%")
    
    # Add Roth IRA and HYS to overvaluation data with dynamic stress indicators
    extended_overvaluation = overvaluation_data.copy()
    
    # Calculate stress indicators based on total average overvaluation
    avg_overval = np.mean(list(overvaluation_data.values()))
    
    # Dynamic stress response: more overvaluation = more aggressive safe haven allocation
    if avg_overval > 30:
        roth_stress = 0.3  # 30% stress multiplier
        hys_stress = 0.3
    elif avg_overval > 20:
        roth_stress = 0.2  # 20% stress multiplier
        hys_stress = 0.2
    elif avg_overval > 10:
        roth_stress = 0.1  # 10% stress multiplier
        hys_stress = 0.1
    else:
        roth_stress = 0.0  # No stress
        hys_stress = 0.0
    
    extended_overvaluation['Roth IRA'] = -roth_stress * 100  # Negative = undervalued
    extended_overvaluation['High-Yield Savings'] = -hys_stress * 100
    
    print(f"\nOVERVALUATION ANALYSIS:")
    print("-" * 25)
    for asset, overval_pct in extended_overvaluation.items():
        print(f"{asset}: {overval_pct:+.1f}% overvalued")
    
    # Calculate adjustment factors
    # More overvalued = reduce allocation
    # Less overvalued = increase allocation
    adjustment_factors = {}
    
    # Calculate average overvaluation (excluding Cash)
    adjustable_assets = [k for k in extended_overvaluation.keys() if k != 'Cash']
    avg_overval = np.mean([extended_overvaluation[asset] for asset in adjustable_assets])
    
    print(f"\nAverage overvaluation (excluding Cash): {avg_overval:+.1f}%")
    
    # Calculate relative adjustments for all adjustable assets
    for asset in adjustable_assets:
        overval_pct = extended_overvaluation[asset]
        # Relative to average: if asset is more overvalued than average, reduce allocation
        relative_overval = overval_pct - avg_overval
        adjustment_factors[asset] = -relative_overval / 100  # Convert to decimal
    
    print(f"\nADJUSTMENT FACTORS:")
    print("-" * 20)
    for asset, factor in adjustment_factors.items():
        print(f"{asset}: {factor:+.3f} ({factor*100:+.1f}%)")
    
    # Apply adjustments to all assets except Cash
    adjusted_allocation = base_allocation.copy()
    
    # Cash stays fixed
    print(f"\nCash stays fixed at: {base_allocation['Cash']:.1f}%")
    
    # Adjust all other assets based on overvaluation
    for asset in adjustable_assets:
        if asset in adjustment_factors:
            # Calculate adjustment based on base allocation (proportional to each asset's size)
            adjustment = adjustment_factors[asset] * base_allocation[asset] * 0.8  # Limit to 80% of base allocation
            
            new_allocation = base_allocation[asset] + adjustment
            
            # Set reasonable bounds for each asset
            if asset == 'Bitcoin':
                new_allocation = max(15.0, min(45.0, new_allocation))  # 15-45%
            elif asset in ['Gold', 'Silver']:
                new_allocation = max(5.0, min(35.0, new_allocation))   # 5-35%
            elif asset in ['Roth IRA', 'High-Yield Savings']:
                new_allocation = max(5.0, min(30.0, new_allocation))   # 5-30%
            
            adjusted_allocation[asset] = new_allocation
    
    # Rebalance to ensure total = 100%
    total_allocation = sum(adjusted_allocation.values())
    if abs(total_allocation - 100.0) > 0.1:
        # Scale all assets except Cash proportionally
        non_cash_total = total_allocation - adjusted_allocation['Cash']
        scale_factor = (100.0 - adjusted_allocation['Cash']) / non_cash_total
        
        for asset in adjusted_allocation:
            if asset != 'Cash':
                adjusted_allocation[asset] *= scale_factor
    
    return adjusted_allocation, adjustment_factors

def calculate_allocation_changes(base_allocation, adjusted_allocation):
    """Calculate the changes in allocation"""
    
    changes = {}
    for asset in base_allocation:
        change = adjusted_allocation[asset] - base_allocation[asset]
        changes[asset] = change
    
    return changes

def save_allocation_results(base_allocation, adjusted_allocation, changes, overvaluation_data):
    """Save the allocation results to a file"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"Portfolio/asset_allocation_results_{timestamp}.txt"
    
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
        f.write("• Cash stays fixed at 6% (never changes)\n")
        f.write("• Roth IRA & HYS have dynamic stress indicators based on total overvaluation\n")
        f.write("• All assets move independently based on relative overvaluation\n")
        f.write("• Drawdown-aware safeguards prevent overshooting into crashing assets\n")
    
    print(f"\nResults saved to: {results_file}")

def main():
    """Main function to run the asset allocation calculator"""
    
    print("ASSET ALLOCATION CALCULATOR")
    print("=" * 40)
    
    # Load overvaluation data
    overvaluation_data = load_overvaluation_results()
    if not overvaluation_data:
        return
    
    # Define base allocation
    base_allocation = {
        'Bitcoin': 30.0,
        'Gold': 22.0,
        'Silver': 12.0,
        'Roth IRA': 20.0,
        'High-Yield Savings': 10.0,
        'Cash': 6.0
    }
    
    # Calculate optimal allocation
    adjusted_allocation, adjustment_factors = calculate_optimal_allocation(overvaluation_data, base_allocation)
    
    # Calculate changes
    changes = calculate_allocation_changes(base_allocation, adjusted_allocation)
    
    # Display results
    print(f"\nOPTIMAL ALLOCATION:")
    print("-" * 20)
    for asset, allocation in adjusted_allocation.items():
        change = changes[asset]
        change_str = f"({change:+.1f})" if change != 0 else ""
        print(f"{asset}: {allocation:.1f}% {change_str}")
    
    print(f"\nTOTAL: {sum(adjusted_allocation.values()):.1f}%")
    
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
    save_allocation_results(base_allocation, adjusted_allocation, changes, overvaluation_data)
    
    print(f"\n✅ Asset allocation analysis completed!")
    print(f"📄 Results saved to: Portfolio/asset_allocation_results_*.txt")

if __name__ == "__main__":
    main() 
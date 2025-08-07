import pandas as pd
import numpy as np
from datetime import datetime
import os

def load_overvaluation_results():
    """Load the most recent overvaluation results"""
    
    # Look for the most recent overvaluation results file
    results_dir = "Portfolio"
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
    for line in lines:
        if 'Overvaluation:' in line:
            # Find the asset name and percentage
            if 'BITCOIN' in line or 'Bitcoin' in line:
                pct = float(line.split('Overvaluation:')[1].split('%')[0].strip())
                overvaluation_data['Bitcoin'] = pct
            elif 'GOLD' in line or 'Gold' in line:
                pct = float(line.split('Overvaluation:')[1].split('%')[0].strip())
                overvaluation_data['Gold'] = pct
            elif 'SILVER' in line or 'Silver' in line:
                pct = float(line.split('Overvaluation:')[1].split('%')[0].strip())
                overvaluation_data['Silver'] = pct
    
    return overvaluation_data

def calculate_optimal_allocation(overvaluation_data, base_allocation=None):
    """
    Calculate optimal asset allocation based on overvaluation percentages.
    
    Strategy:
    1. Start with base allocation
    2. Adjust based on relative overvaluation
    3. More overvalued = reduce allocation
    4. Less overvalued = increase allocation
    5. Keep total allocation at 100%
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
    
    print(f"\nOVERVALUATION ANALYSIS:")
    print("-" * 25)
    for asset, overval_pct in overvaluation_data.items():
        print(f"{asset}: {overval_pct:+.1f}% overvalued")
    
    # Calculate adjustment factors
    # More overvalued = reduce allocation
    # Less overvalued = increase allocation
    adjustment_factors = {}
    
    # Calculate average overvaluation
    avg_overval = np.mean(list(overvaluation_data.values()))
    
    print(f"\nAverage overvaluation: {avg_overval:+.1f}%")
    
    # Calculate relative adjustments
    for asset, overval_pct in overvaluation_data.items():
        # Relative to average: if asset is more overvalued than average, reduce allocation
        relative_overval = overval_pct - avg_overval
        adjustment_factors[asset] = -relative_overval / 100  # Convert to decimal
    
    print(f"\nADJUSTMENT FACTORS:")
    print("-" * 20)
    for asset, factor in adjustment_factors.items():
        print(f"{asset}: {factor:+.3f} ({factor*100:+.1f}%)")
    
    # Apply adjustments to metals only (keep other allocations fixed)
    adjusted_allocation = base_allocation.copy()
    
    # Calculate total metals allocation
    metals_allocation = base_allocation['Gold'] + base_allocation['Silver']
    
    # Adjust metals based on overvaluation
    for asset in ['Gold', 'Silver']:
        if asset in adjustment_factors:
            # Calculate new allocation within metals
            base_metal_ratio = base_allocation[asset] / metals_allocation
            adjustment = adjustment_factors[asset] * metals_allocation * 0.5  # Limit adjustment to 50% of metals allocation
            
            new_allocation = base_allocation[asset] + adjustment
            new_allocation = max(5.0, min(35.0, new_allocation))  # Keep between 5% and 35%
            adjusted_allocation[asset] = new_allocation
    
    # Rebalance metals to maintain total metals allocation
    total_metals_adjusted = adjusted_allocation['Gold'] + adjusted_allocation['Silver']
    if total_metals_adjusted != metals_allocation:
        # Scale back proportionally
        scale_factor = metals_allocation / total_metals_adjusted
        adjusted_allocation['Gold'] *= scale_factor
        adjusted_allocation['Silver'] *= scale_factor
    
    # Bitcoin gets special treatment - more aggressive adjustment
    if 'Bitcoin' in adjustment_factors:
        bitcoin_adjustment = adjustment_factors['Bitcoin'] * base_allocation['Bitcoin'] * 0.8
        new_bitcoin = base_allocation['Bitcoin'] + bitcoin_adjustment
        new_bitcoin = max(15.0, min(45.0, new_bitcoin))  # Keep between 15% and 45%
        adjusted_allocation['Bitcoin'] = new_bitcoin
    
    # Rebalance to ensure total = 100%
    total_allocation = sum(adjusted_allocation.values())
    if abs(total_allocation - 100.0) > 0.1:
        scale_factor = 100.0 / total_allocation
        for asset in adjusted_allocation:
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
        f.write("• Bitcoin gets more aggressive adjustment due to volatility\n")
        f.write("• Metals are rebalanced together to maintain stability\n")
        f.write("• Other allocations (IRA, Savings, Cash) remain fixed\n")
    
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
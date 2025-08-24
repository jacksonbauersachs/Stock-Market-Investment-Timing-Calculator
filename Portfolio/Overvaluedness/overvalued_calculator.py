"""
OVERVALUED CALCULATOR - Updated Version

CHANGES MADE IN THIS VERSION:
================================

1. BITCOIN CUSTOM PRICE INPUT:
   - Added custom price input for Bitcoin (previously only Gold/Silver had this)
   - All assets now support custom price scenarios for "what-if" analysis
   - Bitcoin: Test different price scenarios (e.g., if BTC drops to $50k)
   - Metals: Adjust for premium/discount vs. spot price

2. INPUT FORMAT CLARIFICATION:
   - Clear instructions: Use raw numbers only (no $, no commas)
   - Examples: 50000, 150000.50, 200000 (not $50,000 or $150,000.50)
   - Better error messages with format examples
   - Input help displayed for Bitcoin analysis

3. FILE SAVING FIXES:
   - Fixed save location: Now saves to Portfolio/Overvaluedness/ folder
   - Fixed filename: Uses overvaluation_results_latest.txt
   - File replacement: Each run overwrites previous results (no more timestamped files)

4. USER EXPERIENCE IMPROVEMENTS:
   - Clear input format guidance
   - Better error messages
   - Consistent behavior across all assets
   - Clean file organization

USAGE:
- Run calculator to get current overvaluation analysis
- Input custom prices for any asset to test scenarios
- Results always saved to: Portfolio/Overvaluedness/overvaluation_results_latest.txt
- Each run replaces previous results for clean organization
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import re

def load_models_from_file():
    """Load the best-fitting models from the updated_models.txt file."""
    
    models_file = "Portfolio/Models/updated_models.txt"
    
    if not os.path.exists(models_file):
        print(f"Error: Models file not found - {models_file}")
        return None
    
    models = {}
    
    with open(models_file, 'r') as f:
        content = f.read()
    
    # For Bitcoin, use the original model parameters from bitcoin_growth_model_coefficients_day365.txt
    # This gives us the $107k fair value that matches the GBM simulation
    try:
        with open("Crypto/Bitcoin/Models/Growth/Results/bitcoin_growth_model_coefficients_day365.txt", 'r') as f:
            lines = f.readlines()
            a = float(lines[0].split('=')[1].strip())
            b = float(lines[1].split('=')[1].strip())
        
        models['Bitcoin'] = {
            'type': 'Rainbow Chart',
            'slope': a,
            'intercept': b,
            'formula': f"log10(price) = {a} * ln(day) + {b}"
        }
    except:
        # Fallback to updated models if original file not found
        bitcoin_match = re.search(r'Rainbow Chart Model:\s*\n\s*Formula: log10\(price\) = ([\d.-]+) \* ln\(day\) \+ ([\d.-]+)', content)
        if bitcoin_match:
            models['Bitcoin'] = {
                'type': 'Rainbow Chart',
                'slope': float(bitcoin_match.group(1)),
                'intercept': float(bitcoin_match.group(2)),
                'formula': f"log10(price) = {bitcoin_match.group(1)} * ln(day) + {bitcoin_match.group(2)}"
            }
    
    # Extract Gold best model
    gold_match = re.search(r'Best Gold Model: (\w+)\s*\n\s*Formula: (.+?)\s*\n', content)
    if gold_match:
        model_type = gold_match.group(1)
        formula = gold_match.group(2)
        models['Gold'] = {
            'type': model_type,
            'formula': formula,
            'params': extract_polynomial_params(formula) if model_type == 'Polynomial' else None
        }
    
    # Extract Silver best model
    silver_match = re.search(r'Best Silver Model: (\w+)\s*\n\s*Formula: (.+?)\s*\n', content)
    if silver_match:
        model_type = silver_match.group(1)
        formula = silver_match.group(2)
        models['Silver'] = {
            'type': model_type,
            'formula': formula,
            'params': extract_polynomial_params(formula) if model_type == 'Polynomial' else None
        }
    
    return models

def extract_polynomial_params(formula):
    """Extract polynomial parameters from formula string."""
    # Handle scientific notation and extract coefficients
    # Format: Price = 6.98e-10*Days³ + -8.76e-06*Days² + 0.0451*Days + 250.64
    try:
        # Extract coefficients using regex
        coeffs = re.findall(r'([\d.-]+e?[\d.-]*)', formula)
        if len(coeffs) >= 4:
            return [float(coeffs[0]), float(coeffs[1]), float(coeffs[2]), float(coeffs[3])]
    except:
        pass
    return None

def get_current_prices():
    """Get current prices from the updated CSV files."""
    
    prices = {}
    
    # Bitcoin price
    try:
        btc_df = pd.read_csv("Portfolio/Data/Bitcoin_all_time_price.csv")
        btc_df['Date'] = pd.to_datetime(btc_df['Date'])
        btc_df = btc_df.sort_values('Date')
        prices['Bitcoin'] = float(btc_df['Price'].iloc[-1])
        print(f"Current Bitcoin price: ${prices['Bitcoin']:,.2f}")
    except Exception as e:
        print(f"Error getting Bitcoin price: {e}")
        prices['Bitcoin'] = None
    
    # Gold price
    try:
        gold_df = pd.read_csv("Portfolio/Data/Gold_all_time_price.csv")
        gold_df['Date'] = pd.to_datetime(gold_df['Date'])
        gold_df = gold_df.sort_values('Date')
        # Clean price column
        gold_df['Price'] = gold_df['Price'].astype(str).str.replace('"', '').str.replace(',', '').astype(float)
        prices['Gold'] = float(gold_df['Price'].iloc[-1])
        print(f"Current Gold price: ${prices['Gold']:,.2f}")
    except Exception as e:
        print(f"Error getting Gold price: {e}")
        prices['Gold'] = None
    
    # Silver price
    try:
        silver_df = pd.read_csv("Portfolio/Data/Silver_all_time_price.csv")
        silver_df['Date'] = pd.to_datetime(silver_df['Date'])
        silver_df = silver_df.sort_values('Date')
        prices['Silver'] = float(silver_df['Price'].iloc[-1])
        print(f"Current Silver price: ${prices['Silver']:.3f}")
    except Exception as e:
        print(f"Error getting Silver price: {e}")
        prices['Silver'] = None
    
    return prices

def calculate_fair_value(asset, model, days_since_start):
    """Calculate fair value based on the growth model."""
    
    if asset == 'Bitcoin':
        # Bitcoin Rainbow Chart model: log10(price) = slope * ln(day) + intercept
        if model['type'] == 'Rainbow Chart':
            log_price = model['slope'] * np.log(days_since_start) + model['intercept']
            fair_value = 10 ** log_price
            return fair_value
    
    elif asset in ['Gold', 'Silver']:
        if model['type'] == 'Polynomial' and model['params']:
            # Polynomial model: ax³ + bx² + cx + d
            a, b, c, d = model['params']
            fair_value = a * (days_since_start ** 3) + b * (days_since_start ** 2) + c * days_since_start + d
            return fair_value
        elif model['type'] == 'Linear':
            # Extract linear parameters from formula
            match = re.search(r'Price = ([\d.-]+) \* Days \+ ([\d.-]+)', model['formula'])
            if match:
                slope = float(match.group(1))
                intercept = float(match.group(2))
                fair_value = slope * days_since_start + intercept
                return fair_value
        elif model['type'] == 'Exponential':
            # Extract exponential parameters from formula
            match = re.search(r'Price = ([\d.-]+) \* exp\(([\d.-]+) \* Days\) \+ ([\d.-]+)', model['formula'])
            if match:
                a = float(match.group(1))
                b = float(match.group(2))
                c = float(match.group(3))
                fair_value = a * np.exp(b * days_since_start) + c
                return fair_value
    
    return None

def get_days_since_start(asset):
    """Get the number of days since the start date for each asset."""
    
    start_dates = {
        'Bitcoin': '2010-07-18',
        'Gold': '1975-01-03',
        'Silver': '1970-02-04'
    }
    
    start_date = datetime.strptime(start_dates[asset], '%Y-%m-%d')
    today = datetime.now()
    
    if asset == 'Bitcoin':
        # Bitcoin Rainbow Chart model: days start at 1, skip first 364 days
        # Formula: (current_date - start_date).days + 1
        days = (today - start_date).days + 1
        # The model skips the first 364 days, so we need to ensure we're at least at day 365
        if days < 365:
            days = 365
    else:
        days = (today - start_date).days
    
    return days

def get_premium_adjusted_price(asset):
    """
    Get custom price from user input or use current market price.
    
    CHANGES MADE:
    - Added custom price input for Bitcoin (previously only Gold/Silver had this feature)
    - All assets now support custom price scenarios for "what-if" analysis
    - Bitcoin: Test different price scenarios (e.g., if BTC drops to $50k, what's overvaluation?)
    - Metals: Adjust for premium/discount vs. spot price
    
    INPUT FORMAT:
    - Type 'no' to use current market price from CSV files
    - Type raw number (no $, no commas): 50000, 150000.50, 200000
    - DO NOT include: $, commas, or currency formatting
    - Examples: 113034.04 (not $113,034.04), 150000 (not $150,000)
    """
    
    current_prices = get_current_prices()
    current_price = current_prices.get(asset)
    
    if current_price is None:
        print(f"Could not get current {asset} price from file.")
        return None
    
    # Show input format help for first asset
    if asset == 'Bitcoin':
        print(f"\n💡 BITCOIN CUSTOM PRICE INPUT:")
        print(f"   • Type 'no' to use current market price: ${current_price:,.2f}")
        print(f"   • Type raw number (no $, no commas): 50000, 150000.50, 200000")
        print(f"   • Examples: 113034.04, 150000, 200000.50")
    
    while True:
        if asset == 'Bitcoin':
            user_input = input(f"Input custom price for Bitcoin? (current: ${current_price:,.2f}) or type 'no': ").strip()
        elif asset == 'Gold':
            user_input = input(f"Input premium adjusted price for Gold? (current: ${current_price:,.2f}) or type 'no': ").strip()
        elif asset == 'Silver':
            user_input = input(f"Input premium adjusted price for Silver? (current: ${current_price:.3f}) or type 'no': ").strip()
        else:
            return current_price
        
        if user_input.lower() == 'no':
            return current_price
        else:
            try:
                premium_price = float(user_input)
                return premium_price
            except ValueError:
                print("❌ Invalid input format!")
                print("   • Use raw numbers only: 50000, 150000.50")
                print("   • NO dollar signs ($), NO commas (,), NO currency formatting")
                print("   • Examples: 113034.04, 150000, 200000.50")

def calculate_overvaluation():
    """Main function to calculate overvaluation for all assets."""
    
    print("OVERVALUED CALCULATOR")
    print("=" * 40)
    
    # Load models
    print("\nLoading growth models...")
    models = load_models_from_file()
    
    if not models:
        print("Error: Could not load models.")
        return
    
    print("Models loaded successfully!")
    
    # Get current prices
    print("\nGetting current prices...")
    current_prices = get_current_prices()
    
    results = {}
    
    # Calculate for each asset
    for asset in ['Bitcoin', 'Gold', 'Silver']:
        print(f"\n--- {asset} Analysis ---")
        
        if asset not in models:
            print(f"No model found for {asset}")
            continue
        
        if current_prices.get(asset) is None:
            print(f"No current price available for {asset}")
            continue
        
        # Get days since start
        days = get_days_since_start(asset)
        print(f"Days since start: {days:,}")
        
        # Calculate fair value
        fair_value = calculate_fair_value(asset, models[asset], days)
        
        if fair_value is None:
            print(f"Could not calculate fair value for {asset}")
            continue
        
        # Get current price (with custom price input for all assets)
        current_price = get_premium_adjusted_price(asset)
        
        if current_price is None:
            print(f"Could not get current price for {asset}")
            continue
        
        # Calculate overvaluation percentage
        overvaluation_pct = ((current_price - fair_value) / fair_value) * 100
        
        results[asset] = {
            'fair_value': fair_value,
            'current_price': current_price,
            'overvaluation_pct': overvaluation_pct,
            'model_type': models[asset]['type'],
            'formula': models[asset]['formula']
        }
        
        print(f"Fair Value: ${fair_value:,.2f}")
        print(f"Current Price: ${current_price:,.2f}")
        print(f"Overvaluation: {overvaluation_pct:+.2f}%")
    
    # Save results
    save_results(results)
    
    return results

def save_results(results):
    """
    Save results to a file.
    
    CHANGES MADE:
    - Fixed save location: Now saves to Portfolio/Overvaluedness/ folder (was saving to wrong location)
    - Fixed filename: Now uses overvaluation_results_latest.txt (was creating timestamped files)
    - File replacement: Each run overwrites previous results (was accumulating multiple files)
    
    FILE LOCATION: Portfolio/Overvaluedness/overvaluation_results_latest.txt
    BEHAVIOR: Replaces previous file each time calculator runs
    """
    
    # Use a fixed filename to replace the old results each time
    results_file = "Portfolio/Overvaluedness/overvaluation_results_latest.txt"
    
    with open(results_file, 'w') as f:
        f.write("ASSET OVERVALUATION ANALYSIS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for asset, data in results.items():
            f.write(f"{asset.upper()} ANALYSIS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Model Type: {data['model_type']}\n")
            f.write(f"Formula: {data['formula']}\n")
            f.write(f"Fair Value: ${data['fair_value']:,.2f}\n")
            f.write(f"Current Price: ${data['current_price']:,.2f}\n")
            f.write(f"Overvaluation: {data['overvaluation_pct']:+.2f}%\n\n")
        
        # Summary
        f.write("SUMMARY\n")
        f.write("-" * 10 + "\n")
        for asset, data in results.items():
            status = "OVERVALUED" if data['overvaluation_pct'] > 0 else "UNDERVALUED"
            f.write(f"{asset}: {data['overvaluation_pct']:+.2f}% ({status})\n")
    
    print(f"\nResults saved to: {results_file}")
    print("Note: This file replaces the previous overvaluation results each time you run the calculator.")
    
    # Print summary
    print("\n" + "=" * 50)
    print("ANALYSIS SUMMARY")
    print("=" * 50)
    for asset, data in results.items():
        status = "OVERVALUED" if data['overvaluation_pct'] > 0 else "UNDERVALUED"
        print(f"{asset}: {data['overvaluation_pct']:+.2f}% ({status})")

def main():
    """Main function."""
    
    try:
        results = calculate_overvaluation()
        if results:
            print("\n✅ Analysis completed successfully!")
        else:
            print("\n❌ Analysis failed.")
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")

if __name__ == "__main__":
    main() 
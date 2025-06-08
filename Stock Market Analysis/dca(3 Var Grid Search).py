import csv
from datetime import datetime, timedelta
from itertools import product
import numpy as np


def calculate_reserve_strategy(file_path, reserve_pct, thresholds, deploy_ratios):
    """Calculate investment growth with flexible reserve strategy"""
    # Read and prepare data
    with open(file_path, 'r') as file:
        data = list(csv.DictReader(file))
    for row in data:
        row['Date'] = datetime.strptime(row['Date'], '%m/%d/%Y')
        row['Price'] = float(row['Price'])
    data.sort(key=lambda x: x['Date'])

    # Strategy parameters
    initial_biweekly = 1000
    regular_pct = 1 - reserve_pct  # Dynamic regular investment %
    inflation_adj = 0.035
    dividend_yield = 0.013

    # Initialize tracking variables
    current_date = data[0]['Date']
    end_date = data[-1]['Date']
    shares = 0
    total_invested = 0
    reserve = 0
    current_investment = initial_biweekly
    last_adj_date = current_date
    ath = data[0]['Price']

    # Investment loop
    while current_date <= end_date:
        # Find nearest trading day
        price_data = next((d for d in data if d['Date'] >= current_date), None)

        if price_data:
            price = price_data['Price']
            ath = max(ath, price)
            dip = (ath - price) / ath

            # Deploy reserves according to tiers
            deployed = 0
            remaining_reserve = reserve
            for thresh, ratio in zip(sorted(thresholds), deploy_ratios):
                if dip >= thresh and remaining_reserve > 0:
                    deployed += remaining_reserve * min(ratio, 1.0)
                    remaining_reserve *= (1 - min(ratio, 1.0))

            # Make regular investment and update reserve
            regular_investment = current_investment * regular_pct
            reserve = remaining_reserve + (current_investment * reserve_pct)

            # Execute investments
            total_investment = regular_investment + deployed
            shares += total_investment / price
            total_invested += total_investment

            # Reinvest dividends
            shares += shares * (dividend_yield / 26)

        # Move to next period
        current_date += timedelta(days=14)
        if (current_date - last_adj_date).days >= 365:
            current_investment *= (1 + inflation_adj)
            last_adj_date = current_date

    return shares * data[-1]['Price']


def optimize_strategy(file_path):
    """Find optimal reserve percentage and tier parameters"""
    # Parameter ranges (including near-zero values)
    reserve_options = np.linspace(0.01, 0.5, 10)  # 1% to 50% in 10 steps
    threshold_options = np.linspace(0.01, 0.2, 8)  # 1% to 20% in 8 steps
    deploy_options = np.linspace(0.05, 1.0, 8)  # 5% to 100% in 8 steps

    best = {'value': 0}
    tested = 0

    # Test all 3-tier combinations
    for reserve_pct in reserve_options:
        for thresh_combo in product(threshold_options, repeat=3):
            thresh_combo = sorted(thresh_combo)
            if len(set(thresh_combo)) < 3:
                continue  # Skip duplicate thresholds

            for deploy_combo in product(deploy_options, repeat=3):
                if sum(deploy_combo) > 1.0:
                    continue

                tested += 1
                final_value = calculate_reserve_strategy(
                    file_path, reserve_pct, thresh_combo, deploy_combo
                )

                if final_value > best['value']:
                    best = {
                        'reserve_pct': reserve_pct,
                        'thresholds': thresh_combo,
                        'deploy_ratios': deploy_combo,
                        'value': final_value
                    }
                    print(f"Test {tested}: New best → ${final_value:,.2f}")
                    print(f"  Reserve: {reserve_pct:.1%}")
                    print(f"  Thresholds: {[f'{t:.1%}' for t in thresh_combo]}")
                    print(f"  Deploy Ratios: {[f'{r:.1%}' for r in deploy_combo]}")

    return best


if __name__ == "__main__":
    file_path = "../S&P 500 Data Sets/S&P 500 Total Data Cleaned 2.csv"

    print("1. Full optimization (may take hours)")
    print("2. Quick test (reduced parameters)")
    choice = input("Select option: ")

    if choice == "1":
        result = optimize_strategy(file_path)
    elif choice == "2":
        # Quick test with minimal parameters
        reserve_pct = 0.2
        thresholds = [0.05, 0.10, 0.15]
        deploy_ratios = [0.2, 0.3, 0.5]
        final_value = calculate_reserve_strategy(
            file_path, reserve_pct, thresholds, deploy_ratios
        )
        result = {
            'reserve_pct': reserve_pct,
            'thresholds': thresholds,
            'deploy_ratios': deploy_ratios,
            'value': final_value
        }

    print("\nOptimal Strategy:")
    print(f"Reserve Percentage: {result['reserve_pct']:.1%}")
    print(f"Dip Thresholds: {[f'{t:.1%}' for t in result['thresholds']]}")
    print(f"Deploy Ratios: {[f'{r:.1%}' for r in result['deploy_ratios']]}")
    print(f"Final Portfolio Value: ${result['value']:,.2f}")
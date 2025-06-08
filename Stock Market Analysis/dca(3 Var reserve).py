import csv
from datetime import datetime, timedelta


def calculate_tiered_reserve_strategy(file_path):
    # Read the CSV file
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        data = [row for row in reader]

    # Convert data types
    for row in data:
        row['Date'] = datetime.strptime(row['Date'], '%m/%d/%Y')
        row['Price'] = float(row['Price'])

    # Sort data by date
    data.sort(key=lambda x: x['Date'])

    # Investment parameters
    initial_biweekly = 1000  # $1,000 every other week
    regular_investment_ratio = 0.8  # Invest 80% regularly
    reserve_ratio = 0.2  # Hold 20% in reserve
    inflation_adjustment = 0.035  # 3.5% annual increase
    dividend_yield = 0.013  # 1.3% annual dividend yield

    # Tier thresholds and deployment percentages
    tiers = [
        {'threshold': 0.05, 'deploy_ratio': 0.2},  # 5% drop -> 20% of reserve
        {'threshold': 0.10, 'deploy_ratio': 0.3},  # 10% drop -> 30% of reserve
        {'threshold': 0.15, 'deploy_ratio': 0.5}  # 15% drop -> 50% of reserve
    ]

    # Initialize variables
    current_date = data[0]['Date']
    end_date = data[-1]['Date']
    shares_owned = 0
    total_invested = 0
    reserve_fund = 0
    current_biweekly = initial_biweekly
    last_inflation_adjustment = current_date
    all_time_high = data[0]['Price']  # Track ATH

    # Main investment loop
    while current_date <= end_date:
        # Find the closest trading day
        price_data = None
        for day in data:
            if day['Date'] >= current_date:
                price_data = day
                break

        if price_data:
            current_price = price_data['Price']

            # Update all-time high
            if current_price > all_time_high:
                all_time_high = current_price

            # Calculate current price drop from ATH
            price_dip = (all_time_high - current_price) / all_time_high

            # Determine how much reserve to deploy (if any)
            reserve_deployed = 0
            remaining_reserve = reserve_fund

            # Check each tier in order
            for tier in sorted(tiers, key=lambda x: x['threshold']):
                if price_dip >= tier['threshold'] and remaining_reserve > 0:
                    # Calculate amount to deploy from this tier
                    deploy_amount = min(remaining_reserve, reserve_fund * tier['deploy_ratio'])
                    reserve_deployed += deploy_amount
                    remaining_reserve -= deploy_amount

            # Calculate regular investment (80%)
            regular_investment = current_biweekly * regular_investment_ratio

            # Add to reserve (20%)
            reserve_fund = remaining_reserve + (current_biweekly * reserve_ratio)

            # Total investment this period (regular + any reserve deployment)
            total_investment = regular_investment + reserve_deployed

            # Buy shares
            shares_bought = total_investment / current_price
            shares_owned += shares_bought
            total_invested += total_investment

            # Apply dividend (assuming reinvestment)
            dividend_shares = shares_owned * (dividend_yield / 26)  # 26 biweekly periods per year
            shares_owned += dividend_shares

        # Move to next investment date
        current_date += timedelta(days=14)

        # Adjust for inflation annually
        if (current_date - last_inflation_adjustment).days >= 365:
            current_biweekly *= (1 + inflation_adjustment)
            last_inflation_adjustment = current_date

    # Calculate final results
    final_price = data[-1]['Price']
    final_value = shares_owned * final_price
    total_growth = (final_value - total_invested) / total_invested * 100

    return {
        'total_invested': total_invested,
        'final_value': final_value,
        'total_growth_percent': total_growth,
        'shares_owned': shares_owned,
        'reserve_remaining': reserve_fund,
        'all_time_high': all_time_high,
        'final_price': final_price,
        'price_ratio': final_price / all_time_high
    }


# Run the simulation
if __name__ == "__main__":
    file_path = "../S&P 500 Data Sets/S&P 500 Total Data Cleaned 2.csv"
    results = calculate_tiered_reserve_strategy(file_path)

    print("\nTiered Reserve Strategy Results:")
    print(f"Total Invested: ${results['total_invested']:,.2f}")
    print(f"Final Portfolio Value: ${results['final_value']:,.2f}")
    print(f"Total Growth: {results['total_growth_percent']:.2f}%")
    print(f"Total Shares Owned: {results['shares_owned']:,.2f}")
    print(f"Reserve Remaining: ${results['reserve_remaining']:,.2f}")
    print(f"\nMarket Conditions:")
    print(f"All-Time High: ${results['all_time_high']:,.2f}")
    print(f"Final Price: ${results['final_price']:,.2f}")
    print(f"Price Ratio to ATH: {results['price_ratio']:.2%}")
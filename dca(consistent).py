import csv
from datetime import datetime, timedelta


def calculate_investment_growth(file_path):
    # Read the CSV file
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        data = [row for row in reader]

    # Convert date strings to datetime objects and parse numeric values
    for row in data:
        row['Date'] = datetime.strptime(row['Date'], '%m/%d/%Y')
        row['Price'] = float(row['Price'])

    # Sort data by date (just in case)
    data.sort(key=lambda x: x['Date'])

    # Investment parameters
    initial_investment = 1000  # $1,000 every other week
    inflation_adjustment = 0.035  # 3.5% annual increase
    dividend_yield = 0.013  # 1.3% annual dividend yield

    # Initialize variables
    current_date = data[0]['Date']
    end_date = data[-1]['Date']
    shares_owned = 0
    total_invested = 0
    current_biweekly_investment = initial_investment
    last_inflation_adjustment_date = current_date

    # Main investment loop
    while current_date <= end_date:
        # Find the closest date in the data (since markets are closed some days)
        price_data = None
        for day in data:
            if day['Date'] >= current_date:
                price_data = day
                break

        if price_data:
            # Calculate how many shares we can buy with current investment
            price = price_data['Price']
            shares_bought = current_biweekly_investment / price
            shares_owned += shares_bought
            total_invested += current_biweekly_investment

            # Apply dividend (assuming reinvestment)
            dividend_shares = shares_owned * (dividend_yield / 26)  # 26 biweekly periods per year
            shares_owned += dividend_shares

        # Move to next investment date (every other week)
        current_date += timedelta(days=14)

        # Adjust for inflation annually
        if (current_date - last_inflation_adjustment_date).days >= 365:
            current_biweekly_investment *= (1 + inflation_adjustment)
            last_inflation_adjustment_date = current_date

    # Calculate final value
    final_price = data[-1]['Price']
    final_value = shares_owned * final_price
    total_growth = (final_value - total_invested) / total_invested * 100

    return {
        'total_invested': total_invested,
        'final_value': final_value,
        'total_growth_percent': total_growth,
        'shares_owned': shares_owned
    }


# Example usage
if __name__ == "__main__":
    file_path = "S&P 500 Data Sets/S&P 500 Total Data Cleaned 2.csv"  # Replace with your actual file path
    results = calculate_investment_growth(file_path)

    print(f"Total Invested: ${results['total_invested']:,.2f}")
    print(f"Final Portfolio Value: ${results['final_value']:,.2f}")
    print(f"Total Growth: {results['total_growth_percent']:.2f}%")
    print(f"Total Shares Owned: {results['shares_owned']:,.2f}")
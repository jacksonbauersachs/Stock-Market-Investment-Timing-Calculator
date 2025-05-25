import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def calculate_dca_returns(data_path, investment_amount=100):
    # Load and prepare data - ensure no chained assignments
    df = pd.read_csv(data_path, parse_dates=['Date']).sort_values('Date')
    df = df.copy()  # Create explicit copy to avoid warnings
    df['Price'] = pd.to_numeric(df['Price'].astype(str).str.replace(',', ''), errors='coerce')

    # Historical parameters
    dividend_yield = 0.02  # 2% average dividend yield
    inflation_rate = 0.038  # 3.8% average inflation

    # Calculate total return index without warnings
    price_ratio = df['Price'].values[1:] / df['Price'].values[:-1]
    days = (df['Date'].values[1:] - df['Date'].values[:-1]).astype('timedelta64[D]').astype(int)
    daily_div = (1 + dividend_yield) ** (1 / 365.25)

    total_return = np.zeros(len(df))
    total_return[0] = df['Price'].iloc[0]
    for i in range(1, len(df)):
        total_return[i] = total_return[i - 1] * price_ratio[i - 1] * (daily_div ** days[i - 1])

    df['Total_Return'] = total_return

    # Inflation adjustment
    years_elapsed = (df['Date'] - df['Date'].iloc[0]).dt.days / 365.25
    df['Inflation_Factor'] = (1 + inflation_rate) ** years_elapsed

    results = []

    for start_day in range(14):
        current_date = df['Date'].iloc[0] + timedelta(days=start_day)
        shares = 0.0
        total_invested = 0.0

        while current_date <= df['Date'].iloc[-1]:
            mask = df['Date'] >= current_date
            if not mask.any():
                break

            next_row = df[mask].iloc[0]
            real_price = next_row['Total_Return'] / next_row['Inflation_Factor']
            shares += investment_amount / real_price
            total_invested += investment_amount
            current_date += timedelta(days=14)

        if total_invested == 0:
            continue

        final_value = shares * (df['Total_Return'].iloc[-1] / df['Inflation_Factor'].iloc[-1])
        years = (df['Date'].iloc[-1] - df['Date'].iloc[0]).days / 365.25
        cagr = (final_value / total_invested) ** (1 / years) - 1

        results.append({
            'start_day': start_day,
            'real_cagr': cagr * 100,
            'total_real_return': ((final_value - total_invested) / total_invested) * 100,
            'years': years
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    try:
        results = calculate_dca_returns("S&P 500 Data Sets/S&P 500 Total Data Cleaned.csv")

        print("Accurate Inflation-Adjusted DCA Simulation")
        print("-----------------------------------------")
        print(f"Time Period: {results['years'].mean():.1f} years")
        print(f"Average Real CAGR: {results['real_cagr'].mean():.2f}%")
        print(f"Total Real Return: {results['total_real_return'].mean():.0f}%")
        print("\nNote: Real returns = after inflation")

    except Exception as e:
        print(f"Error: {str(e)}")
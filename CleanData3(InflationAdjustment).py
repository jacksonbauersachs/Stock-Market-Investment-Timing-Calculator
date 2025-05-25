import pandas as pd
import numpy as np
from datetime import datetime


def adjust_to_today_dollars(input_file, output_file, annual_inflation=0.035):
    """
    Adjusts historical S&P 500 prices to today's dollars (inflates past values)
    Args:
        input_file: Path to S&P 500 data CSV
        output_file: Path for output CSV
        annual_inflation: Annual inflation rate (default 3.5%)
    """
    # Load data
    df = pd.read_csv(input_file, parse_dates=['Date'])

    # Calculate days from each date to the most recent date
    current_date = df['Date'].max()
    df['Days'] = (current_date - df['Date']).dt.days
    df['Years'] = df['Days'] / 365.25

    # Calculate inflation factor (how much prices have increased since each date)
    df['Inflation_Factor'] = (1 + annual_inflation) ** df['Years']

    # Adjust all price columns (multiply by inflation factor)
    price_cols = ['Price', 'Open', 'High', 'Low']
    for col in price_cols:
        df[f'Today_{col}'] = (df[col] * df['Inflation_Factor']).round(2)

    # Format output (keeping original Change %)
    result = df[['Date'] + [f'Today_{c}' for c in price_cols] + ['Change %']]
    result.columns = ['Date'] + price_cols + ['Change %']  # Remove 'Today_' prefix

    # Save
    result.to_csv(output_file, index=False)
    print(f"Saved today's-dollar adjusted data to {output_file}")
    print(f"Note: All prices are in {current_date.date()} equivalent dollars")
    return result


if __name__ == "__main__":
    # Example usage
    input_file = "S&P 500 Data Sets/S&P 500 Total Data Cleaned 2.csv"
    output_file = "S&P 500 Data Sets/S&P 500 Total Data Cleaned 3(Inflation Adjusted).csv"
    adjusted_data = adjust_to_today_dollars(input_file, output_file)

    # Show samples
    print("\nFirst 5 rows (oldest, most adjusted):")
    print(adjusted_data.head())
    print("\nLast 5 rows (most recent, least adjusted):")
    print(adjusted_data.tail())
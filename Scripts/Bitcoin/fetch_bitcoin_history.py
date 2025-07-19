import requests
import pandas as pd
import os
from datetime import datetime
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def fetch_bitcoin_daily_data():
    """
    Fetch complete historical daily Bitcoin price data from Alpha Vantage
    """
    # Get API key from environment
    api_key = os.getenv('alpha_vantage_key')
    if not api_key:
        raise ValueError("API key not found. Please check your .env file.")
    
    print("Fetching complete historical Bitcoin daily data from Alpha Vantage...")
    
    # Alpha Vantage Digital Currency Daily endpoint
    url = "https://www.alphavantage.co/query"
    params = {
        'function': 'DIGITAL_CURRENCY_DAILY',
        'symbol': 'BTC',
        'market': 'USD',
        'apikey': api_key,
        'outputsize': 'full'  # Get full historical data
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Check for API errors
        if 'Error Message' in data:
            raise Exception(f"API Error: {data['Error Message']}")
        
        if 'Note' in data:
            print(f"API Note: {data['Note']}")
            return None
        
        # Extract the time series data
        time_series_key = 'Time Series (Digital Currency Daily)'
        if time_series_key not in data:
            print("No time series data found in response")
            print("Response keys:", list(data.keys()))
            return None
        
        time_series = data[time_series_key]
        
        # Convert to DataFrame
        records = []
        for date, values in time_series.items():
            record = {
                'Date': date,
                'Open': float(values['1. open']),
                'High': float(values['2. high']),
                'Low': float(values['3. low']),
                'Close': float(values['4. close']),
                'Volume': float(values['5. volume'])
            }
            records.append(record)
        
        df = pd.DataFrame(records)
        
        # Convert date column to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Sort by date (oldest first)
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Add a Price column (using Close price for consistency)
        df['Price'] = df['Close']
        
        print(f"Successfully fetched {len(df)} days of Bitcoin data")
        print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"Price range: ${df['Price'].min():.2f} to ${df['Price'].max():,.2f}")
        
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def save_bitcoin_data(df, filename=None):
    """
    Save Bitcoin data to CSV file
    """
    if filename is None:
        # Create filename with current date
        current_date = datetime.now().strftime("%Y%m%d")
        filename = f"Data Sets/Bitcoin Data/Bitcoin_Historical_Data_AlphaVantage_{current_date}.csv"
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"Data saved to: {filename}")
    
    return filename

def main():
    """
    Main function to fetch and save Bitcoin historical data
    """
    print("="*60)
    print("BITCOIN HISTORICAL DATA FETCHER")
    print("="*60)
    
    # Fetch data
    df = fetch_bitcoin_daily_data()
    
    if df is not None:
        # Save data
        filename = save_bitcoin_data(df)
        
        # Display summary statistics
        print("\n" + "="*40)
        print("DATA SUMMARY")
        print("="*40)
        print(f"Total days: {len(df):,}")
        print(f"Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
        print(f"Days of data: {(df['Date'].max() - df['Date'].min()).days:,}")
        print(f"Current price: ${df['Price'].iloc[-1]:,.2f}")
        print(f"All-time high: ${df['Price'].max():,.2f}")
        print(f"All-time low: ${df['Price'].min():,.2f}")
        
        # Check for any missing data
        expected_days = (df['Date'].max() - df['Date'].min()).days + 1
        missing_days = expected_days - len(df)
        if missing_days > 0:
            print(f"Missing days: {missing_days}")
        else:
            print("No missing days detected")
        
        print(f"\nData file ready: {filename}")
        print("You can now use this file for your rainbow chart analysis!")
        
    else:
        print("Failed to fetch Bitcoin data. Please check your API key and try again.")

if __name__ == "__main__":
    main() 
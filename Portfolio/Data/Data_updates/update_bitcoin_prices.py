import requests
import pandas as pd
import os
from datetime import datetime, timedelta
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_last_date_from_file(file_path):
    """Get the last date from the existing Bitcoin data file."""
    try:
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        last_date = df['Date'].max()
        return last_date.strftime('%Y-%m-%d')
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

def fetch_bitcoin_data_for_period(start_date, end_date, api_key):
    """
    Fetch Bitcoin data for a specific date range using Alpha Vantage API
    """
    url = "https://www.alphavantage.co/query"
    params = {
        'function': 'DIGITAL_CURRENCY_DAILY',
        'symbol': 'BTC',
        'market': 'USD',
        'apikey': api_key,
        'outputsize': 'full'
    }
    
    try:
        print(f"Fetching Bitcoin data from {start_date} to {end_date}...")
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Check for API errors
        if 'Error Message' in data:
            print(f"API Error: {data['Error Message']}")
            return None
        
        if 'Note' in data:
            print(f"API Note: {data['Note']}")
            return None
        
        # Extract time series data
        time_series_key = 'Time Series (Digital Currency Daily)'
        if time_series_key not in data:
            print("No time series data found")
            return None
        
        time_series = data[time_series_key]
        
        # Filter data for the specified date range
        filtered_data = {}
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        for date_str, values in time_series.items():
            date_dt = datetime.strptime(date_str, '%Y-%m-%d')
            if start_dt <= date_dt <= end_dt:
                filtered_data[date_str] = values
        
        return filtered_data
        
    except Exception as e:
        print(f"Error fetching data for {start_date} to {end_date}: {e}")
        return None

def update_bitcoin_prices():
    """
    Update Bitcoin price data from the last available date to today
    """
    # Get API key
    api_key = os.getenv('alpha_vantage_key')
    if not api_key:
        raise ValueError("API key not found. Please check your .env file.")
    
    # Define file paths
    bitcoin_data_file = "Portfolio/Data/Bitcoin_all_time_price.csv"
    
    # Check if the file exists
    if not os.path.exists(bitcoin_data_file):
        print(f"Error: Bitcoin data file not found at {bitcoin_data_file}")
        return
    
    # Get the last date from the existing file
    last_date = get_last_date_from_file(bitcoin_data_file)
    if not last_date:
        print("Could not determine last date from file.")
        return
    
    # Calculate today's date
    today = datetime.now().strftime('%Y-%m-%d')
    
    print(f"Last date in file: {last_date}")
    print(f"Today's date: {today}")
    
    # Check if we need to update
    if last_date >= today:
        print("Data is already up to date!")
        return
    
    # Calculate the start date for fetching (next day after last date)
    last_date_dt = datetime.strptime(last_date, '%Y-%m-%d')
    start_date = (last_date_dt + timedelta(days=1)).strftime('%Y-%m-%d')
    
    print(f"Fetching new data from {start_date} to {today}...")
    
    # Fetch new data
    new_data = fetch_bitcoin_data_for_period(start_date, today, api_key)
    
    if not new_data:
        print("No new data was fetched.")
        return
    
    # Convert new data to DataFrame
    records = []
    for date, values in new_data.items():
        record = {
            'Date': date,
            'Price': float(values['4. close']),  # Use close price as Price
            'Open': float(values['1. open']),
            'High': float(values['2. high']),
            'Low': float(values['3. low']),
            'Vol.': float(values['5. volume'])
        }
        records.append(record)
    
    new_df = pd.DataFrame(records)
    
    if new_df.empty:
        print("No new records to add.")
        return
    
    # Sort by date
    new_df['Date'] = pd.to_datetime(new_df['Date'])
    new_df = new_df.sort_values('Date').reset_index(drop=True)
    
    print(f"Fetched {len(new_df)} new days of data")
    print(f"Date range: {new_df['Date'].min().strftime('%Y-%m-%d')} to {new_df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"Price range: ${new_df['Price'].min():.2f} to ${new_df['Price'].max():.2f}")
    
    # Read existing data
    existing_df = pd.read_csv(bitcoin_data_file)
    existing_df['Date'] = pd.to_datetime(existing_df['Date'])
    
    # Combine existing and new data
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    
    # Remove duplicates (in case of overlapping dates)
    combined_df = combined_df.drop_duplicates(subset=['Date']).reset_index(drop=True)
    
    # Sort by date
    combined_df = combined_df.sort_values('Date').reset_index(drop=True)
    
    # Convert Date back to string format for consistency
    combined_df['Date'] = combined_df['Date'].dt.strftime('%Y-%m-%d')
    
    # Save updated data
    combined_df.to_csv(bitcoin_data_file, index=False)
    
    print(f"\nSuccessfully updated Bitcoin data!")
    print(f"Total records: {len(combined_df)}")
    print(f"Updated date range: {combined_df['Date'].iloc[0]} to {combined_df['Date'].iloc[-1]}")
    print(f"Added {len(new_df)} new records")
    
    return combined_df

def main():
    """Main function to run the Bitcoin price update."""
    
    print("Bitcoin Price Data Update")
    print("=" * 30)
    
    try:
        updated_df = update_bitcoin_prices()
        if updated_df is not None:
            print("\n✅ Update completed successfully!")
        else:
            print("\n❌ Update failed or no new data available.")
    except Exception as e:
        print(f"\n❌ Error during update: {e}")

if __name__ == "__main__":
    main() 
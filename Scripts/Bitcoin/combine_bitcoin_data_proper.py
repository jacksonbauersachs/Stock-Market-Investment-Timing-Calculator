import requests
import pandas as pd
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def fetch_latest_bitcoin_data():
    """
    Fetch the latest Bitcoin data from Alpha Vantage
    """
    api_key = os.getenv('alpha_vantage_key')
    if not api_key:
        raise ValueError("API key not found. Please check your .env file.")
    
    url = "https://www.alphavantage.co/query"
    params = {
        'function': 'DIGITAL_CURRENCY_DAILY',
        'symbol': 'BTC',
        'market': 'USD',
        'apikey': api_key,
        'outputsize': 'full'
    }
    
    try:
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
        
        # Convert to DataFrame with correct headers
        records = []
        for date, values in time_series.items():
            record = {
                'Date': date,
                'Price': float(values['4. close']),  # Use Close as Price
                'Open': float(values['1. open']),
                'High': float(values['2. high']),
                'Low': float(values['3. low']),
                'Vol.': float(values['5. volume'])  # Use Vol. to match existing format
            }
            records.append(record)
        
        df = pd.DataFrame(records)
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Sort by date (oldest first)
        df = df.sort_values('Date').reset_index(drop=True)
        
        return df
        
    except Exception as e:
        print(f"Error fetching latest data: {e}")
        return None

def load_existing_data():
    """
    Load the existing comprehensive Bitcoin data with correct headers
    """
    try:
        # Try to load the comprehensive data file
        df = pd.read_csv('Data Sets/Bitcoin Data/Bitcoin Historical Data Full.csv')
        
        # Convert date column
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Ensure we have the required columns with correct names
        if 'Price' not in df.columns:
            if 'Close' in df.columns:
                df['Price'] = df['Close']
            else:
                print("No Price or Close column found")
                return None
        
        # Convert Price column to numeric, handling any string values
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        
        # Remove any rows with NaN prices
        df = df.dropna(subset=['Price'])
        
        # Ensure we have the correct column structure
        required_columns = ['Date', 'Price', 'Open', 'High', 'Low', 'Vol.']
        
        # Check if we have all required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing columns: {missing_columns}")
            return None
        
        # Select only the required columns in the correct order
        df = df[required_columns]
        
        # Sort by date
        df = df.sort_values('Date').reset_index(drop=True)
        
        return df
        
    except Exception as e:
        print(f"Error loading existing data: {e}")
        return None

def combine_and_update_data(existing_df, latest_df):
    """
    Combine existing data with latest data, updating overlapping dates
    """
    if existing_df is None or latest_df is None:
        return None
    
    # Find the latest date in existing data
    existing_latest_date = existing_df['Date'].max()
    print(f"Existing data ends at: {existing_latest_date.strftime('%Y-%m-%d')}")
    
    # Find the latest date in new data
    latest_latest_date = latest_df['Date'].max()
    print(f"Latest API data ends at: {latest_latest_date.strftime('%Y-%m-%d')}")
    
    # Filter new data to only include dates after the existing data ends
    new_data = latest_df[latest_df['Date'] > existing_latest_date].copy()
    
    if len(new_data) == 0:
        print("No new data to add - existing data is already up to date!")
        return existing_df
    
    print(f"Adding {len(new_data)} new days of data")
    
    # Combine the data
    combined_df = pd.concat([existing_df, new_data], ignore_index=True)
    combined_df = combined_df.sort_values('Date').reset_index(drop=True)
    
    # Remove any duplicates (in case of overlapping dates)
    combined_df = combined_df.drop_duplicates(subset=['Date']).reset_index(drop=True)
    
    return combined_df

def save_updated_data(df, filename=None):
    """
    Save the updated Bitcoin data
    """
    if filename is None:
        current_date = datetime.now().strftime("%Y%m%d")
        filename = f"Data Sets/Bitcoin Data/Bitcoin_Complete_Historical_Data_{current_date}.csv"
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"Updated data saved to: {filename}")
    
    return filename

def main():
    """
    Main function to update Bitcoin data with correct headers
    """
    print("="*60)
    print("BITCOIN DATA COMBINER (CORRECT HEADERS)")
    print("="*60)
    print("This script will combine your existing Bitcoin data")
    print("with the latest Alpha Vantage data using correct headers.")
    print("="*60)
    
    # Load existing data
    print("Loading existing comprehensive Bitcoin data...")
    existing_df = load_existing_data()
    
    if existing_df is None:
        print("Failed to load existing data. Please check the file path and column structure.")
        return
    
    print(f"✓ Loaded {len(existing_df):,} days of existing data")
    print(f"  Date range: {existing_df['Date'].min().strftime('%Y-%m-%d')} to {existing_df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"  Columns: {list(existing_df.columns)}")
    
    # Fetch latest data
    print("\nFetching latest Bitcoin data from Alpha Vantage...")
    latest_df = fetch_latest_bitcoin_data()
    
    if latest_df is None:
        print("Failed to fetch latest data. Please check your API key.")
        return
    
    print(f"✓ Fetched {len(latest_df):,} days of latest data")
    print(f"  Date range: {latest_df['Date'].min().strftime('%Y-%m-%d')} to {latest_df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"  Columns: {list(latest_df.columns)}")
    
    # Combine and update data
    print("\nCombining and updating data...")
    updated_df = combine_and_update_data(existing_df, latest_df)
    
    if updated_df is None:
        print("Failed to combine data.")
        return
    
    # Save updated data
    filename = save_updated_data(updated_df)
    
    # Display summary
    print("\n" + "="*50)
    print("COMBINATION SUMMARY")
    print("="*50)
    print(f"Total days: {len(updated_df):,}")
    print(f"Date range: {updated_df['Date'].min().strftime('%Y-%m-%d')} to {updated_df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"Days of data: {(updated_df['Date'].max() - updated_df['Date'].min()).days:,}")
    print(f"Current price: ${updated_df['Price'].iloc[-1]:,.2f}")
    print(f"All-time high: ${updated_df['Price'].max():,.2f}")
    print(f"All-time low: ${updated_df['Price'].min():,.2f}")
    
    # Check for any missing data
    expected_days = (updated_df['Date'].max() - updated_df['Date'].min()).days + 1
    missing_days = expected_days - len(updated_df)
    if missing_days > 0:
        print(f"Missing days: {missing_days}")
    else:
        print("No missing days detected")
    
    print(f"\nCombined data file ready: {filename}")
    print("Column structure: Date, Price, Open, High, Low, Vol.")
    print("You can now use this file for your rainbow chart analysis!")

if __name__ == "__main__":
    main() 
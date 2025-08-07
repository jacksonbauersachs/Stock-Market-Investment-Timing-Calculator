import requests
import pandas as pd
import os
from datetime import datetime, timedelta
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_last_date_from_file(file_path):
    """Get the last date from the existing gold data file."""
    try:
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        last_date = df['Date'].max()
        return last_date.strftime('%Y-%m-%d')
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

def fetch_gold_data_for_period(start_date, end_date, api_key):
    """
    Fetch gold data for a specific date range using GoldAPI.io
    """
    headers = {
        'x-access-token': api_key,
        'Content-Type': 'application/json'
    }
    
    try:
        print(f"Fetching gold data from {start_date} to {end_date}...")
        
        # Convert dates to YYYYMMDD format for API
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        filtered_data = {}
        current_dt = start_dt
        
        while current_dt <= end_dt:
            date_str = current_dt.strftime('%Y%m%d')
            url = f"https://www.goldapi.io/api/XAU/USD/{date_str}"
            
            print(f"  Fetching data for {current_dt.strftime('%Y-%m-%d')}...")
            
            try:
                response = requests.get(url, headers=headers)
                
                if response.status_code != 200:
                    print(f"  Error response: {response.text}")
            except Exception as e:
                print(f"  Request error: {e}")
                response = None
            
            if response and response.status_code == 200:
                try:
                    data = response.json()
                    
                    # Convert timestamp to date string (timestamp is in milliseconds)
                    timestamp = data.get('timestamp', 0)
                    if timestamp:
                        # Convert milliseconds to seconds
                        timestamp_seconds = timestamp / 1000
                        date_obj = datetime.fromtimestamp(timestamp_seconds)
                        date_key = date_obj.strftime('%Y-%m-%d')
                        
                        # Store data in format compatible with existing code
                        filtered_data[date_key] = {
                            '1. open': data.get('open_price', 0),
                            '2. high': data.get('high_price', 0),
                            '3. low': data.get('low_price', 0),
                            '4. close': data.get('price', 0),
                            '5. volume': 0  # GoldAPI doesn't provide volume
                        }
                        print(f"  Successfully processed data for {date_key}")
                    else:
                        print(f"  No timestamp in response")
                except Exception as e:
                    print(f"  Error processing response: {e}")
            else:
                print(f"  Skipping date {current_dt.strftime('%Y-%m-%d')} due to failed request")
            
            # Rate limiting - wait between requests
            time.sleep(1)
            current_dt += timedelta(days=1)
        
        return filtered_data
        
    except Exception as e:
        print(f"Error fetching data for {start_date} to {end_date}: {e}")
        return None

def update_gold_prices():
    """
    Update gold price data from the last available date to today
    """
    # Get API key
    api_key = os.getenv('precious_metals_api_key')
    if not api_key:
        raise ValueError("API key not found. Please check your .env file.")
    
    # Define file paths
    gold_data_file = "Portfolio/Data/Gold_all_time_price.csv"
    
    # Check if the file exists
    if not os.path.exists(gold_data_file):
        print(f"Error: Gold data file not found at {gold_data_file}")
        return
    
    # Get the last date from the existing file
    last_date = get_last_date_from_file(gold_data_file)
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
    new_data = fetch_gold_data_for_period(start_date, today, api_key)
    
    if not new_data:
        print("No new data was fetched.")
        return
    
    # Convert new data to DataFrame
    records = []
    for date, values in new_data.items():
        # Format price values to match existing format (with quotes and commas)
        price = float(values['4. close'])
        open_price = float(values['1. open'])
        high_price = float(values['2. high'])
        low_price = float(values['3. low'])
        volume = float(values['5. volume'])
        
        record = {
            'Date': date,
            'Price': f'{price:,.2f}',
            'Open': f'{open_price:,.2f}',
            'High': f'{high_price:,.2f}',
            'Low': f'{low_price:,.2f}',
            'Vol.': f'{volume:,.2f}',
            'Change %': ''
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
    
    # Extract numeric values for price range display
    price_values = []
    for price_str in new_df['Price']:
        # Remove quotes and convert to float
        price_float = float(price_str.replace('"', '').replace(',', ''))
        price_values.append(price_float)
    
    print(f"Price range: ${min(price_values):.2f} to ${max(price_values):.2f}")
    
    # Read existing data
    existing_df = pd.read_csv(gold_data_file)
    existing_df['Date'] = pd.to_datetime(existing_df['Date'])
    
    # Combine existing and new data
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    
    # Remove duplicates (in case of overlapping dates)
    combined_df = combined_df.drop_duplicates(subset=['Date']).reset_index(drop=True)
    
    # Sort by date
    combined_df = combined_df.sort_values('Date').reset_index(drop=True)
    
    # Convert Date back to string format for consistency
    combined_df['Date'] = combined_df['Date'].dt.strftime('%m/%d/%Y')
    
    # Save updated data with proper quoting
    combined_df.to_csv(gold_data_file, index=False, quoting=1)  # QUOTE_ALL
    
    print(f"\nSuccessfully updated gold data!")
    print(f"Total records: {len(combined_df)}")
    print(f"Updated date range: {combined_df['Date'].iloc[0]} to {combined_df['Date'].iloc[-1]}")
    print(f"Added {len(new_df)} new records")
    
    return combined_df

def main():
    """Main function to run the gold price update."""
    
    print("Gold Price Data Update")
    print("=" * 30)
    
    try:
        updated_df = update_gold_prices()
        if updated_df is not None:
            print("\n✅ Update completed successfully!")
        else:
            print("\n❌ Update failed or no new data available.")
    except Exception as e:
        print(f"\n❌ Error during update: {e}")

if __name__ == "__main__":
    main() 
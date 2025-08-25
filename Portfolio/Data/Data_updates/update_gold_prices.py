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
            # Skip weekends (Saturday = 5, Sunday = 6)
            if current_dt.weekday() >= 5:
                print(f"  Skipping {current_dt.strftime('%Y-%m-%d')} (weekend)")
                current_dt += timedelta(days=1)
                continue
            
            # Skip future dates (no data available yet)
            if current_dt > datetime.now():
                print(f"  Skipping {current_dt.strftime('%Y-%m-%d')} (future date)")
                current_dt += timedelta(days=1)
                continue
                
            date_str = current_dt.strftime('%Y%m%d')
            url = f"https://www.goldapi.io/api/XAU/USD/{date_str}"
            
            print(f"  Fetching data for {current_dt.strftime('%Y-%m-%d')}...")
            
            try:
                response = requests.get(url, headers=headers)
                
                if response.status_code != 200:
                    print(f"  Error response: {response.text}")
                    print(f"  Status code: {response.status_code}")
            except Exception as e:
                print(f"  Request error: {e}")
                response = None
            
            if response and response.status_code == 200:
                try:
                    data = response.json()
                    
                    # Debug: Print the raw API response to see what we're getting
                    print(f"  API Response: {data}")
                    
                    # GoldAPI only provides current price, not OHLC data
                    # We'll use the current price as close price and estimate other values
                    close_price = data.get('price')
                    
                    print(f"  Extracted price - Close: {close_price}")
                    
                    # Check if we have valid close price data
                    if close_price is None or close_price <= 0:
                        print(f"  Skipping {current_dt.strftime('%Y-%m-%d')}: Missing or invalid close price")
                        current_dt += timedelta(days=1)
                        continue
                    
                    # Since GoldAPI doesn't provide OHLC, we'll estimate them
                    # Use close price as the base and create reasonable estimates
                    open_price = close_price * 0.999  # Slightly lower than close
                    high_price = close_price * 1.002  # Slightly higher than close
                    low_price = close_price * 0.998   # Slightly lower than close
                    
                    print(f"  Estimated prices - Open: {open_price:.2f}, High: {high_price:.2f}, Low: {low_price:.2f}, Close: {close_price:.2f}")
                    
                    # Convert timestamp to date string (timestamp is in milliseconds)
                    timestamp = data.get('timestamp', 0)
                    if timestamp:
                        # Convert milliseconds to seconds
                        timestamp_seconds = timestamp / 1000
                        date_obj = datetime.fromtimestamp(timestamp_seconds)
                        date_key = date_obj.strftime('%Y-%m-%d')
                        
                        # Store data in format compatible with existing code
                        filtered_data[date_key] = {
                            '1. open': open_price,
                            '2. high': high_price,
                            '3. low': low_price,
                            '4. close': close_price,
                            '5. volume': 0  # GoldAPI doesn't provide volume
                        }
                        print(f"  Successfully processed data for {date_key}")
                        print(f"  Note: OHLC values are estimated from close price (GoldAPI limitation)")
                    else:
                        print(f"  No timestamp in response for {current_dt.strftime('%Y-%m-%d')}")
                except Exception as e:
                    print(f"  Error processing response for {current_dt.strftime('%Y-%m-%d')}: {e}")
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
    
    print(f"Using API key: {api_key[:8]}...{api_key[-4:] if len(api_key) > 12 else '***'}")
    
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
    print("Note: Weekend dates will be automatically skipped (no trading data)")
    print("Note: GoldAPI only provides current price - OHLC values will be estimated")
    
    # Fetch new data
    new_data = fetch_gold_data_for_period(start_date, today, api_key)
    
    if not new_data:
        print("No new data was fetched.")
        print("This could be due to:")
        print("- All requested dates are weekends/holidays")
        print("- API issues or rate limiting")
        print("- No valid trading data available")
        return
    
    # Convert new data to DataFrame
    records = []
    for date, values in new_data.items():
        try:
            # Validate and convert price values with error handling
            close_price = values.get('4. close')
            open_price = values.get('1. open')
            high_price = values.get('2. high')
            low_price = values.get('3. low')
            volume = values.get('5. volume', 0)
            
            # Check if any required price data is missing or invalid
            if any(price is None or price == 0 for price in [close_price, open_price, high_price, low_price]):
                print(f"  Skipping {date}: Invalid or missing price data")
                continue
            
            # Convert to float with validation
            price = float(close_price)
            open_price = float(open_price)
            high_price = float(high_price)
            low_price = float(low_price)
            volume = float(volume)
            
            # Additional validation - check for reasonable price ranges
            if price <= 0 or price > 10000:  # Gold shouldn't be negative or over $10k
                print(f"  Skipping {date}: Price {price} seems unreasonable")
                continue
            
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
            
        except (ValueError, TypeError) as e:
            print(f"  Skipping {date}: Error converting price data - {e}")
            continue
    
    new_df = pd.DataFrame(records)
    
    if new_df.empty:
        print("No new records to add.")
        print("This might be due to:")
        print("- Weekend dates (no trading data)")
        print("- Market holidays")
        print("- API rate limiting or data availability issues")
        print("- Invalid or missing price data from API")
        return
    
    # Sort by date
    new_df['Date'] = pd.to_datetime(new_df['Date'])
    new_df = new_df.sort_values('Date').reset_index(drop=True)
    
    print(f"Fetched {len(new_df)} new days of data")
    print(f"Date range: {new_df['Date'].min().strftime('%Y-%m-%d')} to {new_df['Date'].max().strftime('%Y-%m-%d')}")
    
    # Extract numeric values for price range display
    if not new_df.empty:
        price_values = []
        for price_str in new_df['Price']:
            try:
                # Remove quotes and convert to float
                price_float = float(price_str.replace('"', '').replace(',', ''))
                price_values.append(price_float)
            except (ValueError, TypeError) as e:
                print(f"Warning: Could not parse price '{price_str}': {e}")
                continue
        
        if price_values:
            print(f"Price range: ${min(price_values):.2f} to ${max(price_values):.2f}")
        else:
            print("Warning: No valid price values found for range calculation")
    else:
        print("No new data to display price range")
    
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
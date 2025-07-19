import requests
import os
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv('alpha_vantage_key')
print(f"API Key found: {'Yes' if api_key else 'No'}")

if not api_key:
    print("No API key found in .env file")
    exit()

# Test the API
url = "https://www.alphavantage.co/query"
params = {
    'function': 'DIGITAL_CURRENCY_DAILY',
    'symbol': 'BTC',
    'market': 'USD',
    'apikey': api_key,
    'outputsize': 'full'  # Get full historical data
}

print("Testing Alpha Vantage API...")
print(f"URL: {url}")
print(f"Params: {params}")

try:
    response = requests.get(url, params=params)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print("\nResponse keys:")
        for key in data.keys():
            print(f"  - {key}")
        
        print("\nFull response (first 1000 chars):")
        print(json.dumps(data, indent=2)[:1000])
        
        # Check if we have time series data
        if 'Time Series (Digital Currency Daily)' in data:
            time_series = data['Time Series (Digital Currency Daily)']
            print(f"\nTime series data found with {len(time_series)} entries")
            
            # Show first entry structure
            first_date = list(time_series.keys())[0]
            first_data = time_series[first_date]
            print(f"\nFirst entry ({first_date}):")
            for key, value in first_data.items():
                print(f"  {key}: {value}")
        else:
            print("\nNo time series data found in response")
            
    else:
        print(f"Error: {response.text}")
        
except Exception as e:
    print(f"Error: {e}") 
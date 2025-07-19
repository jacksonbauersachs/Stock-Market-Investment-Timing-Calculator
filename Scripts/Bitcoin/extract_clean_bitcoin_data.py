import pandas as pd
import os
from datetime import datetime

def extract_clean_bitcoin_data():
    """
    Extract and clean Bitcoin data from the existing comprehensive file
    """
    print("="*60)
    print("BITCOIN DATA EXTRACTOR & CLEANER")
    print("="*60)
    print("This script will extract clean Bitcoin data from your existing file.")
    print("="*60)
    
    try:
        # Load the comprehensive data file
        print("Loading existing comprehensive Bitcoin data...")
        df = pd.read_csv('Data Sets/Bitcoin Data/Bitcoin Historical Data Full.csv')
        
        print(f"✓ Loaded {len(df):,} rows of data")
        print(f"Columns found: {list(df.columns)}")
        
        # Convert date column
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Clean the Price column - remove any quotes and convert to numeric
        if 'Price' in df.columns:
            # Remove quotes and convert to numeric
            df['Price'] = df['Price'].astype(str).str.replace('"', '').str.replace(',', '')
            df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        
        # Clean the Open column
        if 'Open' in df.columns:
            df['Open'] = df['Open'].astype(str).str.replace('"', '').str.replace(',', '')
            df['Open'] = pd.to_numeric(df['Open'], errors='coerce')
        
        # Clean the High column
        if 'High' in df.columns:
            df['High'] = df['High'].astype(str).str.replace('"', '').str.replace(',', '')
            df['High'] = pd.to_numeric(df['High'], errors='coerce')
        
        # Clean the Low column
        if 'Low' in df.columns:
            df['Low'] = df['Low'].astype(str).str.replace('"', '').str.replace(',', '')
            df['Low'] = pd.to_numeric(df['Low'], errors='coerce')
        
        # Clean the Vol. column
        if 'Vol.' in df.columns:
            df['Vol.'] = df['Vol.'].astype(str).str.replace('"', '').str.replace(',', '')
            # Convert K to thousands
            df['Vol.'] = df['Vol.'].str.replace('K', '000').str.replace('k', '000')
            df['Vol.'] = pd.to_numeric(df['Vol.'], errors='coerce')
        
        # Remove any rows with NaN prices
        df = df.dropna(subset=['Price'])
        
        # Select only the required columns in the correct order
        required_columns = ['Date', 'Price', 'Open', 'High', 'Low', 'Vol.']
        
        # Check if we have all required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing columns: {missing_columns}")
            return None
        
        # Select only the required columns
        df = df[required_columns]
        
        # Sort by date
        df = df.sort_values('Date').reset_index(drop=True)
        
        print(f"✓ Cleaned {len(df):,} days of data")
        print(f"Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
        print(f"Price range: ${df['Price'].min():.2f} to ${df['Price'].max():,.2f}")
        
        # Save the cleaned data
        current_date = datetime.now().strftime("%Y%m%d")
        filename = f"Data Sets/Bitcoin Data/Bitcoin_Cleaned_Complete_Data_{current_date}.csv"
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Save to CSV
        df.to_csv(filename, index=False)
        print(f"✓ Cleaned data saved to: {filename}")
        
        # Display summary
        print("\n" + "="*50)
        print("CLEANED DATA SUMMARY")
        print("="*50)
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
        
        print(f"\nCleaned data file ready: {filename}")
        print("Column structure: Date, Price, Open, High, Low, Vol.")
        print("You can now use this file for your rainbow chart analysis!")
        
        return df
        
    except Exception as e:
        print(f"Error processing data: {e}")
        return None

if __name__ == "__main__":
    extract_clean_bitcoin_data() 
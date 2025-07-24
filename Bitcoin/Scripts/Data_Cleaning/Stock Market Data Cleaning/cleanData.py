import pandas as pd
import os


def clean_sp500_data(input_file, output_file):
    """Remove 'Vol.' column from S&P 500 data"""
    try:
        # Read CSV with proper encoding and comma handling
        df = pd.read_csv(input_file, thousands=',', encoding='utf-8')

        # Drop Vol. column (check for variations)
        vol_columns = [col for col in df.columns if 'vol' in col.lower()]
        if vol_columns:
            df = df.drop(columns=vol_columns)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        df.to_csv(output_file, index=False)
        print(f"✅ Success! Cleaned data saved to:\n{output_file}")
        return True

    except Exception as e:
        print(f"❌ Error processing {input_file}:\n{e}")
        return False


if __name__ == "__main__":
    # Get absolute paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "../../Data Sets/S&P 500 Data Sets")

    # CORRECTED: Using the actual filename without .csv
    input_filename = "S&P 500 Data (12_26_1979 to 3_14_2025)"  # Removed .csv
    output_filename = "S&P 500 Total Data Cleaned.csv"

    input_path = os.path.join(DATA_DIR, input_filename)
    output_path = os.path.join(DATA_DIR, output_filename)

    # Debugging output
    print(f"Current working directory: {os.getcwd()}")
    print(f"Looking for input file at: {input_path}")

    if not os.path.exists(DATA_DIR):
        print(f"❌ Data directory not found: {DATA_DIR}")
        print("Directory contents:")
        print(os.listdir(BASE_DIR))
    elif not os.path.exists(input_path):
        print(f"❌ Input file not found: {input_filename}")
        print("Available files in data directory:")
        print(os.listdir(DATA_DIR))
    else:
        clean_sp500_data(input_path, output_path)
import pandas as pd
import os


def reverse_sp500_data(input_file, output_file):
    """
    Reverses chronological order of S&P 500 data and saves to new file
    Args:
        input_file: Path to input CSV file
        output_file: Path for output CSV file
    Returns:
        bool: True if successful, False if error occurs
    """
    try:
        # Read CSV file
        df = pd.read_csv(input_file)

        # Reverse the order
        df = df.iloc[::-1]

        # Reset index without adding a column
        df.reset_index(drop=True, inplace=True)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Save to new file
        df.to_csv(output_file, index=False)
        return True

    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return False


if __name__ == "__main__":
    # Define file paths
    input_file = "../S&P 500 Data Sets/S&P 500 Total Data Cleaned.csv"
    output_file = "../S&P 500 Data Sets/S&P 500 Total Data Cleaned 2.csv"

    # Process the file
    if reverse_sp500_data(input_file, output_file):
        print(f"Successfully reversed data saved to: {output_file}")
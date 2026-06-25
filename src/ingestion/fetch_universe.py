import pandas as pd
import json
import os

def extract_tickers(input_path, output_path):
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Read CSV
    try:
        df = pd.read_csv(input_path)
        print(f"Read {len(df)} rows from {input_path}")
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return

    # Extract unique tickers
    if 'Symbol' in df.columns:
        tickers = sorted(df['Symbol'].unique().tolist())
    else:
        print("Error: 'Symbol' column not found in input CSV")
        return

    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(tickers, f, indent=4)
    
    print(f"Successfully extracted {len(tickers)} unique tickers to {output_path}")

if __name__ == "__main__":
    # Define paths relative to project root
    # Assuming script is run from project root
    INPUT_FILE = r'data/raw/sp500_membership.csv'
    OUTPUT_FILE = r'data/interim/canonical/tickers.json'
    
    # Adjust paths if running from src/scripts
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    
    input_path = os.path.join(project_root, INPUT_FILE)
    output_path = os.path.join(project_root, OUTPUT_FILE)
    
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")

    extract_tickers(input_path, output_path)

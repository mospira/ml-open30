import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to sys.path so we can import from `src` when running directly
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import configurations if needed, or define locally
SIDES = ['long', 'short']

def build_instances(daily_bars_path: str, output_path: str, entry_time: str = "09:31:00") -> pd.DataFrame:
    """
    Constructs the base trade instances dataframe.
    Each row represents a potential trade opportunity for a specific `date`, `ticker`, and `side`.
    
    Args:
        daily_bars_path: Path to the daily features or bars (used to determine the daily universe of valid tickers).
        output_path: Path to save the resulting dataset.
        entry_time: The time string representing when a trade is assumed to enter (e.g. '09:31:00').
        
    Returns:
        A pd.DataFrame representing the expanded instances list.
    """
    print(f"Reading daily data from {daily_bars_path} to build universe...")
    df_daily = pd.read_parquet(daily_bars_path)
    
    # Ensure date and ticker are present
    if 'date' not in df_daily.columns or 'ticker' not in df_daily.columns:
        raise ValueError("daily_features must contain 'date' and 'ticker' columns.")
        
    # Take unique combinations of date and ticker
    universe = df_daily[['date', 'ticker']].drop_duplicates()
    
    instances = []
    
    # Every valid ticker/day combination gets a 'long' and a 'short' instance
    for side in SIDES:
        side_df = universe.copy()
        side_df['side'] = side
        
        # Combine date and entry_time into a full timestamp, assuming `date` is a datetime or string object
        # Example convert: '2023-01-01' + ' 09:31:00'
        side_df['entry_ts'] = pd.to_datetime(side_df['date'].astype(str) + f" {entry_time}")
        
        # We can add other dummy placeholders if needed by downstream labelers
        # side_df['entry_price'] = np.nan # To be filled by 1m data join later if needed
        
        instances.append(side_df)
        
    df_instances = pd.concat(instances, ignore_index=True)
    
    # Sort for cleaner chronology
    df_instances = df_instances.sort_values(['date', 'ticker', 'side']).reset_index(drop=True)
    
    print(f"Constructed {len(df_instances)} trade instances.")
    
    # Save the dataframe
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df_instances.to_parquet(output_path, index=False)
    print(f"Saved instances to {output_path}")
    
    return df_instances

if __name__ == "__main__":
    # Example usage (paths would typically come from pipeline.yaml or argparse)
    daily_path = Path(project_root) / "data" / "processed" / "features" / "daily_features.parquet"
    out_path = Path(project_root) / "data" / "processed" / "trade_instances.parquet"
    
    if daily_path.exists():
        build_instances(str(daily_path), str(out_path))
    else:
        print(f"Could not find {daily_path}. Please generate daily features first.")

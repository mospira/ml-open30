import pandas as pd
import numpy as np

def calculate_atr_dist(df: pd.DataFrame, atr_col: str = 'ATR14', k: float = 1.0) -> pd.Series:
    """
    Calculate stop distance based on Average True Range.
    Uses daily ATR from t-1.
    """
    # Assuming df has 'ATR14' calculated from previous daily rollups
    if atr_col not in df.columns:
        # Fallback or error handling for missing ATR column
        raise KeyError(f"Column '{atr_col}' not found in dataframe.")
    return df[atr_col] * k

def calculate_bps_dist(df: pd.DataFrame, price_col: str = 'entry_price', bps: float = 20.0) -> pd.Series:
    """
    Calculate stop distance based on basis points (bps) of entry price.
    Example: 20 bps = 20 / 10000 = 0.002 * price
    """
    if price_col not in df.columns:
        raise KeyError(f"Column '{price_col}' not found in dataframe.")
    return df[price_col] * (bps / 10000.0)

def get_stop_distance(df: pd.DataFrame, dist_config: dict) -> pd.Series:
    """
    Dispatcher to compute stop distance matching the rule specified in labels.yaml.
    Expected config struct: {'dist_rule': 'atr', 'k': 1.0, 'atr_period': 14}
    """
    rule = dist_config.get('dist_rule', 'atr').lower()
    
    if rule == 'atr':
        k = dist_config.get('k', 1.0)
        # Using a hardcoded column 'ATR14' mapping to atr_period in production
        period = dist_config.get('atr_period', 14)
        return calculate_atr_dist(df, atr_col=f'ATR{period}', k=k)
        
    elif rule == 'bps':
        bps = dist_config.get('bps', 20.0)
        return calculate_bps_dist(df, bps=bps)
    
    elif rule == 'intraday_atr':
        k = dist_config.get('k', 1.0)
        col = 'ATR14_30m'
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found. Pre-compute 30-min ATR before calling get_stop_distance with intraday_atr rule.")
        return df[col] * k
        
    else:
        raise ValueError(f"Unknown dist_rule: '{rule}'")

import pandas as pd
import numpy as np

# Outcome Enums (matching configs/labels.yaml)
SL = 0
TP = 1
TIME = 2
AMBIG = 3

def apply_drop_policy(df: pd.DataFrame, m: float) -> pd.DataFrame:
    """
    DROP policy (typically for training):
    Keep y_type = AMBIG, but mark y_R_m_{m} as NaN so the model ignores these instances
    when computing loss or evaluating on clear signals.
    """
    ambig_mask = df[f'y_type_m_{m}'] == AMBIG
    
    # Nullify payoff for ambiguous cases
    df.loc[ambig_mask, f'y_R_m_{m}'] = np.nan
    return df

def apply_worst_case_policy(df: pd.DataFrame, m: float) -> pd.DataFrame:
    """
    WORST_CASE policy (typically for backtesting):
    For conservative evaluation, treat any AMBIG situation as a Stop Loss hit.
    Sets outcome to SL and payoff to -1R.
    """
    ambig_mask = df[f'y_type_m_{m}'] == AMBIG
    
    # Overwrite outcome and Reward
    df.loc[ambig_mask, f'y_type_m_{m}'] = SL
    df.loc[ambig_mask, f'y_R_m_{m}'] = -1.0
    return df

def resolve_ambiguity(df: pd.DataFrame, m: float, policy: str) -> pd.DataFrame:
    """
    Apply the chosen ambiguity resolution policy to a single reward multiple `m`.
    `policy` can be 'DROP' or 'WORST_CASE'.
    """
    policy = policy.upper()
    
    if policy == 'DROP':
        return apply_drop_policy(df, m)
    elif policy == 'WORST_CASE':
        return apply_worst_case_policy(df, m)
    else:
        raise ValueError(f"Unknown ambiguity policy: '{policy}'. Expected 'DROP' or 'WORST_CASE'.")

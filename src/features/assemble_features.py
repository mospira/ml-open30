"""
assemble_features.py
────────────────────
Merges the 4 intermediate feature dataframes:
  1) daily_features.parquet
  2) open_features.parquet
  3) market_context.parquet
  4) sentiment_features.parquet

Applies normalizations from configs/features.yaml (e.g., log1p).
Drops the initial 60-day warm-up period (caused by rolling_beta).
Writes the final unified dataset to data/processed/features_table.parquet.
"""

import os
import yaml
import numpy as np
import pandas as pd

from src.utils import load_universe

# ── Paths ──────────────────────────────────────────────────────────────────
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

FEATURES_CFG = os.path.join(PROJECT_ROOT, "configs", "features.yaml")

# Input parquets
INPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "features")
DAILY_PATH = os.path.join(INPUT_DIR, "daily_features.parquet")
OPEN_PATH = os.path.join(INPUT_DIR, "open_features.parquet")
MKT_PATH = os.path.join(INPUT_DIR, "market_context.parquet")
SENT_PATH = os.path.join(INPUT_DIR, "sentiment_features.parquet")

# Output parquet
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "features_table.parquet")

# ═══════════════════════════════════════════════════════════════════════════
# CORE LOGIC
# ═══════════════════════════════════════════════════════════════════════════

def _symlog1p(x: pd.Series) -> pd.Series:
    """
    Symmetric log1p: sign(x) * log1p(abs(x)).
    Useful for values like `sentiment_sum` that can be negative.
    """
    return np.sign(x) * np.log1p(np.abs(x))

def assemble_features():
    """Main execution: merge, normalize, drop warm-up, save."""

    print("Assemble Features Script started.\n")

    # 1. Load Data
    print("Loading intermediate datasets...")
    try:
        daily_df = pd.read_parquet(DAILY_PATH)
        open_df = pd.read_parquet(OPEN_PATH)
        mkt_df = pd.read_parquet(MKT_PATH)
        sent_df = pd.read_parquet(SENT_PATH)
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find required feature file:\n  {e.filename}")
        print("Please ensure all feature generation scripts have been run first.")
        return

    # Ensure date formats match exactly
    for df in (daily_df, open_df, mkt_df, sent_df):
        df["date"] = pd.to_datetime(df["date"])

    print(f"  daily_features:     {daily_df.shape[0]:,} rows")
    print(f"  open_features:      {open_df.shape[0]:,} rows")
    print(f"  market_context:     {mkt_df.shape[0]:,} rows")
    print(f"  sentiment_features: {sent_df.shape[0]:,} rows")

    # 2. Merge Data
    # Base grid is daily_features (which covers all universe tickers x trading days)
    print("\nMerging datasets on [date, ticker]...")
    
    # Left join everything onto the daily grid
    merged = daily_df.merge(mkt_df, on=["date", "ticker"], how="left")
    merged = merged.merge(open_df, on=["date", "ticker"], how="left")
    merged = merged.merge(sent_df, on=["date", "ticker"], how="left")

    # Handle missing news
    # For days without news, fill base volume/sum metrics and interaction terms with 0.
    # Base sentiment_avg/min/max/std can remain NaN, which trees handle natively as "missing".
    fill_zero_cols = [
        "article_count", "sentiment_sum",
        "sentiment_avg_x_gap", "sentiment_avg_x_atr_pct", "sentiment_avg_x_market_dir"
    ]
    for col in fill_zero_cols:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0)

    # 3. Apply Normalizations from YAML
    print("\nApplying transformations from features.yaml...")
    with open(FEATURES_CFG, "r") as f:
        cfg = yaml.safe_load(f)

    log1p_features = []
    
    # Parse YAML mapping
    for grp_name, grp in cfg["feature_groups"].items():
        for feat in grp["features"]:
            feat_name = feat["name"]
            norm_rule = str(feat.get("normalize", "none")).lower()
            
            if norm_rule == "log1p" and feat_name in merged.columns:
                log1p_features.append(feat_name)

    print(f"  Features needing symlog1p: {log1p_features}")
    for col in log1p_features:
        merged[col] = _symlog1p(merged[col])

    # 4. Drop Warm-Up Period
    # We require rolling_beta_spy_60d (lookback=60 days) to be fully populated 
    # to avoid NaNs early in the dataset.
    print("\nDropping initial ~60 trading day warm-up period...")
    
    # Find the first date where a significant majority of universe stocks have rolling_beta_spy_60d.
    # Alternatively, just drop rows where beta is NaN. We'll simply drop rows with NaN beta.
    before_count = len(merged)
    # We drop any records that lack the 60-day macro alignment, as it's a critical feature.
    # Downstream, LightGBM/XGBoost handles regular intra-window NaNs well, but global warm-up shouldn't be trained on.
    merged = merged.dropna(subset=["rolling_beta_spy_60d"])
    
    after_count = len(merged)
    print(f"  Dropped {before_count - after_count:,} warm-up rows.")

    # 5. Output
    print(f"\nWriting final dataset...")
    
    # Drop noisy features (importance <= 0) based on permutation importance analysis
    noisy_features = [
        "return_10d", "sentiment_std", "is_post_holiday", "early_relative_volume",
        "gap_down_freq_10d", "article_count", "gap_up_freq_10d", "sentiment_avg_x_market_dir",
        "month", "atr_14", "avg_post_gap_30m_drift_10d", "ATR14", "prior_day_trendiness",
        "prior_day_range", "volume_avg_20d", "sentiment_avg_x_atr_pct", "sentiment_min",
        "sentiment_avg_x_gap", "close_location_in_range", "realized_vol_20d", "return_20d",
        "prior_day_return", "gap_pct", "prior_day_range_avg_5", "gap_vs_atr", "return_5d",
        "return_3d", "range_expansion", "volume_zscore", "residual_move", "rolling_slope_10d",
        "open_loc_vs_prior_range", "prior_day_return_avg_5", "realized_vol_10d"
    ]
    
    # We are now keeping the noisy features for dynamic rolling selection
    # cols_to_drop = [c for c in noisy_features if c in merged.columns]
    # merged = merged.drop(columns=cols_to_drop)
    # print(f"  Dropped {len(cols_to_drop)} noisy features: {cols_to_drop}")


    # Sort
    merged = merged.sort_values(["ticker", "date"]).reset_index(drop=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tmp_file = OUTPUT_FILE + ".tmp"
    merged.to_parquet(tmp_file, index=False)
    os.replace(tmp_file, OUTPUT_FILE)

    print(f"✅ Wrote {len(merged):,} rows × {len(merged.columns)} cols → {OUTPUT_FILE}")
    print(f"   Final Columns: {sorted(merged.columns.tolist())}")

if __name__ == "__main__":
    assemble_features()

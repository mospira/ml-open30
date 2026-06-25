"""
daily_features.py
─────────────────
Computes all prior-day and rolling daily features from configs/features.yaml
that are derivable from daily OHLCV bars (resampled from 1-min candles).

Feature groups produced:
  • price_context      – gaps, multi-day returns, rolling slope
  • volatility_regime  – ATR, realized vol, range expansion
  • liquidity          – dollar volume, volume z-score, spread proxy
  • market_alignment   – SPY/QQQ open returns, beta, correlation
  • mean_reversion     – trendiness, close location, gap frequencies
  • calendar           – day-of-week, month, post-holiday, OPEX week

Output: data/processed/features/daily_features.parquet
"""

import os
import json
import numpy as np
import pandas as pd
import yaml

from src.utils import load_pipeline_config, load_universe

# ── Paths ──────────────────────────────────────────────────────────────────
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
CANDLES_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "candles_1m.parquet")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "features")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "daily_features.parquet")

# ── Regular-session window (Eastern) ───────────────────────────────────────
RTH_START = "09:30"
RTH_END = "16:00"


# ═══════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING & DAILY RESAMPLING
# ═══════════════════════════════════════════════════════════════════════════



def load_and_resample(tz: str = "America/New_York") -> pd.DataFrame:
    """
    Read 1-min candles, filter to regular-trading-hours, and resample
    to daily OHLCV per ticker.

    Returns
    -------
    DataFrame with columns:
        date (datetime64, tz-naive, represents trading day in ET)
        ticker, open, high, low, close, volume
    """
    print("Loading 1-min candles …")
    raw = pd.read_parquet(CANDLES_PATH)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)

    # Convert to Eastern and filter to regular trading hours
    raw["ts_et"] = raw["timestamp"].dt.tz_convert(tz)
    raw["time"] = raw["ts_et"].dt.strftime("%H:%M")
    rth = raw[(raw["time"] >= RTH_START) & (raw["time"] < RTH_END)].copy()
    rth["date"] = rth["ts_et"].dt.normalize().dt.tz_localize(None)

    print(f"  {len(raw):,} raw bars → {len(rth):,} RTH bars")

    # Resample to daily OHLCV per ticker
    daily = (
        rth.groupby(["ticker", "date"])
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .reset_index()
        .sort_values(["ticker", "date"])
    )

    print(f"  {len(daily):,} daily rows across {daily['ticker'].nunique()} tickers")
    return daily


def load_and_resample_with_spread(tz: str = "America/New_York"):
    """
    Like load_and_resample, but also computes median(1m high - 1m low)
    per ticker-day for the spread_proxy feature.

    Returns (daily_df, spread_df)
    """
    print("Loading 1-min candles …")
    raw = pd.read_parquet(CANDLES_PATH)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)

    raw["ts_et"] = raw["timestamp"].dt.tz_convert(tz)
    raw["time"] = raw["ts_et"].dt.strftime("%H:%M")
    rth = raw[(raw["time"] >= RTH_START) & (raw["time"] < RTH_END)].copy()
    rth["date"] = rth["ts_et"].dt.normalize().dt.tz_localize(None)

    print(f"  {len(raw):,} raw bars → {len(rth):,} RTH bars")

    # Daily OHLCV
    daily = (
        rth.groupby(["ticker", "date"])
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .reset_index()
        .sort_values(["ticker", "date"])
    )

    # Spread proxy: median of (1m high - 1m low) per ticker-day
    rth["bar_range"] = rth["high"] - rth["low"]
    spread_df = (
        rth.groupby(["ticker", "date"])["bar_range"]
        .median()
        .reset_index()
        .rename(columns={"bar_range": "median_1m_range"})
    )

    print(f"  {len(daily):,} daily rows across {daily['ticker'].nunique()} tickers")
    return daily, spread_df


# ═══════════════════════════════════════════════════════════════════════════
# 2. FEATURE COMPUTATION — one function per feature group
# ═══════════════════════════════════════════════════════════════════════════

def compute_price_context(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Price-context features: gaps, prior-day stats, multi-day returns,
    rolling slope.  All shift-1 safe (no look-ahead).
    """
    df = daily[["date", "ticker", "open", "high", "low", "close"]].copy()

    g = df.groupby("ticker")

    # Prior-day values (shifted)
    df["close_prev"] = g["close"].shift(1)
    df["open_prev"] = g["open"].shift(1)
    df["high_prev"] = g["high"].shift(1)
    df["low_prev"] = g["low"].shift(1)

    # Today's open is already the open of the day (first RTH bar)
    open_930 = df["open"]

    # ── Simple prior-day features ──
    df["gap_pct"] = (open_930 - df["close_prev"]) / df["close_prev"]

    prior_range = df["high_prev"] - df["low_prev"]
    df["open_loc_vs_prior_range"] = np.where(
        prior_range > 0,
        (open_930 - df["low_prev"]) / prior_range,
        np.nan,
    )

    df["prior_day_return"] = (df["close_prev"] - df["open_prev"]) / df["open_prev"]
    df["prior_day_range"] = prior_range / df["close_prev"]

    # True range for t-1  (needs close at t-2)
    close_t2 = g["close"].shift(2)
    tr_comp1 = df["high_prev"] - df["low_prev"]
    tr_comp2 = (df["high_prev"] - close_t2).abs()
    tr_comp3 = (df["low_prev"] - close_t2).abs()
    df["prior_day_true_range"] = pd.concat([tr_comp1, tr_comp2, tr_comp3], axis=1).max(axis=1)

    # Rolling-5 averages
    df["prior_day_return_avg_5"] = g["prior_day_return"].transform(
        lambda s: s.rolling(5, min_periods=5).mean()
    )
    df["prior_day_range_avg_5"] = g["prior_day_range"].transform(
        lambda s: s.rolling(5, min_periods=5).mean()
    )

    # Multi-day returns  (close_prev vs close N days ago)
    for label, shift_n in [("return_3d", 3), ("return_5d", 5),
                           ("return_10d", 10), ("return_20d", 20)]:
        close_back = g["close"].shift(shift_n + 1)  # +1 because close_prev is shift(1)
        df[label] = (df["close_prev"] - close_back) / close_back

    # gap_vs_atr  — computed after ATR is available; we'll merge later.
    # Rolling slope (10 days): OLS slope of close over 10 days, normalized
    def _rolling_slope(s: pd.Series, window: int = 10) -> pd.Series:
        """Normalized OLS slope of values over a rolling window."""
        out = pd.Series(np.nan, index=s.index)
        x = np.arange(window, dtype=float)
        x_demean = x - x.mean()
        denom = (x_demean ** 2).sum()
        vals = s.values
        for i in range(window - 1, len(vals)):
            chunk = vals[i - window + 1 : i + 1]
            if np.any(np.isnan(chunk)):
                continue
            slope = (x_demean * (chunk - chunk.mean())).sum() / denom
            # Normalize by mean price in the window
            mean_price = chunk.mean()
            out.iloc[i] = slope / mean_price if mean_price != 0 else np.nan
        return out

    df["rolling_slope_10d"] = g["close"].transform(lambda s: _rolling_slope(s, 10))

    # We need to shift rolling_slope so it represents data known before today
    df["rolling_slope_10d"] = g["rolling_slope_10d"].shift(1)

    keep_cols = [
        "date", "ticker",
        "gap_pct", "open_loc_vs_prior_range",
        "prior_day_return", "prior_day_range", "prior_day_true_range",
        "prior_day_return_avg_5", "prior_day_range_avg_5",
        "return_3d", "return_5d", "return_10d", "return_20d",
        "rolling_slope_10d",
    ]
    return df[keep_cols]


def compute_volatility_regime(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Volatility-regime features: ATR_14, ATR%, realized vol (10d/20d),
    range expansion.
    """
    df = daily[["date", "ticker", "open", "high", "low", "close"]].copy()
    g = df.groupby("ticker")

    close_prev = g["close"].shift(1)

    # True Range (for each day)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - close_prev).abs()
    tr3 = (df["low"] - close_prev).abs()
    df["true_range"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # ATR-14 (Wilder-style EMA)
    df["atr_14"] = g["true_range"].transform(
        lambda s: s.ewm(span=14, min_periods=14, adjust=False).mean()
    )
    # Shift so it represents yesterday's ATR (known at open)
    df["atr_14"] = g["atr_14"].shift(1)
    df["atr_pct"] = df["atr_14"] / close_prev

    # Realized volatility: std of daily log returns
    df["log_ret"] = np.log(df["close"] / close_prev)
    df["realized_vol_10d"] = g["log_ret"].transform(
        lambda s: s.rolling(10, min_periods=10).std()
    )
    df["realized_vol_20d"] = g["log_ret"].transform(
        lambda s: s.rolling(20, min_periods=20).std()
    )
    # Shift so it's prior-day
    df["realized_vol_10d"] = g["realized_vol_10d"].shift(1)
    df["realized_vol_20d"] = g["realized_vol_20d"].shift(1)

    # Range expansion: true_range_{t-1} / mean(true_range over last 5 days at t-1)
    tr_shifted = g["true_range"].shift(1)
    mean_tr_5 = g["true_range"].transform(
        lambda s: s.rolling(5, min_periods=5).mean()
    )
    # Need shifted version of mean_tr_5
    mean_tr_5_shifted = g.apply(
        lambda sub: sub["true_range"].rolling(5, min_periods=5).mean().shift(1),
        include_groups=False,
    ).droplevel(0)
    df["range_expansion"] = tr_shifted / mean_tr_5_shifted

    keep_cols = [
        "date", "ticker",
        "atr_14", "atr_pct",
        "realized_vol_10d", "realized_vol_20d",
        "range_expansion",
    ]
    return df[keep_cols]


def compute_liquidity(
    daily: pd.DataFrame, spread_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Liquidity features: dollar volume, volume stats, spread proxy.
    """
    df = daily[["date", "ticker", "close", "volume"]].copy()
    g = df.groupby("ticker")

    close_prev = g["close"].shift(1)
    vol_prev = g["volume"].shift(1)

    df["dollar_volume_prev"] = close_prev * vol_prev

    # Rolling averages (shifted so they use only prior data)
    df["dollar_volume"] = df["close"] * df["volume"]
    df["dollar_volume_avg_10d"] = g["dollar_volume"].transform(
        lambda s: s.rolling(10, min_periods=10).mean()
    )
    df["dollar_volume_avg_10d"] = g["dollar_volume_avg_10d"].shift(1)

    df["volume_avg_20d"] = g["volume"].transform(
        lambda s: s.rolling(20, min_periods=20).mean()
    )
    df["volume_avg_20d_shifted"] = g["volume_avg_20d"].shift(1)

    vol_std_20d = g["volume"].transform(
        lambda s: s.rolling(20, min_periods=20).std()
    )
    vol_std_20d_shifted = g.apply(
        lambda sub: sub["volume"].rolling(20, min_periods=20).std().shift(1),
        include_groups=False,
    ).droplevel(0)

    df["volume_zscore"] = np.where(
        vol_std_20d_shifted > 0,
        (vol_prev - df["volume_avg_20d_shifted"]) / vol_std_20d_shifted,
        np.nan,
    )

    # Rename for output
    df["volume_avg_20d"] = df["volume_avg_20d_shifted"]

    # Spread proxy: median(1m high-low) / close_prev  — from prior day
    spread_shifted = spread_df.copy()
    spread_shifted = spread_shifted.sort_values(["ticker", "date"])
    spread_shifted["median_1m_range_prev"] = (
        spread_shifted.groupby("ticker")["median_1m_range"].shift(1)
    )
    df = df.merge(
        spread_shifted[["date", "ticker", "median_1m_range_prev"]],
        on=["date", "ticker"],
        how="left",
    )
    df["spread_proxy"] = df["median_1m_range_prev"] / close_prev

    keep_cols = [
        "date", "ticker",
        "dollar_volume_prev", "dollar_volume_avg_10d",
        "volume_avg_20d", "volume_zscore", "spread_proxy",
    ]
    return df[keep_cols]


# Removed compute_market_alignment and compute_calendar.


def compute_mean_reversion(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Mean-reversion regime features: trendiness, close location,
    gap-up/down frequencies.
    """
    df = daily[["date", "ticker", "open", "high", "low", "close"]].copy()
    g = df.groupby("ticker")

    close_prev = g["close"].shift(1)
    open_prev = g["open"].shift(1)
    high_prev = g["high"].shift(1)
    low_prev = g["low"].shift(1)

    # True range for prior day
    close_t2 = g["close"].shift(2)
    tr1 = high_prev - low_prev
    tr2 = (high_prev - close_t2).abs()
    tr3 = (low_prev - close_t2).abs()
    tr_prev = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # prior_day_trendiness: (close-open) / true_range for t-1
    df["prior_day_trendiness"] = np.where(
        tr_prev > 0,
        (close_prev - open_prev) / tr_prev,
        np.nan,
    )

    # close_location_in_range
    prior_range = high_prev - low_prev
    df["close_location_in_range"] = np.where(
        prior_range > 0,
        (close_prev - low_prev) / prior_range,
        np.nan,
    )

    # Gap frequency over 10 days
    df["gap"] = (df["open"] - close_prev) / close_prev
    df["gap_up"] = (df["gap"] > 0).astype(float)
    df["gap_down"] = (df["gap"] < 0).astype(float)

    df["gap_up_freq_10d"] = g["gap_up"].transform(
        lambda s: s.rolling(10, min_periods=10).mean()
    )
    df["gap_down_freq_10d"] = g["gap_down"].transform(
        lambda s: s.rolling(10, min_periods=10).mean()
    )
    # Shift so we use only prior data
    df["gap_up_freq_10d"] = g["gap_up_freq_10d"].shift(1)
    df["gap_down_freq_10d"] = g["gap_down_freq_10d"].shift(1)

    keep_cols = [
        "date", "ticker",
        "prior_day_trendiness", "close_location_in_range",
        "gap_up_freq_10d", "gap_down_freq_10d",
    ]
    return df[keep_cols]




# ═══════════════════════════════════════════════════════════════════════════
# 3. ASSEMBLY & OUTPUT
# ═══════════════════════════════════════════════════════════════════════════

def merge_gap_vs_atr(
    price_ctx: pd.DataFrame, vol_regime: pd.DataFrame
) -> pd.DataFrame:
    """Add gap_vs_atr = gap_pct / atr_14  (cross-group dependency)."""
    merged = price_ctx.merge(
        vol_regime[["date", "ticker", "atr_14"]],
        on=["date", "ticker"],
        how="left",
    )
    merged["gap_vs_atr"] = np.where(
        merged["atr_14"] > 0,
        merged["gap_pct"] / (merged["atr_14"] / merged.get("close_prev", 1)),
        np.nan,
    )
    # gap_vs_atr is gap_pct / atr_14 in *dollar* terms → gap$ / atr$
    # gap$ = gap_pct * close_prev; atr$ = atr_14
    # So gap_vs_atr = (open - close_prev) / atr_14
    # We'll recalculate correctly here
    merged.drop(columns=["gap_vs_atr", "atr_14"], inplace=True)
    return merged


def build_daily_features() -> None:
    """Main entry point: compute all daily features and write parquet."""
    cfg = load_pipeline_config()
    tz = cfg["project"]["timezone"]

    # Load raw data
    daily, spread_df = load_and_resample_with_spread(tz)

    # Filter to universe tickers only (including benchmarks, used for alignment)
    universe = load_universe()
    daily = daily[daily["ticker"].isin(universe)].copy()
    spread_df = spread_df[spread_df["ticker"].isin(universe)].copy()

    # Non-benchmark tickers for the final output
    benchmarks = {"SPY", "QQQ"}
    stock_tickers = [t for t in universe if t not in benchmarks]

    print("\nComputing feature groups …")

    # ── Compute each group ──
    price_ctx = compute_price_context(daily)
    print("  ✓ price_context")

    vol_regime = compute_volatility_regime(daily)
    print("  ✓ volatility_regime")

    liq = compute_liquidity(daily, spread_df)
    print("  ✓ liquidity")

    mean_rev = compute_mean_reversion(daily)
    print("  ✓ mean_reversion_regime")

    # ── Cross-group: gap_vs_atr ──
    # gap_vs_atr = (open - close_prev) / atr_14
    # We need close_prev and atr_14 from vol_regime to compute this.
    # Merge atr_14 into price_context, then compute.
    combined = price_ctx.merge(
        vol_regime[["date", "ticker", "atr_14"]],
        on=["date", "ticker"],
        how="left",
    )
    # gap$ = gap_pct * close_prev; gap_vs_atr = gap$ / atr_14
    close_prev_lookup = daily.groupby("ticker")["close"].shift(1)
    daily_with_cprev = daily[["date", "ticker"]].copy()
    daily_with_cprev["close_prev"] = close_prev_lookup.values
    combined = combined.merge(
        daily_with_cprev[["date", "ticker", "close_prev"]],
        on=["date", "ticker"],
        how="left",
    )
    gap_dollars = combined["gap_pct"] * combined["close_prev"]
    combined["gap_vs_atr"] = np.where(
        combined["atr_14"] > 0,
        gap_dollars / combined["atr_14"],
        np.nan,
    )
    combined.drop(columns=["atr_14", "close_prev"], inplace=True)

    # ── Merge all groups ──
    # Filter to stock tickers only
    combined = combined[~combined["ticker"].isin(benchmarks)]
    vol_regime_out = vol_regime[~vol_regime["ticker"].isin(benchmarks)]
    liq_out = liq[~liq["ticker"].isin(benchmarks)]
    mean_rev_out = mean_rev[~mean_rev["ticker"].isin(benchmarks)]

    result = combined
    for right_df in [vol_regime_out, liq_out, mean_rev_out]:
        result = result.merge(right_df, on=["date", "ticker"], how="left")

    # Sort and write
    result = result.sort_values(["ticker", "date"]).reset_index(drop=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tmp_file = OUTPUT_FILE + ".tmp"
    result.to_parquet(tmp_file, index=False)
    os.replace(tmp_file, OUTPUT_FILE)

    print(f"\n✅ Wrote {len(result):,} rows × {len(result.columns)} cols → {OUTPUT_FILE}")
    print(f"   Columns: {sorted(result.columns.tolist())}")

    # Quick sanity checks
    non_key = [c for c in result.columns if c not in ("date", "ticker")]
    all_nan = [c for c in non_key if result[c].isna().all()]
    if all_nan:
        print(f"   ⚠  All-NaN columns: {all_nan}")
    else:
        print("   ✓ No all-NaN columns")

    # Count NaN fraction per column
    nan_frac = result[non_key].isna().mean().sort_values(ascending=False)
    high_nan = nan_frac[nan_frac > 0.3]
    if not high_nan.empty:
        print(f"   ⚠  High-NaN columns (>30%):")
        for c, v in high_nan.items():
            print(f"      {c}: {v:.1%}")


if __name__ == "__main__":
    build_daily_features()

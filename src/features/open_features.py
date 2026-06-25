"""
open_features.py
────────────────
Computes entry-time (open-window) features from the first 1–2 minutes
after market open, as defined in configs/features.yaml under:

  • open_window             – first-1-min candle structure & activity
  • mean_reversion_regime   – avg_post_gap_30m_drift_10d (1-min source)

All features are computed from 1-min intraday bars and keyed by
(date, ticker) so they can be joined to daily_features.parquet.

Observation note: entry is at 09:31 ET (pipeline.yaml: entry_time).
  - Features derived solely from the 09:30 bar are known at 09:31.
  - Features referencing 09:31 close are also available at 09:31.
  - avg_post_gap_30m_drift_10d measures 09:30→10:00 drift from *past*
    days, so it is known and leak-free at entry.

Output: data/processed/features/open_features.parquet
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
DAILY_FEATURES_PATH = os.path.join(
    PROJECT_ROOT, "data", "processed", "features", "daily_features.parquet"
)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "features")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "open_features.parquet")

# ── Session times (Eastern) ────────────────────────────────────────────────
BAR_0930 = "09:30"   # first RTH bar open
BAR_1000 = "10:00"   # end of 30-min window (for drift feature)


# ═══════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════



def load_intraday(tz: str = "America/New_York") -> pd.DataFrame:
    """
    Read 1-min candles, localize to Eastern, and filter to intraday bars
    from 09:30 through 10:00 (inclusive) — everything needed for this script.

    Returns a DataFrame with columns:
        date (tz-naive, ET trading day), time (str HH:MM),
        ticker, open, high, low, close, volume
    """
    print("Loading 1-min candles …")
    raw = pd.read_parquet(CANDLES_PATH)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    raw["ts_et"] = raw["timestamp"].dt.tz_convert(tz)
    raw["time"] = raw["ts_et"].dt.strftime("%H:%M")
    raw["date"] = raw["ts_et"].dt.normalize().dt.tz_localize(None)

    # Keep only the bars we need: 09:30 through 10:00
    bars = raw[raw["time"] <= BAR_1000].copy()
    bars = bars[bars["time"] >= BAR_0930].copy()

    print(f"  {len(raw):,} total bars → {len(bars):,} open-window bars (09:30–10:00)")
    return bars[["date", "time", "ticker", "open", "high", "low", "close", "volume"]]


# ═══════════════════════════════════════════════════════════════════════════
# 2.  FEATURE COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════

def compute_open_window(
    bars: pd.DataFrame,
    daily_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Open-window features: first-1-min candle structure, wick ratios,
    early vol and relative volume.

    Features produced
    -----------------
    first_1m_return         : (close_0931 / open_0930) - 1
    opening_range_1m_vs_atr : (high_0930 - low_0930) / atr_14
    first_1m_body_ratio     : (close - open) / (high - low)  for 09:30 bar
    first_1m_upper_wick     : (high - max(open,close)) / (high - low)  for 09:30 bar
    first_1m_lower_wick     : (min(open,close) - low)  / (high - low)  for 09:30 bar
    early_realized_vol_1m   : |return| of the 09:30 bar  (1 bar sum)
    early_relative_volume   : volume_0930 / rolling_avg_volume_0930 (20d)
    """
    # ── Extract the two bars we need ──
    bar_930 = (
        bars[bars["time"] == BAR_0930]
        .set_index(["date", "ticker"])
        [["open", "high", "low", "close", "volume"]]
        .rename(columns={
            "open": "open_930", "high": "high_930",
            "low": "low_930", "close": "close_930", "volume": "vol_930",
        })
    )

    df = bar_930.reset_index()

    # ── Candle structure ──────────────────────────────────────────────────
    bar_range = df["high_930"] - df["low_930"]  # full 09:30 bar range
    body      = df["close_930"] - df["open_930"]

    df["first_1m_return"] = (df["close_930"] / df["open_930"]) - 1

    df["first_1m_body_ratio"] = np.where(
        bar_range > 0, body / bar_range, 0.0
    )
    df["first_1m_upper_wick"] = np.where(
        bar_range > 0,
        (df["high_930"] - df[["open_930", "close_930"]].max(axis=1)) / bar_range,
        0.0,
    )
    df["first_1m_lower_wick"] = np.where(
        bar_range > 0,
        (df[["open_930", "close_930"]].min(axis=1) - df["low_930"]) / bar_range,
        0.0,
    )

    # ── Early realized vol (single-bar absolute return) ───────────────────
    df["early_realized_vol_1m"] = (
        np.log(df["close_930"] / df["open_930"]).abs()
    )

    # ── opening_range_1m_vs_atr  (needs atr_14 from daily_features) ──────
    atr_lookup = daily_features[["date", "ticker", "atr_14"]].copy()
    df = df.merge(atr_lookup, on=["date", "ticker"], how="left")
    df["opening_range_1m_vs_atr"] = np.where(
        df["atr_14"] > 0,
        (df["high_930"] - df["low_930"]) / df["atr_14"],
        np.nan,
    )
    df.drop(columns=["atr_14"], inplace=True)

    # ── Early relative volume  (20-day rolling avg of 09:30 bar volume) ──
    vol_930 = (
        df[["date", "ticker", "vol_930"]]
        .sort_values(["ticker", "date"])
        .copy()
    )
    vol_930["vol_930_avg20"] = vol_930.groupby("ticker")["vol_930"].transform(
        lambda s: s.rolling(20, min_periods=20).mean().shift(1)
    )
    vol_930["vol_930_std20"] = vol_930.groupby("ticker")["vol_930"].transform(
        lambda s: s.rolling(20, min_periods=20).std().shift(1)
    )
    df = df.merge(
        vol_930[["date", "ticker", "vol_930_avg20"]],
        on=["date", "ticker"], how="left",
    )
    df["early_relative_volume"] = np.where(
        df["vol_930_avg20"] > 0,
        df["vol_930"] / df["vol_930_avg20"],
        np.nan,
    )

    keep = [
        "date", "ticker",
        "first_1m_return",
        "opening_range_1m_vs_atr",
        "first_1m_body_ratio",
        "first_1m_upper_wick",
        "first_1m_lower_wick",
        "early_realized_vol_1m",
        "early_relative_volume",
    ]
    return df[keep]


def compute_avg_post_gap_30m_drift(bars: pd.DataFrame) -> pd.DataFrame:
    """
    avg_post_gap_30m_drift_10d
    --------------------------
    Mean of (close_1000 - open_0930) / open_0930  over the past 10
    *gap days* for each ticker.  A "gap day" is any day where
    |open_0930 - close_prev| > 0 (i.e., any gap, up or down).

    This is the only mean-reversion-regime feature sourced from 1-min bars.
    It is computed from historical data only (shift-1) so it is leak-free.
    """
    bar_930 = (
        bars[bars["time"] == BAR_0930]
        .set_index(["date", "ticker"])[["open", "close"]]
        .rename(columns={"open": "open_930", "close": "close_930"})
        .reset_index()
    )
    bar_1000 = (
        bars[bars["time"] == BAR_1000]
        .set_index(["date", "ticker"])[["close"]]
        .rename(columns={"close": "close_1000"})
        .reset_index()
    )

    df = bar_930.merge(bar_1000, on=["date", "ticker"], how="inner")
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Prior close
    df["close_prev"] = df.groupby("ticker")["close_930"].shift(1)

    # Gap flag
    df["is_gap"] = (
        ((df["open_930"] - df["close_prev"]) / df["close_prev"]).abs() > 0
    ).astype(float)
    df["is_gap"] = np.where(df["close_prev"].isna(), np.nan, df["is_gap"])

    # 30-min drift for each day
    df["drift_30m"] = (df["close_1000"] - df["open_930"]) / df["open_930"]
    df["drift_30m_on_gap"] = df["drift_30m"] * df["is_gap"]

    # Rolling mean of drift over last 10 gap days
    # Approach: separate rolling sum of drift and count on gap days
    def _rolling_gap_drift(sub: pd.DataFrame, window: int = 10) -> pd.Series:
        out = pd.Series(np.nan, index=sub.index)
        drift_vals = sub["drift_30m"].values
        gap_flags  = sub["is_gap"].values
        for i in range(1, len(sub)):  # start at 1 to look back
            count = 0
            total = 0.0
            j = i - 1  # look at prior days only
            while j >= 0 and count < window:
                if gap_flags[j] == 1:
                    total += drift_vals[j]
                    count += 1
                j -= 1
            if count == window:
                out.iloc[i] = total / count
        return out

    parts = []
    for ticker, sub in df.groupby("ticker"):
        sub = sub.sort_values("date").copy()
        sub["avg_post_gap_30m_drift_10d"] = _rolling_gap_drift(sub, 10)
        parts.append(sub)

    result = pd.concat(parts)

    return result[["date", "ticker", "avg_post_gap_30m_drift_10d"]]


# ═══════════════════════════════════════════════════════════════════════════
# 3.  ASSEMBLY & OUTPUT
# ═══════════════════════════════════════════════════════════════════════════

def build_open_features() -> None:
    """Main entry point: compute all open-window features and write parquet."""
    cfg = load_pipeline_config()
    tz  = cfg["project"]["timezone"]

    # Universe
    universe  = load_universe()
    benchmarks = {"SPY", "QQQ"}
    stock_tickers = [t for t in universe if t not in benchmarks]

    # Load intraday data (09:30–10:00 only)
    bars = load_intraday(tz)
    bars = bars[bars["ticker"].isin(universe)].copy()

    # Load daily_features for atr_14 (needed by opening_range_1m_vs_atr)
    print("Loading daily_features for ATR lookup …")
    daily_features = pd.read_parquet(DAILY_FEATURES_PATH, columns=["date", "ticker", "atr_14"])

    print("\nComputing feature groups …")

    # ── Open-window ────────────────────────────────────────────────────────
    open_win = compute_open_window(bars, daily_features)
    print(f"  ✓ open_window  ({len(open_win):,} rows)")

    # ── avg_post_gap_30m_drift_10d ─────────────────────────────────────────
    gap_drift = compute_avg_post_gap_30m_drift(bars)
    print(f"  ✓ avg_post_gap_30m_drift_10d  ({len(gap_drift):,} rows)")

    # ── Merge ──────────────────────────────────────────────────────────────
    result = open_win.merge(gap_drift, on=["date", "ticker"], how="left")

    # Filter to stock tickers only (no benchmarks)
    result = result[result["ticker"].isin(stock_tickers)].copy()
    result = result.sort_values(["ticker", "date"]).reset_index(drop=True)

    # ── Write ──────────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tmp_file = OUTPUT_FILE + ".tmp"
    result.to_parquet(tmp_file, index=False)
    os.replace(tmp_file, OUTPUT_FILE)

    print(f"\n✅ Wrote {len(result):,} rows × {len(result.columns)} cols → {OUTPUT_FILE}")
    print(f"   Columns: {sorted(result.columns.tolist())}")

    # ── Sanity checks ──────────────────────────────────────────────────────
    non_key = [c for c in result.columns if c not in ("date", "ticker")]
    all_nan = [c for c in non_key if result[c].isna().all()]
    if all_nan:
        print(f"   ⚠  All-NaN columns: {all_nan}")
    else:
        print("   ✓ No all-NaN columns")

    nan_frac = result[non_key].isna().mean().sort_values(ascending=False)
    high_nan = nan_frac[nan_frac > 0.3]
    if not high_nan.empty:
        print(f"   ⚠  High-NaN columns (>30%):")
        for c, v in high_nan.items():
            print(f"      {c}: {v:.1%}")

    # Value-range spot checks
    checks = {
        "first_1m_return":        (-0.5, 0.5),
        "first_1m_body_ratio":    (-1.0, 1.0),
        "first_1m_upper_wick":    (0.0, 1.0),
        "first_1m_lower_wick":    (0.0, 1.0),
        "early_relative_volume":  (0.0, 50.0),
    }
    for col, (lo, hi) in checks.items():
        if col not in result.columns:
            continue
        col_data = result[col].dropna()
        if col_data.empty:
            continue
        out_of_range = ((col_data < lo) | (col_data > hi)).sum()
        if out_of_range:
            pct = out_of_range / len(col_data)
            print(f"   ℹ  {col}: {out_of_range} ({pct:.1%}) values outside [{lo}, {hi}]")

    print("\nDone.")


if __name__ == "__main__":
    build_open_features()

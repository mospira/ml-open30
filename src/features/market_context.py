"""
market_context.py
─────────────────
Computes market-wide alignment and calendar features from configs/features.yaml
that are derivable from daily OHLCV bars (resampled from 1-min candles) and 
the daily features parquet.

Feature groups produced:
  • market_alignment   – SPY/QQQ open returns, residual move, beta, correlation
  • calendar           – day-of-week, month, post-holiday, OPEX week

Output: data/processed/features/market_context.parquet
"""

import os
import numpy as np
import pandas as pd
import yaml

from src.utils import load_pipeline_config, load_universe

# ── Paths ──────────────────────────────────────────────────────────────────
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

PIPELINE_CFG = os.path.join(PROJECT_ROOT, "configs", "pipeline.yaml")
CANDLES_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "candles_1m.parquet")
DAILY_FEATURES_PATH = os.path.join(
    PROJECT_ROOT, "data", "processed", "features", "daily_features.parquet"
)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "features")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "market_context.parquet")

# ── Regular-session window (Eastern) ───────────────────────────────────────
RTH_START = "09:30"
RTH_END = "16:00"

# ═══════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_and_resample(tz: str = "America/New_York") -> pd.DataFrame:
    """
    Read 1-min candles, filter to regular-trading-hours, and resample
    to daily OHLCV per ticker.
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

# ═══════════════════════════════════════════════════════════════════════════
# 2. FEATURE COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════

def compute_market_alignment(daily: pd.DataFrame, daily_features: pd.DataFrame) -> pd.DataFrame:
    """
    Market-alignment features: SPY/QQQ open returns, rolling beta
    and correlation vs SPY.
    Uses daily_features.gap_pct to compute residual_move.
    """
    benchmarks = {"SPY", "QQQ"}
    stocks = daily[~daily["ticker"].isin(benchmarks)].copy()
    bmk = daily[daily["ticker"].isin(benchmarks)].copy()

    # Benchmark daily returns & open gaps
    bmk_g = bmk.groupby("ticker")
    bmk["close_prev"] = bmk_g["close"].shift(1)
    bmk["daily_ret"] = (bmk["close"] - bmk["close_prev"]) / bmk["close_prev"]

    spy = bmk[bmk["ticker"] == "SPY"][["date", "open", "close_prev", "daily_ret"]].rename(
        columns={"open": "spy_open", "close_prev": "spy_close_prev", "daily_ret": "spy_daily_ret"}
    )
    qqq = bmk[bmk["ticker"] == "QQQ"][["date", "open", "close_prev"]].rename(
        columns={"open": "qqq_open", "close_prev": "qqq_close_prev"}
    )

    # Stock-level daily returns
    stk_g = stocks.groupby("ticker")
    stocks["close_prev"] = stk_g["close"].shift(1)
    stocks["daily_ret"] = (stocks["close"] - stocks["close_prev"]) / stocks["close_prev"]

    # Merge benchmarks
    df = stocks.merge(spy, on="date", how="left")
    df = df.merge(qqq, on="date", how="left")

    # Benchmark open returns
    df["spy_open_return"] = (df["spy_open"] - df["spy_close_prev"]) / df["spy_close_prev"]
    df["qqq_open_return"] = (df["qqq_open"] - df["qqq_close_prev"]) / df["qqq_close_prev"]

    # Rolling beta (60d) and correlation (20d) vs SPY using daily returns
    # Shift so we only use prior-day data
    def _rolling_beta(sub: pd.DataFrame, window: int = 60) -> pd.Series:
        """OLS beta of stock vs SPY daily returns over a rolling window."""
        out = pd.Series(np.nan, index=sub.index)
        stk_ret = sub["daily_ret"].values
        spy_ret = sub["spy_daily_ret"].values
        for i in range(window - 1, len(stk_ret)):
            y = stk_ret[i - window + 1 : i + 1]
            x = spy_ret[i - window + 1 : i + 1]
            mask = ~(np.isnan(y) | np.isnan(x))
            if mask.sum() < window * 0.5:
                continue
            x_m, y_m = x[mask], y[mask]
            x_dm = x_m - x_m.mean()
            denom = (x_dm ** 2).sum()
            if denom == 0:
                continue
            out.iloc[i] = (x_dm * (y_m - y_m.mean())).sum() / denom
        return out

    def _rolling_corr(sub: pd.DataFrame, window: int = 20) -> pd.Series:
        """Pearson correlation of stock vs SPY daily returns."""
        return sub["daily_ret"].rolling(window, min_periods=window).corr(
            sub["spy_daily_ret"]
        )

    parts = []
    for ticker, sub in df.groupby("ticker"):
        sub = sub.sort_values("date").copy()
        sub["rolling_beta_spy_60d"] = _rolling_beta(sub, 60)
        sub["rolling_corr_spy_20d"] = _rolling_corr(sub, 20)
        parts.append(sub)
    df = pd.concat(parts)

    # Shift beta/corr to avoid look-ahead
    g = df.groupby("ticker")
    df["rolling_beta_spy_60d"] = g["rolling_beta_spy_60d"].shift(1)
    df["rolling_corr_spy_20d"] = g["rolling_corr_spy_20d"].shift(1)

    # Residual move: stock gap - beta * SPY gap
    # Stock gap comes from daily_features.parquet
    daily_params = daily_features[["date", "ticker", "gap_pct"]].copy()
    daily_params["date"] = pd.to_datetime(daily_params["date"])
    df = df.merge(daily_params, on=["date", "ticker"], how="left")
    
    df["residual_move"] = df["gap_pct"] - df["rolling_beta_spy_60d"] * df["spy_open_return"]

    keep_cols = [
        "date", "ticker",
        "spy_open_return", "qqq_open_return",
        "residual_move", "rolling_beta_spy_60d", "rolling_corr_spy_20d",
    ]
    return df[keep_cols]


def compute_calendar(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Calendar features: day-of-week, month, post-holiday, OPEX week.
    Uses pandas_market_calendars for NYSE schedule.
    """
    import pandas_market_calendars as mcal

    # Unique trading dates from our dataset
    dates = daily[["date"]].drop_duplicates().sort_values("date").copy()
    dates["day_of_week"] = dates["date"].dt.dayofweek   # Mon=0 … Fri=4
    dates["month"] = dates["date"].dt.month

    # NYSE calendar for holiday detection
    nyse = mcal.get_calendar("NYSE")
    min_date = dates["date"].min() - pd.Timedelta(days=10)
    max_date = dates["date"].max() + pd.Timedelta(days=1)
    nyse_schedule = nyse.valid_days(start_date=min_date, end_date=max_date)
    nyse_days = pd.Series(nyse_schedule.tz_localize(None))

    # is_post_holiday: prior trading day is NOT the calendar-previous business day
    def _is_post_holiday(d):
        idx = nyse_days[nyse_days == d].index
        if len(idx) == 0 or idx[0] == 0:
            return np.nan
        prev_trading = nyse_days.iloc[idx[0] - 1]
        prev_calendar_bday = d - pd.offsets.BDay(1)
        return int(prev_trading < prev_calendar_bday)

    dates["is_post_holiday"] = dates["date"].apply(_is_post_holiday).astype(float)

    # is_opex_week: 3rd Friday of each month
    def _opex_fridays(years):
        """Return set of dates that fall in OPEX week (Mon–Fri of 3rd Friday)."""
        opex_weeks = set()
        for yr in years:
            for mo in range(1, 13):
                # Find the 3rd Friday
                first = pd.Timestamp(yr, mo, 1)
                # day_of_week for 1st: 0=Mon..6=Sun
                dow_first = first.dayofweek
                # First Friday offset
                fri_offset = (4 - dow_first) % 7
                third_friday = first + pd.Timedelta(days=fri_offset + 14)
                # The week (Mon-Fri) containing the 3rd Friday
                week_start = third_friday - pd.Timedelta(days=third_friday.dayofweek)
                for i in range(5):
                    opex_weeks.add(week_start + pd.Timedelta(days=i))
        return opex_weeks

    years = dates["date"].dt.year.unique()
    opex_dates = _opex_fridays(years)
    dates["is_opex_week"] = dates["date"].isin(opex_dates).astype(float)

    # Cross-join with all tickers so calendar features have matching keys
    tickers = daily["ticker"].unique()
    cal_rows = []
    for _, row in dates.iterrows():
        for t in tickers:
            cal_rows.append({
                "date": row["date"],
                "ticker": t,
                "day_of_week": row["day_of_week"],
                "month": row["month"],
                "is_post_holiday": row["is_post_holiday"],
                "is_opex_week": row["is_opex_week"],
            })
    cal_df = pd.DataFrame(cal_rows)

    return cal_df


# ═══════════════════════════════════════════════════════════════════════════
# 3. ASSEMBLY & OUTPUT
# ═══════════════════════════════════════════════════════════════════════════

def build_market_context() -> None:
    """Main entry point: compute all market alignment and calendar features."""
    with open(PIPELINE_CFG, "r") as f:
        cfg = yaml.safe_load(f)
    tz = cfg["project"]["timezone"]

    # Load raw data
    daily = load_and_resample(tz)

    # Filter to universe tickers only (including benchmarks)
    universe = load_universe()
    daily = daily[daily["ticker"].isin(universe)].copy()
    
    # Non-benchmark tickers for the final output
    benchmarks = {"SPY", "QQQ"}
    stock_tickers = [t for t in universe if t not in benchmarks]
    
    # Also load daily_features for gap_pct
    print("\nLoading daily_features.parquet for parameter alignment …")
    daily_features = pd.read_parquet(DAILY_FEATURES_PATH)

    print("\nComputing feature groups …")

    # ── Compute each group ──
    mkt_align = compute_market_alignment(daily, daily_features)
    print("  ✓ market_alignment")

    cal = compute_calendar(daily)
    print("  ✓ calendar")

    # ── Merge all groups ──
    # Filter to stock tickers only
    mkt_align = mkt_align[~mkt_align["ticker"].isin(benchmarks)]
    cal = cal[~cal["ticker"].isin(benchmarks)]

    result = mkt_align.merge(cal, on=["date", "ticker"], how="left")

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


if __name__ == "__main__":
    build_market_context()

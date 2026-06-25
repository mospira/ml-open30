"""
sentiment_features.py

Computes daily sentiment features from Alpha Vantage news data,
as defined in configs/features.yaml under `sentiment`.

Features produced:
  - sentiment_avg, article_count, sentiment_sum
  - sentiment_max, sentiment_min, sentiment_std
  - sentiment_avg_x_gap, sentiment_avg_x_atr_pct, sentiment_avg_x_market_dir

News-to-trading-day alignment (ET):
  For a trading day T, aggregate article timestamps in
  [prev_trading_day 00:00, T 09:30] (inclusive).

This includes:
  - prior trading day news,
  - weekend/non-trading-day news for Monday,
  - same-day pre-open news through 09:30 ET.

Output: data/processed/features/sentiment_features.parquet
"""

import os
import numpy as np
import pandas as pd

from src.utils import load_universe

# Paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

SENTIMENT_PATH = os.path.join(
    PROJECT_ROOT, "data", "interim", "canonical", "sentiment_scores.parquet"
)
DAILY_FEATURES_PATH = os.path.join(
    PROJECT_ROOT, "data", "processed", "features", "daily_features.parquet"
)
MARKET_CONTEXT_PATH = os.path.join(
    PROJECT_ROOT, "data", "processed", "features", "market_context.parquet"
)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "features")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "sentiment_features.parquet")

# 09:30 ET pre-open cutoff for same-day sentiment inclusion.
PREOPEN_CUTOFF = pd.Timedelta(hours=9, minutes=30)


def build_alignment_mapping(daily_features: pd.DataFrame) -> pd.DataFrame:
    """
    Build article-date -> target-trading-date mapping.

    For each trading day T (excluding the first available day), we include:
      1) full calendar dates in [prev_trading_day, T) and
      2) date T itself, but only for timestamps <= T 09:30 ET.

    The first available trading day has no prior trading day in-sample, so
    no sentiment window is constructed for it.
    """
    trading_dates = (
        pd.Series(pd.to_datetime(daily_features["date"].unique()))
        .sort_values()
        .reset_index(drop=True)
    )
    date_map = pd.DataFrame({"date": trading_dates, "prev_date": trading_dates.shift(1)})

    rows = []
    for _, row in date_map.dropna().iterrows():
        t_date = row["date"]
        p_date = row["prev_date"]

        # Full dates from previous trading day up to (but excluding) trading day T.
        curr = p_date
        while curr < t_date:
            rows.append(
                {
                    "cal_date": curr,
                    "target_trading_date": t_date,
                    "preopen_only": False,
                    "preopen_cutoff": pd.NaT,
                }
            )
            curr += pd.Timedelta(days=1)

        # Same trading day T, pre-open only (00:00-09:30 ET inclusive).
        rows.append(
            {
                "cal_date": t_date,
                "target_trading_date": t_date,
                "preopen_only": True,
                "preopen_cutoff": t_date + PREOPEN_CUTOFF,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=["cal_date", "target_trading_date", "preopen_only", "preopen_cutoff"]
        )

    return pd.DataFrame(rows)


def extract_base_features_from_raw(cal_mapping: pd.DataFrame) -> pd.DataFrame:
    """
    Merge raw articles with the alignment mapping, apply pre-open filter for
    same-day rows, then aggregate to (date, ticker).
    """
    if not os.path.exists(SENTIMENT_PATH):
        raise FileNotFoundError(f"Missing canonical sentiment data at {SENTIMENT_PATH}")

    if cal_mapping.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "ticker",
                "article_count",
                "sentiment_avg",
                "sentiment_sum",
                "sentiment_max",
                "sentiment_min",
                "sentiment_std",
            ]
        )

    print("Loading sentiment scores ...")
    df = pd.read_parquet(SENTIMENT_PATH)

    # Canonical timestamps are ET wall-clock timestamps (tz-naive).
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["overall_sentiment_score"] = pd.to_numeric(df["overall_sentiment_score"], errors="coerce")
    df = df.dropna(subset=["overall_sentiment_score"])
    df["cal_date"] = df["timestamp"].dt.normalize()

    print(
        f"  {len(df):,} articles from "
        f"{df['cal_date'].min().date()} to {df['cal_date'].max().date()}"
    )

    mapped = pd.merge(df, cal_mapping, on="cal_date", how="inner")

    # Keep all rows for prior-day/weekend mapping, and pre-open rows for same-day mapping.
    keep_mask = (~mapped["preopen_only"]) | (mapped["timestamp"] <= mapped["preopen_cutoff"])
    mapped = mapped[keep_mask].copy()

    agg = (
        mapped.groupby(["target_trading_date", "ticker"])
        .agg(
            article_count=("overall_sentiment_score", "count"),
            sentiment_avg=("overall_sentiment_score", "mean"),
            sentiment_sum=("overall_sentiment_score", "sum"),
            sentiment_max=("overall_sentiment_score", "max"),
            sentiment_min=("overall_sentiment_score", "min"),
            sentiment_std=("overall_sentiment_score", "std"),
        )
        .reset_index()
    )

    agg.rename(columns={"target_trading_date": "date"}, inplace=True)
    agg["sentiment_std"] = agg["sentiment_std"].fillna(0.0)

    print(f"  Aggregated to {len(agg):,} trading-day ticker records")
    return agg


def build_sentiment_features() -> None:
    """Main entry point: compute all sentiment features and write parquet."""
    universe = load_universe()
    benchmarks = {"SPY", "QQQ"}
    stock_tickers = [t for t in universe if t not in benchmarks]

    print("Loading daily_features for alignment & interactions ...")
    daily_df = pd.read_parquet(
        DAILY_FEATURES_PATH,
        columns=["date", "ticker", "gap_pct", "atr_pct"],
    )
    daily_df["date"] = pd.to_datetime(daily_df["date"])

    print("Loading market_context for benchmark returns ...")
    mkt_df = pd.read_parquet(
        MARKET_CONTEXT_PATH,
        columns=["date", "ticker", "spy_open_return"],
    )
    mkt_df["date"] = pd.to_datetime(mkt_df["date"])
    daily_df = daily_df.merge(mkt_df, on=["date", "ticker"], how="left")

    print("Building trading-day sentiment alignment map ...")
    cal_mapping = build_alignment_mapping(daily_df)
    print(f"  Mapping rows: {len(cal_mapping):,}")

    print("Extracting base sentiment features ...")
    base_features = extract_base_features_from_raw(cal_mapping)

    # Left join onto daily grid to preserve all stock date-ticker rows.
    grid = daily_df[daily_df["ticker"].isin(stock_tickers)].copy()
    result = pd.merge(grid, base_features, on=["date", "ticker"], how="left")

    # Fill missing news with 0 counts/sums; leave avg/min/max/std as NaN.
    result["article_count"] = result["article_count"].fillna(0)
    result["sentiment_sum"] = result["sentiment_sum"].fillna(0.0)

    print("Computing interaction terms ...")
    result["sentiment_avg_x_gap"] = result["sentiment_avg"] * result["gap_pct"]
    result["sentiment_avg_x_atr_pct"] = result["sentiment_avg"] * result["atr_pct"]
    result["sentiment_avg_x_market_dir"] = result["sentiment_avg"] * np.sign(
        result["spy_open_return"]
    )

    keep_cols = [
        "date",
        "ticker",
        "sentiment_avg",
        "article_count",
        "sentiment_sum",
        "sentiment_max",
        "sentiment_min",
        "sentiment_std",
        "sentiment_avg_x_gap",
        "sentiment_avg_x_atr_pct",
        "sentiment_avg_x_market_dir",
    ]
    result = result[keep_cols].sort_values(["ticker", "date"]).reset_index(drop=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tmp_file = OUTPUT_FILE + ".tmp"
    result.to_parquet(tmp_file, index=False)
    os.replace(tmp_file, OUTPUT_FILE)

    print(f"\nWrote {len(result):,} rows x {len(result.columns)} cols -> {OUTPUT_FILE}")
    print(f"Columns: {sorted(result.columns.tolist())}")

    has_news = result[result["article_count"] > 0]
    print(f"Rows with news: {len(has_news):,} ({len(has_news)/len(result):.1%})")
    if not has_news.empty:
        print(f"Max articles in one row: {has_news['article_count'].max()}")


if __name__ == "__main__":
    build_sentiment_features()

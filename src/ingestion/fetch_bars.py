import os
import time
import json
import yaml
import requests
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv
from datetime import datetime
import argparse
from src.utils import load_pipeline_config as load_config, load_universe

# ── paths ──────────────────────────────────────────────────────────
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
ENV_PATH = os.path.join(PROJECT_ROOT, '.env')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'candles_1m.parquet')

# ── config ─────────────────────────────────────────────────────────
load_dotenv(ENV_PATH)
API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
BASE_URL = "https://www.alphavantage.co/query"

# 75 calls / min  →  1 call every 0.8 s  (use 0.85 s for safety)
MIN_INTERVAL = 0.85
LAST_CALL_TIME = 0




def rate_limit_sleep():
    """Enforce ≤75 calls / minute."""
    global LAST_CALL_TIME
    elapsed = time.time() - LAST_CALL_TIME
    if elapsed < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - elapsed)
    LAST_CALL_TIME = time.time()


def generate_months(start_ms, end_ms):
    """Return a list of 'YYYY-MM' strings covering [start_ms, end_ms]."""
    start_dt = datetime.utcfromtimestamp(start_ms / 1000.0)
    end_dt = datetime.utcfromtimestamp(end_ms / 1000.0)

    months = []
    y, m = start_dt.year, start_dt.month
    while (y, m) <= (end_dt.year, end_dt.month):
        months.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months


def fetch_month(ticker, month, api_key):
    """Fetch full 1-min intraday data for *ticker* in *month* (YYYY-MM).

    Returns a list of dicts with keys:
        timestamp, open, high, low, close, volume
    or an empty list on failure / no data.
    """
    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": ticker,
        "interval": "1min",
        "month": month,
        "outputsize": "full",
        "extended_hours": "true",
        "adjusted": "true",
        "datatype": "json",
        "apikey": api_key,
    }

    rate_limit_sleep()

    try:
        resp = requests.get(BASE_URL, params=params, timeout=30)
    except requests.RequestException as e:
        print(f"    Request error for {ticker} {month}: {e}")
        return []

    if resp.status_code != 200:
        print(f"    HTTP {resp.status_code} for {ticker} {month}")
        return []

    data = resp.json()

    # Handle API-level errors / rate-limit notes
    if "Note" in data:
        print(f"    API Note: {data['Note']}")
        if "call frequency" in data["Note"].lower():
            print("    Backing off 60 s …")
            time.sleep(60)
        return []

    if "Information" in data:
        print(f"    API Info: {data['Information']}")
        return []

    ts_key = "Time Series (1min)"
    if ts_key not in data:
        print(f"    No '{ts_key}' key for {ticker} {month}")
        return []

    series = data[ts_key]
    rows = []
    for ts_str, bar in series.items():
        rows.append({
            "timestamp": ts_str,
            "open":   float(bar["1. open"]),
            "high":   float(bar["2. high"]),
            "low":    float(bar["3. low"]),
            "close":  float(bar["4. close"]),
            "volume": int(bar["5. volume"]),
        })
    return rows


def fetch_ticker(ticker, months, api_key):
    """Fetch all months for a single ticker, return a DataFrame."""
    all_rows = []
    for month in months:
        rows = fetch_month(ticker, month, api_key)
        if rows:
            all_rows.extend(rows)
        # If the API returned a rate-limit note with 0 rows, retry once
        if not rows:
            # Retry after a short pause
            time.sleep(2)
            rows = fetch_month(ticker, month, api_key)
            if rows:
                all_rows.extend(rows)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # Alpha Vantage timestamps are US/Eastern (no tz label in the string).
    # Localize → convert to UTC to stay consistent with pipeline.yaml timezone.
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["timestamp"] = (
        df["timestamp"]
        .dt.tz_localize("America/New_York")
        .dt.tz_convert("UTC")
    )
    df["ticker"] = ticker
    df = df[["timestamp", "ticker", "open", "high", "low", "close", "volume"]]
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser(description="Fetch 1-min bars (Alpha Vantage).")
    parser.add_argument("--test", action="store_true",
                        help="Test mode: 1 ticker, 1 month only")
    args = parser.parse_args()

    if not API_KEY:
        print("Error: ALPHAVANTAGE_API_KEY not found in .env")
        return

    config = load_config()
    start_ms = config["data_window"]["start_date"]
    end_ms   = config["data_window"]["end_date"]

    months = generate_months(start_ms, end_ms)
    universe = load_universe()

    if args.test:
        print("── TEST MODE ──")
        universe = universe[:1]
        months = months[:1]
        print(f"  Ticker : {universe[0]}")
        print(f"  Month  : {months[0]}")

    total_calls = len(universe) * len(months)
    est_minutes = total_calls * MIN_INTERVAL / 60
    print(f"Tickers: {len(universe)}  |  Months: {len(months)}  |  "
          f"API calls: {total_calls}  |  Est. time: {est_minutes:.1f} min")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    schema = None
    writer = None

    try:
        if os.path.exists(OUTPUT_FILE):
            os.remove(OUTPUT_FILE)

        for i, ticker in enumerate(universe):
            print(f"[{i+1}/{len(universe)}] {ticker}  ({len(months)} months)")
            df = fetch_ticker(ticker, months, API_KEY)

            if df.empty:
                print(f"  ⚠ No data for {ticker}")
                continue

            print(f"  → {len(df):,} rows")

            table = pa.Table.from_pandas(df)
            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(OUTPUT_FILE, schema)

            if table.schema != schema:
                table = table.cast(schema)

            writer.write_table(table)

    except KeyboardInterrupt:
        print("\nInterrupted – closing writer …")
    except Exception as e:
        print(f"Fatal error: {e}")
    finally:
        if writer:
            writer.close()
            print(f"Saved → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

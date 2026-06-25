import os
import json
import yaml
import requests
import pandas as pd
import time
from datetime import datetime, timezone
from dotenv import load_dotenv
import argparse
import glob
from src.utils import load_pipeline_config as load_config, load_universe

# Setup paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
ENV_PATH = os.path.join(PROJECT_ROOT, '.env')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw', 'news_daily')

# Load configurations
load_dotenv(ENV_PATH)
API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")

LAST_CALL_TIME = 0



def ms_to_av_format(ms_timestamp):
    """Convert millisecond timestamp to YYYYMMDDTHHMM format expected by AlphaVantage."""
    dt = datetime.fromtimestamp(ms_timestamp / 1000.0, tz=timezone.utc)
    return dt.strftime('%Y%m%dT%H%M')

def av_format_to_dt(av_string):
    """Convert YYYYMMDDTHHMMMWWW format to datetime."""
    # The format in response relies on 'time_published' field e.g., "20240202T000000"
    try:
        return datetime.strptime(av_string, '%Y%m%dT%H%M%S')
    except ValueError:
        try:
             # Fallback if seconds are missing or format varies
             return datetime.strptime(av_string, '%Y%m%dT%H%M')
        except ValueError:
             return None

def rate_limit_sleep():
    """Ensure we don't exceed 75 calls per minute (approx 1 call every 0.8s)."""
    global LAST_CALL_TIME
    elapsed = time.time() - LAST_CALL_TIME
    wait_time = 0.9 - elapsed # slightly conservative 0.9s
    if wait_time > 0:
        time.sleep(wait_time)
    LAST_CALL_TIME = time.time()
    
def fetch_news_for_ticker(ticker, start_ms, end_ms, api_key):
    print(f"Fetching news for {ticker}...")
    
    # Initial time_from
    start_dt_str = ms_to_av_format(start_ms)
    end_dt_str = ms_to_av_format(end_ms)
    
    current_time_from_str = start_dt_str
    
    all_articles = []
    
    # We loop until we cover the range or run out of valid responses
    # The API sorts by EARLIEST, so we page forward in time.
    
    page_num = 0
    
    while True:
        page_num += 1
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": ticker,
            "apikey": api_key,
            "time_from": current_time_from_str,
            "time_to": end_dt_str,
            "limit": 1000,
            "sort": "EARLIEST" 
        }
        
        # Only print url for debugging if needed, but hide key
        # print(f"DEBUG: Requesting {ticker} from {current_time_from_str}")
        
        rate_limit_sleep()
        try:
            response = requests.get(url, params=params)
            
            if response.status_code != 200:
                print(f"  Error: Status {response.status_code} - {response.text}")
                break
                
            data = response.json()
            
            if "feed" not in data:
                # Handle API limit or empty responses gracefully
                if "Note" in data:
                    print(f"  API Note: {data['Note']}") 
                    # If rate limit hit aggressively, maybe sleep more?
                    if "Call frequency" in data['Note']:
                         time.sleep(60) # Back off for a minute
                         continue
                if "Information" in data:
                    print(f"  API Info: {data['Information']}")
                # If just empty but no error, break
                if not data:
                    pass
                break
                
            feed = data["feed"]
            if not feed:
                print(f"  No articles found for range starting {current_time_from_str}.")
                break
                
            count = len(feed)
            print(f"  Page {page_num}: retrieved {count} articles starting from {current_time_from_str}")
            
            all_articles.extend(feed)
            
            # If we got fewer than limit, we're likely done with this period
            if count < 1000:
                print("  Reached end of available data for this range.")
                break
                
            # If we hit the limit, we need to paginate.
            last_article = feed[-1]
            last_time_str = last_article.get("time_published")
            
            if not last_time_str:
                print("  Error: Last article has no time_published. Cannot paginate.")
                break
            
            # Check for stagnation
            if current_time_from_str == last_time_str:
                print("  Warning: time_from is failing to advance. Breaking to prevent infinite loop.")
                break
            
            # The API expects YYYYMMDDTHHMM. The response contains seconds (YYYYMMDDTHHMMSS).
            # We need to truncate/format it back to YYYYMMDDTHHMM.
            # However, since sort=EARLIEST, using the same minute might return the same results.
            # We rely on deduplication in save_data.
            if len(last_time_str) > 13:
                 current_time_from_str = last_time_str[:13]
            else:
                 current_time_from_str = last_time_str
            
        except Exception as e:
            print(f"  Error fetching data: {e}")
            break
            
    return all_articles

def save_data(articles, ticker):
    if not articles:
        print(f"No articles to save for {ticker}.")
        return

    # Convert to DataFrame
    df = pd.DataFrame(articles)
    
    # Deduplicate
    initial_len = len(df)
    # Use url as primary key if available
    if 'url' in df.columns:
         df = df.drop_duplicates(subset=['url'])
    else:
         df = df.drop_duplicates()
         
    if len(df) < initial_len:
        print(f"  Deduplicated {initial_len - len(df)} rows.")

    print(f"Saving {len(df)} articles for {ticker}...")
    
    # Ensure output dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    output_file = os.path.join(OUTPUT_DIR, f"{ticker}_news.parquet")
    
    try:
        df.to_parquet(output_file, index=False)
        print(f"Saved -> {output_file}")
    except Exception as e:
        print(f"Error saving parquet: {e}")
        # If failure, try converting complex cols to json strings
        print("  Attempting to convert complex columns to strings...")
        for col in df.columns:
            # Check if column object type
            if df[col].dtype == object:
                # Naive check if it looks like list/dict
                df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, (list, dict)) else x)
        try:
             df.to_parquet(output_file, index=False)
             print(f"Saved with stringified columns -> {output_file}")
        except Exception as e2:
             print(f"  Failed to save even with string conversion: {e2}")

def main():
    parser = argparse.ArgumentParser(description='Fetch news sentiment for tickers.')
    parser.add_argument('--test', action='store_true', help='Run in test mode (1 ticker)')
    args = parser.parse_args()
    
    if not API_KEY:
        print("Error: ALPHAVANTAGE_API_KEY not found in .env")
        return

    config = load_config()
    start_date_ms = config['data_window']['start_date']
    end_date_ms = config['data_window']['end_date']
    
    universe = load_universe()
    
    if args.test:
        print("Running in TEST mode (first 1 ticker).")
        universe = universe[:1]
        
    print(f"Timeframe: {ms_to_av_format(start_date_ms)} to {ms_to_av_format(end_date_ms)}")
    print(f"Tickers to process: {len(universe)}")
    
    for i, ticker in enumerate(universe):
        print(f"[{i+1}/{len(universe)}] Fetching {ticker}...")
        articles = fetch_news_for_ticker(ticker, start_date_ms, end_date_ms, API_KEY)
        save_data(articles, ticker)
        
    print("Done.")

if __name__ == "__main__":
    main()

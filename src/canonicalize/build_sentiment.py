import os
import glob
import json
import pandas as pd

# Setup paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
INPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw', 'news_daily')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'interim', 'canonical')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'sentiment_scores.parquet')

def process_sentiment():
    print(f"Reading parquet files from {INPUT_DIR}...")
    file_pattern = os.path.join(INPUT_DIR, '*_news.parquet')
    parquet_files = glob.glob(file_pattern)
    
    if not parquet_files:
        print("No parquet files found in the input directory.")
        return
        
    all_articles = []
    
    for file_path in parquet_files:
        filename = os.path.basename(file_path)
        ticker = filename.split('_news')[0]
        
        try:
            df = pd.read_parquet(file_path)
            
            if df.empty or 'time_published' not in df.columns or 'overall_sentiment_score' not in df.columns:
                print(f"Skipping {ticker}: Missing required columns or empty dataframe.")
                continue
                
            # 'time_published' format: 20240203T110000 (US/Eastern timezone)
            df['timestamp'] = pd.to_datetime(df['time_published'], format='%Y%m%dT%H%M%S')
            
            # Ensure sentiment score is numeric
            df['overall_sentiment_score'] = pd.to_numeric(df['overall_sentiment_score'], errors='coerce')
            df = df.dropna(subset=['overall_sentiment_score'])
            
            if df.empty:
                continue
            
            # Keep only the columns we need: one row per article
            article_df = df[['timestamp', 'overall_sentiment_score']].copy()
            article_df['ticker'] = ticker
            
            all_articles.append(article_df)
            
        except Exception as e:
            print(f"Error processing {ticker} ({file_path}): {e}")
            
    if not all_articles:
        print("No data processed successfully.")
        return
        
    combined_df = pd.concat(all_articles, ignore_index=True)
    combined_df = combined_df[['timestamp', 'ticker', 'overall_sentiment_score']]
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tmp_file = OUTPUT_FILE + '.tmp'
    combined_df.to_parquet(tmp_file, index=False)
    os.replace(tmp_file, OUTPUT_FILE)
    
    print(f"Successfully saved to {OUTPUT_FILE}")
    print(f"Total rows: {len(combined_df)}")
    print("Sample:\n", combined_df.head())

if __name__ == "__main__":
    process_sentiment()

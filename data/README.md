# Data

Large data artifacts are intentionally not tracked.

The assembled feature and label dataset is hosted on Hugging Face for users
who do not want to rerun Alpha Vantage ingestion and feature generation:

[mospira/open30-equity-features](https://huggingface.co/datasets/mospira/open30-equity-features)

Download `dataset_open30m.parquet` from that dataset and place it at
`data/processed/dataset_open30m.parquet` before running backtests or retraining.

Expected generated paths:

- `data/raw/candles_1m.parquet`
- `data/raw/news_daily/*.parquet`
- `data/interim/canonical/sentiment_scores.parquet`
- `data/processed/features_table.parquet`
- `data/processed/trade_instances.parquet`
- `data/processed/labels/labels.parquet`
- `data/processed/dataset_open30m.parquet`

The tracked universe files are:

- `data/interim/canonical/universe_top25.json`
- `data/interim/canonical/universe_sp500.json`

The default pipeline uses Alpha Vantage for minute bars and news sentiment.
Provide `ALPHAVANTAGE_API_KEY` in a local `.env` file before running ingestion.
The tracked results were generated from an extended history build. The public
pipeline config starts on `2010-02-02`; feature warm-up means the main tracked
backtest rows begin on `2011-10-06`.

If you use a frozen external dataset instead of re-downloading, place files at
the generated paths above and keep a checksum manifest with the dataset release.

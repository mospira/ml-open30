# Dependency Graph

This graph reflects the current implementation in this repository.

## 1) Execution graph

```mermaid
flowchart TD
  A[run_pipeline.py] --> B[src/ingestion/fetch_bars.py]
  A --> C[src/ingestion/fetch_sentiment.py]
  A --> D[src/canonicalize/build_sentiment.py]
  A --> E[src/features/daily_features.py]
  A --> F[src/features/open_features.py]
  A --> G[src/features/market_context.py]
  A --> H[src/features/sentiment_features.py]
  A --> I[src/features/assemble_features.py]
  A --> J[src/dataset/build_instances.py]
  A --> K[src/labeling/generate_labels.py]
  A --> L[src/dataset/assemble_dataset.py]

  B --> R1[data/raw/candles_1m.parquet]
  C --> R2[data/raw/news_daily/*.parquet]
  D --> R3[data/interim/canonical/sentiment_scores.parquet]
  E --> R4[data/processed/features/daily_features.parquet]
  F --> R5[data/processed/features/open_features.parquet]
  G --> R6[data/processed/features/market_context.parquet]
  H --> R7[data/processed/features/sentiment_features.parquet]
  I --> R8[data/processed/features_table.parquet]
  J --> R9[data/processed/trade_instances.parquet]
  K --> R10[data/processed/labels/labels.parquet]
  L --> R11[data/processed/dataset_open30m.parquet]
```

## 2) Artifact dependency graph

```mermaid
flowchart LR
  P1[configs/pipeline.yaml] --> B
  P1 --> C
  P1 --> E
  P1 --> F
  P1 --> G

  P2[configs/features.yaml] --> I
  P3[configs/labels.yaml] --> K

  B[src/ingestion/fetch_bars.py] --> A1[data/raw/candles_1m.parquet]
  C[src/ingestion/fetch_sentiment.py] --> A2[data/raw/news_daily/*.parquet]
  A2 --> D[src/canonicalize/build_sentiment.py]
  D --> A3[data/interim/canonical/sentiment_scores.parquet]

  A1 --> E[src/features/daily_features.py]
  A1 --> F[src/features/open_features.py]
  A1 --> G[src/features/market_context.py]
  A3 --> H[src/features/sentiment_features.py]
  E --> H
  G --> H

  E --> I[src/features/assemble_features.py]
  F --> I
  G --> I
  H --> I
  I --> A4[data/processed/features_table.parquet]

  E --> J[src/dataset/build_instances.py]
  J --> A5[data/processed/trade_instances.parquet]

  A1 --> K[src/labeling/generate_labels.py]
  E --> K
  A5 --> K
  K --> A6[data/processed/labels/labels.parquet]

  A5 --> L[src/dataset/assemble_dataset.py]
  A4 --> L
  A6 --> L
  L --> A7[data/processed/dataset_open30m.parquet]
  A7 --> N[run_backtest.py]
  P4[models/v1/best_params.json optional] --> N
  N --> O1[reports/<arch>/*_trades.csv]
  N --> O2[reports/<arch>/rolling_pnl_chart.png]
  N --> O3[reports/<arch>/*_reliability.png]
```

## 3) Backtest internal flow (current code path)

```mermaid
flowchart TD
  B0[Load dataset_open30m.parquet] --> B1[Loop each date in dataset]
  B1 --> B3{Retrain needed?}
  B3 -->|yes| B4[Build rolling train window]
  B4 --> B5[Filter train rows to side == long]
  B5 --> B6[Temporal split 70/15/15]
  B6 --> B7[Train multi-head XGBoost]
  B7 --> B8[Calibrate each class with isotonic]
  B8 --> B9[Predict probas for day candidates]
  B9 --> B10[Compute EV]
  B10 --> B11[Force short EV to -inf]
  B11 --> B12[Pick best candidate above threshold]
  B12 --> B13[Half-Kelly sizing with floor and cap]
  B13 --> B14[Record trade and account value]
```

## Notes

- In `run_pipeline.py`, category flags skip whole categories, but remaining steps are executed with `force_fresh=True`.
- In `run_backtest.py`, evaluation currently runs on the full assembled dataset timeline.
- Current implemented calibration is isotonic regression.

## 4) Local bundle export flow

```mermaid
flowchart LR
  S0[run_retrain_latest.py] --> S1[data/processed/dataset_open30m.parquet]
  S0 --> S4[models/v1/best_params.json optional]

  S0 --> O1[models/live/YYYYMMDD/model_heads.pkl]
  S0 --> O2[models/live/YYYYMMDD/calibrators.pkl]
  S0 --> O3[models/live/YYYYMMDD/feature_columns.json]
  S0 --> O4[models/live/YYYYMMDD/time_r_lookup.json]
  S0 --> O5[models/live/YYYYMMDD/decision_config.json]
  S0 --> O6[models/live/YYYYMMDD/train_window_meta.json]
  S0 --> O7[models/live/YYYYMMDD/bundle_hash.txt]
  S0 --> O8[models/live/latest.json]
```

# Results

This folder contains curated artifacts copied from completed local backtests.
The full generated report directories are intentionally not tracked.

Tracked result groups:

- `v1/`: raw-EV baseline.
- `v1_1/`, `v1_2/`, `v1_3/`: single-head threshold experiments.
- `v2/`: multi-head expected-return meta selector.
- `stop_distance_sweep/`: ATR stop-distance comparison.
- `improvement_audit/`: research audit and model-experiment summary.

Each architecture folder keeps summary metrics, run metadata, the manifest
snapshot, equity curves, and selected reliability diagnostics. Full daily and
trade CSVs can be regenerated under `reports/`.

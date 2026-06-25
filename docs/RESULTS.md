# Results

The tracked result artifacts are in `results/`. This summary highlights the
main current outputs without requiring the full generated `reports/` tree.

## Single-Head v1.x Branch

All rows use the same assembled dataset path recorded in run metadata. Account
return is not always a clean comparison because `open30_v1` used a different
starting capital from `v1_1` through `v1_3`.
The public pipeline config starts ingestion on `2010-02-02`; feature warm-up
produces the tracked main result start date of `2011-10-06`.

| Architecture | Strategy | Trades | Mean trade return | Profit factor | Max drawdown |
|---|---:|---:|---:|---:|---:|
| open30_v1 | m=1.5 | 3,413 | 23.28 bps | 1.081 | -49.24% |
| v1_1 | m=1.5 | 2,083 | 50.22 bps | 1.166 | -48.31% |
| v1_2 | m=1.5 | 1,957 | 53.90 bps | 1.166 | -47.11% |
| v1_3 | m=1.5 | 1,865 | 56.82 bps | 1.169 | -49.16% |

![v1.x metric comparison](../site/assets/results/summary/v1_metric_comparison.png)

![v1.x normalized equity curves](../site/assets/results/summary/v1_normalized_equity.png)

![v1.3 equity curve](../site/assets/results/v1_3/rolling_pnl_chart.png)

## Multi-Head v2

| Strategy | Trades | Mean trade return | Profit factor | Max drawdown |
|---|---:|---:|---:|---:|
| m=0.5 | 873 | 21.89 bps | 1.103 | -52.60% |
| m=1.0 | 2,089 | 17.40 bps | 1.054 | -76.25% |
| m=1.5 | 1,972 | 42.83 bps | 1.090 | -52.60% |
| m=2.0 | 2,080 | 27.65 bps | 1.125 | -57.29% |
| Best EV | 3,254 | 18.50 bps | 1.038 | -90.66% |

![v2 strategy comparison](../site/assets/results/summary/v2_strategy_comparison.png)

![v2 equity curve](../site/assets/results/v2/rolling_pnl_chart.png)

## ATR Stop-Distance Sweep

The stop sweep covers a shorter recent period than the full-history v1.x
results, so it should not be compared as a clean full-history improvement.

| Mode | Stop k | Trades | Mean trade return | Profit factor | Return |
|---|---:|---:|---:|---:|---:|
| static | 0.1 | 364 | 10.35 bps | 1.147 | 40.23% |
| static | 0.2 | 358 | 20.72 bps | 1.193 | 92.20% |
| static | 0.3 | 358 | 15.75 bps | 1.413 | 65.91% |
| static | 0.4 | 346 | -1.84 bps | 0.935 | -9.84% |
| static | 0.5 | 287 | 23.41 bps | 1.104 | 60.12% |
| dynamic | n/a | 317 | 12.07 bps | 1.228 | 37.45% |

![stop-distance metrics](../site/assets/results/summary/stop_sweep_metrics.png)

![stop-distance sweep](../site/assets/results/stop_distance_sweep/comparison_equity_curves.png)

## Improvement Audit

The improvement audit found that the edge is execution-sensitive. The strongest
tested directions were a fixed cost-aware long EV floor near `0.10`, a separate
short-only model with a stricter threshold, realistic realized costs, and a
point-in-time universe.

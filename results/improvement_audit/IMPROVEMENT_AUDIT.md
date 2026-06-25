# Open30 Improvement Audit

## Bottom line

The long-only `m=1.5` strategy has a real but execution-sensitive edge. The
current v1 walk-forward trades average about 6.4 bps of gross underlying
position return. Using 4 bps round-trip as the intended-cost proxy leaves only
2.4 bps/trade, and the edge turns negative near 7 bps.

The strongest tested improvements are:

1. Use a fixed, cost-aware EV floor near `0.10` for the long strategy.
2. Add a separately trained short-only strategy with a stricter EV floor near
   `0.25`, then share capital when both sides trade.
3. Make costs and realistic entry execution part of realized backtest PnL.

The largest unresolved research risk is the fixed present-day top-25 universe
with no historical membership. This creates survivorship and concentration
risk that can materially overstate the historical edge.

## Comparable results

All net figures below subtract a 4 bps round-trip cost from underlying position
returns. This is an illustrative proxy close to the current `cost_R=0.05`
setting at the average stop distance. The canonical backtest does not currently
deduct `cost_R` from realized PnL.

| Variant | Period | Trades | Net bps/trade | Net return sum | Positive years |
|---|---:|---:|---:|---:|---:|
| Current v1, EV > 0 | 2011-2026 | 3,413 | 2.42 | 0.82 | 9/16 |
| Current v1, fixed EV > 0.10 | 2011-2026 | 2,372 | 6.16 | 1.46 | 13/16 |
| Long, 1,095d lookback / 90d retrain / no dynamic features / EV > 0.10 | 2012-2026 | 2,319 | 5.63 | 1.31 | 12/15 |
| Short-only, same model schedule / EV > 0.25 | 2012-2026 | 828 | 6.26 | 0.52 | 11/15 |
| Long + short capital-sharing combination | 2012-2026 | up to 2/day | n/a | 1.88 | 14/15 |
| Three staggered 90d long models, majority consensus | 2012-2026 | 1,624 | 8.51 | 1.38 | 13/15 |

Monthly block-bootstrap 95% intervals for cost-adjusted return sum:

- Current v1, EV > 0: `[-0.12, 1.80]`
- Current v1, fixed EV > 0.10: `[0.64, 2.28]`
- Long 1,095d / 90d candidate: `[0.50, 2.12]`

## High likelihood

### Fixed cost-aware EV gate

Use a fixed long-side EV floor around `0.10` as the default. It removes weak
trades, raises net edge from 2.4 to 6.2 bps/trade, increases cost-adjusted return
sum from 0.82 to 1.46, and produces a positive monthly-bootstrap interval.

An EV floor around `0.20` trades less and has better consistency/profit factor,
but lower total return. Treat `0.10` as the growth setting and `0.20` as the
more conservative setting.

The current dynamic threshold objective maximizes mean return per selected
trade, which rewards skipping more days. A fixed `0.10` threshold beat the
tested dynamic policies on total cost-adjusted return.

### Separate short-side strategy

Train short candidates in a separate model. Do not combine long and short rows
in the current model because `side` is excluded from the feature set.

The tested short model is weaker than the long model, but its daily returns are
slightly negatively correlated with long returns. With a fixed short EV floor
near `0.25` and equal capital sharing on days when both trade, the combined
cost-adjusted return sum rose about 44%, from 1.31 to 1.88, and 14 of 15 years
were positive.

This requires separate live validation for borrow availability, short
execution, and costs.

### Execution-aware labels and PnL

Deduct execution costs from realized PnL, not only from predicted EV. The
current v1 gross edge is 6.4 bps/trade and becomes negative around 7 bps of
round-trip cost. Entry labels also assume the exact `09:31` open, while a live
market order necessarily arrives later.

Measure live decision-to-fill slippage and rerun labels/backtests using a
realistic delayed entry or slippage distribution. This is the highest-priority
robustness improvement because it determines whether the edge is tradable.

### Historical and broader liquid universe

Replace the fixed current top-25 universe with point-in-time membership and a
liquidity screen, then test a broader candidate set. The current baseline's top
two tickers contribute about 48% of gross return; the improved long candidate
reduces that to about 34%, but concentration remains meaningful.

This may reduce the reported historical result, but it is likely to improve the
strategy's real robustness and opportunity set.

## Medium likelihood

### Staggered retrain ensemble

Longer lookbacks and slower retraining can help, but the result is highly
sensitive to the retrain calendar phase. On the recent controlled period, the
same 1,095-day / 90-day setup produced 1.7 to 5.9 net bps/trade when its start
date shifted by 30 or 60 days.

A three-model staggered ensemble with a majority-consensus gate is more robust.
It produced 8.5 net bps/trade on full history and reduced recent-period
drawdown, but did not always maximize total return.

### Cross-fitted calibration and daily-net threshold objective

The selected-trade reliability tables show material probability
miscalibration. The current workflow also reuses the calibration slice for
permutation feature selection, early stopping, and isotonic calibration.

Use rolling cross-fitted predictions for calibration and tune the EV gate on
net return per available trading day or expected log growth, not mean return
per selected trade.

### Stop-distance and horizon heads

Stop geometry matters, but the recent stop sweep is unstable. Static
`k=0.2` produced the highest recent account return, while the no-lookahead
dynamic selector underperformed several static choices and the best setting
changed sharply by quarter.

Test stop distance and holding horizon jointly as a small set of separate
heads, including 15, 30, 45, and 60 minutes. Require stability across retrain
anchors and time blocks before promotion.

### Focused feature work

Market-alignment features added value in the recent ablation. First-minute
features added a smaller amount. Sentiment was useful in one controlled window
but harmful/unstable in another full-history comparison.

The best feature candidates are richer market regime and opening microstructure
inputs: premarket move/volume, first-minute dollar volume, opening auction
imbalance if available, VIX/volatility regime, sector ETF gap, and cross-sectional
gap rank. Treat sentiment expansion as experimental until timestamp coverage
and stability are improved.

## Low likelihood

### Direct return regression, rankers, and current meta models

On the controlled recent window:

- Direct return regression: 4.0 net bps/trade
- Pairwise ranker: 0.9 net bps/trade
- Best classifier configuration: 5.9 net bps/trade

Second-stage context regressors also failed to beat a simple EV threshold.
Keep the classifier/EV design unless a replacement wins across multiple
walk-forward anchors.

### Dynamic permutation feature selection

Using all features slightly beat rolling permutation selection on the recent
controlled window and trained much faster. The current selector optimizes
classification accuracy rather than trading value.

### More frequent retraining, shorter lookbacks, and deeper tuning

The 365-day and 30-day-retrain variants were weaker and less stable. More
Optuna trials, deeper trees, or neural architectures are unlikely to be the
best use of effort until cost modeling, universe bias, calibration, and
retrain-phase instability are addressed.

### Current dynamic stop selector and multi-head v2 selector

The tested dynamic stop selector underperformed several static stop choices.
The v2 expected-return meta selector also underperformed the stronger
single-head `m=1.5` variants.

## Recommended next experiment order

1. Add realized execution costs and delayed-entry/slippage scenarios to the
   backtest; measure live fills.
2. Promote a research candidate with fixed long EV floor `0.10`, then compare
   `0.10` and `0.20` on fresh data.
3. Implement a separate short-only model with EV floor `0.25` and portfolio
   capital sharing.
4. Rebuild with point-in-time membership and a broader liquid universe.
5. Test staggered retrain consensus and cross-fitted calibration.
6. Run joint stop-distance/horizon heads and focused regime/microstructure
   feature additions.

## Artifacts

- Controlled backtests: `reports/improvement_audit/backtests/`
- Direct regression/ranker results:
  `reports/improvement_audit/model_experiments/summary.csv`
- Exploratory model runner: `run_improvement_model_experiments.py`

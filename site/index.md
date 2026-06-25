---
title: Open30 Research
permalink: /
---

# Open30 Research

<p class="lede">A reproducible research project for a 30-minute open-session
equity strategy, including data construction, labels, walk-forward validation,
strategy manifests, and curated results.</p>

<div class="metric-grid">
  <div class="metric"><strong>195,166</strong><span>dataset rows in tracked run metadata</span></div>
  <div class="metric"><strong>3,983</strong><span>trading days in tracked run metadata</span></div>
  <div class="metric"><strong>5</strong><span>versioned architecture manifests</span></div>
  <div class="metric"><strong>2011-2026</strong><span>main backtest result period</span></div>
</div>

The main public narrative is the [Research Report]({{ '/research_report/' | relative_url }}).

## What Is In The Repo

- The runnable research pipeline and walk-forward backtest code.
- Architecture manifests for the baseline, threshold variants, and multi-head selector.
- Curated results copied from completed local reports.
- GitHub Pages source for a hosted research report.
- README sections for reproducibility, limitations, and research ideas.

## Key Result Snapshot

The strongest tracked full-history single-head result is `v1_3`, with 1,865
trades, 56.82 bps mean trade return, and 1.169 profit factor in the tracked
backtest artifact. The result remains execution-sensitive and should be read
with the limitations in the report.

![v1.3 equity curve](assets/results/v1_3/rolling_pnl_chart.png)

from __future__ import annotations

import unittest

import pandas as pd

from run_v1_stop_distance_sweep import build_dynamic_strategy


class StopDistanceSweepTests(unittest.TestCase):
    def test_dynamic_selector_uses_only_prior_candidate_results(self) -> None:
        dates = pd.date_range("2026-01-01", periods=4, freq="D")
        candidate_daily = {
            0.1: pd.DataFrame(
                {
                    "date": dates,
                    "n_trades": [1, 1, 1, 1],
                    "position_return": [0.10, 0.10, -0.20, -0.20],
                }
            ),
            0.3: pd.DataFrame(
                {
                    "date": dates,
                    "n_trades": [1, 1, 1, 1],
                    "position_return": [-0.10, -0.10, 0.20, 0.20],
                }
            ),
        }

        dynamic, selections = build_dynamic_strategy(
            candidate_daily,
            starting_capital=1000.0,
            step_days=2,
            selector_lookback_days=10,
            selector_min_trades=1,
            initial_k=0.3,
            position_sizing={
                "base_multiplier": 1.0,
                "margin_threshold_equity": 2000.0,
                "margin_multiplier_above_threshold": 2.0,
                "high_margin_threshold_equity": 25000.0,
                "high_margin_multiplier_above_threshold": 4.0,
                "buying_power_utilization": 0.95,
            },
        )

        self.assertEqual(selections.iloc[0]["selected_k"], 0.3)
        self.assertEqual(selections.iloc[1]["selected_k"], 0.1)
        self.assertEqual(dynamic["selected_k"].tolist(), [0.3, 0.3, 0.1, 0.1])
        self.assertAlmostEqual(dynamic.iloc[0]["daily_pnl"], -95.0)


if __name__ == "__main__":
    unittest.main()

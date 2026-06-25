from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

RESEARCH_ROOT = Path(__file__).resolve().parents[1]
if str(RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(RESEARCH_ROOT))

from src.modeling import retrain_window
from src.position_sizing import (
    normalize_position_sizing,
    policy_multiplier,
    position_return_from_row,
    research_sizing_fields,
)


class TestPositionSizing(unittest.TestCase):
    def test_policy_threshold_and_utilization(self) -> None:
        self.assertEqual(policy_multiplier(150.0), 1.0)
        self.assertEqual(policy_multiplier(2000.0), 1.0)
        self.assertEqual(policy_multiplier(2000.01), 2.0)
        self.assertEqual(policy_multiplier(25000.0), 2.0)
        self.assertEqual(policy_multiplier(25000.01), 4.0)

        sizing = research_sizing_fields(
            account_value=150.0,
            entry_price=100.0,
            stop_distance=3.0,
            actual_r=1.5,
        )

        self.assertEqual(sizing["margin_multiplier"], 1.0)
        self.assertEqual(sizing["gross_buying_power"], 150.0)
        self.assertEqual(sizing["sizing_notional"], 142.5)
        self.assertEqual(sizing["position_qty"], 1.425)
        self.assertEqual(sizing["stop_pct"], 0.03)
        self.assertEqual(sizing["position_return"], 0.045)
        self.assertAlmostEqual(sizing["trade_pnl"], 6.4125)

    def test_margin_thresholds_are_strictly_greater_than_cutoffs(self) -> None:
        at_first_threshold = research_sizing_fields(
            account_value=2000.0,
            entry_price=100.0,
            stop_distance=3.0,
            actual_r=1.0,
        )
        above_first_threshold = research_sizing_fields(
            account_value=2000.01,
            entry_price=100.0,
            stop_distance=3.0,
            actual_r=1.0,
        )
        at_high_threshold = research_sizing_fields(
            account_value=25000.0,
            entry_price=100.0,
            stop_distance=3.0,
            actual_r=1.0,
        )
        above_high_threshold = research_sizing_fields(
            account_value=25000.01,
            entry_price=100.0,
            stop_distance=3.0,
            actual_r=1.0,
        )

        self.assertEqual(at_first_threshold["margin_multiplier"], 1.0)
        self.assertEqual(at_first_threshold["sizing_notional"], 1900.0)
        self.assertEqual(above_first_threshold["margin_multiplier"], 2.0)
        self.assertAlmostEqual(above_first_threshold["sizing_notional"], 3800.019)
        self.assertEqual(at_high_threshold["margin_multiplier"], 2.0)
        self.assertEqual(at_high_threshold["sizing_notional"], 47500.0)
        self.assertEqual(above_high_threshold["margin_multiplier"], 4.0)
        self.assertAlmostEqual(above_high_threshold["sizing_notional"], 95000.038)

    def test_legacy_single_margin_config_maps_to_corrected_tiers(self) -> None:
        policy = normalize_position_sizing(
            {
                "base_multiplier": 1.0,
                "margin_threshold_equity": 2000.0,
                "margin_multiplier_above_threshold": 4.0,
                "buying_power_utilization": 0.95,
            }
        )

        self.assertEqual(policy["margin_multiplier_above_threshold"], 2.0)
        self.assertEqual(policy["high_margin_threshold_equity"], 25000.0)
        self.assertEqual(policy["high_margin_multiplier_above_threshold"], 4.0)

    def test_position_return_from_row_uses_stop_pct_not_sized_risk(self) -> None:
        row = pd.Series({"entry_price": 100.0, "ATR14": 10.0})

        self.assertEqual(
            position_return_from_row(
                row,
                actual_r=2.0,
                stop_distance_config={"rule": "atr", "atr_period": 14, "k": 0.3},
            ),
            0.06,
        )

    def test_dynamic_threshold_candidates_use_position_return(self) -> None:
        df_window = pd.DataFrame(
            [
                {
                    "date": "2026-01-02",
                    "ticker": "XOM",
                    "side": "long",
                    "entry_price": 100.0,
                    "ATR14": 10.0,
                    "y_R_m_1.5": 2.0,
                }
            ],
            index=[42],
        )
        eval_mask = pd.Series([True], index=df_window.index)
        fake_candidates = [
            {
                "idx": 42,
                "m": 1.5,
                "ev": 0.25,
                "sized_risk": 0.99,
            }
        ]

        with patch.object(retrain_window, "select_daily_head_candidates", return_value=fake_candidates):
            candidates = retrain_window._collect_threshold_eval_candidates(
                df_train_window=df_window,
                eval_mask_int=eval_mask,
                active_model=None,
                active_calibrators={},
                active_E_R_TIME={},
                active_feature_cols=[],
                cost_R=0.05,
                selection_multiples=[1.5],
                long_only_filter=True,
                risk_pct=0.05,
                kelly_fraction=0.5,
                min_risk_pct=0.01,
                stop_distance_config={"rule": "atr", "atr_period": 14, "k": 0.3},
            )

        self.assertEqual(len(candidates), 1)
        trade_return = float(candidates.iloc[0]["trade_return"])
        self.assertTrue(math.isfinite(trade_return))
        self.assertEqual(trade_return, 0.06)
        self.assertNotEqual(trade_return, 0.99 * 2.0)


if __name__ == "__main__":
    unittest.main()

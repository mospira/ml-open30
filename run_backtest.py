#!/usr/bin/env python3
"""
Walk-forward rolling retraining backtest.

Runs 5 simulations:
  - m=0.5 only, m=1.0 only, m=1.5 only, m=2.0 only
  - Best EV across all m

Periodically retrains the model on an overlapping rolling historical window.
Outputs:
  reports/<architecture_version>/rolling_pnl_chart.png - overlaid equity curves
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import matplotlib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.architecture import bundle_architecture_payload, load_architecture
from src.modeling.common import (
    DEFAULT_DATASET_PATH,
    extract_feature_columns,
    load_labeled_dataset,
    load_xgb_params,
)
from src.modeling.retrain_window import (
    apply_ambiguous_worst_case,
    build_meta_features,
    compute_dynamic_time_r,
    select_daily_head_candidates,
    train_retrain_window,
)
from src.position_sizing import research_sizing_fields, stop_distance_from_row

COLORS = {
    "m=0.5": "#FF9800",
    "m=1.0": "#2196F3",
    "m=1.5": "#9C27B0",
    "m=2.0": "#4CAF50",
    "Best EV": "#E91E63",
}

RR_MULTIPLES = [0.5, 1.0, 1.5, 2.0]
TRAIN_SIDE = "long"
M05_THRESHOLD = 0.10
LONG_ONLY_FILTER = True
KELLY_FRACTION = 0.5
MIN_RISK_PCT = 0.01
SELECTION_MODE = "raw_ev"
META_MODEL_TARGET = "diagnostic_binary"
DYNAMIC_EV_THRESHOLD = {"enabled": False}


def report_subdir_name(architecture: dict) -> str:
    for candidate in [
        architecture.get("architecture_id", ""),
        os.path.splitext(os.path.basename(architecture.get("source_path", "")))[0],
    ]:
        tail = candidate.rsplit("_", 1)[-1]
        if tail.startswith("v") and len(tail) > 1:
            return tail
    return architecture["architecture_id"]


def safe_label_name(label: str) -> str:
    return label.replace(" ", "_").replace("=", "")


def compute_strategy_metrics(
    label: str,
    res: pd.DataFrame,
    starting_capital: float,
) -> dict[str, object]:
    metrics: dict[str, object] = {
        "strategy": label,
        "start_date": None,
        "end_date": None,
        "days": int(len(res)),
        "trades": 0,
        "trade_days": 0,
        "final_account": float(starting_capital),
        "total_pnl": 0.0,
        "return_pct": 0.0,
        "win_rate": np.nan,
        "mean_R": np.nan,
        "median_R": np.nan,
        "mean_trade_pnl": np.nan,
        "mean_trade_return": np.nan,
        "mean_ev": np.nan,
        "median_ev": np.nan,
        "mean_selection_score": np.nan,
        "mean_sized_risk": np.nan,
        "profit_factor": np.nan,
        "max_drawdown_dollars": np.nan,
        "max_drawdown_pct": np.nan,
    }
    if res.empty:
        return metrics

    metrics["start_date"] = str(res["date"].iloc[0])
    metrics["end_date"] = str(res["date"].iloc[-1])

    account_curve = res["account_value"].astype(float)
    running_peak = account_curve.cummax()
    drawdown_dollars = account_curve - running_peak
    drawdown_pct = (account_curve / running_peak) - 1.0
    final_account = float(account_curve.iloc[-1])

    metrics.update(
        {
            "days": int(len(res)),
            "trades": int(res["n_trades"].sum()),
            "trade_days": int((res["n_trades"] > 0).sum()),
            "final_account": final_account,
            "total_pnl": float(final_account - starting_capital),
            "return_pct": float((final_account / starting_capital - 1.0) * 100.0),
            "max_drawdown_dollars": float(drawdown_dollars.min()),
            "max_drawdown_pct": float(drawdown_pct.min() * 100.0),
        }
    )

    trades = res[res["n_trades"] > 0].copy()
    if trades.empty:
        return metrics

    if "trade_return" in trades.columns:
        trade_returns = trades["trade_return"].dropna()
    elif "position_return" in trades.columns:
        trade_returns = trades["position_return"].dropna()
    elif "sized_risk" in trades.columns:
        trade_returns = trades["sized_risk"] * trades["actual_R"]
    else:
        trade_returns = pd.Series(dtype=float)
    gross_profit = trades.loc[trades["daily_pnl"] > 0, "daily_pnl"].sum()
    gross_loss = trades.loc[trades["daily_pnl"] < 0, "daily_pnl"].sum()
    profit_factor = np.nan if gross_loss == 0 else float(gross_profit / abs(gross_loss))

    metrics.update(
        {
            "win_rate": float((trades["actual_R"] > 0).mean()),
            "mean_R": float(trades["actual_R"].mean()),
            "median_R": float(trades["actual_R"].median()),
            "mean_trade_pnl": float(trades["daily_pnl"].mean()),
            "mean_trade_return": float(trade_returns.mean()) if not trade_returns.empty else np.nan,
            "mean_ev": float(trades["best_ev"].mean()) if "best_ev" in trades.columns else np.nan,
            "median_ev": float(trades["best_ev"].median()) if "best_ev" in trades.columns else np.nan,
            "mean_selection_score": float(trades["selection_score"].mean()) if "selection_score" in trades.columns else np.nan,
            "mean_sized_risk": float(trades["sized_risk"].mean()) if "sized_risk" in trades.columns else np.nan,
            "profit_factor": profit_factor,
        }
    )
    return metrics


def write_json(path: str, payload: object) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False)


def rolling_simulate(
    df_all: pd.DataFrame,
    feature_cols: list[str],
    lookback_days: int = 730,
    step_days: int = 30,
    embargo_days: int = 1,
    starting_capital: float = 2000.0,
    risk_pct: float = 0.05,
    ev_threshold: float = 0.0,
    cost_R: float | None = None,
    m_filter: float | None = None,
    xgb_params: dict | None = None,
    dynamic_features: bool = False,
    optuna_tune: bool = False,
    optuna_trials: int = 10,
    use_meta_model: bool = False,
    dynamic_ev_threshold: dict | None = None,
    stop_distance_config: dict | None = None,
    position_sizing_config: dict | None = None,
) -> pd.DataFrame:
    if cost_R is None:
        raise ValueError("cost_R must be provided from the active architecture or CLI.")

    multiples = [m_filter] if m_filter is not None else RR_MULTIPLES.copy()
    if stop_distance_config is None:
        stop_distance_config = {"rule": "atr", "atr_period": 14, "k": 0.3}

    df_backtest = apply_ambiguous_worst_case(df_all, multiples)
    df_backtest = df_backtest.sort_values("date")
    unique_dates = sorted(df_backtest["date"].unique())

    df_all_dt = df_all.copy()
    df_all_dt["date_dt"] = pd.to_datetime(df_all_dt["date"])

    account = starting_capital
    records = []

    active_model = None
    active_meta_model = None
    active_meta_feature_cols: list[str] = []
    active_meta_model_target: str | None = None
    active_E_R_TIME = None
    active_calibrators = {}
    active_feature_cols = feature_cols.copy()
    active_ev_threshold = ev_threshold
    next_retrain_date = pd.to_datetime(unique_dates[0]) if unique_dates else None

    for day_str in unique_dates:
        day_dt = pd.to_datetime(day_str)
        day_df = df_backtest[df_backtest["date"] == day_str]

        if day_df.empty:
            records.append(
                {
                    "date": day_str,
                    "account_value": account,
                    "n_trades": 0,
                    "daily_pnl": 0.0,
                    "best_m": None,
                    "best_ticker": None,
                    "best_side": None,
                    "best_ev": None,
                    "actual_R": None,
                    "pred_p_sl": None,
                    "pred_p_tp": None,
                    "pred_p_time": None,
                    "actual_outcome": None,
                    "meta_p_profitable": None,
                    "meta_expected_return": None,
                    "selection_score": None,
                    "ev_threshold": active_ev_threshold,
                }
            )
            continue

        if active_model is None or day_dt >= next_retrain_date:
            retrain_output = train_retrain_window(
                df_all=df_all_dt,
                feature_cols=feature_cols,
                rr_multiples=RR_MULTIPLES,
                as_of_date=day_dt,
                lookback_days=lookback_days,
                step_days=step_days,
                embargo_days=embargo_days,
                ev_threshold=ev_threshold,
                risk_pct=risk_pct,
                cost_R=cost_R,
                xgb_params=xgb_params,
                dynamic_features=dynamic_features,
                optuna_tune=optuna_tune,
                optuna_trials=optuna_trials,
                use_meta_model=use_meta_model,
                dynamic_ev_threshold=dynamic_ev_threshold,
                train_side=TRAIN_SIDE,
                m05_threshold=M05_THRESHOLD,
                long_only_filter=LONG_ONLY_FILTER,
                kelly_fraction=KELLY_FRACTION,
                min_risk_pct=MIN_RISK_PCT,
                stop_distance_config=stop_distance_config,
                meta_model_target=META_MODEL_TARGET,
                selection_multiples=multiples,
                allow_insufficient_history=True,
                log_prefix="    ",
            )
            if retrain_output is None:
                continue

            active_model = retrain_output["model"]
            active_meta_model = retrain_output["meta_model"]
            active_meta_feature_cols = retrain_output["meta_feature_cols"]
            active_meta_model_target = retrain_output["meta_model_target"]
            active_E_R_TIME = retrain_output["e_r_time"]
            active_calibrators = retrain_output["calibrators"]
            active_feature_cols = retrain_output["active_feature_cols"]
            active_ev_threshold = float(retrain_output["ev_threshold"])
            next_retrain_date = day_dt + pd.Timedelta(days=step_days)

        if active_E_R_TIME is None:
            train_end = day_dt - pd.Timedelta(days=embargo_days)
            train_start = train_end - pd.Timedelta(days=lookback_days)
            mask = (df_all_dt["date_dt"] >= train_start) & (df_all_dt["date_dt"] <= train_end)
            active_E_R_TIME = compute_dynamic_time_r(df_all_dt[mask], RR_MULTIPLES)

        daily_candidates = select_daily_head_candidates(
            day_df=day_df,
            active_feature_cols=active_feature_cols,
            selection_multiples=multiples,
            active_model=active_model,
            active_calibrators=active_calibrators,
            active_E_R_TIME=active_E_R_TIME,
            ev_threshold=active_ev_threshold,
            cost_R=cost_R,
            m05_threshold=M05_THRESHOLD,
            long_only_filter=LONG_ONLY_FILTER,
            risk_pct=risk_pct,
            kelly_fraction=KELLY_FRACTION,
            min_risk_pct=MIN_RISK_PCT,
        )

        if not daily_candidates:
            records.append(
                {
                    "date": day_str,
                    "account_value": account,
                    "n_trades": 0,
                    "daily_pnl": 0.0,
                    "best_m": None,
                    "best_ticker": None,
                    "best_side": None,
                    "best_ev": None,
                    "actual_R": None,
                    "pred_p_sl": None,
                    "pred_p_tp": None,
                    "pred_p_time": None,
                    "actual_outcome": None,
                    "meta_p_profitable": None,
                    "meta_expected_return": None,
                    "selection_score": None,
                    "ev_threshold": active_ev_threshold,
                }
            )
            continue

        best_trade = None
        best_selection_score = None
        if (
            active_meta_model is not None
            and SELECTION_MODE == "meta_expected_return"
            and active_meta_model_target == "expected_return"
        ):
            df_meta_candidates = pd.DataFrame(
                [
                    build_meta_features(
                        primary_m=candidate["m"],
                        primary_ev=candidate["ev"],
                        primary_probas=candidate["probas"],
                        sized_risk=candidate["sized_risk"],
                        primary_stop_pct=(
                            stop_distance_from_row(day_df.loc[candidate["idx"]], stop_distance_config)
                            / float(day_df.loc[candidate["idx"], "entry_price"])
                        ),
                    )
                    for candidate in daily_candidates
                ]
            )
            meta_expected_returns = active_meta_model.predict(df_meta_candidates[active_meta_feature_cols])
            for candidate, meta_expected_return in zip(daily_candidates, meta_expected_returns):
                candidate["meta_expected_return"] = float(meta_expected_return)
                candidate["meta_p_profitable"] = None

            best_trade = max(daily_candidates, key=lambda candidate: candidate["meta_expected_return"])
            best_selection_score = float(best_trade["meta_expected_return"])
            if best_selection_score <= 0:
                records.append(
                    {
                        "date": day_str,
                        "account_value": account,
                        "n_trades": 0,
                        "daily_pnl": 0.0,
                        "best_m": None,
                        "best_ticker": None,
                        "best_side": None,
                        "best_ev": None,
                        "actual_R": None,
                        "pred_p_sl": None,
                        "pred_p_tp": None,
                        "pred_p_time": None,
                        "actual_outcome": None,
                        "meta_p_profitable": None,
                        "meta_expected_return": best_selection_score,
                        "selection_score": best_selection_score,
                        "ev_threshold": active_ev_threshold,
                    }
                )
                continue
        else:
            for candidate in daily_candidates:
                candidate["meta_expected_return"] = None
                candidate["meta_p_profitable"] = None
            best_trade = max(daily_candidates, key=lambda candidate: candidate["ev"])
            best_selection_score = float(best_trade["ev"])

            if active_meta_model is not None and active_meta_model_target == "diagnostic_binary":
                df_curr_meta = pd.DataFrame(
                    [
                        build_meta_features(
                            primary_m=best_trade["m"],
                            primary_ev=best_trade["ev"],
                            primary_probas=best_trade["probas"],
                            sized_risk=best_trade["sized_risk"],
                            primary_stop_pct=(
                                stop_distance_from_row(day_df.loc[best_trade["idx"]], stop_distance_config)
                                / float(day_df.loc[best_trade["idx"], "entry_price"])
                            ),
                        )
                    ]
                )
                meta_p_prof = float(active_meta_model.predict_proba(df_curr_meta[active_meta_feature_cols])[0, 1])
                best_trade["meta_p_profitable"] = meta_p_prof

        best_idx = best_trade["idx"]
        best_m = best_trade["m"]
        best_ev = best_trade["ev"]
        best_probas = best_trade["probas"]
        sized_risk = best_trade["sized_risk"]

        pred_p_sl = float(best_probas[0])
        pred_p_tp = float(best_probas[1])
        pred_p_time = float(best_probas[2]) if len(best_probas) > 2 else 0.0

        if active_meta_model is not None:
            if active_meta_model_target == "expected_return" and best_trade["meta_expected_return"] is not None:
                print(
                    f"      [Meta] E[return]: {best_selection_score:.4f} | "
                    f"Chosen m: {best_m:.1f} | "
                    f"EV: {best_ev:.3f} | Kelly risk: {sized_risk:.3f}"
                )
            elif active_meta_model_target == "diagnostic_binary" and best_trade["meta_p_profitable"] is not None:
                print(
                    f"      [Meta] P(profitable): {best_trade['meta_p_profitable']:.3f} | "
                    f"Chosen m: {best_m:.1f} | "
                    f"EV: {best_ev:.3f} | Kelly risk: {sized_risk:.3f}"
                )

        actual_R = day_df.loc[best_idx, f"y_R_m_{best_m}"]
        if pd.isna(actual_R):
            actual_R = 0.0

        if actual_R <= -0.95:
            actual_outcome = 0
        elif actual_R >= best_m * 0.95:
            actual_outcome = 1
        else:
            actual_outcome = 2

        account_before_trade = account
        row = day_df.loc[best_idx]
        stop_distance = stop_distance_from_row(row, stop_distance_config)
        sizing = research_sizing_fields(
            account_value=account_before_trade,
            entry_price=float(row["entry_price"]),
            stop_distance=stop_distance,
            actual_r=actual_R,
            policy=position_sizing_config,
        )
        risk_amount = account_before_trade * sized_risk
        trade_pnl = sizing["trade_pnl"]
        account += trade_pnl

        records.append(
            {
                "date": day_str,
                "account_value_before": account_before_trade,
                "account_value": account,
                "n_trades": 1,
                "daily_pnl": trade_pnl,
                "best_m": best_m,
                "best_ticker": day_df.loc[best_idx, "ticker"],
                "best_side": day_df.loc[best_idx, "side"],
                "best_ev": best_ev,
                "sized_risk": sized_risk,
                "risk_amount": risk_amount,
                "margin_multiplier": sizing["margin_multiplier"],
                "gross_buying_power": sizing["gross_buying_power"],
                "sizing_notional": sizing["sizing_notional"],
                "position_qty": sizing["position_qty"],
                "stop_pct": sizing["stop_pct"],
                "position_return": sizing["position_return"],
                "trade_return": (
                    trade_pnl / account_before_trade
                    if account_before_trade and np.isfinite(account_before_trade)
                    else np.nan
                ),
                "actual_R": actual_R,
                "pred_p_sl": pred_p_sl,
                "pred_p_tp": pred_p_tp,
                "pred_p_time": pred_p_time,
                "actual_outcome": actual_outcome,
                "meta_p_profitable": best_trade["meta_p_profitable"],
                "meta_expected_return": best_trade["meta_expected_return"],
                "selection_score": best_selection_score,
                "ev_threshold": active_ev_threshold,
            }
        )

    return pd.DataFrame(records)


def main() -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--architecture",
        default=None,
        help="Architecture manifest path. Defaults to the current canonical architecture.",
    )
    pre_args, remaining = pre_parser.parse_known_args()
    architecture = load_architecture(pre_args.architecture)

    global RR_MULTIPLES, TRAIN_SIDE, M05_THRESHOLD, LONG_ONLY_FILTER, KELLY_FRACTION, MIN_RISK_PCT, SELECTION_MODE, META_MODEL_TARGET, DYNAMIC_EV_THRESHOLD
    RR_MULTIPLES = architecture["rr_multiples"]
    TRAIN_SIDE = architecture["training"]["train_side"]
    M05_THRESHOLD = architecture["decision"]["m05_threshold"]
    LONG_ONLY_FILTER = architecture["decision"]["long_only_filter"]
    KELLY_FRACTION = architecture["decision"]["kelly_fraction"]
    MIN_RISK_PCT = architecture["decision"]["min_risk_pct"]
    SELECTION_MODE = architecture["decision"]["selection_mode"]
    META_MODEL_TARGET = architecture["training"]["meta_model_target"]
    DYNAMIC_EV_THRESHOLD = architecture["decision"]["dynamic_ev_threshold"]

    parser = argparse.ArgumentParser(
        description="Rolling walk-forward backtest",
        parents=[pre_parser],
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=architecture["decision"]["ev_threshold"],
        help="Minimum EV (in R) to take a trade.",
    )
    parser.add_argument("--dataset_path", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--model_dir", default="models/v1")
    parser.add_argument("--capital", type=float, default=2000.0)
    parser.add_argument("--risk", type=float, default=architecture["decision"]["risk_pct"])
    parser.add_argument(
        "--cost_R",
        type=float,
        default=architecture["decision"]["cost_R"],
        help="Execution / slippage cost in R units.",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=architecture["training"]["lookback_days"],
        help="Days of history to train on",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=architecture["training"]["step_days"],
        help="Days between model retrains",
    )
    parser.add_argument(
        "--embargo",
        type=int,
        default=architecture["training"]["embargo_days"],
        help="Days of gap between train end and backtest day",
    )
    parser.add_argument("--0.5", dest="m_05", action="store_true", help="Run m=0.5 simulation")
    parser.add_argument("--1.0", dest="m_10", action="store_true", help="Run m=1.0 simulation")
    parser.add_argument("--1.5", dest="m_15", action="store_true", help="Run m=1.5 simulation")
    parser.add_argument("--2.0", dest="m_20", action="store_true", help="Run m=2.0 simulation")
    parser.add_argument(
        "--m",
        dest="m_values",
        action="append",
        type=float,
        default=[],
        help="Run a specific reward multiple. Repeatable.",
    )
    parser.add_argument("--best_ev", dest="best_ev", action="store_true", help="Run Best EV simulation")
    parser.add_argument(
        "--dynamic_features",
        action=argparse.BooleanOptionalAction,
        default=architecture["training"]["dynamic_features"],
        help="Run dynamic rolling feature selection",
    )
    parser.add_argument(
        "--optuna",
        action=argparse.BooleanOptionalAction,
        default=architecture["training"]["optuna"],
        help="Enable Optuna hyperparameter tuning per retrain",
    )
    parser.add_argument(
        "--optuna_trials",
        type=int,
        default=architecture["training"]["optuna_trials"],
        help="Number of Optuna trials per retrain",
    )
    parser.add_argument(
        "--meta_model",
        action=argparse.BooleanOptionalAction,
        default=architecture["training"]["meta_model"],
        help="Enable the architecture-defined meta model behavior",
    )
    parser.add_argument("--output_prefix", type=str, default="", help="Prefix for output files (charts, CSVs)")
    parser.add_argument(
        "--reports_dir",
        default=None,
        help="Optional output directory. Defaults to reports/<architecture_version>.",
    )
    args = parser.parse_args(remaining)

    os.makedirs(args.model_dir, exist_ok=True)
    xgb_params = load_xgb_params(args.model_dir)

    print("Rolling Backtest Config")
    print(f"  Architecture : {architecture['architecture_id']} ({architecture['source_path']})")
    print(f"  EV Threshold : {args.threshold}")
    print(f"  Cost R       : {args.cost_R}")
    print(f"  Selection    : {SELECTION_MODE}")
    print(f"  Dynamic Thresh: {DYNAMIC_EV_THRESHOLD.get('enabled', False)}")
    print(f"  Meta Target  : {META_MODEL_TARGET if args.meta_model else 'disabled'}")
    print(f"  Lookback     : {args.lookback} days")
    print(f"  Step         : {args.step} days")
    print(f"  Embargo      : {args.embargo} days\n")

    print("Loading assembled dataset to construct rolling windows ...")
    df_all = load_labeled_dataset(args.dataset_path)
    feature_cols = extract_feature_columns(df_all)
    missing_multiples = [
        m
        for m in RR_MULTIPLES
        if f"y_type_m_{m}" not in df_all.columns or f"y_R_m_{m}" not in df_all.columns
    ]
    if missing_multiples:
        raise RuntimeError(
            "Dataset is missing label columns for architecture multiples "
            f"{missing_multiples}. Regenerate labels/dataset with "
            f"`python run_pipeline.py --architecture {architecture['source_path']}` first."
        )

    print(f"Dataset: {len(df_all):,} rows  |  {df_all['date'].nunique()} trading days\n")

    requested_ms = list(args.m_values)
    if args.m_05:
        requested_ms.append(0.5)
    if args.m_10:
        requested_ms.append(1.0)
    if args.m_15:
        requested_ms.append(1.5)
    if args.m_20:
        requested_ms.append(2.0)
    requested_ms = [float(m) for m in requested_ms]

    run_all = not any([requested_ms, args.best_ev])

    sim_configs = []
    if run_all:
        for m_val in RR_MULTIPLES:
            sim_configs.append((f"m={m_val}", m_val))
        if len(RR_MULTIPLES) > 1:
            sim_configs.append(("Best EV", None))
    else:
        seen = set()
        for m_val in requested_ms:
            if m_val in seen:
                continue
            seen.add(m_val)
            sim_configs.append((f"m={m_val}", m_val))
        if args.best_ev:
            sim_configs.append(("Best EV", None))

    all_results = {}

    for label, m_val in sim_configs:
        print(f"--- Simulating {label} ---")
        res = rolling_simulate(
            df_all=df_all,
            feature_cols=feature_cols,
            lookback_days=args.lookback,
            step_days=args.step,
            embargo_days=args.embargo,
            starting_capital=args.capital,
            risk_pct=args.risk,
            ev_threshold=args.threshold,
            cost_R=args.cost_R,
            m_filter=m_val,
            xgb_params=xgb_params,
            dynamic_features=args.dynamic_features,
            optuna_tune=args.optuna,
            optuna_trials=args.optuna_trials,
            use_meta_model=args.meta_model,
            dynamic_ev_threshold=DYNAMIC_EV_THRESHOLD,
            stop_distance_config=architecture["stop_distance"],
            position_sizing_config=architecture["position_sizing"],
        )
        all_results[label] = res

        if res.empty:
            print(f"    => {label} Result: no eligible days\n")
            continue

        final = res["account_value"].iloc[-1]
        ret = (final / args.capital - 1) * 100
        trades = int(res["n_trades"].sum())
        print(f"    => {label} Result: ${final:,.2f}  ({ret:+.2f}%)  |  {trades} trades\n")

    print(f"\n{'-' * 60}")
    print(f"  {'Strategy':<12} {'Final $':>10} {'Return':>10} {'Trades':>8} {'Trade Days':>11}")
    print(f"{'-' * 60}")
    for label, res in all_results.items():
        if res.empty:
            continue
        final = res["account_value"].iloc[-1]
        ret = (final / args.capital - 1) * 100
        trades = int(res["n_trades"].sum())
        trade_days = int((res["n_trades"] > 0).sum())
        print(f"  {label:<12} {f'${final:,.2f}':>10} {f'{ret:+.2f}%':>10} {trades:>8} {trade_days:>11}")
    print(f"{'-' * 60}")

    fig, ax = plt.subplots(figsize=(13, 6))

    for label, res in all_results.items():
        if res.empty:
            continue
        linewidth = 2.2 if label == "Best EV" else 1.2
        color = COLORS.get(label)
        dates_dt = pd.to_datetime(res["date"])
        ax.plot(
            dates_dt,
            res["account_value"],
            label=label,
            color=color,
            linewidth=linewidth,
            alpha=0.9 if label == "Best EV" else 0.7,
        )

    ax.axhline(args.capital, ls="--", color="grey", alpha=0.4)
    ax.set_xlabel("Date")
    ax.set_ylabel("Account Value ($)")
    ax.set_title(
        f"Rolling Walk-Forward Backtest  (step={args.step}d, lookback={args.lookback}d)",
        fontweight="bold",
        fontsize=13,
    )
    ax.legend(loc="best", fontsize=9)
    fig.autofmt_xdate()
    fig.tight_layout()

    reports_dir = args.reports_dir or os.path.join("reports", report_subdir_name(architecture))
    if not os.path.isabs(reports_dir):
        reports_dir = os.path.join(PROJECT_ROOT, reports_dir)
    os.makedirs(reports_dir, exist_ok=True)
    prefix = f"{args.output_prefix}_" if args.output_prefix else ""

    write_json(
        os.path.join(reports_dir, f"{prefix}architecture_manifest.json"),
        bundle_architecture_payload(architecture),
    )
    write_json(
        os.path.join(reports_dir, f"{prefix}run_metadata.json"),
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "architecture_id": architecture["architecture_id"],
            "architecture_source_path": architecture["source_path"],
            "reports_subdir": report_subdir_name(architecture),
            "output_prefix": args.output_prefix,
            "dataset_path": args.dataset_path,
            "dataset_rows": int(len(df_all)),
            "trading_days": int(df_all["date"].nunique()),
            "selection_mode": SELECTION_MODE,
            "stop_distance": architecture["stop_distance"],
            "position_sizing": architecture["position_sizing"],
            "meta_model_target": META_MODEL_TARGET if args.meta_model else None,
            "args": vars(args),
            "strategies": [label for label, _ in sim_configs],
        },
    )

    chart_path = os.path.join(reports_dir, f"{prefix}rolling_pnl_chart.png")
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    print(f"\nRolling PnL chart saved -> {chart_path}")

    summary_rows = []
    head_mix_records = []
    equity_curves = None

    for label, res in all_results.items():
        safe_label = safe_label_name(label)
        summary_rows.append(compute_strategy_metrics(label, res, args.capital))

        daily_path = os.path.join(reports_dir, f"{prefix}{safe_label}_daily.csv")
        res.to_csv(daily_path, index=False)

        trades_only = res[res["n_trades"] > 0].copy()
        trades_path = os.path.join(reports_dir, f"{prefix}{safe_label}_trades.csv")
        trades_only.to_csv(trades_path, index=False)

        if not res.empty:
            curve = res[["date", "account_value"]].copy()
            curve = curve.rename(columns={"account_value": safe_label})
            if equity_curves is None:
                equity_curves = curve
            else:
                equity_curves = equity_curves.merge(curve, on="date", how="outer")

        if not trades_only.empty and "best_m" in trades_only.columns:
            counts = trades_only["best_m"].value_counts().sort_index()
            total = int(counts.sum())
            for best_m, count in counts.items():
                head_mix_records.append(
                    {
                        "strategy": label,
                        "best_m": float(best_m),
                        "trades": int(count),
                        "trade_share": float(count / total) if total else np.nan,
                    }
                )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(reports_dir, f"{prefix}summary_metrics.csv")
    summary_df.to_csv(summary_path, index=False)

    if equity_curves is not None:
        equity_curves = equity_curves.sort_values("date")
        equity_path = os.path.join(reports_dir, f"{prefix}equity_curves.csv")
        equity_curves.to_csv(equity_path, index=False)
        print(f"Equity curves saved -> {equity_path}")

    if head_mix_records:
        head_mix_df = pd.DataFrame(head_mix_records)
        head_mix_path = os.path.join(reports_dir, f"{prefix}head_selection_mix.csv")
        head_mix_df.to_csv(head_mix_path, index=False)
        print(f"Head selection mix saved -> {head_mix_path}")

    print(f"Daily results and trade CSVs saved -> {reports_dir}")
    print(f"Summary metrics saved -> {summary_path}")

    for label, res in all_results.items():
        trades = res[res["n_trades"] > 0].copy()
        if len(trades) < 30:
            continue
        if not all(c in trades.columns for c in ["pred_p_sl", "pred_p_tp", "pred_p_time", "actual_outcome"]):
            continue

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        reliability_rows = []
        class_info = [
            (0, "pred_p_sl", "SL (class 0)", "#E53935"),
            (1, "pred_p_tp", "TP (class 1)", "#43A047"),
            (2, "pred_p_time", "TIME (class 2)", "#1E88E5"),
        ]

        for ax, (cls, prob_col, cls_label, color) in zip(axes, class_info):
            y_true_binary = (trades["actual_outcome"] == cls).astype(int).values
            y_prob = trades[prob_col].values

            valid = ~np.isnan(y_prob) & ~np.isnan(y_true_binary)
            y_true_binary = y_true_binary[valid]
            y_prob = y_prob[valid]

            if len(y_prob) < 20 or len(np.unique(y_true_binary)) < 2:
                ax.set_title(f"{cls_label} - insufficient data")
                continue

            try:
                prob_true, prob_pred = calibration_curve(
                    y_true_binary,
                    y_prob,
                    n_bins=10,
                    strategy="uniform",
                )
            except Exception:
                ax.set_title(f"{cls_label} - calibration failed")
                continue

            ax.plot([0, 1], [0, 1], ls="--", color="grey", alpha=0.6, label="Perfect")
            ax.plot(
                prob_pred,
                prob_true,
                "o-",
                color=color,
                linewidth=2,
                markersize=6,
                label=cls_label,
            )
            ax.set_xlabel("Mean Predicted Probability")
            ax.set_ylabel("Observed Frequency")
            ax.set_title(cls_label, fontweight="bold")
            ax.legend(loc="upper left", fontsize=8)
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)

            for pred_val, true_val in zip(prob_pred, prob_true):
                reliability_rows.append(
                    {
                        "strategy": label,
                        "class_id": cls,
                        "class_label": cls_label,
                        "mean_predicted_probability": float(pred_val),
                        "observed_frequency": float(true_val),
                    }
                )

        safe_label = safe_label_name(label)
        fig.suptitle(
            f"Reliability Diagram - {label} ({len(trades)} trades)",
            fontweight="bold",
            fontsize=13,
        )
        fig.tight_layout()
        rel_path = os.path.join(reports_dir, f"{prefix}{safe_label}_reliability.png")
        fig.savefig(rel_path, dpi=150)
        plt.close(fig)
        print(f"Reliability diagram saved -> {rel_path}")

        if reliability_rows:
            reliability_df = pd.DataFrame(reliability_rows)
            rel_csv_path = os.path.join(reports_dir, f"{prefix}{safe_label}_reliability.csv")
            reliability_df.to_csv(rel_csv_path, index=False)
            print(f"Reliability data saved -> {rel_csv_path}")


if __name__ == "__main__":
    main()

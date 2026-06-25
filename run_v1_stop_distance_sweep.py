#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run_backtest import compute_strategy_metrics
from src.architecture import apply_architecture_to_labels_config, load_architecture
from src.labeling.generate_labels import generate_labels
from src.position_sizing import policy_multiplier

DEFAULT_K_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5]
DEFAULT_SELECTOR_LOOKBACK_DAYS = 110
DEFAULT_SELECTOR_MIN_TRADES = 20


def resolve_project_path(raw: str | Path) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def default_source_root() -> Path:
    return PROJECT_ROOT.resolve()


def first_existing(candidates: list[Path], label: str) -> Path:
    for path in candidates:
        if path.exists():
            return path
    formatted = [str(path) for path in candidates]
    raise FileNotFoundError(f"Missing {label}. Tried: {formatted}")


def k_slug(k: float) -> str:
    return f"k_{float(k):.1f}".replace(".", "_")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False)
        f.write("\n")


def write_yaml(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def _normalize_date_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dates = pd.to_datetime(out["date"])
    if dates.dt.tz is not None:
        dates = dates.dt.tz_localize(None)
    out["date"] = dates
    return out


def load_source_frames(source_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source_root = source_root.resolve()
    minute_path = first_existing(
        [
            source_root / "minute_bars.parquet",
            source_root / "data" / "raw" / "minute_bars.parquet",
            source_root / "data" / "raw" / "candles_1m.parquet",
        ],
        "minute bars for stop-distance sweep",
    )
    trades_path = first_existing(
        [
            source_root / "trade_instances.parquet",
            source_root / "data" / "processed" / "trade_instances.parquet",
        ],
        "trade instances for stop-distance sweep",
    )
    features_path = first_existing(
        [
            source_root / "features_table.parquet",
            source_root / "data" / "processed" / "features_table.parquet",
        ],
        "feature table for stop-distance sweep",
    )

    print(f"Loading source data from {source_root} ...")
    trades = _normalize_date_column(pd.read_parquet(trades_path))
    features = _normalize_date_column(pd.read_parquet(features_path))
    try:
        minute_bars = pd.read_parquet(
            minute_path,
            columns=["date", "time", "ticker", "high", "low"],
        )
        minute_bars = _normalize_date_column(minute_bars)
    except Exception:
        minute_bars = pd.read_parquet(
            minute_path,
            columns=["timestamp", "ticker", "high", "low"],
        )
        local_ts = pd.to_datetime(minute_bars["timestamp"], utc=True).dt.tz_convert("America/New_York")
        minute_bars["date"] = local_ts.dt.tz_localize(None).dt.normalize()
        minute_bars["time"] = local_ts.dt.strftime("%H:%M:%S")
        minute_bars = minute_bars[["date", "time", "ticker", "high", "low"]]

    minute_bars["time"] = minute_bars["time"].astype(str).str.strip()
    has_seconds = minute_bars["time"].str.len() >= 8
    minute_bars.loc[~has_seconds, "time"] = minute_bars.loc[~has_seconds, "time"] + ":00"
    minute_bars = minute_bars[
        (minute_bars["time"] >= "09:31:00")
        & (minute_bars["time"] <= "09:59:00")
    ].copy()

    required_trade_cols = {"date", "ticker", "side", "entry_price", "exit_price"}
    missing_trade_cols = sorted(required_trade_cols - set(trades.columns))
    if missing_trade_cols:
        raise ValueError(f"Trade instances are missing required columns: {missing_trade_cols}")

    if "ATR14" not in trades.columns:
        atr_col = "ATR14" if "ATR14" in features.columns else "atr_14"
        if atr_col not in features.columns:
            raise ValueError("Features table is missing ATR14/atr_14.")
        atr_lookup = features[["date", "ticker", atr_col]].rename(columns={atr_col: "ATR14"})
        trades = trades.merge(atr_lookup, on=["date", "ticker"], how="left")

    before = len(trades)
    trades = trades.dropna(subset=["entry_price", "exit_price", "ATR14"]).copy()
    print(
        f"  trades={len(trades):,} ({before - len(trades):,} dropped) "
        f"minutes={len(minute_bars):,} features={len(features):,}"
    )
    print(
        f"  coverage={trades['date'].min().date()} -> {trades['date'].max().date()} "
        f"({trades['date'].nunique()} trading days)"
    )
    return trades, features, minute_bars


def build_candidate_manifest(base_raw: dict[str, Any], k: float) -> dict[str, Any]:
    raw = copy.deepcopy(base_raw)
    slug = k_slug(k)
    raw["architecture_id"] = f"v1_atr_{slug}"
    raw["name"] = f"Open30 v1 ATR Stop {k:.1f}x"
    base_description = str(raw.get("description", "")).strip()
    raw["description"] = (
        f"{base_description} Stop-distance sweep candidate using {k:.1f} x ATR14."
    ).strip()
    raw.setdefault("stop_distance", {})
    raw["stop_distance"].update({"rule": "atr", "atr_period": 14, "k": float(k)})
    return raw


def prepare_candidate_dataset(
    *,
    k: float,
    base_raw: dict[str, Any],
    labels_config: dict[str, Any],
    trades: pd.DataFrame,
    features: pd.DataFrame,
    minute_bars: pd.DataFrame,
    output_root: Path,
    force: bool,
) -> tuple[Path, Path]:
    slug = k_slug(k)
    candidate_root = output_root / slug
    manifest_path = candidate_root / "architecture.yaml"
    labels_path = candidate_root / "labels.parquet"
    dataset_path = candidate_root / "dataset.parquet"

    manifest_raw = build_candidate_manifest(base_raw, k)
    write_yaml(manifest_path, manifest_raw)
    if dataset_path.exists() and not force:
        print(f"[{slug}] Reusing dataset {dataset_path}")
        return manifest_path, dataset_path

    architecture = load_architecture(str(manifest_path))
    config = apply_architecture_to_labels_config(labels_config, architecture)
    print(f"[{slug}] Generating labels ...")
    labels = generate_labels(trades.copy(), minute_bars, config)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(labels_path, index=False)

    keys = ["date", "ticker", "side"]
    target_cols = [
        col
        for col in labels.columns
        if col.startswith(("y_type_m_", "y_R_m_", "y_hit_minute_m_", "y_ambig_m_"))
    ]
    label_cols = keys + ["entry_price", "exit_price"] + target_cols
    instances = trades[keys].drop_duplicates()
    dataset = instances.merge(features, on=["date", "ticker"], how="inner")
    dataset = dataset.merge(labels[label_cols], on=keys, how="inner")
    dataset.to_parquet(dataset_path, index=False)

    write_json(
        candidate_root / "dataset_metadata.json",
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "stop_k": float(k),
            "rows": int(len(dataset)),
            "trading_days": int(dataset["date"].nunique()),
            "start_date": str(dataset["date"].min()),
            "end_date": str(dataset["date"].max()),
            "manifest_path": str(manifest_path),
            "labels_path": str(labels_path),
            "dataset_path": str(dataset_path),
        },
    )
    print(f"[{slug}] Saved {len(dataset):,} rows -> {dataset_path}")
    return manifest_path, dataset_path


def run_candidate_backtest(
    *,
    k: float,
    manifest_path: Path,
    dataset_path: Path,
    reports_root: Path,
    model_dir: str,
    force: bool,
) -> Path:
    report_dir = reports_root / k_slug(k)
    summary_path = report_dir / "summary_metrics.csv"
    if summary_path.exists() and not force:
        print(f"[{k_slug(k)}] Reusing backtest {summary_path}")
        return report_dir

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "run_backtest.py"),
        "--architecture",
        str(manifest_path),
        "--dataset_path",
        str(dataset_path),
        "--reports_dir",
        str(report_dir),
        "--model_dir",
        model_dir,
        "--m",
        "1.5",
    ]
    print(f"[{k_slug(k)}] Running walk-forward backtest ...")
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    return report_dir


def _candidate_score(
    daily: pd.DataFrame,
    *,
    as_of_date: pd.Timestamp,
    lookback_days: int,
    min_trades: int,
) -> tuple[float, int]:
    start = as_of_date - pd.Timedelta(days=lookback_days)
    window = daily[(daily["date"] < as_of_date) & (daily["date"] >= start)]
    trades = window[window["n_trades"] > 0]
    if len(trades) < min_trades:
        return -math.inf, int(len(trades))
    values = pd.to_numeric(trades["position_return"], errors="coerce").dropna()
    if values.empty:
        return -math.inf, int(len(trades))
    return float(values.mean()), int(len(trades))


def build_dynamic_strategy(
    candidate_daily: dict[float, pd.DataFrame],
    *,
    starting_capital: float,
    step_days: int,
    selector_lookback_days: int,
    selector_min_trades: int,
    initial_k: float,
    position_sizing: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prepared: dict[float, pd.DataFrame] = {}
    all_dates: set[pd.Timestamp] = set()
    for k, raw in candidate_daily.items():
        daily = raw.copy()
        daily["date"] = pd.to_datetime(daily["date"])
        daily = daily.sort_values("date").drop_duplicates("date", keep="last").set_index("date")
        prepared[float(k)] = daily
        all_dates.update(daily.index.tolist())

    dates = sorted(all_dates)
    if not dates:
        return pd.DataFrame(), pd.DataFrame()

    available_k = sorted(prepared)
    fallback_k = min(available_k, key=lambda value: abs(value - initial_k))
    account = float(starting_capital)
    active_k: float | None = None
    next_selection_date: pd.Timestamp | None = None
    records: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []

    for date in dates:
        if active_k is None or next_selection_date is None or date >= next_selection_date:
            scores: dict[float, float] = {}
            trade_counts: dict[float, int] = {}
            for k, daily in prepared.items():
                score, trades = _candidate_score(
                    daily.reset_index(),
                    as_of_date=date,
                    lookback_days=selector_lookback_days,
                    min_trades=selector_min_trades,
                )
                scores[k] = score
                trade_counts[k] = trades

            eligible = [k for k, score in scores.items() if math.isfinite(score)]
            active_k = max(eligible, key=lambda k: scores[k]) if eligible else fallback_k
            next_selection_date = date + pd.Timedelta(days=step_days)
            selection = {
                "selection_date": date,
                "selected_k": active_k,
                "next_selection_date": next_selection_date,
                "used_fallback": not eligible,
            }
            for k in available_k:
                selection[f"score_{k_slug(k)}"] = None if not math.isfinite(scores[k]) else scores[k]
                selection[f"trades_{k_slug(k)}"] = trade_counts[k]
            selections.append(selection)

        daily = prepared[active_k]
        selected = daily.loc[date] if date in daily.index else None
        traded = selected is not None and int(selected.get("n_trades", 0)) > 0
        position_return = (
            float(selected.get("position_return", 0.0))
            if traded and pd.notna(selected.get("position_return"))
            else 0.0
        )
        account_before = account
        multiplier = policy_multiplier(account_before, position_sizing)
        gross_buying_power = account_before * multiplier
        sizing_notional = gross_buying_power * float(position_sizing["buying_power_utilization"])
        daily_pnl = sizing_notional * position_return if traded else 0.0
        account += daily_pnl

        record: dict[str, Any] = {
            "date": date,
            "selected_k": active_k,
            "account_value_before": account_before,
            "account_value": account,
            "n_trades": 1 if traded else 0,
            "daily_pnl": daily_pnl,
            "margin_multiplier": multiplier,
            "gross_buying_power": gross_buying_power,
            "sizing_notional": sizing_notional,
            "position_return": position_return if traded else None,
            "trade_return": daily_pnl / account_before if traded and account_before else None,
        }
        if selected is not None:
            for col in [
                "best_m",
                "best_ticker",
                "best_side",
                "best_ev",
                "sized_risk",
                "stop_pct",
                "actual_R",
                "pred_p_sl",
                "pred_p_tp",
                "pred_p_time",
                "actual_outcome",
                "selection_score",
                "ev_threshold",
            ]:
                record[col] = selected.get(col)
        records.append(record)

    return pd.DataFrame(records), pd.DataFrame(selections)


def write_comparison_outputs(
    *,
    k_values: list[float],
    report_dirs: dict[float, Path],
    reports_root: Path,
    base_architecture: dict[str, Any],
    starting_capital: float,
    selector_lookback_days: int,
    selector_min_trades: int,
    initial_k: float,
) -> None:
    candidate_daily: dict[float, pd.DataFrame] = {}
    summary_rows: list[dict[str, Any]] = []
    for k in k_values:
        report_dir = report_dirs[k]
        daily_path = report_dir / "m1.5_daily.csv"
        summary_path = report_dir / "summary_metrics.csv"
        candidate_daily[k] = pd.read_csv(daily_path)
        static_summary = pd.read_csv(summary_path).iloc[0].to_dict()
        static_summary.update({"mode": "static", "stop_k": float(k)})
        summary_rows.append(static_summary)

    dynamic_daily, selection_history = build_dynamic_strategy(
        candidate_daily,
        starting_capital=starting_capital,
        step_days=int(base_architecture["training"]["step_days"]),
        selector_lookback_days=selector_lookback_days,
        selector_min_trades=selector_min_trades,
        initial_k=initial_k,
        position_sizing=base_architecture["position_sizing"],
    )
    dynamic_summary = compute_strategy_metrics("dynamic_stop_k", dynamic_daily, starting_capital)
    dynamic_summary.update({"mode": "dynamic", "stop_k": None})
    summary_rows.append(dynamic_summary)

    reports_root.mkdir(parents=True, exist_ok=True)
    dynamic_daily.to_csv(reports_root / "dynamic_daily.csv", index=False)
    dynamic_daily[dynamic_daily["n_trades"] > 0].to_csv(reports_root / "dynamic_trades.csv", index=False)
    selection_history.to_csv(reports_root / "dynamic_selection_history.csv", index=False)
    comparison = pd.DataFrame(summary_rows)
    comparison.to_csv(reports_root / "comparison.csv", index=False)

    fig, ax = plt.subplots(figsize=(13, 6))
    for k in k_values:
        daily = candidate_daily[k]
        ax.plot(pd.to_datetime(daily["date"]), daily["account_value"], label=f"static {k:.1f}x", alpha=0.65)
    ax.plot(
        pd.to_datetime(dynamic_daily["date"]),
        dynamic_daily["account_value"],
        label="dynamic",
        color="black",
        linewidth=2.4,
    )
    ax.axhline(starting_capital, ls="--", color="grey", alpha=0.4)
    ax.set_title("Open30 v1 ATR Stop-Distance Sweep")
    ax.set_xlabel("Date")
    ax.set_ylabel("Account Value ($)")
    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(reports_root / "comparison_equity_curves.png", dpi=150)
    plt.close(fig)

    write_json(
        reports_root / "dynamic_run_metadata.json",
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "method": "Select stop distance only at retrain boundaries using prior out-of-sample candidate trades.",
            "candidate_stop_k": k_values,
            "step_days": int(base_architecture["training"]["step_days"]),
            "selector_lookback_days": selector_lookback_days,
            "selector_min_trades": selector_min_trades,
            "selector_objective": "mean_position_return",
            "initial_k": initial_k,
            "starting_capital": starting_capital,
        },
    )
    print(f"Comparison saved -> {reports_root / 'comparison.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run isolated Open30 v1 static and dynamic ATR stop-distance experiments."
    )
    parser.add_argument("--base-architecture", default="architectures/open30_v1.yaml")
    parser.add_argument("--labels-config", default="configs/labels.yaml")
    parser.add_argument("--source-root", default=str(default_source_root()))
    parser.add_argument("--output-root", default="data/experiments/v1_atr_stop_sweep")
    parser.add_argument("--reports-root", default="reports/v1_atr_stop_sweep")
    parser.add_argument("--model-dir", default="models/v1")
    parser.add_argument("--k", nargs="+", type=float, default=DEFAULT_K_VALUES)
    parser.add_argument("--capital", type=float, default=2000.0)
    parser.add_argument("--initial-k", type=float, default=0.3)
    parser.add_argument("--selector-lookback-days", type=int, default=DEFAULT_SELECTOR_LOOKBACK_DAYS)
    parser.add_argument("--selector-min-trades", type=int, default=DEFAULT_SELECTOR_MIN_TRADES)
    parser.add_argument("--force-labels", action="store_true")
    parser.add_argument("--force-backtests", action="store_true")
    parser.add_argument("--skip-backtests", action="store_true")
    args = parser.parse_args()

    k_values = sorted({float(k) for k in args.k})
    if any(k <= 0 for k in k_values):
        raise ValueError("All stop-distance k values must be positive.")

    base_architecture_path = resolve_project_path(args.base_architecture)
    labels_config_path = resolve_project_path(args.labels_config)
    source_root = resolve_project_path(args.source_root) if not Path(args.source_root).is_absolute() else Path(args.source_root)
    output_root = resolve_project_path(args.output_root)
    reports_root = resolve_project_path(args.reports_root)

    with base_architecture_path.open("r", encoding="utf-8") as f:
        base_raw = yaml.safe_load(f) or {}
    with labels_config_path.open("r", encoding="utf-8") as f:
        labels_config = yaml.safe_load(f) or {}
    base_architecture = load_architecture(str(base_architecture_path))

    if base_architecture["rr_multiples"] != [1.5]:
        raise ValueError("This experiment runner expects the single-head open30_v1 m=1.5 architecture.")

    trades, features, minute_bars = load_source_frames(source_root)
    candidates: dict[float, tuple[Path, Path]] = {}
    for k in k_values:
        candidates[k] = prepare_candidate_dataset(
            k=k,
            base_raw=base_raw,
            labels_config=labels_config,
            trades=trades,
            features=features,
            minute_bars=minute_bars,
            output_root=output_root,
            force=args.force_labels,
        )

    if args.skip_backtests:
        print("Datasets prepared; backtests skipped.")
        return

    report_dirs: dict[float, Path] = {}
    for k in k_values:
        manifest_path, dataset_path = candidates[k]
        report_dirs[k] = run_candidate_backtest(
            k=k,
            manifest_path=manifest_path,
            dataset_path=dataset_path,
            reports_root=reports_root,
            model_dir=args.model_dir,
            force=args.force_backtests,
        )

    write_comparison_outputs(
        k_values=k_values,
        report_dirs=report_dirs,
        reports_root=reports_root,
        base_architecture=base_architecture,
        starting_capital=args.capital,
        selector_lookback_days=args.selector_lookback_days,
        selector_min_trades=args.selector_min_trades,
        initial_k=args.initial_k,
    )


if __name__ == "__main__":
    main()

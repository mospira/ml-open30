import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.architecture import apply_architecture_to_labels_config, load_architecture
from src.labeling.ambiguity import AMBIG, SL, TIME, TP, resolve_ambiguity
from src.labeling.dist_rules import get_stop_distance


def determine_outcomes(df_trades: pd.DataFrame, df_min: pd.DataFrame, m: float, side: str) -> pd.DataFrame:
    """
    Vectorized 1-minute bar scan from 09:31 to 09:59 ET.
    """
    col_type = f"y_type_m_{m}"
    col_r = f"y_R_m_{m}"
    col_minute = f"y_hit_minute_m_{m}"
    col_ambig = f"y_ambig_m_{m}"

    df_trades[col_type] = TIME
    df_trades[col_r] = 0.0
    df_trades[col_minute] = -1
    df_trades[col_ambig] = False

    if side == "long":
        df_trades["stop_level"] = df_trades["entry_price"] - df_trades["dist"]
        df_trades["target_level"] = df_trades["entry_price"] + (df_trades["dist"] * m)
    elif side == "short":
        df_trades["stop_level"] = df_trades["entry_price"] + df_trades["dist"]
        df_trades["target_level"] = df_trades["entry_price"] - (df_trades["dist"] * m)
    else:
        raise ValueError(f"Unknown side: '{side}'")

    if "time" not in df_min.columns:
        return df_trades

    window = df_min[(df_min["time"] >= "09:31:00") & (df_min["time"] <= "09:59:00")].copy()
    if window.empty:
        return df_trades

    if hasattr(window["date"].dtype, "tz") and window["date"].dt.tz is not None:
        window["date"] = window["date"].dt.tz_localize(None)

    df_trades["_tidx"] = np.arange(len(df_trades))

    cols_trade = ["_tidx", "date", "ticker", "stop_level", "target_level"]
    cols_bar = ["date", "ticker", "time", "high", "low"]
    merged = df_trades[cols_trade].merge(window[cols_bar], on=["date", "ticker"], how="inner")

    if merged.empty:
        df_trades.drop(columns=["_tidx"], inplace=True)
        return df_trades

    merged = merged.sort_values(["_tidx", "time"])

    if side == "long":
        merged["tp_hit"] = merged["high"] >= merged["target_level"]
        merged["sl_hit"] = merged["low"] <= merged["stop_level"]
    else:
        merged["tp_hit"] = merged["low"] <= merged["target_level"]
        merged["sl_hit"] = merged["high"] >= merged["stop_level"]

    merged["any_hit"] = merged["tp_hit"] | merged["sl_hit"]
    merged["both_hit"] = merged["tp_hit"] & merged["sl_hit"]

    hits = merged[merged["any_hit"]].copy()
    if hits.empty:
        df_trades.drop(columns=["_tidx"], inplace=True)
        return df_trades

    first = hits.groupby("_tidx").first().reset_index()
    first["minute_offset"] = first["time"].str[3:5].astype(int) - 31

    ambig_mask = first["both_hit"]
    tp_only_mask = first["tp_hit"] & ~first["sl_hit"]
    sl_only_mask = first["sl_hit"] & ~first["tp_hit"]

    idx_map = first.set_index("_tidx")

    for tidx in idx_map.index[ambig_mask.values]:
        pos = df_trades.index[df_trades["_tidx"] == tidx]
        df_trades.loc[pos, col_type] = AMBIG
        df_trades.loc[pos, col_r] = np.nan
        df_trades.loc[pos, col_minute] = int(idx_map.loc[tidx, "minute_offset"])
        df_trades.loc[pos, col_ambig] = True

    for tidx in idx_map.index[tp_only_mask.values]:
        pos = df_trades.index[df_trades["_tidx"] == tidx]
        df_trades.loc[pos, col_type] = TP
        df_trades.loc[pos, col_r] = float(m)
        df_trades.loc[pos, col_minute] = int(idx_map.loc[tidx, "minute_offset"])

    for tidx in idx_map.index[sl_only_mask.values]:
        pos = df_trades.index[df_trades["_tidx"] == tidx]
        df_trades.loc[pos, col_type] = SL
        df_trades.loc[pos, col_r] = -1.0
        df_trades.loc[pos, col_minute] = int(idx_map.loc[tidx, "minute_offset"])

    df_trades.drop(columns=["_tidx"], inplace=True)
    return df_trades


def generate_labels(df_trades: pd.DataFrame, df_min_bars: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Generate TP/SL/TIME/AMBIG labels for the configured reward multiples.
    """
    dist_config = config.get("stop_distance", {})
    multiples = config.get("rr_multiples", [0.5, 1.0, 1.5, 2.0])
    policy = config.get("ambiguity_policy", {}).get("training", "DROP")
    sides = config.get("instances", {}).get("sides", ["long", "short"])

    df_trades["dist"] = get_stop_distance(df_trades, dist_config)

    for m in multiples:
        print(f"  Labeling m={m} ...")
        col_type = f"y_type_m_{m}"
        col_r = f"y_R_m_{m}"

        for side in sides:
            mask = df_trades["side"] == side
            if not mask.any():
                continue

            sub = df_trades.loc[mask].copy()
            sub = determine_outcomes(sub, df_min_bars, m, side)

            for col in [col_type, f"y_R_m_{m}", f"y_hit_minute_m_{m}", f"y_ambig_m_{m}"]:
                df_trades.loc[mask, col] = sub[col].values

            df_trades.loc[mask, "stop_level"] = sub["stop_level"].values
            df_trades.loc[mask, "target_level"] = sub["target_level"].values

        time_mask = df_trades[col_type] == TIME

        long_time = time_mask & (df_trades["side"] == "long")
        df_trades.loc[long_time, col_r] = (
            df_trades.loc[long_time, "exit_price"] - df_trades.loc[long_time, "entry_price"]
        ) / df_trades.loc[long_time, "dist"]

        short_time = time_mask & (df_trades["side"] == "short")
        df_trades.loc[short_time, col_r] = (
            df_trades.loc[short_time, "entry_price"] - df_trades.loc[short_time, "exit_price"]
        ) / df_trades.loc[short_time, "dist"]

        df_trades = resolve_ambiguity(df_trades, m, policy)

    cols_to_drop = ["dist", "stop_level", "target_level"]
    df_trades.drop(columns=[c for c in cols_to_drop if c in df_trades.columns], inplace=True)
    return df_trades


if __name__ == "__main__":
    import yaml

    parser = argparse.ArgumentParser(description="Generate TP/SL/TIME labels for the open30 strategy.")
    parser.add_argument(
        "--labels-config",
        default=None,
        help="Path to labels config YAML. Defaults to configs/labels.yaml.",
    )
    parser.add_argument(
        "--architecture",
        default=None,
        help="Optional architecture manifest. When provided, rr_multiples and stop_distance override labels config.",
    )
    args = parser.parse_args()

    root = Path(project_root)

    instances_path = root / "data" / "processed" / "trade_instances.parquet"
    candles_path = root / "data" / "raw" / "candles_1m.parquet"
    daily_feat_path = root / "data" / "processed" / "features" / "daily_features.parquet"
    labels_cfg_path = Path(args.labels_config) if args.labels_config else root / "configs" / "labels.yaml"
    if not labels_cfg_path.is_absolute():
        labels_cfg_path = (root / labels_cfg_path).resolve()
    output_dir = root / "data" / "processed" / "labels"

    for p in [instances_path, candles_path, daily_feat_path, labels_cfg_path]:
        if not p.exists():
            print(f"Missing required input: {p}")
            sys.exit(1)

    with open(labels_cfg_path) as f:
        config = yaml.safe_load(f)
    if args.architecture:
        architecture = load_architecture(args.architecture)
        config = apply_architecture_to_labels_config(config, architecture)
        print(
            f"Loaded architecture {architecture['architecture_id']} "
            f"from {architecture['source_path']} for label overrides"
        )

    print("Loading trade instances ...")
    df_trades = pd.read_parquet(instances_path)
    print(f"  {len(df_trades)} instances")

    print("Loading 1-min candles (this may take a moment) ...")
    df_min = pd.read_parquet(candles_path)

    if df_min["timestamp"].dt.tz is not None:
        df_min["timestamp"] = df_min["timestamp"].dt.tz_convert("America/New_York")
    else:
        df_min["timestamp"] = df_min["timestamp"].dt.tz_localize("UTC").dt.tz_convert("America/New_York")

    df_min["date"] = df_min["timestamp"].dt.normalize()
    df_min["time"] = df_min["timestamp"].dt.strftime("%H:%M:%S")

    entry_bars = df_min[df_min["time"] == "09:31:00"][["date", "ticker", "open"]].copy()
    entry_bars = entry_bars.rename(columns={"open": "entry_price"})
    entry_bars["date"] = entry_bars["date"].dt.tz_localize(None)

    exit_bars = df_min[df_min["time"] == "09:59:00"][["date", "ticker", "close"]].copy()
    exit_bars = exit_bars.rename(columns={"close": "exit_price"})
    exit_bars["date"] = exit_bars["date"].dt.tz_localize(None)

    df_trades["date"] = pd.to_datetime(df_trades["date"]).dt.tz_localize(None)

    df_trades = df_trades.merge(entry_bars, on=["date", "ticker"], how="left")
    df_trades = df_trades.merge(exit_bars, on=["date", "ticker"], how="left")

    before = len(df_trades)
    df_trades = df_trades.dropna(subset=["entry_price", "exit_price"])
    after = len(df_trades)
    print(f"  Dropped {before - after} instances with missing entry/exit prices")

    print("Loading daily features for ATR lookup ...")
    df_daily = pd.read_parquet(daily_feat_path, columns=["date", "ticker", "atr_14"])
    df_daily = df_daily.rename(columns={"atr_14": "ATR14"})
    df_daily["date"] = pd.to_datetime(df_daily["date"]).dt.tz_localize(None)

    df_trades = df_trades.merge(df_daily, on=["date", "ticker"], how="left")
    missing_atr = df_trades["ATR14"].isna().sum()
    if missing_atr > 0:
        print(f"  Warning: {missing_atr} instances missing ATR14, dropping ...")
        df_trades = df_trades.dropna(subset=["ATR14"])

    print(f"  {len(df_trades)} instances ready for labeling")

    print("Generating labels ...")
    df_labeled = generate_labels(df_trades, df_min, config)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "labels.parquet"
    df_labeled.to_parquet(out_path, index=False)
    print(f"\nSaved {len(df_labeled)} labeled rows -> {out_path}")

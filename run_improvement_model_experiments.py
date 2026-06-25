#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRanker, XGBRegressor

from src.modeling.common import extract_feature_columns, load_labeled_dataset


DEFAULT_DATASET = "data/processed/dataset_open30m.parquet"
DEFAULT_OUTPUT_DIR = "reports/improvement_audit/model_experiments"
NET_COST = 0.0004


def build_model(kind: str):
    common = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 4,
        "min_child_weight": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "device": "cpu",
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
    }
    if kind == "regression":
        return XGBRegressor(objective="reg:squarederror", **common)
    if kind == "ranker":
        return XGBRanker(objective="rank:pairwise", **common)
    raise ValueError(f"Unknown model kind: {kind}")


def selected_daily_rows(df: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    scored = df[["date", "ticker", "gross_return_eval"]].copy()
    scored["score"] = scores
    selected_idx = scored.groupby("date")["score"].idxmax()
    return scored.loc[selected_idx].sort_values("date").reset_index(drop=True)


def tune_gate(selected: pd.DataFrame, validation_days: int) -> tuple[float, dict]:
    score_quantiles = [0.0, 0.25, 0.5, 0.65, 0.75, 0.85, 0.9]
    thresholds = [-np.inf]
    thresholds.extend(float(selected["score"].quantile(q)) for q in score_quantiles)
    thresholds = sorted(set(thresholds))

    candidates = []
    for threshold in thresholds:
        trades = selected[selected["score"] > threshold]
        net_sum = float((trades["gross_return_eval"] - NET_COST).sum())
        candidates.append(
            {
                "threshold": float(threshold),
                "trades": int(len(trades)),
                "mean_daily_net": net_sum / validation_days,
                "mean_trade_net": (
                    float((trades["gross_return_eval"] - NET_COST).mean())
                    if len(trades)
                    else float("nan")
                ),
            }
        )

    eligible = [row for row in candidates if row["trades"] >= 30]
    best = max(eligible or candidates, key=lambda row: (row["mean_daily_net"], -row["threshold"]))
    return float(best["threshold"]), best


def fit_window_model(
    kind: str,
    train_fit: pd.DataFrame,
    feature_cols: list[str],
):
    model = build_model(kind)
    train_fit = train_fit.sort_values(["date", "ticker"]).copy()
    X = train_fit[feature_cols]

    if kind == "regression":
        model.fit(X, train_fit["gross_return_train"])
        return model

    train_fit["rank_target"] = train_fit.groupby("date")["gross_return_train"].rank(
        method="dense",
        ascending=True,
    )
    qid = pd.factorize(train_fit["date"], sort=True)[0]
    model.fit(X, train_fit["rank_target"], qid=qid)
    return model


def run_model(
    df: pd.DataFrame,
    feature_cols: list[str],
    kind: str,
    lookback_days: int,
    step_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique_dates = sorted(pd.to_datetime(df["date"]).unique())
    next_retrain_date = pd.Timestamp(unique_dates[0])
    model = None
    active_threshold = -np.inf
    active_gate_meta: dict = {}
    records = []
    retrains = []

    for current_date_raw in unique_dates:
        current_date = pd.Timestamp(current_date_raw)
        if model is None or current_date >= next_retrain_date:
            train_end = current_date - pd.Timedelta(days=1)
            train_start = train_end - pd.Timedelta(days=lookback_days)
            window = df[(df["date"] >= train_start) & (df["date"] <= train_end)].copy()
            window_dates = sorted(window["date"].unique())
            if len(window_dates) < 500:
                continue

            validation_cutoff = window_dates[int(len(window_dates) * 0.80)]
            train_fit = window[
                (window["date"] < validation_cutoff)
                & window["gross_return_train"].notna()
            ].copy()
            validation = window[window["date"] >= validation_cutoff].copy()
            if train_fit.empty or validation.empty:
                continue

            model = fit_window_model(kind, train_fit, feature_cols)
            validation_selected = selected_daily_rows(
                validation,
                model.predict(validation[feature_cols]),
            )
            active_threshold, active_gate_meta = tune_gate(
                validation_selected,
                validation_days=validation["date"].nunique(),
            )
            next_retrain_date = current_date + pd.Timedelta(days=step_days)
            retrains.append(
                {
                    "model": kind,
                    "as_of_date": current_date,
                    "train_start": train_start,
                    "train_end": train_end,
                    "validation_cutoff": validation_cutoff,
                    "train_rows": len(train_fit),
                    "validation_days": validation["date"].nunique(),
                    **active_gate_meta,
                }
            )

        if model is None:
            continue

        day = df[df["date"] == current_date].copy()
        selected = selected_daily_rows(day, model.predict(day[feature_cols])).iloc[0]
        take_trade = bool(selected["score"] > active_threshold)
        gross_return = float(selected["gross_return_eval"]) if take_trade else 0.0
        records.append(
            {
                "model": kind,
                "date": current_date,
                "trade": int(take_trade),
                "ticker": selected["ticker"] if take_trade else None,
                "score": float(selected["score"]),
                "threshold": active_threshold,
                "gross_return": gross_return,
                "net_return_4bps": gross_return - NET_COST if take_trade else 0.0,
            }
        )

    return pd.DataFrame(records), pd.DataFrame(retrains)


def summarize(daily: pd.DataFrame) -> dict:
    trades = daily[daily["trade"] == 1]
    net = trades["net_return_4bps"]
    annual = daily.groupby(daily["date"].dt.year)["net_return_4bps"].sum()
    gross_loss = abs(net[net < 0].sum())
    return {
        "model": daily["model"].iloc[0],
        "start_date": daily["date"].min(),
        "end_date": daily["date"].max(),
        "days": len(daily),
        "trades": len(trades),
        "gross_bps_per_trade": float(trades["gross_return"].mean() * 10000),
        "net_4bps_per_trade": float(net.mean() * 10000),
        "net_4bps_sum": float(net.sum()),
        "mean_daily_net_bps": float(daily["net_return_4bps"].mean() * 10000),
        "net_profit_factor": float(net[net > 0].sum() / gross_loss) if gross_loss else np.nan,
        "positive_years": int((annual > 0).sum()),
        "years": int(len(annual)),
        "worst_year_net": float(annual.min()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--lookback", type=int, default=1095)
    parser.add_argument("--step", type=int, default=90)
    parser.add_argument("--start-date", default="2017-01-01")
    args = parser.parse_args()

    df = load_labeled_dataset(args.dataset)
    if args.start_date:
        df = df[pd.to_datetime(df["date"]) >= pd.Timestamp(args.start_date)].copy()
    feature_cols = extract_feature_columns(df)
    df = df[df["side"] == "long"].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["stop_pct"] = 0.3 * df["ATR14"] / df["entry_price"]
    df["gross_return_train"] = df["y_R_m_1.5"] * df["stop_pct"]
    df["gross_return_eval"] = (
        df["y_R_m_1.5"].fillna(-1.0) * df["stop_pct"]
    )
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for kind in ["regression", "ranker"]:
        print(f"Running {kind}...")
        daily, retrains = run_model(
            df=df,
            feature_cols=feature_cols,
            kind=kind,
            lookback_days=args.lookback,
            step_days=args.step,
        )
        daily.to_csv(output_dir / f"{kind}_daily.csv", index=False)
        retrains.to_csv(output_dir / f"{kind}_retrains.csv", index=False)
        summaries.append(summarize(daily))

    summary = pd.DataFrame(summaries)
    summary.to_csv(output_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

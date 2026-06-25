#!/usr/bin/env python3
"""
Train and export the latest live model bundle using run_backtest-aligned retraining logic.

This script mirrors the model-retraining branch in run_backtest.py and writes a bundle
that a separate execution app can consume.
"""

import argparse
import hashlib
import json
import os
import pickle
import shutil
import sys
from datetime import datetime, timezone

import pandas as pd

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
from src.modeling.retrain_window import resolve_as_of_date, train_retrain_window

RR_MULTIPLES = [0.5, 1.0, 1.5, 2.0]
TRAIN_SIDE = "long"
M05_THRESHOLD = 0.10
LONG_ONLY_FILTER = True
KELLY_FRACTION = 0.5
MIN_RISK_PCT = 0.01
SELECTION_MODE = "raw_ev"
META_MODEL_TARGET = "diagnostic_binary"
DYNAMIC_EV_THRESHOLD = {"enabled": False}


def _save_json(path: str, payload: dict | list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False)


def _sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def retrain_latest_bundle(
    df_all: pd.DataFrame,
    feature_cols: list[str],
    architecture: dict,
    as_of_date: pd.Timestamp,
    lookback_days: int,
    step_days: int,
    embargo_days: int,
    ev_threshold: float,
    risk_pct: float,
    cost_R: float,
    xgb_params: dict,
    dynamic_features: bool,
    optuna_tune: bool,
    optuna_trials: int,
    use_meta_model: bool,
    dynamic_ev_threshold: dict | None,
) -> dict:
    retrain_output = train_retrain_window(
        df_all=df_all,
        feature_cols=feature_cols,
        rr_multiples=RR_MULTIPLES,
        as_of_date=as_of_date,
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
        stop_distance_config=architecture["stop_distance"],
        meta_model_target=META_MODEL_TARGET,
        selection_multiples=RR_MULTIPLES,
    )
    if retrain_output is None:
        raise RuntimeError("Retraining did not produce a model bundle.")

    retrain_output["architecture"] = architecture
    return retrain_output


def _serialize_time_r(e_r_time: dict) -> dict:
    out = {}
    for m in RR_MULTIPLES:
        out[str(m)] = {}
        for side in ["long", "short"]:
            out[str(m)][side] = float(e_r_time.get((m, side), 0.0))
    return out


def export_bundle(
    retrain_output: dict,
    output_root: str,
    overwrite: bool,
) -> str:
    as_of_date = retrain_output["as_of_date"]
    version = as_of_date.strftime("%Y%m%d")
    bundle_dir = os.path.join(output_root, version)

    if os.path.exists(bundle_dir) and not overwrite:
        raise FileExistsError(
            f"Bundle directory already exists: {bundle_dir}. "
            "Use --overwrite to replace it."
        )
    if os.path.exists(bundle_dir) and overwrite:
        shutil.rmtree(bundle_dir)
    os.makedirs(bundle_dir, exist_ok=True)

    model = retrain_output["model"]
    calibrators = retrain_output["calibrators"]
    active_feature_cols = retrain_output["active_feature_cols"]
    meta_model = retrain_output["meta_model"]
    architecture = retrain_output["architecture"]

    model_path = os.path.join(bundle_dir, "model_heads.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "rr_multiples": RR_MULTIPLES,
                "estimator_params": retrain_output["params_to_use"],
                "models": model.models,
                "trained_heads": retrain_output["trained_heads"],
            },
            f,
        )

    calibrator_path = os.path.join(bundle_dir, "calibrators.pkl")
    with open(calibrator_path, "wb") as f:
        pickle.dump(calibrators, f)

    if meta_model is not None:
        with open(os.path.join(bundle_dir, "meta_model.pkl"), "wb") as f:
            pickle.dump(meta_model, f)
        _save_json(os.path.join(bundle_dir, "meta_feature_columns.json"), retrain_output["meta_feature_cols"])

    _save_json(
        os.path.join(bundle_dir, "architecture_manifest.json"),
        bundle_architecture_payload(architecture),
    )
    _save_json(os.path.join(bundle_dir, "feature_columns.json"), active_feature_cols)
    _save_json(
        os.path.join(bundle_dir, "feature_schema.json"),
        {"feature_dtypes": retrain_output.get("feature_dtypes", {})},
    )
    _save_json(os.path.join(bundle_dir, "time_r_lookup.json"), _serialize_time_r(retrain_output["e_r_time"]))
    _save_json(
        os.path.join(bundle_dir, "decision_config.json"),
        {
            "architecture_id": architecture["architecture_id"],
            "architecture_source_path": architecture["source_path"],
            "multiples": RR_MULTIPLES,
            "cost_R": retrain_output["cost_R"],
            "ev_threshold": retrain_output["ev_threshold"],
            "base_ev_threshold": retrain_output.get("base_ev_threshold", retrain_output["ev_threshold"]),
            "dynamic_ev_threshold": retrain_output.get("dynamic_ev_threshold", {"enabled": False}),
            "risk_pct": retrain_output["risk_pct"],
            "m05_threshold": M05_THRESHOLD,
            "long_only_filter": LONG_ONLY_FILTER,
            "kelly_fraction": KELLY_FRACTION,
            "min_risk_pct": MIN_RISK_PCT,
            "train_side": TRAIN_SIDE,
            "calibration_method": architecture["training"]["calibration_method"],
            "stop_distance": architecture["stop_distance"],
            "position_sizing": architecture["position_sizing"],
            "selection_mode": SELECTION_MODE,
            "meta_model_target": retrain_output["meta_model_target"],
            "meta_feature_columns": retrain_output["meta_feature_cols"],
        },
    )
    _save_json(
        os.path.join(bundle_dir, "train_window_meta.json"),
        {
            "retrained_at_utc": datetime.now(timezone.utc).isoformat(),
            "as_of_date": retrain_output["as_of_date"].date().isoformat(),
            "train_start": retrain_output["train_start"].date().isoformat(),
            "train_end": retrain_output["train_end"].date().isoformat(),
            "cal_cutoff": retrain_output["cal_cutoff"].date().isoformat(),
            "meta_cutoff": retrain_output["meta_cutoff"].date().isoformat(),
            "lookback_days": retrain_output["lookback_days"],
            "step_days": retrain_output["step_days"],
            "embargo_days": retrain_output["embargo_days"],
            "trained_heads": retrain_output["trained_heads"],
            "feature_count": len(active_feature_cols),
            "architecture_id": architecture["architecture_id"],
            "architecture_source_path": architecture["source_path"],
            "dynamic_ev_threshold": retrain_output.get("dynamic_ev_threshold", {"enabled": False}),
        },
    )
    _save_json(os.path.join(bundle_dir, "xgb_params_used.json"), retrain_output["params_to_use"])

    file_hashes = {}
    for name in sorted(os.listdir(bundle_dir)):
        if name == "bundle_hash.txt":
            continue
        path = os.path.join(bundle_dir, name)
        if os.path.isfile(path):
            file_hashes[name] = _sha256_of_file(path)

    aggregate = hashlib.sha256(
        json.dumps(file_hashes, sort_keys=True).encode("utf-8")
    ).hexdigest()
    with open(os.path.join(bundle_dir, "bundle_hash.txt"), "w", encoding="utf-8") as f:
        f.write(aggregate + "\n")

    os.makedirs(output_root, exist_ok=True)
    latest_path = os.path.join(output_root, "latest.json")
    _save_json(
        latest_path,
        {
            "version": version,
            "architecture_id": architecture["architecture_id"],
            "bundle_dir": os.path.relpath(bundle_dir, PROJECT_ROOT).replace("\\", "/"),
            "bundle_hash": aggregate,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )

    return bundle_dir


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
        description="Retrain and export the latest live model bundle.",
        parents=[pre_parser],
    )
    parser.add_argument("--dataset_path", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--model_dir", default="models/v1", help="Path containing optional best_params.json")
    parser.add_argument("--output_root", default="models/live")
    parser.add_argument("--as_of_date", default=None, help="YYYY-MM-DD. Defaults to latest available date.")
    parser.add_argument("--lookback", type=int, default=architecture["training"]["lookback_days"])
    parser.add_argument(
        "--step",
        type=int,
        default=architecture["training"]["step_days"],
        help="Stored in metadata for parity; not used for single retrain.",
    )
    parser.add_argument("--embargo", type=int, default=architecture["training"]["embargo_days"])
    parser.add_argument(
        "--threshold",
        type=float,
        default=architecture["decision"]["ev_threshold"],
        help="Stored in decision config.",
    )
    parser.add_argument(
        "--risk",
        type=float,
        default=architecture["decision"]["risk_pct"],
        help="Stored in decision config.",
    )
    parser.add_argument(
        "--cost_R",
        type=float,
        default=architecture["decision"]["cost_R"],
        help="Execution cost in R units.",
    )
    parser.add_argument(
        "--dynamic_features",
        action=argparse.BooleanOptionalAction,
        default=architecture["training"]["dynamic_features"],
    )
    parser.add_argument(
        "--optuna",
        action=argparse.BooleanOptionalAction,
        default=architecture["training"]["optuna"],
    )
    parser.add_argument("--optuna_trials", type=int, default=architecture["training"]["optuna_trials"])
    parser.add_argument(
        "--meta_model",
        action=argparse.BooleanOptionalAction,
        default=architecture["training"]["meta_model"],
        help="Train the architecture-defined meta model.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing version directory if present.")
    args = parser.parse_args(remaining)

    print("Loading assembled dataset...")
    df_all = load_labeled_dataset(args.dataset_path)
    feature_cols = extract_feature_columns(df_all)
    missing_multiples = [
        m for m in RR_MULTIPLES
        if f"y_type_m_{m}" not in df_all.columns or f"y_R_m_{m}" not in df_all.columns
    ]
    if missing_multiples:
        raise RuntimeError(
            "Dataset is missing label columns for architecture multiples "
            f"{missing_multiples}. Regenerate labels/dataset with "
            f"`python run_pipeline.py --architecture {architecture['source_path']}` first."
        )

    as_of_date = resolve_as_of_date(df_all, args.as_of_date)
    print(f"As-of date: {as_of_date.date()} | rows={len(df_all):,} | features={len(feature_cols)}")
    print(
        f"Architecture: {architecture['architecture_id']} "
        f"({architecture['source_path']}) | rr_multiples={RR_MULTIPLES}"
    )

    xgb_params = load_xgb_params(args.model_dir)

    retrain_output = retrain_latest_bundle(
        df_all=df_all,
        feature_cols=feature_cols,
        architecture=architecture,
        as_of_date=as_of_date,
        lookback_days=args.lookback,
        step_days=args.step,
        embargo_days=args.embargo,
        ev_threshold=args.threshold,
        risk_pct=args.risk,
        cost_R=args.cost_R,
        xgb_params=xgb_params,
        dynamic_features=args.dynamic_features,
        optuna_tune=args.optuna,
        optuna_trials=args.optuna_trials,
        use_meta_model=args.meta_model,
        dynamic_ev_threshold=DYNAMIC_EV_THRESHOLD,
    )

    output_root = os.path.join(PROJECT_ROOT, args.output_root)
    bundle_dir = export_bundle(
        retrain_output=retrain_output,
        output_root=output_root,
        overwrite=args.overwrite,
    )

    print("\nBundle export complete.")
    print(f"  Bundle dir: {bundle_dir}")
    print(f"  Architecture: {architecture['architecture_id']}")
    print(f"  Trained heads: {retrain_output['trained_heads']}")
    print(f"  EV threshold: {retrain_output['ev_threshold']}")
    print(f"  Active feature count: {len(retrain_output['active_feature_cols'])}")
    print(f"  Latest pointer: {os.path.join(output_root, 'latest.json')}")


if __name__ == "__main__":
    main()

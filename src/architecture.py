from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

from src.position_sizing import normalize_position_sizing

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHITECTURE_PATH = PROJECT_ROOT / "architectures" / "open30_v2.yaml"
DEFAULT_DYNAMIC_EV_THRESHOLD = {
    "enabled": False,
    "grid": [0.0],
    "min_trades": 20,
    "objective": "mean_trade_return",
}


def resolve_architecture_path(path: str | None = None) -> Path:
    candidate = Path(path) if path else DEFAULT_ARCHITECTURE_PATH
    if not candidate.is_absolute():
        candidate = (PROJECT_ROOT / candidate).resolve()
    return candidate


def load_architecture(path: str | None = None) -> dict[str, Any]:
    arch_path = resolve_architecture_path(path)
    if not arch_path.exists():
        raise FileNotFoundError(f"Architecture manifest not found: {arch_path}")

    with arch_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    rr_multiples = [float(x) for x in raw.get("model", {}).get("rr_multiples", [0.5, 1.0, 1.5, 2.0])]
    if not rr_multiples:
        raise ValueError(f"Architecture manifest has no rr_multiples: {arch_path}")

    training = raw.get("training", {})
    decision = raw.get("decision", {})
    stop_distance = raw.get("stop_distance", {})
    position_sizing = normalize_position_sizing(raw.get("position_sizing"))
    dynamic_ev_threshold = _parse_dynamic_ev_threshold(decision.get("dynamic_ev_threshold"))

    result = {
        "schema_version": int(raw.get("schema_version", 1)),
        "architecture_id": str(raw.get("architecture_id", "open30_custom")),
        "name": str(raw.get("name", raw.get("architecture_id", "open30_custom"))),
        "description": str(raw.get("description", "")).strip(),
        "source_path": os.path.relpath(arch_path, PROJECT_ROOT).replace("\\", "/"),
        "rr_multiples": rr_multiples,
        "training": {
            "lookback_days": int(training.get("lookback_days", 730)),
            "step_days": int(training.get("step_days", 30)),
            "embargo_days": int(training.get("embargo_days", 1)),
            "dynamic_features": bool(training.get("dynamic_features", False)),
            "optuna": bool(training.get("optuna", False)),
            "optuna_trials": int(training.get("optuna_trials", 10)),
            "meta_model": bool(training.get("meta_model", False)),
            "meta_model_target": str(training.get("meta_model_target", "diagnostic_binary")).lower(),
            "train_side": str(training.get("train_side", "long")).lower(),
            "calibration_method": str(training.get("calibration_method", "isotonic")).lower(),
        },
        "decision": {
            "ev_threshold": float(decision.get("ev_threshold", 0.0)),
            "dynamic_ev_threshold": dynamic_ev_threshold,
            "risk_pct": float(decision.get("risk_pct", 0.05)),
            "cost_R": float(decision.get("cost_R", 0.05)),
            "m05_threshold": float(decision.get("m05_threshold", 0.10)),
            "long_only_filter": bool(decision.get("long_only_filter", True)),
            "kelly_fraction": float(decision.get("kelly_fraction", 0.5)),
            "min_risk_pct": float(decision.get("min_risk_pct", 0.01)),
            "selection_mode": str(decision.get("selection_mode", "raw_ev")).lower(),
        },
        "stop_distance": {
            "rule": str(stop_distance.get("rule", "atr")).lower(),
            "atr_period": int(stop_distance.get("atr_period", 14)),
            "k": float(stop_distance.get("k", 0.3)),
            "bps": float(stop_distance.get("bps", 20.0)),
            "atr_column": stop_distance.get("atr_column"),
        },
        "position_sizing": position_sizing,
        "raw": raw,
    }

    train_side = result["training"]["train_side"]
    if train_side not in {"long", "short", "both"}:
        raise ValueError(
            f"Unsupported train_side='{train_side}' in architecture manifest {arch_path}. "
            "Use one of: long, short, both."
        )

    meta_model_target = result["training"]["meta_model_target"]
    if meta_model_target not in {"diagnostic_binary", "expected_return"}:
        raise ValueError(
            f"Unsupported meta_model_target='{meta_model_target}' in architecture manifest {arch_path}. "
            "Use one of: diagnostic_binary, expected_return."
        )

    selection_mode = result["decision"]["selection_mode"]
    if selection_mode not in {"raw_ev", "meta_expected_return"}:
        raise ValueError(
            f"Unsupported selection_mode='{selection_mode}' in architecture manifest {arch_path}. "
            "Use one of: raw_ev, meta_expected_return."
        )

    if selection_mode == "meta_expected_return" and meta_model_target != "expected_return":
        raise ValueError(
            f"Architecture manifest {arch_path} uses selection_mode='meta_expected_return' "
            "but meta_model_target is not 'expected_return'."
        )

    return result


def _parse_dynamic_ev_threshold(raw: Any) -> dict[str, Any]:
    if raw is None:
        return copy.deepcopy(DEFAULT_DYNAMIC_EV_THRESHOLD)

    if isinstance(raw, bool):
        out = copy.deepcopy(DEFAULT_DYNAMIC_EV_THRESHOLD)
        out["enabled"] = raw
        return out

    if not isinstance(raw, dict):
        raise ValueError("decision.dynamic_ev_threshold must be a mapping or boolean.")

    out = copy.deepcopy(DEFAULT_DYNAMIC_EV_THRESHOLD)
    out.update(raw)
    out["enabled"] = bool(out.get("enabled", False))

    grid = out.get("grid", [0.0])
    if not isinstance(grid, list) or not grid:
        raise ValueError("decision.dynamic_ev_threshold.grid must be a non-empty list.")
    out["grid"] = sorted({float(x) for x in grid})

    out["min_trades"] = int(out.get("min_trades", 20))
    if out["min_trades"] < 1:
        raise ValueError("decision.dynamic_ev_threshold.min_trades must be >= 1.")

    out["objective"] = str(out.get("objective", "mean_trade_return")).lower()
    if out["objective"] != "mean_trade_return":
        raise ValueError("Only dynamic EV threshold objective 'mean_trade_return' is supported.")

    return out


def apply_architecture_to_labels_config(base_config: dict[str, Any], architecture: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base_config)
    out["rr_multiples"] = architecture["rr_multiples"]
    out.setdefault("stop_distance", {})
    stop_distance = copy.deepcopy(architecture["stop_distance"])
    if "rule" in stop_distance:
        stop_distance["dist_rule"] = stop_distance["rule"]
    out["stop_distance"].update(stop_distance)
    return out


def bundle_architecture_payload(architecture: dict[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(architecture["raw"])
    payload["architecture_id"] = architecture["architecture_id"]
    payload["source_path"] = architecture["source_path"]
    payload["resolved_rr_multiples"] = architecture["rr_multiples"]
    payload["resolved_training"] = copy.deepcopy(architecture["training"])
    payload["resolved_decision"] = copy.deepcopy(architecture["decision"])
    payload["resolved_stop_distance"] = copy.deepcopy(architecture["stop_distance"])
    payload["position_sizing"] = copy.deepcopy(architecture["position_sizing"])
    payload["resolved_position_sizing"] = copy.deepcopy(architecture["position_sizing"])
    return payload

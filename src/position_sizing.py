from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import math


DEFAULT_POSITION_SIZING: dict[str, float] = {
    "base_multiplier": 1.0,
    "margin_threshold_equity": 2000.0,
    "margin_multiplier_above_threshold": 2.0,
    "high_margin_threshold_equity": 25000.0,
    "high_margin_multiplier_above_threshold": 4.0,
    "buying_power_utilization": 0.95,
}


@dataclass(frozen=True)
class PositionSizingPolicy:
    base_multiplier: float = DEFAULT_POSITION_SIZING["base_multiplier"]
    margin_threshold_equity: float = DEFAULT_POSITION_SIZING["margin_threshold_equity"]
    margin_multiplier_above_threshold: float = DEFAULT_POSITION_SIZING["margin_multiplier_above_threshold"]
    high_margin_threshold_equity: float = DEFAULT_POSITION_SIZING["high_margin_threshold_equity"]
    high_margin_multiplier_above_threshold: float = DEFAULT_POSITION_SIZING["high_margin_multiplier_above_threshold"]
    buying_power_utilization: float = DEFAULT_POSITION_SIZING["buying_power_utilization"]


def normalize_position_sizing(raw: dict[str, Any] | None = None) -> dict[str, float]:
    source = dict(DEFAULT_POSITION_SIZING)
    if raw:
        source.update(raw)
        if (
            "high_margin_threshold_equity" not in raw
            and "high_margin_multiplier_above_threshold" not in raw
            and float(source["margin_threshold_equity"]) == DEFAULT_POSITION_SIZING["margin_threshold_equity"]
            and float(source["margin_multiplier_above_threshold"])
            == DEFAULT_POSITION_SIZING["high_margin_multiplier_above_threshold"]
        ):
            source["margin_multiplier_above_threshold"] = DEFAULT_POSITION_SIZING[
                "margin_multiplier_above_threshold"
            ]
    normalized = {
        "base_multiplier": float(source["base_multiplier"]),
        "margin_threshold_equity": float(source["margin_threshold_equity"]),
        "margin_multiplier_above_threshold": float(source["margin_multiplier_above_threshold"]),
        "high_margin_threshold_equity": float(source["high_margin_threshold_equity"]),
        "high_margin_multiplier_above_threshold": float(source["high_margin_multiplier_above_threshold"]),
        "buying_power_utilization": float(source["buying_power_utilization"]),
    }
    if normalized["base_multiplier"] <= 0:
        raise ValueError("position_sizing.base_multiplier must be > 0")
    if normalized["margin_threshold_equity"] < 0:
        raise ValueError("position_sizing.margin_threshold_equity must be >= 0")
    if normalized["margin_multiplier_above_threshold"] <= 0:
        raise ValueError("position_sizing.margin_multiplier_above_threshold must be > 0")
    if normalized["high_margin_threshold_equity"] <= normalized["margin_threshold_equity"]:
        raise ValueError("position_sizing.high_margin_threshold_equity must be > margin_threshold_equity")
    if normalized["high_margin_multiplier_above_threshold"] <= 0:
        raise ValueError("position_sizing.high_margin_multiplier_above_threshold must be > 0")
    if normalized["buying_power_utilization"] <= 0 or normalized["buying_power_utilization"] > 1:
        raise ValueError("position_sizing.buying_power_utilization must be > 0 and <= 1")
    return normalized


def policy_from_config(raw: dict[str, Any] | None = None) -> PositionSizingPolicy:
    normalized = normalize_position_sizing(raw)
    return PositionSizingPolicy(**normalized)


def policy_multiplier(equity: float, policy: PositionSizingPolicy | dict[str, Any] | None = None) -> float:
    resolved = policy if isinstance(policy, PositionSizingPolicy) else policy_from_config(policy)
    equity_value = float(equity)
    if equity_value > resolved.high_margin_threshold_equity:
        return resolved.high_margin_multiplier_above_threshold
    if equity_value > resolved.margin_threshold_equity:
        return resolved.margin_multiplier_above_threshold
    return resolved.base_multiplier


def stop_pct_from_values(entry_price: float, stop_distance: float) -> float:
    entry = float(entry_price)
    dist = float(stop_distance)
    if not math.isfinite(entry) or entry <= 0 or not math.isfinite(dist) or dist <= 0:
        return float("nan")
    return dist / entry


def position_return_from_r(actual_r: float, entry_price: float, stop_distance: float) -> float:
    stop_pct = stop_pct_from_values(entry_price, stop_distance)
    if not math.isfinite(stop_pct):
        return float("nan")
    try:
        r_value = float(actual_r)
    except (TypeError, ValueError):
        return float("nan")
    if not math.isfinite(r_value):
        return float("nan")
    return r_value * stop_pct


def stop_distance_from_row(row: Any, stop_distance_config: dict[str, Any]) -> float:
    rule = str(stop_distance_config.get("rule", stop_distance_config.get("dist_rule", "atr"))).lower()
    if rule == "atr":
        atr_period = int(stop_distance_config.get("atr_period", 14))
        candidates = [
            stop_distance_config.get("atr_column"),
            f"ATR{atr_period}",
            f"atr_{atr_period}",
        ]
        atr_val = float("nan")
        for col in candidates:
            if not col:
                continue
            try:
                raw = row[col]
            except Exception:
                continue
            try:
                atr_val = float(raw)
            except (TypeError, ValueError):
                continue
            if math.isfinite(atr_val) and atr_val > 0:
                break
        return float(stop_distance_config.get("k", 0.3)) * atr_val
    if rule == "bps":
        try:
            entry = float(row["entry_price"])
        except Exception:
            return float("nan")
        return entry * (float(stop_distance_config.get("bps", 20.0)) / 10000.0)
    return float("nan")


def position_return_from_row(
    row: Any,
    actual_r: float,
    stop_distance_config: dict[str, Any],
) -> float:
    try:
        entry = float(row["entry_price"])
    except Exception:
        return float("nan")
    return position_return_from_r(
        actual_r=actual_r,
        entry_price=entry,
        stop_distance=stop_distance_from_row(row, stop_distance_config),
    )


def research_sizing_fields(
    *,
    account_value: float,
    entry_price: float,
    stop_distance: float,
    actual_r: float,
    policy: PositionSizingPolicy | dict[str, Any] | None = None,
) -> dict[str, float]:
    resolved = policy if isinstance(policy, PositionSizingPolicy) else policy_from_config(policy)
    account = float(account_value)
    entry = float(entry_price)
    multiplier = policy_multiplier(account, resolved)
    gross_buying_power = account * multiplier
    sizing_notional = gross_buying_power * resolved.buying_power_utilization
    position_qty = sizing_notional / entry if math.isfinite(entry) and entry > 0 else float("nan")
    stop_pct = stop_pct_from_values(entry, stop_distance)
    position_return = position_return_from_r(actual_r, entry, stop_distance)
    trade_pnl = sizing_notional * position_return if math.isfinite(position_return) else float("nan")
    return {
        "margin_multiplier": float(multiplier),
        "gross_buying_power": float(gross_buying_power),
        "sizing_notional": float(sizing_notional),
        "position_qty": float(position_qty),
        "stop_pct": float(stop_pct),
        "position_return": float(position_return),
        "trade_pnl": float(trade_pnl),
    }

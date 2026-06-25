from __future__ import annotations

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.backtest.decision import compute_ev
from src.modeling.common import compute_sample_weights
from src.modeling.modeling import MultiHeadModel
from src.position_sizing import position_return_from_row, stop_distance_from_row, stop_pct_from_values

META_DIAGNOSTIC_FEATURE_COLS = [
    "meta_primary_m",
    "meta_primary_ev",
    "meta_primary_p_0",
    "meta_primary_p_1",
    "meta_primary_p_2",
]
META_RETURN_FEATURE_COLS = [
    "meta_primary_m",
    "meta_primary_ev",
    "meta_primary_stop_pct",
    "meta_primary_p_0",
    "meta_primary_p_1",
    "meta_primary_p_2",
]
MIN_HISTORY_FRACTION = 0.5


def training_side_mask(df: pd.DataFrame, train_side: str) -> pd.Series:
    if train_side == "both":
        return pd.Series(True, index=df.index)
    return df["side"] == train_side


def get_meta_feature_cols(meta_model_target: str) -> list[str]:
    if meta_model_target == "diagnostic_binary":
        return META_DIAGNOSTIC_FEATURE_COLS.copy()
    if meta_model_target == "expected_return":
        return META_RETURN_FEATURE_COLS.copy()
    raise ValueError(f"Unsupported meta_model_target='{meta_model_target}'.")


def apply_ambiguous_worst_case(
    df: pd.DataFrame,
    rr_multiples: list[float],
) -> pd.DataFrame:
    df_out = df.copy()
    for m in rr_multiples:
        ambig_col = f"y_ambig_m_{m}"
        type_col = f"y_type_m_{m}"
        r_col = f"y_R_m_{m}"
        if ambig_col not in df_out.columns:
            continue
        mask = df_out[ambig_col] == True
        df_out.loc[mask, type_col] = 0
        df_out.loc[mask, r_col] = -1.0
    return df_out


def apply_long_only_filter(
    ev: np.ndarray,
    sides: np.ndarray,
    long_only_filter: bool,
) -> np.ndarray:
    if not long_only_filter:
        return ev
    filtered = ev.copy()
    filtered[sides == "short"] = -np.inf
    return filtered


def compute_dynamic_time_r(
    df_window: pd.DataFrame,
    rr_multiples: list[float],
) -> dict[tuple[float, str], float]:
    e_r_time: dict[tuple[float, str], float] = {}
    for m_test in rr_multiples:
        for side in ["long", "short"]:
            mask_time = (df_window[f"y_type_m_{m_test}"] == 2) & (df_window["side"] == side)
            if mask_time.any():
                avg_r = df_window.loc[mask_time, f"y_R_m_{m_test}"].mean()
                e_r_time[(m_test, side)] = float(avg_r)
            else:
                e_r_time[(m_test, side)] = 0.0
    return e_r_time


def compute_sized_risk(
    ev: float,
    m: float,
    risk_pct: float,
    kelly_fraction: float,
    min_risk_pct: float,
) -> float:
    raw_fraction = (ev / m) * kelly_fraction
    return float(max(min_risk_pct, min(raw_fraction, risk_pct)))


def build_meta_features(
    primary_m: float,
    primary_ev: float,
    primary_probas: np.ndarray,
    sized_risk: float,
    primary_stop_pct: float = float("nan"),
) -> dict[str, float]:
    return {
        "meta_primary_m": float(primary_m),
        "meta_primary_ev": float(primary_ev),
        "meta_primary_stop_pct": float(primary_stop_pct),
        "meta_primary_p_0": float(primary_probas[0]),
        "meta_primary_p_1": float(primary_probas[1]),
        "meta_primary_p_2": float(primary_probas[2]) if len(primary_probas) > 2 else 0.0,
    }


def select_daily_head_candidates(
    day_df: pd.DataFrame,
    active_feature_cols: list[str],
    selection_multiples: list[float],
    active_model: MultiHeadModel,
    active_calibrators: dict[float, dict[int, IsotonicRegression]],
    active_E_R_TIME: dict[tuple[float, str], float],
    ev_threshold: float,
    cost_R: float,
    m05_threshold: float,
    long_only_filter: bool,
    risk_pct: float,
    kelly_fraction: float,
    min_risk_pct: float,
) -> list[dict]:
    if day_df.empty:
        return []

    X_day = day_df[active_feature_cols]
    candidates = []

    for m in selection_multiples:
        r_col = f"y_R_m_{m}"
        if r_col not in day_df.columns:
            continue

        valid = day_df[r_col].notna()
        if not valid.any():
            continue

        try:
            probas = predict_calibrated_probas(
                active_model,
                active_calibrators,
                m,
                X_day.loc[valid.values],
            )
        except Exception:
            continue

        sides = day_df.loc[valid.values, "side"].values
        ev = compute_ev(
            probas,
            m=m,
            sides=sides,
            cost_R=cost_R,
            custom_E_R_TIME=active_E_R_TIME,
        )
        ev = apply_long_only_filter(ev, sides, long_only_filter)
        threshold = m05_threshold if m == 0.5 else ev_threshold

        if not np.isfinite(ev).any():
            continue

        local_best = int(ev.argmax())
        local_best_ev = float(ev[local_best])
        if not np.isfinite(local_best_ev) or local_best_ev <= threshold:
            continue

        best_idx = day_df.index[valid.values][local_best]
        best_probas = probas[local_best]
        sized_risk = compute_sized_risk(
            ev=local_best_ev,
            m=m,
            risk_pct=risk_pct,
            kelly_fraction=kelly_fraction,
            min_risk_pct=min_risk_pct,
        )

        candidates.append(
            {
                "idx": best_idx,
                "m": m,
                "ev": local_best_ev,
                "probas": best_probas,
                "sized_risk": sized_risk,
            }
        )

    return candidates


def _threshold_selected_trades(
    df_candidates: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    eligible = df_candidates[df_candidates["ev"] > threshold].copy()
    if eligible.empty:
        return eligible
    selected_idx = eligible.groupby("date")["ev"].idxmax()
    return eligible.loc[selected_idx].copy()


def _collect_threshold_eval_candidates(
    df_train_window: pd.DataFrame,
    eval_mask_int: pd.Series,
    active_model: MultiHeadModel,
    active_calibrators: dict[float, dict[int, IsotonicRegression]],
    active_E_R_TIME: dict[tuple[float, str], float],
    active_feature_cols: list[str],
    cost_R: float,
    selection_multiples: list[float],
    long_only_filter: bool,
    risk_pct: float,
    kelly_fraction: float,
    min_risk_pct: float,
    stop_distance_config: dict,
) -> pd.DataFrame:
    records = []
    df_eval_window = apply_ambiguous_worst_case(df_train_window[eval_mask_int], selection_multiples)

    for day_str in df_eval_window["date"].unique():
        day_df_eval = df_eval_window[df_eval_window["date"] == day_str]
        candidates = select_daily_head_candidates(
            day_df=day_df_eval,
            active_feature_cols=active_feature_cols,
            selection_multiples=selection_multiples,
            active_model=active_model,
            active_calibrators=active_calibrators,
            active_E_R_TIME=active_E_R_TIME,
            ev_threshold=-np.inf,
            cost_R=cost_R,
            m05_threshold=-np.inf,
            long_only_filter=long_only_filter,
            risk_pct=risk_pct,
            kelly_fraction=kelly_fraction,
            min_risk_pct=min_risk_pct,
        )

        for candidate in candidates:
            actual_r = day_df_eval.loc[candidate["idx"], f"y_R_m_{candidate['m']}"]
            if pd.isna(actual_r):
                continue
            row = day_df_eval.loc[candidate["idx"]]
            position_return = position_return_from_row(row, actual_r, stop_distance_config)
            if not np.isfinite(position_return):
                continue
            records.append(
                {
                    "date": day_str,
                    "m": float(candidate["m"]),
                    "ev": float(candidate["ev"]),
                    "sized_risk": float(candidate["sized_risk"]),
                    "actual_R": float(actual_r),
                    "position_return": float(position_return),
                    "trade_return": float(position_return),
                }
            )

    return pd.DataFrame(records)


def select_dynamic_ev_threshold(
    df_train_window: pd.DataFrame,
    eval_mask_int: pd.Series,
    active_model: MultiHeadModel,
    active_calibrators: dict[float, dict[int, IsotonicRegression]],
    active_E_R_TIME: dict[tuple[float, str], float],
    active_feature_cols: list[str],
    fallback_threshold: float,
    dynamic_ev_threshold: dict | None,
    cost_R: float,
    selection_multiples: list[float],
    long_only_filter: bool,
    risk_pct: float,
    kelly_fraction: float,
    min_risk_pct: float,
    stop_distance_config: dict,
    log_prefix: str,
) -> tuple[float, dict]:
    config = dynamic_ev_threshold or {}
    if not bool(config.get("enabled", False)):
        return float(fallback_threshold), {
            "enabled": False,
            "selected_threshold": float(fallback_threshold),
            "reason": "disabled",
            "threshold_scores": [],
        }

    grid = sorted({float(x) for x in config.get("grid", [fallback_threshold])})
    min_trades = int(config.get("min_trades", 20))
    dynamic_fallback_threshold = float(grid[0])

    df_candidates = _collect_threshold_eval_candidates(
        df_train_window=df_train_window,
        eval_mask_int=eval_mask_int,
        active_model=active_model,
        active_calibrators=active_calibrators,
        active_E_R_TIME=active_E_R_TIME,
        active_feature_cols=active_feature_cols,
        cost_R=cost_R,
        selection_multiples=selection_multiples,
        long_only_filter=long_only_filter,
        risk_pct=risk_pct,
        kelly_fraction=kelly_fraction,
        min_risk_pct=min_risk_pct,
        stop_distance_config=stop_distance_config,
    )

    if df_candidates.empty:
        print(f"{log_prefix}[!] Dynamic EV threshold fallback: no held-out candidates.")
        return dynamic_fallback_threshold, {
            "enabled": True,
            "selected_threshold": dynamic_fallback_threshold,
            "fallback_threshold": float(fallback_threshold),
            "dynamic_fallback_threshold": dynamic_fallback_threshold,
            "reason": "no_heldout_candidates",
            "min_trades": min_trades,
            "candidate_rows": 0,
            "threshold_scores": [],
        }

    threshold_scores = []
    for threshold in grid:
        selected = _threshold_selected_trades(df_candidates, threshold)
        trade_count = int(len(selected))
        mean_trade_return = float(selected["trade_return"].mean()) if trade_count else float("nan")
        mean_R = float(selected["actual_R"].mean()) if trade_count else float("nan")
        win_rate = float((selected["actual_R"] > 0).mean()) if trade_count else float("nan")
        threshold_scores.append(
            {
                "threshold": float(threshold),
                "trades": trade_count,
                "mean_trade_return": mean_trade_return,
                "mean_R": mean_R,
                "win_rate": win_rate,
                "eligible": bool(trade_count >= min_trades and np.isfinite(mean_trade_return)),
            }
        )

    eligible_scores = [row for row in threshold_scores if row["eligible"]]
    if not eligible_scores:
        print(
            f"{log_prefix}[!] Dynamic EV threshold fallback: no threshold had "
            f"at least {min_trades} held-out trades."
        )
        return dynamic_fallback_threshold, {
            "enabled": True,
            "selected_threshold": dynamic_fallback_threshold,
            "fallback_threshold": float(fallback_threshold),
            "dynamic_fallback_threshold": dynamic_fallback_threshold,
            "reason": "no_threshold_met_min_trades",
            "min_trades": min_trades,
            "candidate_rows": int(len(df_candidates)),
            "threshold_scores": threshold_scores,
        }

    best = sorted(
        eligible_scores,
        key=lambda row: (-row["mean_trade_return"], row["threshold"]),
    )[0]
    print(
        f"{log_prefix}[+] Dynamic EV threshold selected {best['threshold']:.4f} "
        f"from {best['trades']} held-out trades "
        f"(mean_trade_return={best['mean_trade_return']:.4f}, win_rate={best['win_rate']:.3f})."
    )
    return float(best["threshold"]), {
        "enabled": True,
        "selected_threshold": float(best["threshold"]),
        "fallback_threshold": float(fallback_threshold),
        "dynamic_fallback_threshold": dynamic_fallback_threshold,
        "reason": "selected",
        "objective": str(config.get("objective", "mean_trade_return")),
        "min_trades": min_trades,
        "candidate_rows": int(len(df_candidates)),
        "threshold_scores": threshold_scores,
    }


def resolve_as_of_date(df_all: pd.DataFrame, as_of_date: str | None) -> pd.Timestamp:
    dates = sorted(pd.to_datetime(df_all["date"]).unique())
    if not dates:
        raise ValueError("No dates found in dataset.")

    if as_of_date is None:
        return pd.Timestamp(dates[-1])

    requested = pd.to_datetime(as_of_date)
    eligible = [d for d in dates if pd.Timestamp(d) <= requested]
    if not eligible:
        raise ValueError(
            f"Requested as_of_date={requested.date()} is before available data range "
            f"starting {pd.Timestamp(dates[0]).date()}."
        )
    return pd.Timestamp(eligible[-1])


def predict_calibrated_probas(
    model: MultiHeadModel,
    calibrators: dict[float, dict[int, IsotonicRegression]],
    m: float,
    X: pd.DataFrame,
) -> np.ndarray:
    raw_probas = model.models[m].predict_proba(X)
    calibrated_probas = np.zeros_like(raw_probas)
    head_calibrators = calibrators[m]

    for c in range(raw_probas.shape[1]):
        calibrated_probas[:, c] = head_calibrators[c].predict(raw_probas[:, c])

    row_sums = calibrated_probas.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return calibrated_probas / row_sums


def _add_missing_classes(
    X_t: pd.DataFrame,
    y_t: pd.Series,
) -> tuple[pd.DataFrame, pd.Series, list[int]]:
    missing_classes = sorted(set([0, 1, 2]) - set(np.unique(y_t)))
    if not missing_classes:
        return X_t, y_t, missing_classes

    dummy_X = [X_t.iloc[0:1].copy() for _ in missing_classes]
    X_t = pd.concat([X_t] + dummy_X, ignore_index=True)
    y_t = pd.concat([y_t, pd.Series(missing_classes)], ignore_index=True)
    return X_t, y_t, missing_classes


def _fit_sample_weights(y_t: pd.Series, missing_classes: list[int]) -> np.ndarray:
    sample_weights = compute_sample_weights(y_t.values)
    if missing_classes:
        sample_weights[-len(missing_classes):] = 0.00001
    return sample_weights


def _select_dynamic_features(
    df_train_window: pd.DataFrame,
    feature_cols: list[str],
    t_mask_int: pd.Series,
    cal_mask_int: pd.Series,
    xgb_params: dict,
    rr_multiples: list[float],
    log_prefix: str,
) -> list[str]:
    print(f"{log_prefix}[*] Running permutation importance over {len(feature_cols)} base features...")
    perm_importances = np.zeros(len(feature_cols))
    valid_heads = 0

    for m in rr_multiples:
        ambig_col = f"y_ambig_m_{m}"
        type_col = f"y_type_m_{m}"

        t_rows_fs = t_mask_int & (df_train_window[ambig_col] != True)
        v_rows_fs = cal_mask_int & (df_train_window[ambig_col] != True)
        X_t_fs = df_train_window.loc[t_rows_fs, feature_cols].copy()
        y_t_fs = df_train_window.loc[t_rows_fs, type_col].astype(int).copy()
        X_v_fs = df_train_window.loc[v_rows_fs, feature_cols].copy()
        y_v_fs = df_train_window.loc[v_rows_fs, type_col].astype(int).copy()

        if len(X_t_fs) == 0 or len(X_v_fs) == 0 or len(np.unique(y_t_fs)) < 2:
            continue

        X_t_fs, y_t_fs, missing_classes_fs = _add_missing_classes(X_t_fs, y_t_fs)
        sw_fs = _fit_sample_weights(y_t_fs, missing_classes_fs)

        base_params = xgb_params.copy()
        base_params.pop("early_stopping_rounds", None)
        clf_fs = XGBClassifier(**base_params)
        clf_fs.fit(X_t_fs, y_t_fs, sample_weight=sw_fs)

        perm_result = permutation_importance(
            clf_fs,
            X_v_fs,
            y_v_fs,
            n_repeats=5,
            random_state=42,
            scoring="accuracy",
        )
        perm_importances += perm_result.importances_mean
        valid_heads += 1

    if valid_heads == 0:
        print(f"{log_prefix}[!] Dynamic feature selection skipped (no valid heads). Using full feature set.")
        return feature_cols.copy()

    perm_importances /= valid_heads
    active_feature_cols = [f for f, imp in zip(feature_cols, perm_importances) if imp > 0]
    dropped = [f for f, imp in zip(feature_cols, perm_importances) if imp <= 0]
    print(
        f"{log_prefix}[+] Selected {len(active_feature_cols)}/{len(feature_cols)} features "
        "via permutation importance."
    )
    print(f"{log_prefix}[+] Kept: {active_feature_cols}")
    print(f"{log_prefix}[-] Dropped: {dropped}")

    if len(active_feature_cols) < 3:
        print(
            f"{log_prefix}[!] Too few features selected; "
            f"falling back to all {len(feature_cols)} features."
        )
        return feature_cols.copy()

    return active_feature_cols


def _run_optuna(
    df_train_window: pd.DataFrame,
    active_feature_cols: list[str],
    t_mask_int: pd.Series,
    cal_mask_int: pd.Series,
    base_params: dict,
    rr_multiples: list[float],
    trials: int,
    log_prefix: str,
) -> dict:
    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }
        trial_params = base_params.copy()
        trial_params.update(params)

        val_losses = []
        for m in rr_multiples:
            ambig_col = f"y_ambig_m_{m}"
            type_col = f"y_type_m_{m}"

            t_rows = t_mask_int & (df_train_window[ambig_col] != True)
            v_rows = cal_mask_int & (df_train_window[ambig_col] != True)

            X_t = df_train_window.loc[t_rows, active_feature_cols].copy()
            y_t = df_train_window.loc[t_rows, type_col].astype(int).copy()
            X_v = df_train_window.loc[v_rows, active_feature_cols].copy()
            y_v = df_train_window.loc[v_rows, type_col].astype(int).copy()

            if len(X_t) == 0 or len(X_v) == 0 or len(np.unique(y_t)) < 2:
                continue

            X_t, y_t, missing_classes = _add_missing_classes(X_t, y_t)
            sw = _fit_sample_weights(y_t, missing_classes)

            clf = XGBClassifier(**trial_params)
            clf.fit(
                X_t,
                y_t,
                sample_weight=sw,
                eval_set=[(X_v, y_v)],
                verbose=0,
            )
            res = clf.evals_result()
            val_losses.append(min(res["validation_0"]["mlogloss"]))

        if not val_losses:
            return float("inf")
        return float(np.mean(val_losses))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    print(f"{log_prefix}[*] Running Optuna optimization ({trials} trials)...")
    study.optimize(objective, n_trials=trials)
    print(f"{log_prefix}[*] Best Optuna params: {study.best_params}")

    params_to_use = base_params.copy()
    params_to_use.update(study.best_params)
    return params_to_use


def _train_calibrated_heads(
    df_train_window: pd.DataFrame,
    active_feature_cols: list[str],
    t_mask_int: pd.Series,
    cal_mask_int: pd.Series,
    params_to_use: dict,
    rr_multiples: list[float],
    log_prefix: str,
) -> tuple[MultiHeadModel, dict[float, dict[int, IsotonicRegression]], list[float]]:
    model = MultiHeadModel(
        rr_multiples=rr_multiples,
        base_estimator_class=XGBClassifier,
        estimator_params=params_to_use,
    )
    calibrators: dict[float, dict[int, IsotonicRegression]] = {}
    trained_heads: list[float] = []

    for m in model.rr_multiples:
        ambig_col = f"y_ambig_m_{m}"
        type_col = f"y_type_m_{m}"

        t_rows = t_mask_int & (df_train_window[ambig_col] != True)
        v_rows = cal_mask_int & (df_train_window[ambig_col] != True)

        X_t = df_train_window.loc[t_rows, active_feature_cols].copy()
        y_t = df_train_window.loc[t_rows, type_col].astype(int).copy()
        X_v = df_train_window.loc[v_rows, active_feature_cols].copy()
        y_v = df_train_window.loc[v_rows, type_col].astype(int).copy()

        if len(X_t) == 0 or len(X_v) == 0 or len(np.unique(y_t)) < 2:
            print(f"{log_prefix}[!] Skipping head m={m} due to insufficient train/calibration data.")
            continue

        X_t, y_t, missing_classes = _add_missing_classes(X_t, y_t)
        sw = _fit_sample_weights(y_t, missing_classes)

        model.models[m].fit(
            X_t,
            y_t,
            sample_weight=sw,
            eval_set=[(X_t, y_t), (X_v, y_v)],
            verbose=0,
        )

        cal_probas = model.models[m].predict_proba(X_v)
        head_calibrators = {}
        for c in range(cal_probas.shape[1]):
            y_cal_binary = (y_v.values == c).astype(int)
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(cal_probas[:, c], y_cal_binary)
            head_calibrators[c] = ir

        calibrators[m] = head_calibrators
        trained_heads.append(m)
        print(
            f"{log_prefix}[+] Trained + calibrated head m={m} "
            f"on {len(X_t)} train rows / {len(X_v)} cal rows."
        )

    return model, calibrators, trained_heads


def _train_meta_model(
    df_train_window: pd.DataFrame,
    meta_mask_int: pd.Series,
    active_model: MultiHeadModel,
    active_calibrators: dict[float, dict[int, IsotonicRegression]],
    active_E_R_TIME: dict[tuple[float, str], float],
    active_feature_cols: list[str],
    ev_threshold: float,
    cost_R: float,
    selection_multiples: list[float],
    m05_threshold: float,
    long_only_filter: bool,
    risk_pct: float,
    kelly_fraction: float,
    min_risk_pct: float,
    stop_distance_config: dict,
    meta_model_target: str,
    log_prefix: str,
) -> LogisticRegression | RandomForestRegressor | None:
    meta_records = []
    df_meta_window = apply_ambiguous_worst_case(df_train_window[meta_mask_int], selection_multiples)
    meta_feature_cols = get_meta_feature_cols(meta_model_target)

    for day_str in df_meta_window["date"].unique():
        day_df_meta = df_meta_window[df_meta_window["date"] == day_str]
        candidates = select_daily_head_candidates(
            day_df=day_df_meta,
            active_feature_cols=active_feature_cols,
            selection_multiples=selection_multiples,
            active_model=active_model,
            active_calibrators=active_calibrators,
            active_E_R_TIME=active_E_R_TIME,
            ev_threshold=ev_threshold,
            cost_R=cost_R,
            m05_threshold=m05_threshold,
            long_only_filter=long_only_filter,
            risk_pct=risk_pct,
            kelly_fraction=kelly_fraction,
            min_risk_pct=min_risk_pct,
        )
        if not candidates:
            continue

        meta_candidates = candidates
        if meta_model_target == "diagnostic_binary":
            meta_candidates = [max(candidates, key=lambda candidate: candidate["ev"])]

        for candidate in meta_candidates:
            actual_r = day_df_meta.loc[candidate["idx"], f"y_R_m_{candidate['m']}"]
            if pd.isna(actual_r):
                continue

            row_features = build_meta_features(
                primary_m=candidate["m"],
                primary_ev=candidate["ev"],
                primary_probas=candidate["probas"],
                sized_risk=candidate["sized_risk"],
                primary_stop_pct=stop_pct_from_values(
                    float(day_df_meta.loc[candidate["idx"], "entry_price"]),
                    stop_distance_from_row(day_df_meta.loc[candidate["idx"]], stop_distance_config),
                ),
            )
            if meta_model_target == "diagnostic_binary":
                row_features["meta_target_binary"] = 1 if actual_r > 0 else 0
            else:
                position_return = position_return_from_row(
                    day_df_meta.loc[candidate["idx"]],
                    actual_r,
                    stop_distance_config,
                )
                if not np.isfinite(position_return):
                    continue
                row_features["meta_target_return"] = float(position_return)
            meta_records.append(row_features)

    if len(meta_records) <= 20:
        print(f"{log_prefix}[!] Not enough simulated trades ({len(meta_records)}) to train meta-model.")
        return None

    df_meta = pd.DataFrame(meta_records)
    X_meta = df_meta[meta_feature_cols]

    if meta_model_target == "diagnostic_binary":
        y_meta = df_meta["meta_target_binary"]
        if len(np.unique(y_meta)) < 2:
            print(f"{log_prefix}[!] Only one class in meta labels, skipping meta-model.")
            return None

        meta_model = LogisticRegression(C=1.0, max_iter=1000)
        meta_model.fit(X_meta, y_meta)
        n_pos = int(y_meta.sum())
        n_neg = len(y_meta) - n_pos
        print(
            f"{log_prefix}[+] Diagnostic meta-model trained on {len(meta_records)} trades "
            f"({n_pos} profitable, {n_neg} unprofitable)."
        )
        return meta_model

    y_meta = df_meta["meta_target_return"]
    if np.isclose(float(y_meta.std(ddof=0)), 0.0):
        print(f"{log_prefix}[!] Meta target return is constant, skipping meta-model.")
        return None

    meta_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=4,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
    )
    meta_model.fit(X_meta, y_meta)
    mean_target = float(y_meta.mean())
    positive_share = float((y_meta > 0).mean())
    print(
        f"{log_prefix}[+] Expected-return meta-model trained on {len(meta_records)} head winners "
        f"(mean return={mean_target:.4f}, positive_share={positive_share:.3f})."
    )
    return meta_model


def train_retrain_window(
    df_all: pd.DataFrame,
    feature_cols: list[str],
    rr_multiples: list[float],
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
    train_side: str,
    m05_threshold: float,
    long_only_filter: bool,
    kelly_fraction: float,
    min_risk_pct: float,
    stop_distance_config: dict | None,
    meta_model_target: str,
    selection_multiples: list[float] | None = None,
    allow_insufficient_history: bool = False,
    log_prefix: str = "",
) -> dict | None:
    if selection_multiples is None:
        selection_multiples = rr_multiples
    if stop_distance_config is None:
        stop_distance_config = {"rule": "atr", "atr_period": 14, "k": 0.3}

    if "date_dt" in df_all.columns:
        df_all_dt = df_all
    else:
        df_all_dt = df_all.copy()
        df_all_dt["date_dt"] = pd.to_datetime(df_all_dt["date"])

    as_of_date = pd.to_datetime(as_of_date)
    train_end = as_of_date - pd.Timedelta(days=embargo_days)
    train_start = train_end - pd.Timedelta(days=lookback_days)
    mask = (
        (df_all_dt["date_dt"] >= train_start)
        & (df_all_dt["date_dt"] <= train_end)
        & training_side_mask(df_all_dt, train_side)
    )
    df_train_window = df_all_dt[mask].copy()

    dates_in_window = sorted(df_train_window["date"].unique())
    min_history_days = lookback_days * MIN_HISTORY_FRACTION
    if len(dates_in_window) < min_history_days:
        message = (
            f"Insufficient training history: {len(dates_in_window)} days found, "
            f"need at least {min_history_days:.0f}."
        )
        if allow_insufficient_history:
            print(f"{log_prefix}[!] {message}")
            return None
        raise RuntimeError(message)

    print(
        f"{log_prefix}[+] Retraining as-of {as_of_date.date()} "
        f"(train window: {train_start.date()} -> {train_end.date()}, days={len(dates_in_window)})"
    )

    cal_cutoff = dates_in_window[int(len(dates_in_window) * 0.70)]
    meta_cutoff = dates_in_window[int(len(dates_in_window) * 0.85)]
    t_mask_int = df_train_window["date"] < cal_cutoff
    cal_mask_int = (df_train_window["date"] >= cal_cutoff) & (df_train_window["date"] < meta_cutoff)
    meta_mask_int = df_train_window["date"] >= meta_cutoff

    if dynamic_features:
        active_feature_cols = _select_dynamic_features(
            df_train_window=df_train_window,
            feature_cols=feature_cols,
            t_mask_int=t_mask_int,
            cal_mask_int=cal_mask_int,
            xgb_params=xgb_params,
            rr_multiples=rr_multiples,
            log_prefix=log_prefix,
        )
    else:
        active_feature_cols = feature_cols.copy()

    if optuna_tune:
        params_to_use = _run_optuna(
            df_train_window=df_train_window,
            active_feature_cols=active_feature_cols,
            t_mask_int=t_mask_int,
            cal_mask_int=cal_mask_int,
            base_params=xgb_params,
            rr_multiples=rr_multiples,
            trials=optuna_trials,
            log_prefix=log_prefix,
        )
    else:
        params_to_use = xgb_params.copy()
        print(f"{log_prefix}[*] Using default/tuned params without Optuna.")

    model, calibrators, trained_heads = _train_calibrated_heads(
        df_train_window=df_train_window,
        active_feature_cols=active_feature_cols,
        t_mask_int=t_mask_int,
        cal_mask_int=cal_mask_int,
        params_to_use=params_to_use,
        rr_multiples=rr_multiples,
        log_prefix=log_prefix,
    )
    if not trained_heads:
        message = "No model heads were trained."
        if allow_insufficient_history:
            print(f"{log_prefix}[!] {message}")
            return None
        raise RuntimeError(f"{message} Cannot export bundle.")

    e_r_time = compute_dynamic_time_r(df_train_window, rr_multiples)
    selected_ev_threshold, dynamic_ev_threshold_result = select_dynamic_ev_threshold(
        df_train_window=df_train_window,
        eval_mask_int=meta_mask_int,
        active_model=model,
        active_calibrators=calibrators,
        active_E_R_TIME=e_r_time,
        active_feature_cols=active_feature_cols,
        fallback_threshold=ev_threshold,
        dynamic_ev_threshold=dynamic_ev_threshold,
        cost_R=cost_R,
        selection_multiples=selection_multiples,
        long_only_filter=long_only_filter,
        risk_pct=risk_pct,
        kelly_fraction=kelly_fraction,
        min_risk_pct=min_risk_pct,
        stop_distance_config=stop_distance_config,
        log_prefix=log_prefix,
    )

    meta_model = None
    if use_meta_model:
        print(f"{log_prefix}[*] Training meta-model ({meta_model_target})...")
        meta_model = _train_meta_model(
            df_train_window=df_train_window,
            meta_mask_int=meta_mask_int,
            active_model=model,
            active_calibrators=calibrators,
            active_E_R_TIME=e_r_time,
            active_feature_cols=active_feature_cols,
            ev_threshold=selected_ev_threshold,
            cost_R=cost_R,
            selection_multiples=selection_multiples,
            m05_threshold=m05_threshold,
            long_only_filter=long_only_filter,
            risk_pct=risk_pct,
            kelly_fraction=kelly_fraction,
            min_risk_pct=min_risk_pct,
            stop_distance_config=stop_distance_config,
            meta_model_target=meta_model_target,
            log_prefix=log_prefix,
        )

    return {
        "model": model,
        "calibrators": calibrators,
        "active_feature_cols": active_feature_cols,
        "feature_dtypes": {
            c: str(df_train_window[c].dtype)
            for c in active_feature_cols
            if c in df_train_window.columns
        },
        "e_r_time": e_r_time,
        "meta_model": meta_model,
        "meta_model_target": meta_model_target if use_meta_model else None,
        "meta_feature_cols": get_meta_feature_cols(meta_model_target) if use_meta_model else [],
        "trained_heads": trained_heads,
        "params_to_use": params_to_use,
        "train_start": train_start,
        "train_end": train_end,
        "cal_cutoff": pd.to_datetime(cal_cutoff),
        "meta_cutoff": pd.to_datetime(meta_cutoff),
        "as_of_date": as_of_date,
        "lookback_days": lookback_days,
        "step_days": step_days,
        "embargo_days": embargo_days,
        "ev_threshold": selected_ev_threshold,
        "base_ev_threshold": ev_threshold,
        "dynamic_ev_threshold": dynamic_ev_threshold_result,
        "risk_pct": risk_pct,
        "cost_R": cost_R,
    }

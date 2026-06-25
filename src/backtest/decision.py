"""
EV computation and trade-filtering logic.

EV(m) = P(TP)*m  -  P(SL)*1  +  P(TIME)*avg_R_time  -  cost_R

A trade is taken only when EV > threshold (default 0).
"""

import numpy as np


# Historical E[R|TIME] from training data, keyed by (m, side).
# Computed from the training split — represents the average R payoff
# when neither stop nor target is hit by 10:00 AM.
E_R_TIME = {
    (0.5,  "long"):  -0.0453,
    (0.5,  "short"): -0.0558,
    (1.0,  "long"):  +0.0065,
    (1.0,  "short"): -0.0065,
    (1.5,  "long"):  +0.0121,
    (1.5,  "short"): -0.0002,
    (2.0,  "long"):  +0.0133,
    (2.0,  "short"): +0.0010,
}


def compute_ev(
    probas: np.ndarray,
    m: float,
    sides: np.ndarray | None = None,
    cost_R: float = 0.05,
    custom_E_R_TIME: dict | None = None,
) -> np.ndarray:
    """
    Compute per-row Expected Value in R-units.

    Parameters
    ----------
    probas : (N, 3) array — columns [P(SL), P(TP), P(TIME)]
    m      : reward multiple
    sides  : (N,) array of "long"/"short" strings. If provided, uses
             per-side E[R|TIME]. Otherwise falls back to 0.
    cost_R : execution / slippage cost in R-units

    Returns
    -------
    (N,) array of EV values
    """
    P_SL   = probas[:, 0]
    P_TP   = probas[:, 1]
    P_TIME = probas[:, 2]

    if sides is not None:
        dict_to_use = custom_E_R_TIME if custom_E_R_TIME is not None else E_R_TIME
        avg_R_time = np.array([
            dict_to_use.get((m, s), 0.0) for s in sides
        ])
    else:
        avg_R_time = 0.0

    return (P_TP * m) - (P_SL * 1.0) + (P_TIME * avg_R_time) - cost_R


def filter_trades(ev: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """Return a boolean mask selecting trades whose EV exceeds *threshold*."""
    return ev > threshold

"""Causal evaluation metrics: Qini Curve, AUUC, and Uplift Decile Tables.

Provides specialized validation tools for Uplift models where standard
supervised metrics (AUC-ROC, RMSE) fail due to the Fundamental Problem of Causal Inference.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


def compute_qini_curve(
    y_true: np.ndarray,
    t_true: np.ndarray,
    cate_pred: np.ndarray,
    n_bins: int = 100,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Compute the cumulative Qini Curve and Area Under Uplift Curve (AUUC).

    Formula:
    Q(u) = Y_1(u) - (N_1(u) / N_0(u)) * Y_0(u)

    Parameters
    ----------
    y_true : np.ndarray
        Binary observed outcome (1 = converted/retained, 0 = churned).
    t_true : np.ndarray
        Binary treatment indicator (1 = treated, 0 = control).
    cate_pred : np.ndarray
        Estimated treatment effects (uplift) used to rank customers descending.
    n_bins : int
        Number of quantile evaluation points.

    Returns
    -------
    fractions : np.ndarray
        Fraction of customer base targeted [0.0 to 1.0].
    qini_values : np.ndarray
        Cumulative incremental gains Q(u).
    auuc : float
        Area Under the Uplift Curve.
    qini_score : float
        Normalized Qini score (Model AUUC minus Random baseline AUUC).
    """
    y = np.asarray(y_true, dtype=np.float64)
    t = np.asarray(t_true, dtype=np.int32)
    cate = np.asarray(cate_pred, dtype=np.float64)
    N = len(y)

    # Sort descending by predicted CATE (best persuadables targeted first)
    order = np.argsort(-cate)
    y_sorted = y[order]
    t_sorted = t[order]

    # Cumulative treated and control responses
    cum_t = np.cumsum(t_sorted)
    cum_c = np.cumsum(1 - t_sorted)
    cum_y_t = np.cumsum(y_sorted * t_sorted)
    cum_y_c = np.cumsum(y_sorted * (1 - t_sorted))

    # Evaluate at regular fractional intervals
    # Since idx is 0-indexed, the cumulative count evaluated at idx is (idx + 1)
    indices = np.unique(np.linspace(0, N - 1, min(n_bins, N), dtype=int))
    fractions = (indices + 1).astype(np.float64) / float(N)

    qini_vals = []
    for idx in indices:
        n1 = cum_t[idx]
        n0 = cum_c[idx]
        y1 = cum_y_t[idx]
        y0 = cum_y_c[idx]

        if n0 == 0:
            gain = float(y1)
        else:
            # Traditional Radcliffe (2007) Qini formula
            gain = float(y1 - (n1 / n0) * y0)
        qini_vals.append(gain)

    # Prepend origin (0.0, 0.0)
    fractions = np.concatenate([[0.0], fractions])
    qini_values = np.concatenate([[0.0], np.array(qini_vals, dtype=np.float64)])

    # Compute AUUC via trapezoidal numerical integration
    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    auuc = float(trapz_fn(qini_values, fractions))

    # Random baseline connects (0, 0) to (1.0, Final Gain)
    final_gain = qini_values[-1]
    auuc_random = 0.5 * final_gain
    qini_score = auuc - auuc_random

    return fractions, qini_values, auuc, qini_score


def compute_uplift_by_decile(
    y_true: np.ndarray,
    t_true: np.ndarray,
    cate_pred: np.ndarray,
    n_deciles: int = 10,
) -> pd.DataFrame:
    """Generate Decile-level Uplift Table (Treatment vs Control Conversion Rates).

    Validates that Top Deciles exhibit significant positive conversion uplift
    while Bottom Deciles contain neutral or negative responders (Sleeping Dogs).
    """
    df = pd.DataFrame({
        "y": y_true,
        "t": t_true,
        "cate": cate_pred,
    })

    # Decile assignment (1 = highest predicted uplift, 10 = lowest)
    df["decile"] = pd.qcut(df["cate"].rank(method="first", ascending=False), q=n_deciles, labels=range(1, n_deciles + 1))

    stats = []
    for d, group in df.groupby("decile", observed=False):
        t_group = group[group["t"] == 1]
        c_group = group[group["t"] == 0]

        n_t = len(t_group)
        n_c = len(c_group)

        conv_t = t_group["y"].mean() if n_t > 0 else 0.0
        conv_c = c_group["y"].mean() if n_c > 0 else 0.0
        uplift = conv_t - conv_c

        stats.append({
            "decile": int(d),
            "size": len(group),
            "treated_count": n_t,
            "control_count": n_c,
            "conv_rate_treated": round(float(conv_t), 4),
            "conv_rate_control": round(float(conv_c), 4),
            "incremental_uplift": round(float(uplift), 4),
            "mean_predicted_cate": round(float(group["cate"].mean()), 4),
        })

    return pd.DataFrame(stats)

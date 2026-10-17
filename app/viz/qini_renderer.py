"""Qini curve and decile uplift visualization renderer.

Generates publication-quality charts for model evaluation, comparing
CATE Meta-Learners against Random and Naïve Churn Targeting baselines.
"""

from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def render_qini_and_uplift_plots(
    fractions: np.ndarray,
    qini_curves: Dict[str, np.ndarray],
    uplift_deciles: pd.DataFrame,
    output_path: str = "demo_qini_retention.png",
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """Render a 2-panel figure: (1) Qini Cumulative Curves, (2) Decile Uplift Bars.

    Parameters
    ----------
    fractions : np.ndarray
        Fraction of customer base targeted [0.0, 1.0].
    qini_curves : Dict[str, np.ndarray]
        Mapping of model name to cumulative Qini array.
    uplift_deciles : pd.DataFrame
        Decile uplift stats (conv_rate_treated, conv_rate_control, decile).
    output_path : str
        Path to save output image.

    Returns
    -------
    fig, (ax1, ax2)
        Matplotlib figure and axis objects.
    """
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=130)

    # Panel 1: Qini Curves
    colors = {
        "X-Learner (Künzel 2019)": "#10b981",  # Emerald
        "Doubly Robust Learner": "#3b82f6",    # Blue
        "T-Learner (Baseline)": "#f59e0b",     # Amber
        "Random Targeting": "#94a3b8",         # Slate Gray
    }

    for name, q_vals in qini_curves.items():
        color = colors.get(name, "#6366f1")
        linestyle = "--" if name == "Random Targeting" else "-"
        linewidth = 2.0 if name != "Random Targeting" else 1.5
        ax1.plot(fractions * 100.0, q_vals, label=name, color=color, linestyle=linestyle, linewidth=linewidth)

    ax1.set_title("Causal Qini Cumulative Gains Curve", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Percentage of Customers Targeted (%)", fontsize=11)
    ax1.set_ylabel("Cumulative Incremental Conversions", fontsize=11)
    ax1.legend(loc="upper left", frameon=True, fontsize=10)
    ax1.set_xlim(0, 100)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Panel 2: Uplift by Decile (Treatment vs Control Conversion Rates)
    deciles = uplift_deciles["decile"].values
    conv_t = uplift_deciles["conv_rate_treated"].values * 100.0
    conv_c = uplift_deciles["conv_rate_control"].values * 100.0
    uplift = uplift_deciles["incremental_uplift"].values * 100.0

    x = np.arange(len(deciles))
    width = 0.35

    ax2.bar(x - width/2, conv_t, width, label="Treated Group (Voucher)", color="#10b981", alpha=0.85)
    ax2.bar(x + width/2, conv_c, width, label="Control Group (No Voucher)", color="#cbd5e1", alpha=0.85)

    # Add uplift delta labels on top
    for i, (ct, cc, u) in enumerate(zip(conv_t, conv_c, uplift)):
        sign = "+" if u >= 0 else ""
        text_color = "#047857" if u >= 0 else "#b91c1c"
        y_pos = max(ct, cc) + 0.8
        ax2.text(i, y_pos, f"{sign}{u:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=text_color)

    ax2.set_title("Conversion Rate & Incremental Uplift by Decile (Persuadables to Sleeping Dogs)", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("Customer Decile (Ranked by Predicted Uplift)", fontsize=11)
    ax2.set_ylabel("Conversion Rate (%)", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"D{d}" for d in deciles])
    ax2.legend(loc="upper right", frameon=True, fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    return fig, (ax1, ax2)

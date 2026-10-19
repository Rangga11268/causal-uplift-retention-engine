import numpy as np
import pandas as pd
import pytest

from app.core.metrics import compute_qini_curve, compute_uplift_by_decile


def test_qini_curve_properties():
    # Synthetic perfect ordering: high CATE has high treatment response
    np.random.seed(42)
    N = 1000
    t = np.random.binomial(1, 0.5, N)
    # Perfect alignment: high cate -> y=1 when treated, y=0 when control
    cate = np.linspace(1.0, -1.0, N)
    y = np.where(t == 1, (cate > 0).astype(int), 0)

    fracs, qini_vals, auuc, qini_score = compute_qini_curve(y, t, cate, n_bins=50)

    assert len(fracs) == 51
    assert len(qini_vals) == 51
    assert fracs[0] == 0.0
    assert np.isclose(fracs[-1], 1.0, atol=0.01)
    assert qini_vals[0] == 0.0
    # Qini score for superior ordering must be strictly positive
    assert qini_score > 0.0
    assert auuc > 0.0


def test_uplift_by_decile_structure():
    np.random.seed(42)
    N = 1000
    t = np.random.binomial(1, 0.5, N)
    y = np.random.binomial(1, 0.4, N)
    cate = np.random.randn(N)

    deciles_df = compute_uplift_by_decile(y, t, cate, n_deciles=10)

    assert len(deciles_df) == 10
    assert list(deciles_df.columns) == [
        "decile",
        "size",
        "treated_count",
        "control_count",
        "conv_rate_treated",
        "conv_rate_control",
        "incremental_uplift",
        "mean_predicted_cate",
    ]
    assert np.sum(deciles_df["size"]) == N


def test_metrics_input_validation():
    with pytest.raises(ValueError, match="At least 2 samples required"):
        compute_qini_curve(np.array([1]), np.array([1]), np.array([0.5]))

    with pytest.raises(ValueError, match="Dimension mismatch"):
        compute_qini_curve(np.array([1, 0]), np.array([1]), np.array([0.5, 0.2]))

    with pytest.raises(ValueError, match="must not contain NaN"):
        compute_qini_curve(np.array([1, np.nan]), np.array([1, 0]), np.array([0.5, 0.2]))

    with pytest.raises(ValueError, match="must not contain Inf"):
        compute_qini_curve(np.array([1, 0]), np.array([1, 0]), np.array([np.inf, 0.2]))

    with pytest.raises(ValueError, match="must be binary"):
        compute_qini_curve(np.array([1, 0]), np.array([2, 0]), np.array([0.5, 0.2]))

    with pytest.raises(ValueError, match="Sample count .* must be >= n_deciles"):
        compute_uplift_by_decile(np.array([1, 0]), np.array([1, 0]), np.array([0.5, 0.2]), n_deciles=10)


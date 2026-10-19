"""Synthetic yet realistic customer retention dataset generator.

Generates realistic observational customer features with selection bias
and ground-truth CATE under all four canonical uplift archetypes:
1. Persuadables: tau(x) >> 0 (converts only if treated)
2. Sure Things: tau(x) ~ 0, Y(0) = 1 (converts regardless)
3. Lost Causes: tau(x) ~ 0, Y(0) = 0 (churns regardless)
4. Sleeping Dogs: tau(x) < 0 (churns if disturbed by promotion)
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd


def generate_customer_retention_data(
    n_samples: int = 10000,
    random_state: int = 42,
    treatment_ratio: float = 0.35,  # Observational bias: 35% targeted
    confounding_strength: float = 1.0,  # 0.0 = RCT (unconfounded), 1.0 = standard observational
    near_zero_cate: bool = False,       # True = simulate global null hypothesis tau_i ~ 0
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic customer retention logs with known counterfactuals.

    Parameters
    ----------
    n_samples : int
        Number of customer records.
    random_state : int
        Seed for reproducibility.
    treatment_ratio : float
        Mean propensity of receiving promo treatment.
    confounding_strength : float, default=1.0
        Multiplier on observational selection bias (0.0 = randomized trial / RCT, 1.0 = observational).
    near_zero_cate : bool, default=False
        If True, simulates the null hypothesis where treatment has no heterogeneous effect (tau ~ 0).

    Returns
    -------
    X : pd.DataFrame
        Customer feature matrix (recency, frequency, monetary, app engagement, tenure, discount sensitivity).
    T : np.ndarray
        Binary treatment indicator (1 = received discount/voucher, 0 = control).
    Y : np.ndarray
        Observed binary conversion/retention outcome.
    tau_true : np.ndarray
        True individual treatment effect (uplift).
    propensity_true : np.ndarray
        True conditional probability of receiving treatment P(T=1|X).
    """
    rng = np.random.default_rng(random_state)

    # 1. Generate realistic customer behavioral features
    # Tenure: months with the platform [1 to 60]
    tenure_months = rng.gamma(shape=3.0, scale=8.0, size=n_samples)
    tenure_months = np.clip(tenure_months, 1.0, 60.0)

    # Recency: days since last purchase [0 to 90]
    recency_days = rng.exponential(scale=20.0, size=n_samples)
    recency_days = np.clip(recency_days, 0.0, 90.0)

    # Historical frequency: total orders past 12 months
    hist_orders = rng.poisson(lam=1.5 + 0.1 * tenure_months)
    hist_orders = np.clip(hist_orders, 0, 50)

    # Historical monetary value: avg order value ($)
    avg_order_value = rng.lognormal(mean=3.5, sigma=0.5, size=n_samples)

    # In-app engagement score: [0.0 to 1.0]
    app_engagement = rng.beta(a=2.0, b=3.0, size=n_samples)

    # Discount sensitivity: [0.0 to 1.0]
    discount_sensitivity = rng.beta(a=2.5, b=2.5, size=n_samples)

    # Assemble feature DataFrame
    X = pd.DataFrame({
        "tenure_months": np.round(tenure_months, 1),
        "recency_days": np.round(recency_days, 1),
        "hist_orders": hist_orders,
        "avg_order_value": np.round(avg_order_value, 2),
        "app_engagement": np.round(app_engagement, 3),
        "discount_sensitivity": np.round(discount_sensitivity, 3),
    })

    # 2. Confounding / Observational Selection Bias (Propensity)
    # Marketing teams naturally target engaged users with higher discount sensitivity
    logits_bias = (
        0.8 * discount_sensitivity
        + 0.5 * app_engagement
        - 0.02 * recency_days
        + 0.015 * hist_orders
    )
    logits_p = -1.2 + float(confounding_strength) * logits_bias
    propensity_true = 1.0 / (1.0 + np.exp(-logits_p))
    propensity_true = np.clip(propensity_true, 0.05, 0.95)

    # Realized treatment assignment T ~ Bernoulli(propensity)
    T = rng.binomial(n=1, p=propensity_true)

    # 3. Ground-truth Causal Archetypes & Counterfactual Potential Outcomes
    # Baseline probability of conversion without treatment Y(0):
    # Loyal/engaged users convert naturally (Sure Things)
    base_logits = (
        -0.8
        + 1.2 * app_engagement
        + 0.03 * hist_orders
        - 0.035 * recency_days
        + 0.02 * tenure_months
    )
    p_y0 = 1.0 / (1.0 + np.exp(-base_logits))

    # Treatment effect tau(x):
    # - High discount sensitivity + moderate engagement -> Large positive uplift (Persuadables)
    # - Low discount sensitivity + high tenure -> Negative effect from marketing spam (Sleeping Dogs)
    if near_zero_cate:
        tau_true = rng.normal(loc=0.0, scale=0.005, size=n_samples)
    else:
        tau_raw = (
            0.35 * discount_sensitivity
            + 0.15 * (1.0 - np.clip(recency_days / 45.0, 0.0, 1.0))
            - 0.18 * (1.0 - discount_sensitivity) * (tenure_months / 40.0)
        )
        tau_true = np.clip(tau_raw, -0.30, 0.60)

    # Probability of conversion with treatment Y(1) = clip(Y(0) + tau, 0, 1)
    p_y1 = np.clip(p_y0 + tau_true, 0.01, 0.99)
    # Recompute exact tau to match clamped probabilities
    tau_true = p_y1 - p_y0

    # 4. Generate realized observed outcome Y
    y0_realized = rng.binomial(n=1, p=p_y0)
    y1_realized = rng.binomial(n=1, p=p_y1)
    Y = np.where(T == 1, y1_realized, y0_realized)

    return X, T, Y, tau_true, propensity_true

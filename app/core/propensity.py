"""Propensity score estimation and calibration.

Estimates e(x) = P(T=1 | X=x) to correct for observational selection bias
and weighting in X-Learner and Doubly Robust estimators.
"""

from typing import Optional, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from app.config import DEFAULT_PROPENSITY, PropensityConfig


class PropensityEstimator:
    """Estimates and clips treatment propensity scores e(x) = P(T=1 | X=x)."""

    def __init__(
        self,
        model_type: str = "gradient_boosting",
        config: PropensityConfig = DEFAULT_PROPENSITY,
    ):
        self.model_type = model_type.lower()
        self.config = config

        if self.model_type in ("gradient_boosting", "lightgbm"):
            self.model = HistGradientBoostingClassifier(
                max_iter=100,
                learning_rate=0.05,
                max_depth=4,
                random_state=config.random_state,
            )
        elif self.model_type == "logistic":
            self.model = LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=config.random_state,
            )
        else:
            raise ValueError(f"Unsupported model_type: {model_type}. Use 'gradient_boosting' or 'logistic'.")

        self.is_fitted = False

    def fit(self, X: Union[np.ndarray, pd.DataFrame], T: np.ndarray) -> "PropensityEstimator":
        """Fit the propensity classifier."""
        self.model.fit(X, T)
        self.is_fitted = True
        return self

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict clipped propensity scores in [clip_min, clip_max]."""
        if not self.is_fitted:
            raise RuntimeError("PropensityEstimator must be fitted before calling predict_proba.")

        # Probability of class 1 (Treatment)
        p1 = self.model.predict_proba(X)[:, 1]
        # Positivity assumption: strictly clip away from 0.0 and 1.0 to avoid unbounded inverse propensity weights
        return np.clip(p1, self.config.clip_min, self.config.clip_max)

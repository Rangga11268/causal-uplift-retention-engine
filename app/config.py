"""Configuration constants and hyperparameters for Causal Uplift & Retention Engine."""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class PropensityConfig:
    """Propensity estimation configuration."""
    clip_min: float = 0.01
    clip_max: float = 0.99
    cv_folds: int = 5
    random_state: int = 42


@dataclass(frozen=True)
class MetaLearnerConfig:
    """Hyperparameters for CATE Meta-Learners (T-Learner, X-Learner, DR-Learner)."""
    n_estimators: int = 100
    learning_rate: float = 0.05
    max_depth: int = 4
    min_child_samples: int = 20
    random_state: int = 42
    n_cross_fit_folds: int = 5  # Cross-fitting folds for Doubly Robust estimation


@dataclass(frozen=True)
class BudgetConfig:
    """Budget optimization settings."""
    default_daily_budget: float = 5000.0  # Total promo budget units
    default_voucher_cost: float = 10.0    # Mean cost per intervention
    default_clv: float = 85.0             # Mean customer lifetime value / conversion gain
    min_cate_threshold: float = 0.01      # Minimum expected uplift required to consider targeting


DEFAULT_PROPENSITY = PropensityConfig()
DEFAULT_META_LEARNER = MetaLearnerConfig()
DEFAULT_BUDGET = BudgetConfig()

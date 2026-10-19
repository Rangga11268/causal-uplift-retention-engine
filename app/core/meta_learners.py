"""Meta-Learners for estimating Heterogeneous Treatment Effects (CATE).

Implements:
1. S-Learner (Single Model Baseline)
2. T-Learner (Two-Model Baseline)
3. X-Learner (Künzel et al., PNAS 2019)
4. Doubly Robust Learner (Semenova & Chernozhukov, Econometrics Journal 2021)
"""

from abc import ABC, abstractmethod
from typing import Optional, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

from app.config import DEFAULT_META_LEARNER, MetaLearnerConfig
from app.core.propensity import PropensityEstimator


def _create_regressor(cfg: MetaLearnerConfig) -> HistGradientBoostingRegressor:
    """Instantiate robust cross-platform Gradient Boosted Regressor."""
    return HistGradientBoostingRegressor(
        max_iter=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        max_depth=cfg.max_depth,
        min_samples_leaf=cfg.min_child_samples,
        random_state=cfg.random_state,
    )


class BaseMetaLearner(ABC):
    """Abstract base class for CATE Meta-Learners."""

    @abstractmethod
    def fit(self, X: Union[np.ndarray, pd.DataFrame], T: np.ndarray, y: np.ndarray) -> "BaseMetaLearner":
        """Fit the causal model."""
        pass

    @abstractmethod
    def predict_cate(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Estimate Conditional Average Treatment Effect tau(x) = E[Y(1) - Y(0) | X=x]."""
        pass


class SLearner(BaseMetaLearner):
    """S-Learner (Single Learner): concatenates treatment indicator T as an extra feature."""

    def __init__(self, config: MetaLearnerConfig = DEFAULT_META_LEARNER):
        self.config = config
        self.model = _create_regressor(config)
        self.is_fitted = False

    def fit(self, X: Union[np.ndarray, pd.DataFrame], T: np.ndarray, y: np.ndarray) -> "SLearner":
        X_mat = np.asarray(X)
        T_col = np.asarray(T).reshape(-1, 1)
        X_aug = np.hstack([X_mat, T_col])
        self.model.fit(X_aug, y)
        self.is_fitted = True
        return self

    def predict_cate(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict_cate.")
        X_mat = np.asarray(X)
        N = X_mat.shape[0]

        X_treated = np.hstack([X_mat, np.ones((N, 1))])
        X_control = np.hstack([X_mat, np.zeros((N, 1))])

        pred_treated = self.model.predict(X_treated)
        pred_control = self.model.predict(X_control)
        return pred_treated - pred_control


class TLearner(BaseMetaLearner):
    """T-Learner (Two Learners): trains separate models for control and treatment groups."""

    def __init__(self, config: MetaLearnerConfig = DEFAULT_META_LEARNER):
        self.config = config
        self.model_0 = _create_regressor(config)
        self.model_1 = _create_regressor(config)
        self.is_fitted = False

    def fit(self, X: Union[np.ndarray, pd.DataFrame], T: np.ndarray, y: np.ndarray) -> "TLearner":
        X_mat = np.asarray(X)
        T_arr = np.asarray(T)
        y_arr = np.asarray(y)

        mask_0 = (T_arr == 0)
        mask_1 = (T_arr == 1)

        self.model_0.fit(X_mat[mask_0], y_arr[mask_0])
        self.model_1.fit(X_mat[mask_1], y_arr[mask_1])
        self.is_fitted = True
        return self

    def predict_cate(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict_cate.")
        X_mat = np.asarray(X)
        return self.model_1.predict(X_mat) - self.model_0.predict(X_mat)


class XLearner(BaseMetaLearner):
    """X-Learner (Künzel et al. 2019): Cross-imputed heterogeneous treatment effect estimator."""

    def __init__(
        self,
        config: MetaLearnerConfig = DEFAULT_META_LEARNER,
        propensity_estimator: Optional[PropensityEstimator] = None,
    ):
        self.config = config
        self.propensity_estimator = propensity_estimator or PropensityEstimator()

        self.mu_0 = _create_regressor(config)
        self.mu_1 = _create_regressor(config)
        self.tau_0 = _create_regressor(config)
        self.tau_1 = _create_regressor(config)
        self.is_fitted = False

    def fit(self, X: Union[np.ndarray, pd.DataFrame], T: np.ndarray, y: np.ndarray) -> "XLearner":
        X_mat = np.asarray(X)
        T_arr = np.asarray(T)
        y_arr = np.asarray(y)

        mask_0 = (T_arr == 0)
        mask_1 = (T_arr == 1)

        # Step 0: Fit propensity model if not already fitted
        if not self.propensity_estimator.is_fitted:
            self.propensity_estimator.fit(X_mat, T_arr)

        # Step 1: Estimate first-stage response functions mu_0 and mu_1
        self.mu_0.fit(X_mat[mask_0], y_arr[mask_0])
        self.mu_1.fit(X_mat[mask_1], y_arr[mask_1])

        # Step 2: Impute counterfactuals
        # D_1 = Y_1 - mu_0(X_1) (Effect on treated)
        D_1 = y_arr[mask_1] - self.mu_0.predict(X_mat[mask_1])
        # D_0 = mu_1(X_0) - Y_0 (Effect on control)
        D_0 = self.mu_1.predict(X_mat[mask_0]) - y_arr[mask_0]

        # Step 3: Estimate second-stage effect functions tau_1 and tau_0
        self.tau_1.fit(X_mat[mask_1], D_1)
        self.tau_0.fit(X_mat[mask_0], D_0)

        self.is_fitted = True
        return self

    def predict_cate(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict_cate.")
        X_mat = np.asarray(X)

        e_x = self.propensity_estimator.predict_proba(X_mat)
        pred_tau_0 = self.tau_0.predict(X_mat)
        pred_tau_1 = self.tau_1.predict(X_mat)

        # Step 4: Adaptive propensity weighting: tau(x) = e(x)*tau_0(x) + (1 - e(x))*tau_1(x)
        return e_x * pred_tau_0 + (1.0 - e_x) * pred_tau_1


class DoublyRobustLearner(BaseMetaLearner):
    """Doubly Robust (DR) Learner with K-fold cross-fitting and Neyman-orthogonal scores."""

    def __init__(
        self,
        config: MetaLearnerConfig = DEFAULT_META_LEARNER,
        propensity_estimator: Optional[PropensityEstimator] = None,
    ):
        self.config = config
        self.propensity_estimator = propensity_estimator or PropensityEstimator()
        self.effect_model = _create_regressor(config)
        self.is_fitted = False

    def fit(self, X: Union[np.ndarray, pd.DataFrame], T: np.ndarray, y: np.ndarray) -> "DoublyRobustLearner":
        X_mat = np.asarray(X)
        T_arr = np.asarray(T)
        y_arr = np.asarray(y)
        N = len(y_arr)

        n_treated = int(np.sum(T_arr == 1))
        n_control = int(np.sum(T_arr == 0))
        n_folds = self.config.n_cross_fit_folds

        if n_treated < n_folds or n_control < n_folds:
            raise ValueError(
                f"Insufficient sample coverage for {n_folds}-fold cross-fitting: "
                f"treated={n_treated}, control={n_control}. Each group must have at least {n_folds} observations."
            )

        # Cross-fitting to prevent overfitting bias on nuisance estimates
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=self.config.random_state)

        pseudo_outcomes = np.zeros(N, dtype=np.float64)

        for fold_i, (train_idx, val_idx) in enumerate(kf.split(X_mat)):
            X_tr, y_tr, T_tr = X_mat[train_idx], y_arr[train_idx], T_arr[train_idx]
            X_val, y_val, T_val = X_mat[val_idx], y_arr[val_idx], T_arr[val_idx]

            n_tr_treated = np.sum(T_tr == 1)
            n_tr_control = np.sum(T_tr == 0)
            if n_tr_treated < 2 or n_tr_control < 2:
                raise ValueError(
                    f"Fold {fold_i} lacks sufficient treatment coverage: "
                    f"treated={n_tr_treated}, control={n_tr_control}. Both groups must have >= 2 samples."
                )

            # 1. Fit nuisance response models on fold train
            m0 = _create_regressor(self.config)
            m1 = _create_regressor(self.config)
            m0.fit(X_tr[T_tr == 0], y_tr[T_tr == 0])
            m1.fit(X_tr[T_tr == 1], y_tr[T_tr == 1])

            # 2. Fit nuisance propensity model on fold train
            prop_fold = PropensityEstimator()
            prop_fold.fit(X_tr, T_tr)

            # 3. Predict nuisance on fold val
            mu0_val = m0.predict(X_val)
            mu1_val = m1.predict(X_val)
            e_val = prop_fold.predict_proba(X_val)

            # 4. Construct Neyman-orthogonal Doubly Robust pseudo-outcome Gamma_i:
            # Gamma_i = (mu1 - mu0) + [T * (Y - mu1) / e] - [(1 - T) * (Y - mu0) / (1 - e)]
            dr_signal = (
                (mu1_val - mu0_val)
                + (T_val * (y_val - mu1_val) / e_val)
                - ((1.0 - T_val) * (y_val - mu0_val) / (1.0 - e_val))
            )
            pseudo_outcomes[val_idx] = dr_signal

        # Step 5: Fit final CATE regression model on the orthogonalized pseudo-outcomes
        self.effect_model.fit(X_mat, pseudo_outcomes)
        self.is_fitted = True
        return self

    def predict_cate(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict_cate.")
        return self.effect_model.predict(np.asarray(X))

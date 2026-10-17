import time
import numpy as np
import pytest

from app.config import DEFAULT_META_LEARNER, DEFAULT_PROPENSITY
from app.core.meta_learners import DoublyRobustLearner, SLearner, TLearner, XLearner
from app.core.propensity import PropensityEstimator
from app.data.generator import generate_customer_retention_data


@pytest.fixture(scope="module")
def retention_data():
    X, T, Y, tau_true, prop_true = generate_customer_retention_data(n_samples=1500, random_state=42)
    return X, T, Y, tau_true, prop_true


def test_propensity_estimator(retention_data):
    X, T, _, _, _ = retention_data
    prop = PropensityEstimator(model_type="gradient_boosting")
    prop.fit(X, T)
    scores = prop.predict_proba(X)
    assert len(scores) == len(X)
    # Strictly bounded by clipping parameters
    assert np.all(scores >= DEFAULT_PROPENSITY.clip_min)
    assert np.all(scores <= DEFAULT_PROPENSITY.clip_max)
    # Mean propensity should approximate true mean treatment rate
    assert np.isclose(np.mean(scores), np.mean(T), atol=0.10)


def test_s_learner(retention_data):
    X, T, Y, tau_true, _ = retention_data
    model = SLearner()
    model.fit(X, T, Y)
    cate = model.predict_cate(X)
    assert len(cate) == len(X)
    assert np.all(np.isfinite(cate))


def test_t_learner(retention_data):
    X, T, Y, tau_true, _ = retention_data
    model = TLearner()
    model.fit(X, T, Y)
    cate = model.predict_cate(X)
    assert len(cate) == len(X)
    assert np.all(np.isfinite(cate))
    # Correlation with true CATE should be positive
    corr = np.corrcoef(cate, tau_true)[0, 1]
    assert corr > 0.25, f"Expected positive correlation, got {corr:.2f}"


def test_x_learner(retention_data):
    X, T, Y, tau_true, _ = retention_data
    model = XLearner()
    model.fit(X, T, Y)
    cate = model.predict_cate(X)
    assert len(cate) == len(X)
    assert np.all(np.isfinite(cate))
    corr = np.corrcoef(cate, tau_true)[0, 1]
    assert corr > 0.25, f"Expected positive correlation, got {corr:.2f}"


def test_doubly_robust_learner(retention_data):
    X, T, Y, tau_true, _ = retention_data
    model = DoublyRobustLearner()
    model.fit(X, T, Y)
    cate = model.predict_cate(X)
    assert len(cate) == len(X)
    assert np.all(np.isfinite(cate))
    corr = np.corrcoef(cate, tau_true)[0, 1]
    assert corr > 0.15, f"Expected positive correlation for DR-Learner, got {corr:.2f}"

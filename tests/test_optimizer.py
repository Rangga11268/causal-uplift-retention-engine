import numpy as np
import pytest

from app.core.optimizer import KnapsackBudgetOptimizer


def test_optimizer_respects_budget():
    optimizer = KnapsackBudgetOptimizer()
    cate = np.array([0.3, 0.25, 0.15, 0.05, -0.10, -0.20])
    costs = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
    values = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
    budget = 20.0

    res = optimizer.optimize(cate, values, costs, budget)

    assert res.total_cost <= budget
    assert res.targeted_customers == 2
    # Should pick indices 0 and 1 (highest CATE density)
    assert set(res.selected_indices) == {0, 1}
    assert res.expected_incremental_conversions == pytest.approx(0.55)
    assert res.expected_incremental_revenue == pytest.approx(55.0)


def test_optimizer_filters_sleeping_dogs():
    optimizer = KnapsackBudgetOptimizer()
    # All negative responders (Sleeping Dogs)
    cate = np.array([-0.05, -0.15, -0.30])
    res = optimizer.optimize(cate, budget=100.0)

    assert res.targeted_customers == 0
    assert res.total_cost == 0.0
    assert len(res.selected_indices) == 0


def test_optimizer_zero_budget():
    optimizer = KnapsackBudgetOptimizer()
    cate = np.array([0.5, 0.4])
    res = optimizer.optimize(cate, budget=0.0)
    assert res.targeted_customers == 0
    assert res.total_cost == 0.0

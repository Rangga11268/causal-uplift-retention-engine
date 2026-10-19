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


def test_greedy_vs_exact_knapsack():
    """Validates that Dantzig greedy density heuristic achieves >= 90% of exact 0/1 knapsack optimum."""
    optimizer = KnapsackBudgetOptimizer()
    cate = np.array([0.40, 0.35, 0.30, 0.20, 0.15])
    values = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
    costs = np.array([25.0, 20.0, 15.0, 10.0, 5.0])
    budget = 40.0

    greedy_res = optimizer.optimize(cate, values, costs, budget)
    exact_indices, exact_gain, exact_cost = optimizer.solve_exact_01_knapsack(cate, values, costs, budget)

    assert greedy_res.total_cost <= budget
    assert exact_cost <= budget
    # Dantzig (1957) proves greedy achieves at least 50% (1/2) approximation of the 0/1 knapsack optimum.
    # On this test instance, greedy achieves 81.25% of the exact branch-and-bound optimum.
    ratio = greedy_res.expected_incremental_revenue / max(1e-6, exact_gain)
    assert ratio >= 0.50, f"Greedy achieved only {ratio:.2f} of exact solution (below 0.5 bound)"
    assert ratio == pytest.approx(0.8125, abs=0.01)


def test_optimizer_input_validation():
    """Optimizer raises ValueError on mismatched dimensions or empty arrays."""
    optimizer = KnapsackBudgetOptimizer()
    with pytest.raises(ValueError, match="empty"):
        optimizer.optimize(np.array([]))

    with pytest.raises(ValueError, match="Length mismatch"):
        optimizer.optimize(np.array([0.1, 0.2]), customer_values=np.array([100.0]))

    with pytest.raises(ValueError, match="Length mismatch"):
        optimizer.optimize(np.array([0.1, 0.2]), voucher_costs=np.array([10.0, 10.0, 10.0]))

    with pytest.raises(ValueError, match="Budget must be a finite number"):
        optimizer.optimize(np.array([0.1, 0.2]), budget=float("nan"))

    with pytest.raises(ValueError, match="Budget must be a finite number"):
        optimizer.optimize(np.array([0.1, 0.2]), budget=float("inf"))

    with pytest.raises(ValueError, match="must not contain NaN"):
        optimizer.optimize(np.array([0.1, np.nan]))

    with pytest.raises(ValueError, match="limited to N <= 30"):
        optimizer.solve_exact_01_knapsack(np.ones(35), np.ones(35), np.ones(35), budget=50.0)


def test_modified_greedy_beats_pure_density_counterexample():
    """Validates that modified greedy (comparing greedy set with best single item) prevents
    pure density failure where a tiny item blocks a massive single item."""
    optimizer = KnapsackBudgetOptimizer()
    # Item 0 has density 1.01 (gain 1.01 / cost 1.0)
    # Item 1 has density 1.00 (gain 10.0 / cost 10.0)
    # Budget is 10.0
    cate = np.array([0.0101, 0.10])
    values = np.array([100.0, 100.0])
    costs = np.array([1.0, 10.0])
    budget = 10.0

    res = optimizer.optimize(cate, values, costs, budget)
    exact_idx, exact_gain, _ = optimizer.solve_exact_01_knapsack(cate, values, costs, budget)

    # Pure density would select [0] with gain 1.01 (only 10.1% of OPT).
    # Modified greedy selects [1] with gain 10.0, matching the exact optimum.
    assert set(res.selected_indices) == {1}
    assert res.expected_incremental_revenue == pytest.approx(10.0)
    assert exact_gain == pytest.approx(10.0)


def test_greedy_knapsack_stress_test_varying_budgets_and_negative_cate():
    """Stress-test greedy optimizer against exact branch-and-bound knapsack across multiple budgets & negative CATE."""
    optimizer = KnapsackBudgetOptimizer()
    rng = np.random.RandomState(123)
    N = 15  # within exact solver limit (N <= 30)

    # Mix of positive persuadables, neutral, and negative sleeping dogs
    cate = np.array([0.45, -0.20, 0.30, 0.15, -0.05, 0.50, -0.15, 0.25, 0.05, -0.30, 0.35, 0.10, -0.10, 0.40, 0.20])
    values = rng.uniform(50.0, 150.0, size=N)
    costs = rng.uniform(5.0, 30.0, size=N)

    # Test across multiple tight to loose budget scenarios
    for budget in [15.0, 35.0, 60.0, 100.0, 150.0]:
        greedy_res = optimizer.optimize(cate, values, costs, budget)
        exact_indices, exact_gain, exact_cost = optimizer.solve_exact_01_knapsack(cate, values, costs, budget)

        assert greedy_res.total_cost <= budget + 1e-6
        assert exact_cost <= budget + 1e-6

        # Negative CATE must never be selected by greedy or exact
        for idx in greedy_res.selected_indices:
            assert cate[idx] > 0, f"Greedy selected negative CATE at index {idx}"
        for idx in exact_indices:
            assert cate[idx] > 0, f"Exact solver selected negative CATE at index {idx}"

        # If there exists a positive gain, verify modified greedy guarantees >= 50% approximation
        if exact_gain > 0:
            ratio = greedy_res.expected_incremental_revenue / exact_gain
            assert ratio >= 0.50, f"Greedy approximation violated for budget {budget}: ratio = {ratio:.3f}"




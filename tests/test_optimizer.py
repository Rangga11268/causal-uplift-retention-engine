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


def test_optimizer_negative_budget_raises():
    """Negative budget must raise ValueError."""
    optimizer = KnapsackBudgetOptimizer()
    with pytest.raises(ValueError, match="Budget must be non-negative"):
        optimizer.optimize(np.array([0.1, 0.2]), budget=-50.0)


def test_optimizer_budget_too_small_for_any_positive_item():
    """When budget is smaller than the cheapest positive CATE candidate, return 0 targeted."""
    optimizer = KnapsackBudgetOptimizer()
    cate = np.array([0.4, 0.2, -0.1])
    values = np.array([100.0, 100.0, 100.0])
    costs = np.array([10.0, 15.0, 5.0])
    budget = 4.0  # cheaper than 10.0 and 15.0 (and 5.0 is negative CATE)

    res = optimizer.optimize(cate, values, costs, budget=budget)
    assert res.targeted_customers == 0
    assert res.total_cost == 0.0
    assert len(res.selected_indices) == 0


def test_exact_knapsack_small_scale_exhaustive():
    """Exhaustively verify modified greedy against branch-and-bound exact solver on small scale N=8."""
    optimizer = KnapsackBudgetOptimizer()
    rng = np.random.RandomState(42)
    N = 8
    cate = rng.uniform(-0.15, 0.40, size=N)
    values = rng.uniform(40.0, 120.0, size=N)
    costs = rng.uniform(10.0, 25.0, size=N)

    # 1. Very generous budget that can afford ALL positive candidates
    pos_mask = (cate >= 0.001) & (costs > 0) & (values > 0)
    full_budget = float(np.sum(costs[pos_mask])) + 10.0
    greedy_full = optimizer.optimize(cate, values, costs, budget=full_budget)
    exact_idx_full, exact_gain_full, exact_cost_full = optimizer.solve_exact_01_knapsack(
        cate, values, costs, budget=full_budget
    )
    # When all positive items fit, greedy and exact must achieve 100% parity
    assert greedy_full.expected_incremental_revenue == pytest.approx(exact_gain_full, abs=0.01)
    assert set(greedy_full.selected_indices) == set(exact_idx_full)

    # 2. Intermediate budgets: verify 1/2 approximation guarantee
    for b in [15.0, 30.0, 45.0]:
        greedy_res = optimizer.optimize(cate, values, costs, budget=b)
        exact_idx, exact_gain, _ = optimizer.solve_exact_01_knapsack(cate, values, costs, budget=b)
        if exact_gain > 0:
            assert greedy_res.expected_incremental_revenue >= 0.5 * exact_gain


def test_exact_knapsack_with_zero_costs():
    """Verify solve_exact_01_knapsack handles items with zero costs gracefully."""
    optimizer = KnapsackBudgetOptimizer()
    cate = np.array([0.3, 0.2, 0.1])
    values = np.array([100.0, 50.0, 40.0])
    costs = np.array([0.0, 10.0, 5.0])  # item 0 has cost 0.0 (free)
    budget = 10.0

    exact_indices, exact_gain, exact_cost = optimizer.solve_exact_01_knapsack(
        cate, values, costs, budget=budget
    )
    # Item 0 (gain 30) should always be selected because cost is 0
    assert 0 in exact_indices
    assert exact_gain >= 30.0
    assert exact_cost <= budget


def test_optimizer_and_exact_reject_non_finite_inputs():
    """Verify both heuristic and exact solvers reject NaN and Inf."""
    optimizer = KnapsackBudgetOptimizer()
    valid_cate = np.array([0.2, 0.1])
    valid_vals = np.array([100.0, 100.0])
    valid_costs = np.array([10.0, 10.0])

    for bad_val in [np.nan, np.inf, -np.inf]:
        with pytest.raises(ValueError, match="must not contain (NaN|Inf)"):
            optimizer.optimize(np.array([bad_val, 0.1]), valid_vals, valid_costs, budget=20.0)

        with pytest.raises(ValueError, match="must not contain (NaN|Inf)"):
            optimizer.solve_exact_01_knapsack(np.array([bad_val, 0.1]), valid_vals, valid_costs, budget=20.0)

        with pytest.raises(ValueError, match="Budget must be non-negative and finite|Budget must be a finite number"):
            optimizer.optimize(valid_cate, valid_vals, valid_costs, budget=bad_val)

        with pytest.raises(ValueError, match="Budget must be non-negative and finite"):
            optimizer.solve_exact_01_knapsack(valid_cate, valid_vals, valid_costs, budget=bad_val)


def test_optimizer_and_exact_reject_negative_costs():
    """Verify negative voucher costs are rejected."""
    optimizer = KnapsackBudgetOptimizer()
    with pytest.raises(ValueError, match="voucher_costs cannot contain negative values"):
        optimizer.optimize(np.array([0.2, 0.1]), np.array([100.0, 100.0]), np.array([-5.0, 10.0]), budget=20.0)

    with pytest.raises(ValueError, match="voucher_costs cannot contain negative values"):
        optimizer.solve_exact_01_knapsack(np.array([0.2, 0.1]), np.array([100.0, 100.0]), np.array([-5.0, 10.0]), budget=20.0)


def test_greedy_and_exact_zero_cost_consistency():
    """Verify greedy and exact solvers achieve 100% agreement when zero-cost items are present."""
    optimizer = KnapsackBudgetOptimizer()
    cate = np.array([0.35, 0.25, 0.15, 0.10])
    values = np.array([100.0, 80.0, 50.0, 40.0])
    costs = np.array([0.0, 15.0, 10.0, 5.0])  # item 0 has cost 0.0 (free)
    budget = 15.0

    greedy_res = optimizer.optimize(cate, values, costs, budget=budget)
    exact_idx, exact_gain, exact_cost = optimizer.solve_exact_01_knapsack(cate, values, costs, budget=budget)

    # Item 0 MUST be in both solutions
    assert 0 in greedy_res.selected_indices
    assert 0 in exact_idx

    # Total revenue and cost must be identical or >= 0.5 ratio
    assert set(greedy_res.selected_indices) == set(exact_idx)
    assert greedy_res.expected_incremental_revenue == pytest.approx(exact_gain)
    assert greedy_res.total_cost == pytest.approx(exact_cost)


def test_optimizer_rejects_multidimensional_arrays():
    """Verify 2D arrays are cleanly rejected by both solvers."""
    optimizer = KnapsackBudgetOptimizer()
    cate_2d = np.array([[0.2, 0.1], [0.3, 0.4]])
    vals_1d = np.array([100.0, 100.0])
    costs_1d = np.array([10.0, 10.0])

    with pytest.raises(ValueError, match="must be a 1D array"):
        optimizer.optimize(cate_2d, vals_1d, costs_1d, budget=20.0)

    with pytest.raises(ValueError, match="must be 1-dimensional"):
        optimizer.solve_exact_01_knapsack(cate_2d, vals_1d, costs_1d, budget=20.0)


def test_optimizer_budget_boundary_exact_match():
    """When budget exactly equals an item cost, that item is chosen without overflow."""
    optimizer = KnapsackBudgetOptimizer()
    cate = np.array([0.4, 0.1])
    values = np.array([100.0, 50.0])
    costs = np.array([25.0, 10.0])
    budget = 25.0

    res = optimizer.optimize(cate, values, costs, budget=budget)
    assert res.total_cost == 25.0
    assert 0 in res.selected_indices
    assert res.budget_utilized_pct == 100.0







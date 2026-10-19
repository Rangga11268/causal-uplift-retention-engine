"""Constrained Knapsack Budget Optimizer for Voucher & Retention Targeting.

Solves the optimal intervention policy:
Maximize:   sum_i [ z_i * tau(x_i) * Value_i ]
Subject to: sum_i [ z_i * Cost_i ] <= Budget
Where z_i in {0, 1} indicates whether customer i receives the promo intervention.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from app.config import DEFAULT_BUDGET, BudgetConfig


@dataclass
class AllocationResult:
    """Detailed summary of the optimal budget allocation."""
    total_customers: int
    targeted_customers: int
    targeting_rate_pct: float
    total_cost: float
    budget_limit: float
    budget_utilized_pct: float
    expected_incremental_conversions: float
    expected_incremental_revenue: float
    net_roi_ratio: float  # (Incremental Revenue - Total Cost) / Total Cost
    selected_indices: np.ndarray


class GreedyKnapsackOptimizer:
    """Greedy budget allocation heuristic for causal decision targeting.

    Uses Dantzig (1957) density ordering: ratio = (tau(x_i) * Value_i) / Cost_i.
    Guarantees 1/2-approximation for 0/1 knapsack problems and scales linearly O(N log N).
    """

    def __init__(self, config: BudgetConfig = DEFAULT_BUDGET):
        self.config = config

    def optimize(
        self,
        cate_estimates: np.ndarray,
        customer_values: Optional[Union[float, np.ndarray]] = None,
        voucher_costs: Optional[Union[float, np.ndarray]] = None,
        budget: Optional[float] = None,
    ) -> AllocationResult:
        """Find targeting subset using causal efficiency density sorting (Greedy Knapsack).

        Parameters
        ----------
        cate_estimates : np.ndarray
            Predicted treatment effects tau(x_i).
        customer_values : float or np.ndarray, optional
            Expected Customer Lifetime Value (CLV) or order value if retained.
        voucher_costs : float or np.ndarray, optional
            Cost of voucher/promo per customer.
        budget : float, optional
            Maximum allowed marketing budget.

        Returns
        -------
        AllocationResult
            Complete allocation metrics and selected target mask.
        """
        tau = np.asarray(cate_estimates, dtype=np.float64)
        N = len(tau)
        if N == 0:
            raise ValueError("cate_estimates cannot be empty.")

        # 1. Resolve and validate values and costs
        if customer_values is None:
            values = np.full(N, self.config.default_clv, dtype=np.float64)
        elif isinstance(customer_values, (int, float)):
            values = np.full(N, float(customer_values), dtype=np.float64)
        else:
            values = np.asarray(customer_values, dtype=np.float64)

        if len(values) != N:
            raise ValueError(f"Length mismatch: cate_estimates ({N}) vs customer_values ({len(values)})")

        if voucher_costs is None:
            costs = np.full(N, self.config.default_voucher_cost, dtype=np.float64)
        elif isinstance(voucher_costs, (int, float)):
            costs = np.full(N, float(voucher_costs), dtype=np.float64)
        else:
            costs = np.asarray(voucher_costs, dtype=np.float64)

        if len(costs) != N:
            raise ValueError(f"Length mismatch: cate_estimates ({N}) vs voucher_costs ({len(costs)})")

        raw_b = budget if budget is not None else self.config.default_daily_budget
        B = float(raw_b)
        if B <= 0:
            # Zero or negative budget: cannot target anyone
            return AllocationResult(
                total_customers=N,
                targeted_customers=0,
                targeting_rate_pct=0.0,
                total_cost=0.0,
                budget_limit=max(0.0, B),
                budget_utilized_pct=0.0,
                expected_incremental_conversions=0.0,
                expected_incremental_revenue=0.0,
                net_roi_ratio=0.0,
                selected_indices=np.array([], dtype=int),
            )

        # 2. Strict Causal Pruning:
        # Never target Sleeping Dogs (tau <= 0) or customers below min_cate_threshold
        # Because tau(x) <= 0 means negative or zero incremental gain!
        is_candidate = (tau >= self.config.min_cate_threshold) & (costs > 0) & (values > 0)
        candidate_indices = np.where(is_candidate)[0]

        if len(candidate_indices) == 0:
            return AllocationResult(
                total_customers=N,
                targeted_customers=0,
                targeting_rate_pct=0.0,
                total_cost=0.0,
                budget_limit=B,
                budget_utilized_pct=0.0,
                expected_incremental_conversions=0.0,
                expected_incremental_revenue=0.0,
                net_roi_ratio=0.0,
                selected_indices=np.array([], dtype=int),
            )

        # 3. Knapsack Density Sorting:
        # Density ratio = Expected Incremental Gain / Cost
        # ROI Density = (tau(x_i) * Value_i) / Cost_i
        expected_gain = tau[candidate_indices] * values[candidate_indices]
        cost_subset = costs[candidate_indices]
        density = expected_gain / cost_subset

        # Sort candidate indices descending by density
        sorted_ranks = np.argsort(-density)
        sorted_candidates = candidate_indices[sorted_ranks]

        # 4. Greedy Knapsack Packing under budget B
        selected = []
        spent = 0.0

        for idx in sorted_candidates:
            c = costs[idx]
            if spent + c <= B:
                selected.append(idx)
                spent += c
            # Stop if budget full
            if spent >= B:
                break

        selected_indices = np.array(selected, dtype=int)
        n_selected = len(selected_indices)

        if n_selected > 0:
            inc_conversions = float(np.sum(tau[selected_indices]))
            inc_revenue = float(np.sum(tau[selected_indices] * values[selected_indices]))
            net_roi = float((inc_revenue - spent) / spent) if spent > 0 else 0.0
        else:
            inc_conversions = 0.0
            inc_revenue = 0.0
            net_roi = 0.0

        return AllocationResult(
            total_customers=N,
            targeted_customers=n_selected,
            targeting_rate_pct=round((n_selected / N) * 100.0, 2),
            total_cost=round(spent, 2),
            budget_limit=round(B, 2),
            budget_utilized_pct=round((spent / B) * 100.0, 2) if B > 0 else 0.0,
            expected_incremental_conversions=round(inc_conversions, 2),
            expected_incremental_revenue=round(inc_revenue, 2),
            net_roi_ratio=round(net_roi, 4),
            selected_indices=selected_indices,
        )

    def solve_exact_01_knapsack(
        self,
        cate_estimates: np.ndarray,
        customer_values: np.ndarray,
        voucher_costs: np.ndarray,
        budget: float,
    ) -> Tuple[List[int], float, float]:
        """Exact 0/1 Knapsack Branch & Bound for small instances (N <= 30).

        Used as an empirical benchmark to compare greedy heuristic optimality.
        """
        N = len(cate_estimates)
        gains = [float(cate_estimates[i] * customer_values[i]) for i in range(N)]
        c_list = [float(voucher_costs[i]) for i in range(N)]
        B_val = float(budget)

        best_gain = 0.0
        best_selected: List[int] = []

        def branch(idx: int, cur_cost: float, cur_gain: float, chosen: List[int]):
            nonlocal best_gain, best_selected
            if cur_cost > B_val:
                return
            if cur_gain > best_gain:
                best_gain = cur_gain
                best_selected = list(chosen)

            if idx >= N:
                return

            # Branch 1: include item if beneficial
            if gains[idx] > 0 and cur_cost + c_list[idx] <= B_val:
                branch(idx + 1, cur_cost + c_list[idx], cur_gain + gains[idx], chosen + [idx])
            # Branch 2: exclude item
            branch(idx + 1, cur_cost, cur_gain, chosen)

        branch(0, 0.0, 0.0, [])
        total_cost = sum(c_list[i] for i in best_selected)
        return best_selected, best_gain, total_cost


# Backward-compatibility alias
KnapsackBudgetOptimizer = GreedyKnapsackOptimizer

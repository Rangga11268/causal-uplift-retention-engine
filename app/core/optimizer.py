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

        if np.isnan(tau).any() or np.isnan(values).any() or np.isnan(costs).any():
            raise ValueError("Inputs to optimize must not contain NaN")
        if np.isinf(tau).any() or np.isinf(values).any() or np.isinf(costs).any():
            raise ValueError("Inputs to optimize must not contain Inf")

        raw_b = budget if budget is not None else self.config.default_daily_budget
        B = float(raw_b)
        if np.isnan(B) or np.isinf(B):
            raise ValueError(f"Budget must be a finite number, got {raw_b}")
        if B < 0:
            raise ValueError(f"Budget must be non-negative, got {raw_b}")
        if B == 0:
            # Zero budget: cannot target anyone
            return AllocationResult(
                total_customers=N,
                targeted_customers=0,
                targeting_rate_pct=0.0,
                total_cost=0.0,
                budget_limit=0.0,
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

        # 4. Modified Greedy Knapsack Packing with strict 1/2-approximation guarantee
        # (Standard Dantzig greedy density packing + single best item comparison)
        selected_greedy = []
        spent_greedy = 0.0

        for idx in sorted_candidates:
            c = costs[idx]
            if spent_greedy + c <= B:
                selected_greedy.append(idx)
                spent_greedy += c
            if spent_greedy >= B:
                break

        gain_greedy = float(np.sum(tau[selected_greedy] * values[selected_greedy])) if selected_greedy else 0.0

        # Find best single item that fits within budget
        best_single_idx = None
        best_single_gain = 0.0
        for idx in candidate_indices:
            if costs[idx] <= B:
                g = float(tau[idx] * values[idx])
                if g > best_single_gain:
                    best_single_gain = g
                    best_single_idx = idx

        # Modified Greedy guarantee: max(Greedy, best_single) >= 0.5 * OPT
        if best_single_idx is not None and best_single_gain > gain_greedy:
            selected = [best_single_idx]
            spent = float(costs[best_single_idx])
        else:
            selected = selected_greedy
            spent = spent_greedy

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
        """Exact 0/1 Knapsack Branch & Bound with Fractional Relaxation Pruning (N <= 30).

        Used as an empirical benchmark to compare greedy heuristic optimality.
        Uses Dantzig Linear Relaxation (Fractional Knapsack) as an admissible upper bound
        to aggressively prune subtrees, eliminating worst-case 2^N exponential explosion.
        """
        N = len(cate_estimates)
        if N > 30:
            raise ValueError(f"solve_exact_01_knapsack is limited to N <= 30 to prevent combinatorial explosion (got N={N})")
        B_val = float(budget)
        if np.isnan(B_val) or np.isinf(B_val) or B_val < 0:
            raise ValueError(f"Budget must be non-negative and finite, got {budget}")

        # Filter beneficial items that fit into budget
        raw_items = []
        for i in range(N):
            g = float(cate_estimates[i] * customer_values[i])
            c = float(voucher_costs[i])
            if g > 0 and c <= B_val:
                raw_items.append((i, g, c, g / c if c > 0 else float("inf")))

        if not raw_items or B_val <= 0:
            return [], 0.0, 0.0

        # Sort items descending by density for fast upper-bounding and early high-gain discovery
        raw_items.sort(key=lambda x: x[3], reverse=True)
        M = len(raw_items)

        orig_indices = [item[0] for item in raw_items]
        gains = [item[1] for item in raw_items]
        costs = [item[2] for item in raw_items]

        best_gain = 0.0
        best_chosen_mask: List[int] = []

        def get_fractional_upper_bound(start_idx: int, rem_capacity: float) -> float:
            """Admissible upper bound via fractional knapsack relaxation."""
            ub = 0.0
            cap = rem_capacity
            for j in range(start_idx, M):
                if costs[j] <= cap:
                    ub += gains[j]
                    cap -= costs[j]
                else:
                    ub += cap * (gains[j] / costs[j])
                    break
            return ub

        def branch(idx: int, cur_cost: float, cur_gain: float, chosen: List[int]):
            nonlocal best_gain, best_chosen_mask

            if cur_gain > best_gain:
                best_gain = cur_gain
                best_chosen_mask = list(chosen)

            if idx >= M:
                return

            rem_cap = B_val - cur_cost
            # Prune subtree if fractional relaxation upper bound cannot beat best_gain
            if cur_gain + get_fractional_upper_bound(idx, rem_cap) <= best_gain:
                return

            # Branch 1: include item idx (if it fits)
            if costs[idx] <= rem_cap:
                branch(idx + 1, cur_cost + costs[idx], cur_gain + gains[idx], chosen + [orig_indices[idx]])

            # Branch 2: exclude item idx
            branch(idx + 1, cur_cost, cur_gain, chosen)

        branch(0, 0.0, 0.0, [])
        total_cost = sum(float(voucher_costs[i]) for i in best_chosen_mask)
        return best_chosen_mask, best_gain, total_cost



# Backward-compatibility alias
KnapsackBudgetOptimizer = GreedyKnapsackOptimizer

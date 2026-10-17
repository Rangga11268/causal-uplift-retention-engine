"""FastAPI endpoints for Causal Uplift & Retention Decision Engine."""

import time
from typing import Dict, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd

from app.api.schemas import (
    CATEBatchRequest,
    CATEBatchResponse,
    CATECustomerResponse,
    CustomerFeatures,
    HealthResponse,
    OptimizationRequest,
    OptimizationResponse,
)
from app.config import DEFAULT_BUDGET, DEFAULT_META_LEARNER
from app.core.meta_learners import DoublyRobustLearner, SLearner, TLearner, XLearner
from app.core.optimizer import KnapsackBudgetOptimizer
from app.data.generator import generate_customer_retention_data

app = FastAPI(
    title="Causal Uplift & Retention Engine API",
    description="CATE Meta-Learners (X-Learner, DR-Learner) & Knapsack Budget Allocation",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global models cache
_MODELS = {}
_OPTIMIZER = KnapsackBudgetOptimizer(config=DEFAULT_BUDGET)


def _initialize_models():
    """Train models on synthetic observational retention data for instant serving."""
    if not _MODELS:
        X_df, T, Y, _, _ = generate_customer_retention_data(n_samples=5000, random_state=42)
        xl = XLearner()
        xl.fit(X_df, T, Y)
        _MODELS["xlearner"] = xl

        tl = TLearner()
        tl.fit(X_df, T, Y)
        _MODELS["tlearner"] = tl

        drl = DoublyRobustLearner()
        drl.fit(X_df, T, Y)
        _MODELS["drlearner"] = drl


# Initialize on startup
_initialize_models()


def _classify_archetype(cate: float, base_recency: float) -> str:
    """Classify into 4 causal archetypes."""
    if cate >= 0.08:
        return "Persuadable"
    elif cate <= -0.02:
        return "Sleeping Dog"
    elif base_recency <= 15.0:
        return "Sure Thing"
    else:
        return "Lost Cause"


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """System health and ready status."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        default_learner="X-Learner (Künzel et al. 2019)",
        models_ready=len(_MODELS) > 0,
    )


@app.post("/v1/estimate-cate", response_model=CATEBatchResponse)
async def estimate_cate(request: CATEBatchRequest):
    """Estimate CATE uplift tau(x) for a batch of customers."""
    t0 = time.perf_counter()
    l_type = request.learner_type.lower()
    if l_type not in _MODELS:
        raise HTTPException(status_code=400, detail=f"Learner '{request.learner_type}' not available. Use 'xlearner', 'drlearner', or 'tlearner'.")

    model = _MODELS[l_type]

    # Convert Pydantic batch to DataFrame
    df = pd.DataFrame([
        {
            "tenure_months": c.tenure_months,
            "recency_days": c.recency_days,
            "hist_orders": c.hist_orders,
            "avg_order_value": c.avg_order_value,
            "app_engagement": c.app_engagement,
            "discount_sensitivity": c.discount_sensitivity,
        }
        for c in request.customers
    ])

    cate_preds = model.predict_cate(df)

    results = []
    for c, cate in zip(request.customers, cate_preds):
        cate_float = float(round(cate, 4))
        archetype = _classify_archetype(cate_float, c.recency_days)
        # Recommend targeting only if positive uplift
        rec = cate_float >= DEFAULT_BUDGET.min_cate_threshold
        results.append(CATECustomerResponse(
            customer_id=c.customer_id,
            predicted_cate=cate_float,
            archetype=archetype,
            targeting_recommendation=rec,
        ))

    latency = (time.perf_counter() - t0) * 1000.0

    return CATEBatchResponse(
        learner_used=l_type,
        total_customers=len(results),
        mean_predicted_cate=float(round(np.mean(cate_preds), 4)),
        results=results,
        latency_ms=float(round(latency, 2)),
    )


@app.post("/v1/optimize-allocation", response_model=OptimizationResponse)
async def optimize_budget_allocation(request: OptimizationRequest):
    """Solve constrained knapsack promo allocation given budget limit."""
    t0 = time.perf_counter()
    l_type = request.learner_type.lower()
    if l_type not in _MODELS:
        raise HTTPException(status_code=400, detail=f"Learner '{request.learner_type}' not found.")

    model = _MODELS[l_type]

    # Extract features, values, and costs
    rows = []
    values = []
    costs = []
    ids = []

    for c in request.customers:
        ids.append(c.customer_id)
        rows.append({
            "tenure_months": c.tenure_months,
            "recency_days": c.recency_days,
            "hist_orders": c.hist_orders,
            "avg_order_value": c.avg_order_value,
            "app_engagement": c.app_engagement,
            "discount_sensitivity": c.discount_sensitivity,
        })
        values.append(c.customer_clv if c.customer_clv is not None else request.default_clv)
        costs.append(c.voucher_cost if c.voucher_cost is not None else request.default_voucher_cost)

    df = pd.DataFrame(rows)
    cate_preds = model.predict_cate(df)

    alloc_res = _OPTIMIZER.optimize(
        cate_estimates=cate_preds,
        customer_values=np.array(values),
        voucher_costs=np.array(costs),
        budget=request.budget,
    )

    selected_ids = [ids[i] for i in alloc_res.selected_indices]
    latency = (time.perf_counter() - t0) * 1000.0

    return OptimizationResponse(
        total_candidates=alloc_res.total_customers,
        targeted_count=alloc_res.targeted_customers,
        targeting_rate_pct=alloc_res.targeting_rate_pct,
        total_budget_spent=alloc_res.total_cost,
        budget_limit=alloc_res.budget_limit,
        budget_utilization_pct=alloc_res.budget_utilized_pct,
        expected_incremental_conversions=alloc_res.expected_incremental_conversions,
        expected_incremental_revenue=alloc_res.expected_incremental_revenue,
        net_roi_ratio=alloc_res.net_roi_ratio,
        selected_customer_ids=selected_ids,
        latency_ms=float(round(latency, 2)),
    )

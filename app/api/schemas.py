"""Pydantic schemas for Causal Uplift & Retention REST API."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    customer_id: str = Field(..., description="Unique customer identifier")
    tenure_months: float = Field(..., description="Tenure with platform in months")
    recency_days: float = Field(..., description="Days since last purchase")
    hist_orders: int = Field(..., description="Total orders past 12 months")
    avg_order_value: float = Field(..., description="Average order value in currency units")
    app_engagement: float = Field(..., description="App engagement index [0.0, 1.0]")
    discount_sensitivity: float = Field(..., description="Discount sensitivity score [0.0, 1.0]")
    customer_clv: Optional[float] = Field(None, description="Optional custom CLV/Order value")
    voucher_cost: Optional[float] = Field(None, description="Optional specific voucher cost")


class CATEBatchRequest(BaseModel):
    customers: List[CustomerFeatures] = Field(..., description="Batch of customers to score")
    learner_type: str = Field("xlearner", description="'xlearner', 'drlearner', 'tlearner', or 'slearner'")


class CATECustomerResponse(BaseModel):
    customer_id: str
    predicted_cate: float
    archetype: str  # Persuadable, Sure Thing, Lost Cause, or Sleeping Dog
    targeting_recommendation: bool


class CATEBatchResponse(BaseModel):
    learner_used: str
    total_customers: int
    mean_predicted_cate: float
    results: List[CATECustomerResponse]
    latency_ms: float


class OptimizationRequest(BaseModel):
    customers: List[CustomerFeatures] = Field(..., description="Pool of candidates for promotion")
    budget: float = Field(5000.0, description="Available marketing budget")
    default_voucher_cost: float = Field(10.0, description="Default cost per voucher if not specified")
    default_clv: float = Field(85.0, description="Default customer lifetime value if not specified")
    learner_type: str = Field("xlearner", description="Learner to use for CATE estimation")


class OptimizationResponse(BaseModel):
    total_candidates: int
    targeted_count: int
    targeting_rate_pct: float
    total_budget_spent: float
    budget_limit: float
    budget_utilization_pct: float
    expected_incremental_conversions: float
    expected_incremental_revenue: float
    net_roi_ratio: float
    selected_customer_ids: List[str]
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    version: str
    default_learner: str
    models_ready: bool

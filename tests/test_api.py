import pytest
from fastapi.testclient import TestClient

from app.api.routes import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["models_ready"] is True
    assert "version" in data


def test_estimate_cate_endpoint():
    payload = {
        "learner_type": "xlearner",
        "customers": [
            {
                "customer_id": "cust_1",
                "tenure_months": 12.0,
                "recency_days": 5.0,
                "hist_orders": 8,
                "avg_order_value": 45.0,
                "app_engagement": 0.8,
                "discount_sensitivity": 0.9,
            },
            {
                "customer_id": "cust_2",
                "tenure_months": 45.0,
                "recency_days": 80.0,
                "hist_orders": 1,
                "avg_order_value": 20.0,
                "app_engagement": 0.1,
                "discount_sensitivity": 0.1,
            },
        ],
    }
    response = client.post("/v1/estimate-cate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_customers"] == 2
    assert "results" in data
    assert len(data["results"]) == 2
    assert "predicted_cate" in data["results"][0]
    assert "archetype" in data["results"][0]
    assert "targeting_recommendation" in data["results"][0]


def test_optimize_allocation_endpoint():
    payload = {
        "budget": 50.0,
        "default_voucher_cost": 10.0,
        "default_clv": 85.0,
        "learner_type": "xlearner",
        "customers": [
            {
                "customer_id": f"c_{i}",
                "tenure_months": float(i * 3 + 2),
                "recency_days": float(i * 5),
                "hist_orders": i + 1,
                "avg_order_value": 40.0,
                "app_engagement": 0.5,
                "discount_sensitivity": 0.8 if i < 3 else 0.1,
            }
            for i in range(10)
        ],
    }
    response = client.post("/v1/optimize-allocation", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_candidates"] == 10
    assert data["total_budget_spent"] <= 50.0
    assert "selected_customer_ids" in data
    assert "net_roi_ratio" in data
    assert "latency_ms" in data


import pandas as pd
import pytest

from inventory import (
    get_mock_forecasts,
    inventory_plan,
    compute_stockout_risks,
    recommend_promotions,
    simulate_inventory_scenario,
    compute_inventory_kpis,
)


# ---------------------------------------------------------
# Sample data used for testing
# ---------------------------------------------------------

@pytest.fixture
def historical_data():
    return pd.DataFrame({
        "date": pd.to_datetime([
            "2026-01-01",
            "2026-01-02",
            "2026-01-03",
            "2026-01-04",
            "2026-01-05",
            "2026-01-06",
            "2026-01-07",
        ]),
        "store_id": ["S001"] * 7,
        "product_id": ["P001"] * 7,
        "category": ["Electronics"] * 7,
        "price_usd": [100.0] * 7,
        "promo_flag": [0, 0, 1, 1, 0, 0, 1],
        "holiday_flag": [0] * 7,
        "is_weekend": [0, 0, 1, 1, 0, 0, 1],
        "units_sold": [20, 22, 18, 25, 21, 23, 24],
        "stock": [30, 28, 25, 30, 29, 27, 20],
        "stockout": [0, 0, 0, 0, 0, 0, 1],
    })


@pytest.fixture
def current_stock():
    return pd.DataFrame({
        "store_id": ["S001"],
        "product_id": ["P001"],
        "stock": [20],
        "price_usd": [100.0],
    })


@pytest.fixture
def forecast_data():
    return pd.DataFrame({
        "date": pd.date_range("2026-01-08", periods=14),
        "store_id": ["S001"] * 14,
        "product_id": ["P001"] * 14,
        "expected_demand": [22.0] * 14,
        "demand_lead_time_sigma": [5.0] * 14,
    })


# ---------------------------------------------------------
# Test 1: Mock Forecasts
# ---------------------------------------------------------

def test_get_mock_forecasts(historical_data):

    result = get_mock_forecasts(
        historical_data,
        horizon_days=14
    )

    assert isinstance(result, pd.DataFrame)

    assert len(result) == 14

    assert "date" in result.columns
    assert "store_id" in result.columns
    assert "product_id" in result.columns
    assert "expected_demand" in result.columns
    assert "demand_lead_time_sigma" in result.columns

    assert (result["expected_demand"] >= 0).all()


# ---------------------------------------------------------
# Test 2: Inventory Plan
# ---------------------------------------------------------

def test_inventory_plan(current_stock, forecast_data):

    result = inventory_plan(
        current_stock,
        forecast_data,
        lead_time_days=7,
        service_level=0.95
    )

    assert isinstance(result, pd.DataFrame)

    assert len(result) == 1

    assert "store_id" in result.columns
    assert "product_id" in result.columns
    assert "safety_stock" in result.columns
    assert "reorder_point" in result.columns
    assert "average_daily_demand" in result.columns
    assert "suggested_order_quantity" in result.columns

    assert result["safety_stock"].iloc[0] >= 0
    assert result["reorder_point"].iloc[0] >= 0
    assert result["suggested_order_quantity"].iloc[0] >= 0


# ---------------------------------------------------------
# Test 3: Stockout Risk
# ---------------------------------------------------------

def test_compute_stockout_risks(current_stock, forecast_data):

    result = compute_stockout_risks(
        current_stock,
        forecast_data,
        risk_horizon_days=14
    )

    assert isinstance(result, pd.DataFrame)

    assert "store_id" in result.columns
    assert "product_id" in result.columns
    assert "stockout_date" in result.columns
    assert "days_of_cover" in result.columns
    assert "risk_score" in result.columns

    assert (result["risk_score"] >= 0).all()
    assert (result["risk_score"] <= 100).all()


# ---------------------------------------------------------
# Test 4: Promotion Recommendations
# ---------------------------------------------------------

def test_recommend_promotions(historical_data, forecast_data):

    result = recommend_promotions(
        historical_data,
        forecast_data
    )

    assert isinstance(result, pd.DataFrame)

    assert "store_id" in result.columns
    assert "product_id" in result.columns
    assert "incremental_units" in result.columns
    assert "incremental_revenue" in result.columns

    assert len(result) <= 5


# ---------------------------------------------------------
# Test 5: Inventory Scenario Simulation
# ---------------------------------------------------------

def test_simulate_inventory_scenario(current_stock, forecast_data):

    base_plan, new_plan, comparison = simulate_inventory_scenario(
        current_stock,
        forecast_data,
        base_lead_time=7,
        new_lead_time=10,
        service_level=0.95
    )

    assert isinstance(base_plan, pd.DataFrame)
    assert isinstance(new_plan, pd.DataFrame)
    assert isinstance(comparison, dict)

    assert "base_lead_time_days" in comparison
    assert "new_lead_time_days" in comparison
    assert "service_level" in comparison


# ---------------------------------------------------------
# Test 6: Inventory KPIs
# ---------------------------------------------------------

def test_compute_inventory_kpis(historical_data):

    result = compute_inventory_kpis(
        historical_data
    )

    assert isinstance(result, dict)

    assert "stockout_rate_percent" in result
    assert "estimated_lost_sales_units" in result
    assert "inventory_holding_value_usd" in result
    assert "historical_service_level_percent" in result

    assert result["stockout_rate_percent"] >= 0
    assert result["historical_service_level_percent"] >= 0


# ---------------------------------------------------------
# Test 7: Invalid Service Level
# ---------------------------------------------------------

def test_invalid_service_level(current_stock, forecast_data):

    with pytest.raises(ValueError):

        inventory_plan(
            current_stock,
            forecast_data,
            lead_time_days=7,
            service_level=1.5
        )


# ---------------------------------------------------------
# Test 8: Empty Data
# ---------------------------------------------------------

def test_empty_data():

    empty_df = pd.DataFrame()

    with pytest.raises(ValueError):

        get_mock_forecasts(
            empty_df,
            horizon_days=14
        )


# ---------------------------------------------------------
# Test 9: Negative Stock
# ---------------------------------------------------------

def test_negative_stock(forecast_data):

    bad_stock = pd.DataFrame({
        "store_id": ["S001"],
        "product_id": ["P001"],
        "stock": [-10],
    })

    with pytest.raises(ValueError):

        inventory_plan(
            bad_stock,
            forecast_data,
            lead_time_days=7,
            service_level=0.95
        )


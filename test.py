# ============================================
# TEST 7-DAY BASELINE
# ============================================

from forecasting import seven_day_baseline

previous_7_days = [20, 25, 30, 22, 28, 24, 26]

prediction = seven_day_baseline(previous_7_days)

print("7-day baseline:", prediction)


# ============================================
# TEST FUTURE FORECAST
# ============================================

from forecasting import forecast_demand

forecast = forecast_demand(
    "S001",
    "P0001",
    5
)

print("\nFuture demand forecast:")
print(forecast)


# ============================================
# TEST STOCKOUT
# ============================================

from stockout import predict_stockout

result = predict_stockout(
    "S001",
    "P0001",
    forecast,
    100
)

print("\nStockout result:")
print(result)


# ============================================
# TEST METRICS
# ============================================

from metrics import calculate_metrics

actual = [20, 30, 25, 40]
predicted = [22, 28, 27, 35]

mae, rmse, mape = calculate_metrics(
    actual,
    predicted
)

print("\nMetrics:")
print("MAE:", mae)
print("RMSE:", rmse)
print("MAPE:", mape)


# ============================================
# TEST REORDER QUANTITY
# ============================================

from reorder import calculate_reorder_quantity

average_demand = sum(forecast) / len(forecast)

quantity = calculate_reorder_quantity(
    forecast_demand=average_demand,
    current_stock=100,
    lead_time=5,
    safety_stock=20
)

print("\nReorder quantity:")
print(quantity)


# ============================================
# TEST HIGH-RISK SKUS
# ============================================

from stockout import get_high_risk_skus

high_risk = get_high_risk_skus(
    region="North",
    days=14
)

print("\nHIGH-RISK SKUs")
print("================")
print(high_risk)
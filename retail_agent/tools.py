
"""Adapters connecting the orchestrator to teammate modules."""

SAMPLE_MODE = False


def _get_shared_forecast(context):
    """Return the forecast result shared by the Forecast Agent."""
    shared = context.get("forecast_result")

    # New format: result returned by forecast_tool().
    if isinstance(shared, dict):
        daily = shared.get("daily_forecast")
        if daily is not None:
            return [float(x) for x in daily]

    # Backward compatibility with an older context format.
    existing = context.get("forecast_demand")
    if existing is not None:
        if isinstance(existing, (list, tuple)):
            return [float(x) for x in existing]
        return [float(existing)]

    return None


def forecast_tool(question, context):
    from forecasting import forecast_demand

    store_id = context.get("store_id", "S001")
    product_id = context.get("product_id", "P0001")
    days = int(context.get("days", 7))

    forecast = forecast_demand(
        store_id=store_id,
        product_id=product_id,
        days=days,
    )

    daily = [float(x) for x in forecast]

    return {
        "store_id": store_id,
        "product_id": product_id,
        "days": days,
        "daily_forecast": daily,
        "total_forecast_demand": round(sum(daily), 2),
    }


def stockout_tool(question, context):
    from stockout import predict_stockout
    from forecasting import forecast_demand as get_forecast

    store_id = context.get("store_id", "S001")
    product_id = context.get("product_id", "P0001")
    days = int(context.get("days", 7))
    current_stock = float(context.get("current_stock", 100))

    # Reuse the Forecast Agent's result when available.
    forecast = _get_shared_forecast(context)

    # Otherwise, calculate a forecast as a fallback.
    if forecast is None:
        forecast = get_forecast(
            store_id=store_id,
            product_id=product_id,
            days=days,
        )
        forecast = [float(x) for x in forecast]

    return predict_stockout(
        store_id,
        product_id,
        forecast,
        current_stock,
    )


def inventory_tool(question, context):
    from reorder import calculate_reorder_quantity
    from forecasting import forecast_demand as get_forecast

    current_stock = float(context.get("current_stock", 100))
    lead_time = float(context.get("lead_time", 7))
    safety_stock = float(context.get("safety_stock", 20))
    days = int(context.get("days", 7))

    # Reuse the shared forecast if one was passed.
    daily_forecast = _get_shared_forecast(context)

    # Fallback for inventory-only questions.
    if daily_forecast is None:
        store_id = context.get("store_id", "S001")
        product_id = context.get("product_id", "P0001")

        daily_forecast = get_forecast(
            store_id=store_id,
            product_id=product_id,
            days=days,
        )
        daily_forecast = [float(x) for x in daily_forecast]

    if not daily_forecast:
        raise ValueError("Forecast returned no demand values.")

    average_daily_demand = (
        sum(daily_forecast) / len(daily_forecast)
    )

    quantity = calculate_reorder_quantity(
        average_daily_demand,
        current_stock,
        lead_time,
        safety_stock,
    )

    return {
        "current_stock": current_stock,
        "average_daily_demand": round(average_daily_demand, 2),
        "lead_time_days": lead_time,
        "safety_stock": safety_stock,
        "reorder_quantity": round(quantity, 2),
    }


def promo_tool(question, context):
    return {
        "sample": True,
        "recommendation": (
            "Compare promo lift and margin before selecting "
            "a promotion."
        ),
        "note": (
            "Replace with teammate's promotion function "
            "when available."
        ),
    }
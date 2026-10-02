def calculate_reorder_quantity(
    forecast_demand,
    current_stock,
    lead_time,
    safety_stock
):

    lead_time_demand = (
        forecast_demand * lead_time
    )

    reorder_quantity = (
        lead_time_demand
        + safety_stock
        - current_stock
    )

    return max(0, reorder_quantity)
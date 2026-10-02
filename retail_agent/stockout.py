def predict_stockout(
    store_id,
    product_id,
    forecast_demand,
    current_stock
):

    total_demand = sum(forecast_demand)

    if total_demand > current_stock:
        risk = "HIGH"

    elif total_demand >= current_stock * 0.8:
        risk = "MEDIUM"

    else:
        risk = "LOW"

    return {
        "store_id": store_id,
        "product_id": product_id,
        "forecast_demand": total_demand,
        "current_stock": current_stock,
        "risk": risk
    }
def get_high_risk_skus(
    region=None,
    days=14
):

    import pandas as pd

    # Load dataset
    data = pd.read_csv("retail_feature_table.csv")

    # Convert date
    data["date"] = pd.to_datetime(data["date"])

    # Filter region if provided
    if region is not None:
        data = data[
            data["region"] == region
        ]

    # Sort data
    data = data.sort_values(
        ["store_id", "product_id", "date"]
    )

    # Get latest record for each store-product
    latest_data = data.groupby(
        ["store_id", "product_id"]
    ).tail(1).copy()

    # Estimate demand for given number of days
    latest_data["estimated_demand"] = (
        latest_data["rolling_mean_7"] * days
    )

    # Find high-risk SKUs
    high_risk_skus = latest_data[
        latest_data["estimated_demand"] >
        latest_data["stock"]
    ].copy()

    # Select required columns
    high_risk_skus = high_risk_skus[
        [
            "store_id",
            "product_id",
            "category",
            "region",
            "stock",
            "estimated_demand"
        ]
    ]

    return high_risk_skus
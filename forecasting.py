import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


# ============================================
# 1. LOAD DATA
# ============================================

df = pd.read_csv("retail_feature_table.csv")

# Convert date
df["date"] = pd.to_datetime(df["date"])

# Sort data
df = df.sort_values(
    ["store_id", "product_id", "date"]
).reset_index(drop=True)


# ============================================
# 2. 7-DAY BASELINE
# ============================================

def seven_day_baseline(previous_7_days):

    if len(previous_7_days) == 0:
        return 0

    return sum(previous_7_days) / len(previous_7_days)


# ============================================
# 3. SELECT FEATURES
# ============================================

features = [
    "lag_1",
    "lag_7",
    "rolling_mean_7",
    "day_of_week",
    "month",
    "price_change",
    "promo_flag",
    "holiday_flag",
    "is_weekend"
]

target = "units_sold"


# ============================================
# 4. REMOVE MISSING VALUES
# ============================================

df = df.dropna(
    subset=features + [target]
).copy()


print("Rows after removing missing values:", len(df))


# ============================================
# 5. TIME-BASED TRAIN TEST SPLIT
# ============================================

split_date = df["date"].quantile(0.80)

train = df[
    df["date"] < split_date
].copy()

test = df[
    df["date"] >= split_date
].copy()


print("\nTRAINING DATA")
print("Rows:", len(train))
print(
    "Date:",
    train["date"].min(),
    "to",
    train["date"].max()
)

print("\nTESTING DATA")
print("Rows:", len(test))
print(
    "Date:",
    test["date"].min(),
    "to",
    test["date"].max()
)


# ============================================
# 6. PREPARE TRAINING DATA
# ============================================

X_train = train[features]
y_train = train[target]

X_test = test[features]
y_test = test[target]


# ============================================
# 7. CREATE ML MODEL
# ============================================

model = HistGradientBoostingRegressor(
    max_iter=300,
    learning_rate=0.05,
    random_state=42
)


# ============================================
# 8. TRAIN MODEL
# ============================================

model.fit(
    X_train,
    y_train
)


# ============================================
# 9. ML PREDICTIONS
# ============================================

ml_predictions = model.predict(
    X_test
)

# Demand cannot be negative
ml_predictions = np.maximum(
    ml_predictions,
    0
)


# ============================================
# 10. 7-DAY BASELINE PREDICTIONS
# ============================================

baseline_predictions = test[
    "rolling_mean_7"
].values

baseline_predictions = np.maximum(
    baseline_predictions,
    0
)


# ============================================
# 11. METRICS FUNCTION
# ============================================

def calculate_metrics(actual, predicted):

    actual = np.array(actual)
    predicted = np.array(predicted)

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    # Avoid division by zero
    mask = actual != 0

    if mask.sum() > 0:

        mape = np.mean(
            np.abs(
                (actual[mask] - predicted[mask])
                / actual[mask]
            )
        ) * 100

    else:

        mape = 0

    return mae, rmse, mape


# ============================================
# 12. CALCULATE ML METRICS
# ============================================

ml_mae, ml_rmse, ml_mape = calculate_metrics(
    y_test,
    ml_predictions
)


# ============================================
# 13. CALCULATE BASELINE METRICS
# ============================================

baseline_mae, baseline_rmse, baseline_mape = calculate_metrics(
    y_test,
    baseline_predictions
)


# ============================================
# 14. PRINT RESULTS
# ============================================

print("\n============================================")
print("ML MODEL RESULTS")
print("============================================")

print("MAE :", ml_mae)
print("RMSE:", ml_rmse)
print("MAPE:", ml_mape)


print("\n============================================")
print("7-DAY BASELINE RESULTS")
print("============================================")

print("MAE :", baseline_mae)
print("RMSE:", baseline_rmse)
print("MAPE:", baseline_mape)


# ============================================
# 15. MODEL COMPARISON
# ============================================

comparison = pd.DataFrame({

    "Model": [
        "7-day baseline",
        "ML model"
    ],

    "MAE": [
        baseline_mae,
        ml_mae
    ],

    "RMSE": [
        baseline_rmse,
        ml_rmse
    ],

    "MAPE": [
        baseline_mape,
        ml_mape
    ]
})


print("\n============================================")
print("MODEL COMPARISON")
print("============================================")

print(comparison)


# ============================================
# 16. SAVE MODEL COMPARISON
# ============================================

comparison.to_csv(
    "model_comparison.csv",
    index=False
)


# ============================================
# 17. SAVE TRAINED MODEL
# ============================================

joblib.dump(
    model,
    "demand_forecasting_model.pkl"
)

print(
    "\nModel saved as demand_forecasting_model.pkl"
)

print(
    "Comparison saved as model_comparison.csv"
)


# ============================================
# 18. FUTURE DEMAND FORECAST
# ============================================

def forecast_demand(
    store_id,
    product_id,
    days
):

    # Get selected store and product
    product_data = df[
        (df["store_id"] == store_id) &
        (df["product_id"] == product_id)
    ].copy()

    # If store/product does not exist
    if len(product_data) == 0:
        return []

    # Sort by date
    product_data = product_data.sort_values(
        "date"
    )

    # Latest available record
    latest = product_data.iloc[-1]

    # Last 7 actual demand values
    previous_demand = list(
        product_data["units_sold"].tail(7)
    )

    # If no previous demand
    if len(previous_demand) == 0:
        return []

    forecasts = []

    # Forecast one day at a time
    for i in range(days):

        # Future date
        future_date = (
            latest["date"]
            + pd.Timedelta(days=i + 1)
        )

        # Previous day's demand
        lag_1 = previous_demand[-1]

        # Demand from 7 observations ago
        if len(previous_demand) >= 7:

            lag_7 = previous_demand[-7]

        else:

            lag_7 = previous_demand[0]

        # Average of previous 7 demand values
        last_7 = previous_demand[-7:]

        rolling_mean_7 = (
            sum(last_7) / len(last_7)
        )

        # Date features
        day_of_week = future_date.dayofweek

        month = future_date.month

        # Weekend
        if day_of_week >= 5:

            is_weekend = 1

        else:

            is_weekend = 0

        # Latest known price change
        price_change = latest["price_change"]

        if pd.isna(price_change):

            price_change = 0

        # Latest known promotion
        promo_flag = latest["promo_flag"]

        # Latest known holiday
        holiday_flag = latest["holiday_flag"]


        # Create future feature row
        future_features = pd.DataFrame([{

            "lag_1": lag_1,

            "lag_7": lag_7,

            "rolling_mean_7": rolling_mean_7,

            "day_of_week": day_of_week,

            "month": month,

            "price_change": price_change,

            "promo_flag": promo_flag,

            "holiday_flag": holiday_flag,

            "is_weekend": is_weekend

        }])


        # Predict demand
        prediction = model.predict(
            future_features[features]
        )[0]


        # Demand cannot be negative
        prediction = max(
            0,
            prediction
        )


        # Store prediction
        forecasts.append(
            round(prediction, 2)
        )


        # Use prediction for next day's lag
        previous_demand.append(
            prediction
        )


    return forecasts


# ============================================
# 19. TEST FUTURE FORECAST
# ============================================

if __name__ == "__main__":

    forecast = forecast_demand(
        "S001",
        "P0001",
        5
    )

    print("\n============================================")
    print("FUTURE DEMAND FORECAST")
    print("============================================")

    print(forecast)
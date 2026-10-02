import numpy as np
import pandas as pd
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_FORECAST_HORIZON = 14
DEFAULT_LEAD_TIME_DAYS = 7
DEFAULT_SERVICE_LEVEL = 0.95
DEFAULT_TARGET_STOCK_MULTIPLIER = 1.5


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_dataframe(
    dataframe: pd.DataFrame,
    required_columns: list[str],
    dataframe_name: str
) -> None:
    """
    Validate that a DataFrame exists and contains required columns.

    Args:
        dataframe: DataFrame to validate.
        required_columns: Columns required by the calculation.
        dataframe_name: Human-readable DataFrame name for error messages.

    Raises:
        TypeError: If dataframe is not a pandas DataFrame.
        ValueError: If dataframe is empty or required columns are missing.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(f"{dataframe_name} must be a pandas DataFrame.")

    if dataframe.empty:
        raise ValueError(f"{dataframe_name} is empty.")

    missing_columns = [
        column for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{dataframe_name} is missing required columns: "
            f"{missing_columns}"
        )


def _validate_positive_integer(value: int, parameter_name: str) -> None:
    """
    Validate a positive integer parameter.

    Args:
        value: Value to validate.
        parameter_name: Parameter name used in the error message.

    Raises:
        ValueError: If value is not a positive integer.
    """
    if not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(
            f"{parameter_name} must be a positive integer."
        )


def _validate_service_level(service_level: float) -> None:
    """
    Validate an inventory service-level probability.

    Args:
        service_level: Probability between 0 and 1.

    Raises:
        ValueError: If service level is outside the valid range.
    """
    if not isinstance(service_level, (int, float, np.integer, np.floating)):
        raise TypeError("service_level must be numeric.")

    if not 0 < float(service_level) < 1:
        raise ValueError(
            "service_level must be greater than 0 and less than 1."
        )


def _validate_non_negative_stock(dataframe: pd.DataFrame) -> None:
    """
    Validate that inventory quantities are not negative.

    Args:
        dataframe: DataFrame containing a stock column.

    Raises:
        ValueError: If negative stock values are found.
    """
    if "stock" in dataframe.columns:
        numeric_stock = pd.to_numeric(
            dataframe["stock"],
            errors="coerce"
        )

        if numeric_stock.isna().any():
            raise ValueError(
                "The stock column contains missing or non-numeric values."
            )

        if (numeric_stock < 0).any():
            raise ValueError(
                "The stock column contains negative inventory values."
            )


def _prepare_forecast_keys(forecast_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare forecast data and verify SKU-store identifiers.

    Args:
        forecast_df: Forecast DataFrame.

    Returns:
        Cleaned forecast DataFrame.

    Raises:
        ValueError: If required identifier columns are missing.
    """
    _validate_dataframe(
        forecast_df,
        ["store_id", "product_id", "expected_demand"],
        "forecast_df"
    )

    result = forecast_df.copy()

    result["expected_demand"] = pd.to_numeric(
        result["expected_demand"],
        errors="coerce"
    )

    if result["expected_demand"].isna().any():
        raise ValueError(
            "forecast_df contains missing or non-numeric expected_demand."
        )

    # Negative demand is not meaningful.
    result["expected_demand"] = result["expected_demand"].clip(lower=0)

    return result


# ---------------------------------------------------------------------------
# 1. Temporary Mock Forecast
# ---------------------------------------------------------------------------

def get_mock_forecasts(
    actual_data_df: pd.DataFrame,
    horizon_days: int = DEFAULT_FORECAST_HORIZON
) -> pd.DataFrame:
    """
    Generate a temporary demand forecast for testing.

    The function creates daily forecasts for every unique store/product
    combination in the historical dataset.

    A simple historical moving average is used as the baseline. Small,
    deterministic random noise is added to make the forecast look realistic
    while keeping tests reproducible.

    The function produces the important Member 2 interface columns:

        expected_demand
        demand_lead_time_sigma
        date

    It also includes:
        store_id
        product_id

    Args:
        actual_data_df:
            Historical sales DataFrame containing:
            store_id, product_id, date, and units_sold.

        horizon_days:
            Number of future days for which forecasts should be generated.

    Returns:
        A pandas DataFrame containing one forecast row per
        store/product/day.

    Raises:
        ValueError:
            If the input is empty, columns are missing, or horizon is invalid.

    Business logic:
        Baseline demand = rolling 7-day historical average where available.

        Forecast demand =
            baseline demand + small deterministic random noise.

        Demand-lead-time sigma is initially estimated as the historical
        daily standard deviation. The inventory planning function converts
        this to lead-time demand uncertainty when necessary.
    """
    _validate_dataframe(
        actual_data_df,
        ["date", "store_id", "product_id", "units_sold"],
        "actual_data_df"
    )

    _validate_positive_integer(horizon_days, "horizon_days")

    data = actual_data_df.copy()

    data["date"] = pd.to_datetime(data["date"], errors="coerce")

    if data["date"].isna().any():
        raise ValueError("actual_data_df contains invalid dates.")

    data["units_sold"] = pd.to_numeric(
        data["units_sold"],
        errors="coerce"
    )

    if data["units_sold"].isna().any():
        raise ValueError(
            "units_sold contains missing or non-numeric values."
        )

    data["units_sold"] = data["units_sold"].clip(lower=0)

    data = data.sort_values(
        ["store_id", "product_id", "date"]
    )

    forecast_rows = []

    # Fixed seed makes the temporary forecast reproducible.
    random_generator = np.random.default_rng(42)

    latest_date = data["date"].max()

    grouped = data.groupby(
        ["store_id", "product_id"],
        dropna=False
    )

    for (store_id, product_id), group in grouped:
        group = group.sort_values("date")

        # Seven-day moving average.
        rolling_average = (
            group["units_sold"]
            .rolling(window=7, min_periods=1)
            .mean()
            .iloc[-1]
        )

        # Historical daily demand variability.
        historical_sigma = float(group["units_sold"].std(ddof=1))

        if not np.isfinite(historical_sigma):
            historical_sigma = 0.0

        future_dates = pd.date_range(
            start=latest_date + pd.Timedelta(days=1),
            periods=horizon_days,
            freq="D"
        )

        for future_date in future_dates:
            # Small noise around the moving average.
            noise = random_generator.normal(
                loc=0.0,
                scale=max(0.05 * rolling_average, 0.01)
            )

            expected_demand = max(
                0.0,
                float(rolling_average) + float(noise)
            )

            forecast_rows.append(
                {
                    "date": future_date,
                    "store_id": store_id,
                    "product_id": product_id,
                    "expected_demand": expected_demand,
                    "demand_lead_time_sigma": historical_sigma
                }
            )

    forecast_df = pd.DataFrame(forecast_rows)

    return forecast_df


# ---------------------------------------------------------------------------
# 2. Core Inventory Planning
# ---------------------------------------------------------------------------

def inventory_plan(
    current_stock_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    lead_time_days: int = DEFAULT_LEAD_TIME_DAYS,
    service_level: float = DEFAULT_SERVICE_LEVEL
) -> pd.DataFrame:
    """
    Calculate SKU-store inventory replenishment recommendations.

    The calculation uses forecast demand supplied by Member 2 or the
    temporary mock forecasting function.

    Formulas:

        Z = norm.ppf(service_level)

        Safety Stock =
            Z × demand_lead_time_sigma

        Reorder Point =
            (Average Daily Demand × Lead Time) + Safety Stock

        Target Stock =
            Reorder Point × 1.5

        Suggested Order Quantity =
            max(0, Target Stock - Current Inventory Position)

    For this version:
        Incoming stock = 0

    Args:
        current_stock_df:
            Current inventory DataFrame containing:
            store_id, product_id, stock.

            price_usd is optional.

        forecast_df:
            Forecast DataFrame containing:
            store_id, product_id, expected_demand.

            demand_lead_time_sigma is optional. If unavailable, it is
            estimated from forecast demand variability.

        lead_time_days:
            Number of days between placing an order and receiving it.

        service_level:
            Desired probability of not stocking out during lead time.
            Example: 0.95 = 95%.

    Returns:
        DataFrame with one row per SKU-store combination and columns:

            store_id
            product_id
            average_daily_demand
            safety_stock
            reorder_point
            current_stock
            incoming_stock
            inventory_position
            target_stock_level
            suggested_order_quantity
            days_of_cover
            reorder_required
            service_level
            z_score
            recommendation

    Raises:
        ValueError:
            For invalid input data or parameters.

    Business logic:
        Historical demand should not be confused with forecast demand.
        This function uses forecast demand for future inventory planning.
    """
    _validate_dataframe(
        current_stock_df,
        ["store_id", "product_id", "stock"],
        "current_stock_df"
    )

    _validate_service_level(service_level)
    _validate_positive_integer(lead_time_days, "lead_time_days")

    _validate_non_negative_stock(current_stock_df)

    forecasts = _prepare_forecast_keys(forecast_df)

    stock_data = current_stock_df.copy()

    stock_data["stock"] = pd.to_numeric(
        stock_data["stock"],
        errors="coerce"
    )

    if stock_data["stock"].isna().any():
        raise ValueError(
            "current_stock_df contains invalid stock values."
        )

    # If multiple rows exist for a SKU-store, use the latest available
    # inventory record when a date is available.
    if "date" in stock_data.columns:
        stock_data["date"] = pd.to_datetime(
            stock_data["date"],
            errors="coerce"
        )

        stock_data = stock_data.sort_values("date")

        stock_data = (
            stock_data
            .drop_duplicates(
                subset=["store_id", "product_id"],
                keep="last"
            )
        )
    else:
        stock_data = (
            stock_data
            .drop_duplicates(
                subset=["store_id", "product_id"],
                keep="last"
            )
        )

    # Average forecast demand per SKU-store.
    forecast_summary = (
        forecasts
        .groupby(
            ["store_id", "product_id"],
            as_index=False
        )
        .agg(
            average_daily_demand=("expected_demand", "mean")
        )
    )

    # Obtain lead-time sigma.
    if "demand_lead_time_sigma" in forecasts.columns:
        sigma_summary = (
            forecasts
            .groupby(
                ["store_id", "product_id"],
                as_index=False
            )
            .agg(
                demand_lead_time_sigma=(
                    "demand_lead_time_sigma",
                    "first"
                )
            )
        )

        forecast_summary = forecast_summary.merge(
            sigma_summary,
            on=["store_id", "product_id"],
            how="left"
        )
    else:
        sigma_summary = (
            forecasts
            .groupby(
                ["store_id", "product_id"],
                as_index=False
            )["expected_demand"]
            .std()
            .rename(
                columns={"expected_demand": "demand_lead_time_sigma"}
            )
        )

        forecast_summary = forecast_summary.merge(
            sigma_summary,
            on=["store_id", "product_id"],
            how="left"
        )

    # Missing standard deviation means there is insufficient evidence
    # about demand variability. For calculation purposes only, a single
    # forecast observation has sigma = 0.
    forecast_summary["demand_lead_time_sigma"] = (
        forecast_summary["demand_lead_time_sigma"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .clip(lower=0)
    )

    result = stock_data[
        ["store_id", "product_id", "stock"]
    ].merge(
        forecast_summary,
        on=["store_id", "product_id"],
        how="inner"
    )

    if result.empty:
        raise ValueError(
            "No matching store_id/product_id combinations were found "
            "between current_stock_df and forecast_df."
        )

    z_score = float(norm.ppf(service_level))

    result["z_score"] = z_score

    # Safety Stock:
    # z × sigma of demand during lead time.
    result["safety_stock"] = (
        result["z_score"]
        * result["demand_lead_time_sigma"]
    )

    result["reorder_point"] = (
        result["average_daily_demand"] * lead_time_days
        + result["safety_stock"]
    )

    # Current inventory position:
    # current stock + incoming stock.
    # Incoming stock is unavailable in the supplied schema, therefore
    # this implementation explicitly uses the business assumption 0.
    result["incoming_stock"] = 0.0

    result["inventory_position"] = (
        result["stock"] + result["incoming_stock"]
    )

    result["target_stock_level"] = (
        result["reorder_point"]
        * DEFAULT_TARGET_STOCK_MULTIPLIER
    )

    result["suggested_order_quantity"] = np.where(
        result["stock"] < result["reorder_point"],
        np.maximum(
            0,
            result["target_stock_level"]
            - result["inventory_position"]
        ),
        0.0
    )

    # Days of cover based on forecast demand.
    result["days_of_cover"] = np.where(
        result["average_daily_demand"] > 0,
        result["stock"] / result["average_daily_demand"],
        np.nan
    )

    result["reorder_required"] = (
        result["stock"] < result["reorder_point"]
    )

    result["recommendation"] = np.where(
        result["reorder_required"],
        "Reorder recommended: current stock is below reorder point.",
        "No reorder required: current stock is at or above reorder point."
    )

    result = result.rename(
        columns={
            "stock": "current_stock"
        }
    )
    result["service_level"] = service_level

    return result[
        [
            "store_id",
            "product_id",
            "average_daily_demand",
            "demand_lead_time_sigma",
            "safety_stock",
            "reorder_point",
            "current_stock",
            "incoming_stock",
            "inventory_position",
            "target_stock_level",
            "suggested_order_quantity",
            "days_of_cover",
            "reorder_required",
            "service_level",
            "z_score",
            "recommendation"
        ]
    ].sort_values(
        ["reorder_required", "suggested_order_quantity"],
        ascending=[False, False]
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Stockout Risk Analysis
# ---------------------------------------------------------------------------

def compute_stockout_risks(
    current_stock_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    risk_horizon_days: int = DEFAULT_FORECAST_HORIZON
) -> pd.DataFrame:
    """
    Calculate and rank future stockout risks.

    Days of cover is calculated using the cumulative forecast demand for
    each SKU-store combination. Demand is subtracted from available stock
    day by day until inventory reaches zero.

    Risk score is a transparent heuristic based on:
        1. Days-of-cover risk
        2. Historical stockout rate
        3. Revenue at risk

    The score is normalized to 0-100.

    Historical stockout rate is used only when a stockout column exists.

    Revenue at risk is estimated as:
        forecasted units during the risk horizon × price_usd

    This is an exposure estimate, not guaranteed lost revenue.

    Args:
        current_stock_df:
            Inventory DataFrame containing:
            store_id, product_id, stock.

            Optional:
                price_usd
                stockout
                date

        forecast_df:
            Forecast DataFrame containing:
            store_id, product_id, date, expected_demand.

        risk_horizon_days:
            Number of future days to examine.

    Returns:
        DataFrame ranked by descending risk score.

    Raises:
        ValueError:
            If required inputs are invalid.
    """
    _validate_dataframe(
        current_stock_df,
        ["store_id", "product_id", "stock"],
        "current_stock_df"
    )

    _validate_dataframe(
        forecast_df,
        ["store_id", "product_id", "date", "expected_demand"],
        "forecast_df"
    )

    _validate_positive_integer(
        risk_horizon_days,
        "risk_horizon_days"
    )

    _validate_non_negative_stock(current_stock_df)

    stock_data = current_stock_df.copy()
    forecasts = forecast_df.copy()

    forecasts["date"] = pd.to_datetime(
        forecasts["date"],
        errors="coerce"
    )

    if forecasts["date"].isna().any():
        raise ValueError("forecast_df contains invalid dates.")

    forecasts["expected_demand"] = pd.to_numeric(
        forecasts["expected_demand"],
        errors="coerce"
    )

    if forecasts["expected_demand"].isna().any():
        raise ValueError(
            "forecast_df contains invalid expected_demand values."
        )

    forecasts["expected_demand"] = forecasts[
        "expected_demand"
    ].clip(lower=0)

    # Use the first risk_horizon_days available per SKU-store.
    forecasts = (
        forecasts
        .sort_values(["store_id", "product_id", "date"])
        .groupby(
            ["store_id", "product_id"],
            group_keys=False
        )
        .head(risk_horizon_days)
    )

    # Current inventory: use latest record where dates are available.
    if "date" in stock_data.columns:
        stock_data["date"] = pd.to_datetime(
            stock_data["date"],
            errors="coerce"
        )
        stock_data = stock_data.sort_values("date")

    stock_data = stock_data.drop_duplicates(
        subset=["store_id", "product_id"],
        keep="last"
    )

    historical_stockout_rates = {}

    if "stockout" in stock_data.columns:
        stockout_values = pd.to_numeric(
            stock_data["stockout"],
            errors="coerce"
        )

        # If current_stock_df has one row per SKU-store, this is simply
        # the available historical indicator.
        for _, row in stock_data.iterrows():
            key = (row["store_id"], row["product_id"])

            if pd.notna(row["stockout"]):
                historical_stockout_rates[key] = float(
                    np.clip(row["stockout"], 0, 1)
                )

    # If the input includes multiple historical records, calculate actual
    # historical stockout rate from them before deduplication.
    if "stockout" in current_stock_df.columns:
        history = current_stock_df.copy()

        history["stockout"] = pd.to_numeric(
            history["stockout"],
            errors="coerce"
        )

        history = history.dropna(subset=["stockout"])

        if not history.empty:
            history["stockout"] = history["stockout"].clip(0, 1)

            rate_table = (
                history
                .groupby(
                    ["store_id", "product_id"]
                )["stockout"]
                .mean()
            )

            historical_stockout_rates = {
                key: float(value)
                for key, value in rate_table.items()
            }

    # Price lookup.
    price_lookup = {}

    if "price_usd" in stock_data.columns:
        price_values = pd.to_numeric(
            stock_data["price_usd"],
            errors="coerce"
        )

        for index, row in stock_data.iterrows():
            price = price_values.loc[index]

            if pd.notna(price) and price >= 0:
                price_lookup[
                    (row["store_id"], row["product_id"])
                ] = float(price)

    risk_rows = []

    for (store_id, product_id), group in forecasts.groupby(
        ["store_id", "product_id"]
    ):
        stock_matches = stock_data[
            (stock_data["store_id"] == store_id)
            & (stock_data["product_id"] == product_id)
        ]

        if stock_matches.empty:
            continue

        current_stock = float(
            stock_matches.iloc[-1]["stock"]
        )

        remaining_stock = current_stock
        stockout_date = pd.NaT
        cumulative_demand = 0.0

        for _, forecast_row in group.sort_values("date").iterrows():
            daily_demand = float(
                forecast_row["expected_demand"]
            )

            cumulative_demand += daily_demand
            remaining_stock -= daily_demand

            if remaining_stock <= 0 and pd.isna(stockout_date):
                stockout_date = forecast_row["date"]

        total_forecast_demand = cumulative_demand

        average_daily_demand = (
            total_forecast_demand / len(group)
            if len(group) > 0
            else 0.0
        )

        if average_daily_demand > 0:
            days_of_cover = current_stock / average_daily_demand
        else:
            days_of_cover = np.nan

        if pd.notna(stockout_date):
            first_forecast_date = group["date"].min()

            days_until_stockout = (
                stockout_date - first_forecast_date
            ).days

            # A stockout on the first forecast day is treated as day 1.
            days_until_stockout = max(
                1,
                days_until_stockout + 1
            )
        else:
            days_until_stockout = np.nan

        price = price_lookup.get(
            (store_id, product_id),
            np.nan
        )

        if pd.notna(price):
            revenue_at_risk = (
                total_forecast_demand * price
                if pd.notna(stockout_date)
                else 0.0
            )
        else:
            revenue_at_risk = np.nan

        historical_stockout_rate = historical_stockout_rates.get(
            (store_id, product_id),
            np.nan
        )

        # Calculate individual normalized risk components.
        # Days-of-cover component:
        # fewer days of cover = higher risk.
        if pd.isna(days_of_cover):
            cover_risk = 0.0
        else:
            cover_risk = min(
                1.0,
                1.0 / max(days_of_cover, 1.0)
            )

        # Historical stockout component.
        if pd.isna(historical_stockout_rate):
            history_risk = 0.0
            history_available = False
        else:
            history_risk = float(
                np.clip(historical_stockout_rate, 0, 1)
            )
            history_available = True

        # Revenue component.
        if pd.isna(revenue_at_risk):
            revenue_risk = 0.0
            revenue_available = False
        else:
            # Normalize against this SKU's forecast exposure.
            maximum_revenue = (
                total_forecast_demand * price
                if pd.notna(price)
                else 0.0
            )

            if maximum_revenue > 0:
                revenue_risk = (
                    revenue_at_risk / maximum_revenue
                )
            else:
                revenue_risk = 0.0

            revenue_available = True

        # Future stockout component.
        stockout_within_horizon = (
            pd.notna(days_until_stockout)
            and days_until_stockout <= risk_horizon_days
        )

        if stockout_within_horizon:
            urgency_risk = max(
                0.0,
                1.0
                - (
                    days_until_stockout
                    / risk_horizon_days
                )
            )
        else:
            urgency_risk = 0.0

        # Transparent heuristic.
        risk_score = 100.0 * (
            0.40 * urgency_risk
            + 0.25 * cover_risk
            + 0.20 * history_risk
            + 0.15 * revenue_risk
        )

        risk_score = float(
            np.clip(risk_score, 0, 100)
        )

        if risk_score >= 70:
            risk_level = "High"
        elif risk_score >= 40:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        if stockout_within_horizon:
            explanation = (
                f"Projected stockout within {risk_horizon_days} days. "
                f"Estimated days of cover: "
                f"{days_of_cover:.2f}."
            )
        elif pd.isna(days_of_cover):
            explanation = (
                "Forecast demand is zero or unavailable, so days "
                "of cover cannot be estimated reliably."
            )
        else:
            explanation = (
                f"No stockout projected within the {risk_horizon_days}-day "
                f"horizon. Estimated days of cover: "
                f"{days_of_cover:.2f}."
            )

        if not history_available:
            explanation += (
                " Historical stockout rate was unavailable."
            )

        if not revenue_available:
            explanation += (
                " Price data was unavailable, so revenue exposure "
                "could not be calculated."
            )

        risk_rows.append(
            {
                "store_id": store_id,
                "product_id": product_id,
                "current_stock": current_stock,
                "average_daily_demand": average_daily_demand,
                "days_of_cover": days_of_cover,
                "days_until_stockout": days_until_stockout,
                "stockout_date": stockout_date,
                "stockout_within_horizon": stockout_within_horizon,
                "historical_stockout_rate": (
                    historical_stockout_rate
                ),
                "revenue_at_risk": revenue_at_risk,
                "risk_score": risk_score,
                "risk_level": risk_level,
                "risk_explanation": explanation
            }
        )

    if not risk_rows:
        return pd.DataFrame(
            columns=[
                "store_id",
                "product_id",
                "current_stock",
                "average_daily_demand",
                "days_of_cover",
                "days_until_stockout",
                "stockout_date",
                "stockout_within_horizon",
                "historical_stockout_rate",
                "revenue_at_risk",
                "risk_score",
                "risk_level",
                "risk_explanation"
            ]
        )

    result = pd.DataFrame(risk_rows)

    return result.sort_values(
        ["risk_score", "stockout_within_horizon"],
        ascending=[False, False]
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. Promotion Recommendations
# ---------------------------------------------------------------------------

def recommend_promotions(
    historical_data_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    category_lift_map: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """
    Estimate historical promotion lift and recommend products for promotion.

    Promotion lift is calculated by comparing average historical sales during
    promotion and non-promotion periods:

        Lift =
            (Promo Average - Non-Promo Average)
            / Non-Promo Average

    If category_lift_map is provided, its values override the calculated
    historical category lift.

    Future demand is then estimated as:

        Lifted Demand =
            Forecast Demand × (1 + Promotion Lift)

        Incremental Units =
            Lifted Demand - Forecast Demand

        Incremental Revenue =
            Incremental Units × Price

    This function reports an observed historical association. It does not
    claim that promotion caused the observed sales difference.

    Args:
        historical_data_df:
            Historical data containing:
            store_id, product_id, category, promo_flag, units_sold.

            price_usd is recommended.

        forecast_df:
            Future forecast data containing:
            store_id, product_id, expected_demand.

            category and price_usd may also be supplied here.

        category_lift_map:
            Optional manually supplied category lift percentages.
            Example:
                {"Beverages": 0.20, "Snacks": 0.15}

            Values must be >= 0.

    Returns:
        DataFrame containing the top five promotion candidates, ranked by
        estimated incremental revenue.

    Raises:
        ValueError:
            If required data is missing or invalid.
    """
    _validate_dataframe(
        historical_data_df,
        [
            "store_id",
            "product_id",
            "category",
            "promo_flag",
            "units_sold"
        ],
        "historical_data_df"
    )

    _validate_dataframe(
        forecast_df,
        ["store_id", "product_id", "expected_demand"],
        "forecast_df"
    )

    history = historical_data_df.copy()
    forecasts = forecast_df.copy()

    history["promo_flag"] = pd.to_numeric(
        history["promo_flag"],
        errors="coerce"
    )

    history["units_sold"] = pd.to_numeric(
        history["units_sold"],
        errors="coerce"
    )

    if history["promo_flag"].isna().any():
        raise ValueError("promo_flag contains invalid values.")

    if history["units_sold"].isna().any():
        raise ValueError("units_sold contains invalid values.")

    history["units_sold"] = history["units_sold"].clip(lower=0)

    # ------------------------------------------------------------------
    # Calculate category-level historical promotion lift.
    # ------------------------------------------------------------------

    promo_average = (
        history[history["promo_flag"] == 1]
        .groupby("category")["units_sold"]
        .mean()
        .rename("promo_average")
    )

    non_promo_average = (
        history[history["promo_flag"] == 0]
        .groupby("category")["units_sold"]
        .mean()
        .rename("non_promo_average")
    )

    lift_table = pd.concat(
        [promo_average, non_promo_average],
        axis=1
    ).reset_index()

    def calculate_lift(row: pd.Series) -> float:
        """
        Calculate safe percentage lift for one category.
        """
        baseline = row["non_promo_average"]
        promo = row["promo_average"]

        if pd.isna(baseline) or pd.isna(promo):
            return np.nan

        if baseline <= 0:
            return np.nan

        return max(
            0.0,
            float((promo - baseline) / baseline)
        )

    lift_table["promotion_lift"] = lift_table.apply(
        calculate_lift,
        axis=1
    )

    lift_lookup = dict(
        zip(
            lift_table["category"],
            lift_table["promotion_lift"]
        )
    )

    # User-supplied category map overrides historical estimates.
    if category_lift_map is not None:
        for category, lift in category_lift_map.items():
            if not isinstance(
                lift,
                (int, float, np.integer, np.floating)
            ):
                raise TypeError(
                    f"Promotion lift for {category} must be numeric."
                )

            if lift < 0:
                raise ValueError(
                    f"Promotion lift for {category} cannot be negative."
                )

            lift_lookup[category] = float(lift)

    # ------------------------------------------------------------------
    # Prepare future forecast.
    # ------------------------------------------------------------------

    forecasts["expected_demand"] = pd.to_numeric(
        forecasts["expected_demand"],
        errors="coerce"
    )

    if forecasts["expected_demand"].isna().any():
        raise ValueError(
            "forecast_df contains invalid expected_demand values."
        )

    forecasts["expected_demand"] = forecasts[
        "expected_demand"
    ].clip(lower=0)

    # Category lookup from history.
    category_lookup = (
        history
        .drop_duplicates(
            ["store_id", "product_id"]
        )
        [["store_id", "product_id", "category"]]
    )

    forecasts = forecasts.merge(
        category_lookup,
        on=["store_id", "product_id"],
        how="left",
        suffixes=("", "_historical")
    )

    # If category already exists in forecast, retain it.
    if "category_historical" in forecasts.columns:
        if "category" not in forecast_df.columns:
            forecasts["category"] = forecasts[
                "category_historical"
            ]

        forecasts = forecasts.drop(
            columns=["category_historical"]
        )

    # Price lookup from historical data.
    if "price_usd" in history.columns:
        price_history = history.copy()

        price_history["price_usd"] = pd.to_numeric(
            price_history["price_usd"],
            errors="coerce"
        )

        price_lookup = (
            price_history
            .dropna(subset=["price_usd"])
            .groupby(
                ["store_id", "product_id"]
            )["price_usd"]
            .last()
            .reset_index()
        )

        forecasts = forecasts.merge(
            price_lookup,
            on=["store_id", "product_id"],
            how="left",
            suffixes=("", "_historical")
        )

        if "price_usd_historical" in forecasts.columns:
            if "price_usd" not in forecast_df.columns:
                forecasts["price_usd"] = forecasts[
                    "price_usd_historical"
                ]

            forecasts = forecasts.drop(
                columns=["price_usd_historical"]
            )

    # ------------------------------------------------------------------
    # Calculate promotion impact.
    # ------------------------------------------------------------------

    if "category" not in forecasts.columns:
        forecasts["category"] = np.nan

    forecasts["promotion_lift"] = forecasts[
        "category"
    ].map(lift_lookup)

    # Products without enough historical evidence are not eligible.
    forecasts["promotion_lift_available"] = (
        forecasts["promotion_lift"].notna()
    )

    forecasts["lifted_demand"] = np.where(
        forecasts["promotion_lift_available"],
        forecasts["expected_demand"]
        * (1 + forecasts["promotion_lift"]),
        np.nan
    )

    forecasts["incremental_units"] = np.where(
        forecasts["promotion_lift_available"],
        forecasts["lifted_demand"]
        - forecasts["expected_demand"],
        np.nan
    )

    if "price_usd" in forecasts.columns:
        forecasts["price_usd"] = pd.to_numeric(
            forecasts["price_usd"],
            errors="coerce"
        )

        forecasts["incremental_revenue"] = (
            forecasts["incremental_units"]
            * forecasts["price_usd"]
        )
    else:
        forecasts["incremental_revenue"] = np.nan

    result = forecasts[
        forecasts["promotion_lift_available"]
    ].copy()

    if result.empty:
        return pd.DataFrame(
            columns=[
                "store_id",
                "product_id",
                "category",
                "expected_demand",
                "promotion_lift",
                "lifted_demand",
                "incremental_units",
                "price_usd",
                "incremental_revenue",
                "recommendation_basis"
            ]
        )

    # Aggregate future days by SKU-store.
    aggregation = {
        "expected_demand": "sum",
        "lifted_demand": "sum",
        "incremental_units": "sum",
        "incremental_revenue": "sum",
        "promotion_lift": "first",
        "category": "first"
    }

    if "price_usd" in result.columns:
        aggregation["price_usd"] = "last"

    result = (
        result
        .groupby(
            ["store_id", "product_id"],
            as_index=False
        )
        .agg(aggregation)
    )

    result["recommendation_basis"] = (
        "Historical promotional sales difference was used to "
        "estimate future promotional lift; this is not a causal estimate."
    )

    result = result.sort_values(
        "incremental_revenue",
        ascending=False,
        na_position="last"
    )

    return result.head(5).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 5. What-If Analysis
# ---------------------------------------------------------------------------

def simulate_inventory_scenario(
    current_stock_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    base_lead_time: int,
    new_lead_time: int,
    service_level: float
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """
    Compare two inventory planning scenarios.

    The base scenario and new scenario use the same current inventory and
    demand forecast while changing only the lead time.

    This allows the team to examine the impact of supply lead-time changes
    on safety stock and suggested replenishment quantities.

    Args:
        current_stock_df:
            Current inventory DataFrame.

        forecast_df:
            Forecast DataFrame.

        base_lead_time:
            Existing/baseline lead time in days.

        new_lead_time:
            Alternative lead time in days.

        service_level:
            Target service level used in both scenarios.

    Returns:
        Tuple containing:

        base_plan:
            Inventory plan under base lead time.

        new_plan:
            Inventory plan under new lead time.

        comparison:
            Dictionary containing total safety stock and order quantity
            differences.

    Raises:
        ValueError:
            If lead times or service level are invalid.
    """
    _validate_positive_integer(
        base_lead_time,
        "base_lead_time"
    )

    _validate_positive_integer(
        new_lead_time,
        "new_lead_time"
    )

    _validate_service_level(service_level)

    base_plan = inventory_plan(
        current_stock_df=current_stock_df,
        forecast_df=forecast_df,
        lead_time_days=base_lead_time,
        service_level=service_level
    )

    new_plan = inventory_plan(
        current_stock_df=current_stock_df,
        forecast_df=forecast_df,
        lead_time_days=new_lead_time,
        service_level=service_level
    )

    base_safety_stock = float(
        base_plan["safety_stock"].sum()
    )

    new_safety_stock = float(
        new_plan["safety_stock"].sum()
    )

    base_order_quantity = float(
        base_plan["suggested_order_quantity"].sum()
    )

    new_order_quantity = float(
        new_plan["suggested_order_quantity"].sum()
    )

    comparison = {
        "base_lead_time_days": float(base_lead_time),
        "new_lead_time_days": float(new_lead_time),
        "service_level": float(service_level),
        "base_total_safety_stock": base_safety_stock,
        "new_total_safety_stock": new_safety_stock,
        "safety_stock_change": (
            new_safety_stock - base_safety_stock
        ),
        "base_total_suggested_order_quantity": (
            base_order_quantity
        ),
        "new_total_suggested_order_quantity": (
            new_order_quantity
        ),
        "suggested_order_quantity_change": (
            new_order_quantity - base_order_quantity
        )
    }

    return base_plan, new_plan, comparison


# ---------------------------------------------------------------------------
# 6. Inventory KPIs
# ---------------------------------------------------------------------------

def compute_inventory_kpis(
    historical_data_df: pd.DataFrame,
    simulation_results: Optional[pd.DataFrame] = None
) -> Dict[str, float]:
    """
    Calculate historical inventory KPIs.

    Metrics:

        Stockout Rate:
            Number of records with stockout = 1 divided by total
            valid observations.

        Estimated Lost Sales:
            Sum of units_sold on stockout records.

            IMPORTANT:
            This is only a proxy because the dataset does not contain
            actual unmet-demand quantities.

        Inventory Holding Value:
            Sum(stock × price_usd).

            This is retail-value exposure, not accounting carrying cost.

        Historical Service Level:
            1 - stockout rate.

    If simulation_results is supplied, an optional simulated order quantity
    metric is included.

    Args:
        historical_data_df:
            Historical inventory/sales data containing:
            stockout, units_sold, stock.

            price_usd is required for inventory holding value.

        simulation_results:
            Optional DataFrame containing
            suggested_order_quantity.

    Returns:
        Dictionary containing KPI names and numerical values.

    Raises:
        ValueError:
            If required historical columns are missing or invalid.
    """
    _validate_dataframe(
        historical_data_df,
        ["stockout", "units_sold", "stock", "price_usd"],
        "historical_data_df"
    )

    data = historical_data_df.copy()

    data["stockout"] = pd.to_numeric(
        data["stockout"],
        errors="coerce"
    )

    data["units_sold"] = pd.to_numeric(
        data["units_sold"],
        errors="coerce"
    )

    data["stock"] = pd.to_numeric(
        data["stock"],
        errors="coerce"
    )

    data["price_usd"] = pd.to_numeric(
        data["price_usd"],
        errors="coerce"
    )

    if data[
        ["stockout", "units_sold", "stock", "price_usd"]
    ].isna().any().any():
        raise ValueError(
            "KPI input contains missing or non-numeric values."
        )

    if (data["stock"] < 0).any():
        raise ValueError(
            "Historical stock contains negative values."
        )

    data["stockout"] = data["stockout"].clip(0, 1)
    data["units_sold"] = data["units_sold"].clip(lower=0)
    data["price_usd"] = data["price_usd"].clip(lower=0)

    total_observations = len(data)

    stockout_observations = int(
        (data["stockout"] == 1).sum()
    )

    stockout_rate = (
        stockout_observations / total_observations
        if total_observations > 0
        else np.nan
    )

    service_level = (
        1.0 - stockout_rate
        if pd.notna(stockout_rate)
        else np.nan
    )

    # This follows the requested project definition but is only a proxy
    # for lost sales because actual unmet demand is unavailable.
    estimated_lost_sales = float(
        data.loc[
            data["stockout"] == 1,
            "units_sold"
        ].sum()
    )

    inventory_holding_value = float(
        (data["stock"] * data["price_usd"]).sum()
    )

    kpis: Dict[str, float] = {
        "stockout_rate_percent": (
            float(stockout_rate * 100)
            if pd.notna(stockout_rate)
            else np.nan
        ),
        "estimated_lost_sales_units": estimated_lost_sales,
        "inventory_holding_value_usd": inventory_holding_value,
        "historical_service_level_percent": (
            float(service_level * 100)
            if pd.notna(service_level)
            else np.nan
        )
    }

    if simulation_results is not None:
        if not isinstance(simulation_results, pd.DataFrame):
            raise TypeError(
                "simulation_results must be a pandas DataFrame."
            )

        if "suggested_order_quantity" in simulation_results.columns:
            order_quantities = pd.to_numeric(
                simulation_results["suggested_order_quantity"],
                errors="coerce"
            ).fillna(0).clip(lower=0)

            kpis["simulated_suggested_order_units"] = float(
                order_quantities.sum()
            )

    return kpis


# ---------------------------------------------------------------------------
# Demonstration / Local Testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("RETAIL DEMAND FORECASTING AI AGENT")
    print("MEMBER 3 - INVENTORY LOGIC DEMONSTRATION")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Synthetic sample data.
    #
    # This is NOT the actual retail-demand.csv.
    # It is only used to demonstrate that inventory.py works.
    # ------------------------------------------------------------------

    sample_dates = pd.date_range(
        start="2026-09-01",
        periods=14,
        freq="D"
    )

    sample_rows = []

    for date_index, current_date in enumerate(sample_dates):
        sample_rows.extend(
            [
                {
                    "date": current_date,
                    "store_id": "S001",
                    "product_id": "P0001",
                    "category": "Beverages",
                    "price_usd": 10.0,
                    "promo_flag": 0 if date_index % 3 else 1,
                    "stock": max(5, 100 - date_index * 5),
                    "units_sold": 20 + (date_index % 4) * 2,
                    "stockout": 0
                },
                {
                    "date": current_date,
                    "store_id": "S001",
                    "product_id": "P0002",
                    "category": "Snacks",
                    "price_usd": 5.0,
                    "promo_flag": 1 if date_index % 3 == 0 else 0,
                    "stock": max(2, 40 - date_index * 4),
                    "units_sold": 10 + (date_index % 3),
                    "stockout": (
                        1
                        if date_index >= 10
                        else 0
                    )
                },
                {
                    "date": current_date,
                    "store_id": "S002",
                    "product_id": "P0001",
                    "category": "Beverages",
                    "price_usd": 12.0,
                    "promo_flag": 1 if date_index % 4 == 0 else 0,
                    "stock": max(10, 120 - date_index * 3),
                    "units_sold": 15 + (date_index % 5),
                    "stockout": 0
                }
            ]
        )

    sample_data = pd.DataFrame(sample_rows)

    print("\n1. SAMPLE HISTORICAL DATA")
    print("-" * 70)
    print(sample_data.head(10).to_string(index=False))

    # ------------------------------------------------------------------
    # Mock Forecast
    # ------------------------------------------------------------------

    mock_forecast = get_mock_forecasts(
        sample_data,
        horizon_days=14
    )

    print("\n2. MOCK FORECAST")
    print("-" * 70)
    print(
        mock_forecast.head(10).to_string(index=False)
    )

    # ------------------------------------------------------------------
    # Current stock.
    #
    # We pass the latest stock record for each SKU-store.
    # ------------------------------------------------------------------

    current_stock = (
        sample_data
        .sort_values("date")
        .drop_duplicates(
            ["store_id", "product_id"],
            keep="last"
        )
    )

    print("\n3. CURRENT STOCK")
    print("-" * 70)
    print(
        current_stock[
            [
                "store_id",
                "product_id",
                "stock",
                "price_usd"
            ]
        ].to_string(index=False)
    )

    # ------------------------------------------------------------------
    # Inventory Plan
    # ------------------------------------------------------------------

    inventory_results = inventory_plan(
        current_stock_df=current_stock,
        forecast_df=mock_forecast,
        lead_time_days=7,
        service_level=0.95
    )

    print("\n4. INVENTORY PLAN")
    print("-" * 70)
    print(
        inventory_results.to_string(index=False)
    )

    # ------------------------------------------------------------------
    # Stockout Risk
    # ------------------------------------------------------------------

    risk_results = compute_stockout_risks(
        current_stock_df=sample_data,
        forecast_df=mock_forecast,
        risk_horizon_days=14
    )

    print("\n5. STOCKOUT RISKS")
    print("-" * 70)
    print(
        risk_results.to_string(index=False)
    )

    # ------------------------------------------------------------------
    # Promotion Recommendations
    # ------------------------------------------------------------------

    promotion_results = recommend_promotions(
        historical_data_df=sample_data,
        forecast_df=mock_forecast,
        category_lift_map=None
    )

    print("\n6. PROMOTION RECOMMENDATIONS")
    print("-" * 70)
    print(
        promotion_results.to_string(index=False)
    )

    # ------------------------------------------------------------------
    # What-if Scenario
    # ------------------------------------------------------------------

    base_plan, new_plan, scenario_comparison = (
        simulate_inventory_scenario(
            current_stock_df=current_stock,
            forecast_df=mock_forecast,
            base_lead_time=7,
            new_lead_time=3,
            service_level=0.95
        )
    )

    print("\n7. WHAT-IF ANALYSIS")
    print("-" * 70)

    for metric_name, metric_value in scenario_comparison.items():
        print(
            f"{metric_name}: {metric_value:.2f}"
            if isinstance(metric_value, float)
            else f"{metric_name}: {metric_value}"
        )

    # ------------------------------------------------------------------
    # KPIs
    # ------------------------------------------------------------------

    kpis = compute_inventory_kpis(
        historical_data_df=sample_data,
        simulation_results=inventory_results
    )

    print("\n8. INVENTORY KPIs")
    print("-" * 70)

    for metric_name, metric_value in kpis.items():
        if isinstance(metric_value, float):
            print(f"{metric_name}: {metric_value:.2f}")
        else:
            print(f"{metric_name}: {metric_value}")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETED")
    print("The data above is SYNTHETIC sample data for testing only.")
    print("=" * 70)
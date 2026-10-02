# ============================================================
# RETAIL DEMAND FORECASTING PROJECT
# MEMBER 1 - DATA & EDA LEAD
# STEP 1: DATA LOADING AND BASIC DATA CHECK
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
plt.ion()

# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------

df = pd.read_csv("retail-demand.csv")


# ------------------------------------------------------------
# 2. CONVERT DATE COLUMN TO DATETIME
# ------------------------------------------------------------

df["date"] = pd.to_datetime(df["date"])


# ------------------------------------------------------------
# 3. DISPLAY FIRST 5 ROWS
# ------------------------------------------------------------

print("\n===== FIRST 5 ROWS =====")
print(df.head())


# ------------------------------------------------------------
# 4. CHECK DATA TYPES
# ------------------------------------------------------------

print("\n===== DATA TYPES =====")
print(df.dtypes)


# ------------------------------------------------------------
# 5. DATASET SIZE
# ------------------------------------------------------------

print("\n===== DATASET SIZE =====")
print("Number of rows:", len(df))
print("Number of columns:", len(df.columns))


# ------------------------------------------------------------
# 6. LIST ALL COLUMNS
# ------------------------------------------------------------

print("\n===== COLUMNS =====")
print(df.columns.tolist())


# ------------------------------------------------------------
# 7. CHECK MISSING VALUES
# ------------------------------------------------------------

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())


# ------------------------------------------------------------
# 8. CHECK DUPLICATE ROWS
# ------------------------------------------------------------

print("\n===== DUPLICATE ROWS =====")
print("Number of duplicate rows:", df.duplicated().sum())


# ------------------------------------------------------------
# 9. UNIQUE STORES, PRODUCTS, CATEGORIES AND REGIONS
# ------------------------------------------------------------

print("\n===== UNIQUE VALUES =====")

print("Number of stores:", df["store_id"].nunique())
print("Number of products:", df["product_id"].nunique())
print("Number of categories:", df["category"].nunique())
print("Number of regions:", df["region"].nunique())


# ------------------------------------------------------------
# 10. CATEGORIES
# ------------------------------------------------------------

print("\n===== CATEGORIES =====")
print(df["category"].unique())


# ------------------------------------------------------------
# 11. REGIONS
# ------------------------------------------------------------

print("\n===== REGIONS =====")
print(df["region"].unique())


# ------------------------------------------------------------
# 12. STORES
# ------------------------------------------------------------

print("\n===== STORES =====")
print(df["store_id"].unique())


# ------------------------------------------------------------
# 13. PRODUCTS
# ------------------------------------------------------------

print("\n===== PRODUCTS =====")
print(df["product_id"].unique())


# ------------------------------------------------------------
# 14. DATE RANGE
# ------------------------------------------------------------

print("\n===== DATE RANGE =====")
print("Start date:", df["date"].min().date())
print("End date:", df["date"].max().date())


# ------------------------------------------------------------
# 15. STATISTICAL SUMMARY
# ------------------------------------------------------------

print("\n===== STATISTICAL SUMMARY =====")
print(df.describe())


# ------------------------------------------------------------
# 16. INVALID VALUE CHECKS
# ------------------------------------------------------------

print("\n===== INVALID VALUE CHECKS =====")

print("Negative units sold:",
      (df["units_sold"] < 0).sum())

print("Price less than or equal to zero:",
      (df["price_usd"] <= 0).sum())

print("Negative stock:",
      (df["stock"] < 0).sum())


# ------------------------------------------------------------
# 17. STOCKOUT SUMMARY
# ------------------------------------------------------------

print("\n===== STOCKOUT SUMMARY =====")

print("Total stockout records:",
      df["stockout"].sum())

print("Stockout rate:",
      round(df["stockout"].mean() * 100, 2), "%")


# ------------------------------------------------------------
# 18. PROMOTION SUMMARY
# ------------------------------------------------------------

print("\n===== PROMOTION SUMMARY =====")

print("Promotion records:",
      df["promo_flag"].sum())

print("Promotion percentage:",
      round(df["promo_flag"].mean() * 100, 2), "%")


# ------------------------------------------------------------
# 19. HOLIDAY SUMMARY
# ------------------------------------------------------------

print("\n===== HOLIDAY SUMMARY =====")

print("Holiday records:",
      df["holiday_flag"].sum())

print("Holiday percentage:",
      round(df["holiday_flag"].mean() * 100, 2), "%")


# ------------------------------------------------------------
# 20. FINAL DATASET INFORMATION
# ------------------------------------------------------------

print("\n===== FINAL DATASET INFORMATION =====")
df.info()
# ============================================================
# EDA 1 - OVERALL DAILY DEMAND TREND
# ============================================================

daily_demand = df.groupby("date")["units_sold"].sum().reset_index()

plt.figure(figsize=(14, 6))

sns.lineplot(
    data=daily_demand,
    x="date",
    y="units_sold"
)

plt.title("Daily Retail Demand Trend", fontsize=16)
plt.xlabel("Date")
plt.ylabel("Total Units Sold")
plt.xticks(rotation=45)

plt.tight_layout()

# Save the chart
plt.savefig("daily_demand.png", dpi=300, bbox_inches="tight")

# Close the chart
plt.close()
# ============================================================
# EDA 2 - MONTHLY DEMAND PATTERN
# ============================================================

df["month"] = df["date"].dt.month

monthly_demand = (
    df.groupby("month")["units_sold"]
      .sum()
      .reset_index()
)

monthly_demand["month_name"] = pd.to_datetime(
    monthly_demand["month"],
    format="%m"
).dt.strftime("%B")

plt.figure(figsize=(12, 6))

sns.barplot(
    data=monthly_demand,
    x="month_name",
    y="units_sold"
)

plt.title("Monthly Retail Demand", fontsize=16)
plt.xlabel("Month")
plt.ylabel("Total Units Sold")
plt.xticks(rotation=45)

plt.tight_layout()

# Save the chart
plt.savefig("monthly_demand.png", dpi=300, bbox_inches="tight")

# Close the chart
plt.close()
# ============================================================
# CHART 3 - CATEGORY-WISE DEMAND
# ============================================================

# Calculate total units sold for each category
category_demand = (
    df.groupby("category")["units_sold"]
      .sum()
      .reset_index()
)

# Create the chart
plt.figure(figsize=(10, 6))

sns.barplot(
    data=category_demand,
    x="category",
    y="units_sold"
)

# Title and labels
plt.title("Demand by Product Category", fontsize=16)
plt.xlabel("Product Category")
plt.ylabel("Total Units Sold")

# Rotate labels if necessary
plt.xticks(rotation=20)

# Adjust spacing
plt.tight_layout()

# Save the chart
plt.savefig(
    "category_demand.png",
    dpi=300,
    bbox_inches="tight"
)

# Close the chart
plt.close()

print("Chart 3 saved as: category_demand.png")
# ============================================================
# CHART 4 - REGION-WISE DEMAND
# ============================================================

region_demand = (
    df.groupby("region")["units_sold"]
      .sum()
      .reset_index()
)

plt.figure(figsize=(10, 6))

sns.barplot(
    data=region_demand,
    x="region",
    y="units_sold"
)

plt.title("Demand by Region", fontsize=16)
plt.xlabel("Region")
plt.ylabel("Total Units Sold")
plt.tight_layout()

plt.savefig(
    "region_demand.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Chart 4 saved as: region_demand.png")
# ============================================================
# CHART 5 - PROMOTION VS NON-PROMOTION DEMAND
# ============================================================

promo_demand = (
    df.groupby("promo_flag")["units_sold"]
      .mean()
      .reset_index()
)

promo_demand["promotion"] = promo_demand["promo_flag"].map({
    0: "No Promotion",
    1: "Promotion"
})

plt.figure(figsize=(8, 6))

sns.barplot(
    data=promo_demand,
    x="promotion",
    y="units_sold"
)

plt.title("Average Demand: Promotion vs No Promotion", fontsize=14)
plt.xlabel("Promotion Status")
plt.ylabel("Average Units Sold")
plt.tight_layout()

plt.savefig(
    "promotion_demand.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Chart 5 saved as: promotion_demand.png")
# ============================================================
# CHART 6 - HOLIDAY VS NON-HOLIDAY DEMAND
# ============================================================

holiday_demand = (
    df.groupby("holiday_flag")["units_sold"]
      .mean()
      .reset_index()
)

holiday_demand["holiday"] = holiday_demand["holiday_flag"].map({
    0: "Non-Holiday",
    1: "Holiday"
})

plt.figure(figsize=(8, 6))

sns.barplot(
    data=holiday_demand,
    x="holiday",
    y="units_sold"
)

plt.title("Average Demand: Holiday vs Non-Holiday", fontsize=14)
plt.xlabel("Day Type")
plt.ylabel("Average Units Sold")
plt.tight_layout()

plt.savefig(
    "holiday_demand.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Chart 6 saved as: holiday_demand.png")
# ============================================================
# CHART 7 - STOCKOUT RATE BY REGION
# ============================================================

stockout_region = (
    df.groupby("region")["stockout"]
      .mean()
      .reset_index()
)

stockout_region["stockout_rate"] = stockout_region["stockout"] * 100

plt.figure(figsize=(10, 6))

sns.barplot(
    data=stockout_region,
    x="region",
    y="stockout_rate"
)

plt.title("Stockout Rate by Region", fontsize=16)
plt.xlabel("Region")
plt.ylabel("Stockout Rate (%)")

plt.tight_layout()

plt.savefig(
    "stockout_by_region.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Chart 7 saved as: stockout_by_region.png")
# ============================================================
# CHART 8 - STOCKOUT RATE BY PRODUCT CATEGORY
# ============================================================

stockout_category = (
    df.groupby("category")["stockout"]
      .mean()
      .reset_index()
)

stockout_category["stockout_rate"] = (
    stockout_category["stockout"] * 100
)

plt.figure(figsize=(10, 6))

sns.barplot(
    data=stockout_category,
    x="category",
    y="stockout_rate"
)

plt.title("Stockout Rate by Product Category", fontsize=16)
plt.xlabel("Product Category")
plt.ylabel("Stockout Rate (%)")
plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(
    "stockout_by_category.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Chart 8 saved as: stockout_by_category.png")
# ============================================================
# STEP 9 - WEEKLY AND MONTHLY AGGREGATES
# ============================================================

# WEEKLY DEMAND
weekly_demand = (
    df.set_index("date")
      .resample("W")["units_sold"]
      .sum()
      .reset_index()
)

weekly_demand.to_csv(
    "weekly_demand.csv",
    index=False
)

# MONTHLY DEMAND
monthly_demand = (
    df.set_index("date")
      .resample("ME")["units_sold"]
      .sum()
      .reset_index()
)

monthly_demand.to_csv(
    "monthly_demand.csv",
    index=False
)

print("Weekly aggregation saved as: weekly_demand.csv")
print("Monthly aggregation saved as: monthly_demand.csv")
# ============================================================
# STEP 10 - SKU LIFECYCLE ANALYSIS
# ============================================================

sku_lifecycle = (
    df.groupby("product_id")
      .agg(
          first_sale=("date", "min"),
          last_sale=("date", "max"),
          total_units_sold=("units_sold", "sum")
      )
      .reset_index()
)

sku_lifecycle["lifecycle_days"] = (
    sku_lifecycle["last_sale"] -
    sku_lifecycle["first_sale"]
).dt.days

print("\nSKU LIFECYCLE:")
print(sku_lifecycle)

sku_lifecycle.to_csv(
    "sku_lifecycle.csv",
    index=False
)

print("\nSKU lifecycle saved as: sku_lifecycle.csv")
# ============================================================
# STEP 11 - FEATURE TABLE FOR MEMBER 2
# ============================================================

# Sort data by store, product and date
df = df.sort_values(
    ["store_id", "product_id", "date"]
).copy()

group_cols = ["store_id", "product_id"]

# ------------------------------------------------------------
# LAG FEATURES
# ------------------------------------------------------------

df["lag_1"] = (
    df.groupby(group_cols)["units_sold"]
      .shift(1)
)

df["lag_7"] = (
    df.groupby(group_cols)["units_sold"]
      .shift(7)
)

df["lag_14"] = (
    df.groupby(group_cols)["units_sold"]
      .shift(14)
)

df["lag_28"] = (
    df.groupby(group_cols)["units_sold"]
      .shift(28)
)

# ------------------------------------------------------------
# ROLLING MEAN
# ------------------------------------------------------------

df["rolling_mean_7"] = (
    df.groupby(group_cols)["units_sold"]
      .transform(
          lambda x: x.shift(1).rolling(7).mean()
      )
)

# ------------------------------------------------------------
# DATE FEATURES
# ------------------------------------------------------------

df["day_of_week"] = df["date"].dt.dayofweek

df["month"] = df["date"].dt.month

# ------------------------------------------------------------
# PRICE CHANGE
# ------------------------------------------------------------

df["price_change"] = (
    df.groupby(group_cols)["price_usd"]
      .pct_change()
)

# ------------------------------------------------------------
# PROMOTION AND HOLIDAY FLAGS
# ------------------------------------------------------------

df["promo_flag"] = df["promo_flag"].astype(int)

df["holiday_flag"] = df["holiday_flag"].astype(int)

# ------------------------------------------------------------
# SAVE FEATURE TABLE
# ------------------------------------------------------------

df.to_csv(
    "retail_feature_table.csv",
    index=False
)

print("\nFeature table created successfully!")

print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nNew feature columns:")
print([
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "day_of_week",
    "month",
    "price_change",
    "promo_flag",
    "holiday_flag"
])

print("\nFeature table saved as: retail_feature_table.csv")
# ============================================================
# STEP 12 - FINAL CLEANING OF FEATURE TABLE
# ============================================================

# Remove rows where required lag/rolling features are unavailable
feature_df = df.dropna(
    subset=[
        "lag_1",
        "lag_7",
        "lag_14",
        "lag_28",
        "rolling_mean_7"
    ]
).copy()

print("\nBefore removing incomplete feature rows:")
print("Rows:", len(df))

print("\nAfter cleaning:")
print("Rows:", len(feature_df))
print("Columns:", len(feature_df.columns))

print("\nMissing values in feature table:")
print(feature_df.isnull().sum())

# Save cleaned feature table
feature_df.to_csv(
    "clean_feature_table.csv",
    index=False
)

print("\nClean feature table saved as: clean_feature_table.csv")
# ============================================================
# STEP 13 - SAVE FINAL DATASETS AS PARQUET
# ============================================================

# Clean original dataset
clean_df = df.drop_duplicates().copy()

clean_df.to_parquet(
    "clean_retail_demand.parquet",
    index=False
)

# Clean feature table
feature_df.to_parquet(
    "retail_feature_table.parquet",
    index=False
)

print("\nParquet files created successfully!")
print("1. clean_retail_demand.parquet")
print("2. retail_feature_table.parquet")

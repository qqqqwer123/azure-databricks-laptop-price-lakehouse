# Databricks notebook source
import pandas as pd
import numpy as np
from datetime import datetime

df_spark = spark.table("databricks0501.silver.laptop_cleaned")
df = df_spark.toPandas()

df.head()

# COMMAND ----------

df.columns

# COMMAND ----------

from sklearn.model_selection import train_test_split

numeric_features = ["ram_gb", "harddisk_gb", "screen_size"]
categorical_features = ["brand"]
target = "price"

required_cols = numeric_features + categorical_features + [target]

df_model = df.dropna(subset=required_cols).copy()

df_model["brand"] = df_model["brand"].astype(str).str.lower().str.strip()
df_model["price"] = pd.to_numeric(df_model["price"], errors="coerce")
df_model["ram_gb"] = pd.to_numeric(df_model["ram_gb"], errors="coerce")
df_model["harddisk_gb"] = pd.to_numeric(df_model["harddisk_gb"], errors="coerce")
df_model["screen_size"] = pd.to_numeric(df_model["screen_size"], errors="coerce")

df_model = df_model.dropna(subset=required_cols).copy()

# 避免樣本太少的品牌造成 stratify error
brand_counts = df_model["brand"].value_counts()
valid_brands = brand_counts[brand_counts >= 3].index
df_model = df_model[df_model["brand"].isin(valid_brands)].copy()

X = df_model[numeric_features + categorical_features]
y = df_model[target]

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=X["brand"]
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.25,
    random_state=42,
    stratify=X_train_val["brand"]
)

print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)

# COMMAND ----------

#訓練 Random Forest
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

baseline_model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", LinearRegression())
])

baseline_model.fit(X_train, y_train)
val_pred_baseline = baseline_model.predict(X_val)

baseline_mae = mean_absolute_error(y_val, val_pred_baseline)
baseline_rmse = np.sqrt(mean_squared_error(y_val, val_pred_baseline))
baseline_r2 = r2_score(y_val, val_pred_baseline)


rf_model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", RandomForestRegressor(
        n_estimators=300,
        random_state=42
    ))
])

rf_model.fit(X_train, y_train)
val_pred_rf = rf_model.predict(X_val)

rf_mae = mean_absolute_error(y_val, val_pred_rf)
rf_rmse = np.sqrt(mean_squared_error(y_val, val_pred_rf))
rf_r2 = r2_score(y_val, val_pred_rf)

model_results = pd.DataFrame({
    "model": ["Linear Regression", "Random Forest"],
    "dataset": ["validation", "validation"],
    "MAE": [baseline_mae, rf_mae],
    "RMSE": [baseline_rmse, rf_rmse],
    "R2": [baseline_r2, rf_r2],
    "trained_at": [datetime.now().isoformat(), datetime.now().isoformat()]
})

model_results

# COMMAND ----------

#Gold Table 1：模型評估表
spark.createDataFrame(model_results).write.mode("overwrite").format("delta").saveAsTable(
    "databricks0501.gold.model_evaluation_metrics"
)

# COMMAND ----------

#Gold Table 2：Feature Importance
feature_names = rf_model.named_steps["preprocess"].get_feature_names_out()
importances = rf_model.named_steps["model"].feature_importances_

feature_importance = (
    pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    })
    .sort_values("importance", ascending=False)
    .reset_index(drop=True)
)

feature_importance["rank"] = feature_importance.index + 1

feature_importance

spark.createDataFrame(feature_importance).write.mode("overwrite").format("delta").saveAsTable(
    "databricks0501.gold.feature_importance_summary"
)

# COMMAND ----------

#Gold Table 3：Brand Premium Residual

numeric_features_q3 = ["ram_gb", "harddisk_gb", "screen_size"]

X_q3 = df_model[numeric_features_q3]
y_q3 = df_model["price"]

X_train_val_q3, X_test_q3, y_train_val_q3, y_test_q3 = train_test_split(
    X_q3,
    y_q3,
    test_size=0.2,
    random_state=42
)

X_train_q3, X_val_q3, y_train_q3, y_val_q3 = train_test_split(
    X_train_val_q3,
    y_train_val_q3,
    test_size=0.25,
    random_state=42
)

q3_preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features_q3)
    ]
)

q3_model = Pipeline(steps=[
    ("preprocess", q3_preprocess),
    ("model", RandomForestRegressor(
        n_estimators=300,
        random_state=42
    ))
])

q3_model.fit(X_train_q3, y_train_q3)

q3_pred = q3_model.predict(X_test_q3)

q3_result = df_model.loc[X_test_q3.index, ["brand", "price", "ram_gb", "harddisk_gb", "screen_size"]].copy()
q3_result["predicted_price"] = q3_pred
q3_result["residual"] = q3_result["price"] - q3_result["predicted_price"]

brand_premium = (
    q3_result
    .groupby("brand")
    .agg(
        avg_actual_price=("price", "mean"),
        avg_predicted_price=("predicted_price", "mean"),
        avg_residual=("residual", "mean"),
        median_residual=("residual", "median"),
        sample_count=("residual", "count")
    )
    .reset_index()
    .sort_values("avg_residual", ascending=False)
)

brand_premium_filtered = brand_premium[brand_premium["sample_count"] >= 10].copy()

brand_premium_filtered


spark.createDataFrame(brand_premium_filtered).write.mode("overwrite").format("delta").saveAsTable(
    "databricks0501.gold.brand_premium_residual"
)

# COMMAND ----------

#Gold Table 4：What-if Prediction

def assign_price_band(price):
    if price < 500:
        return "Low"
    elif price < 1000:
        return "Mid"
    elif price < 1500:
        return "High"
    else:
        return "Premium"


what_if_cases = pd.DataFrame([
    {"brand": "dell", "ram_gb": 8,  "harddisk_gb": 256,  "screen_size": 15.6},
    {"brand": "dell", "ram_gb": 16, "harddisk_gb": 512,  "screen_size": 15.6},
    {"brand": "dell", "ram_gb": 32, "harddisk_gb": 1024, "screen_size": 15.6},

    {"brand": "hp", "ram_gb": 8,  "harddisk_gb": 256,  "screen_size": 15.6},
    {"brand": "hp", "ram_gb": 16, "harddisk_gb": 512,  "screen_size": 15.6},
    {"brand": "hp", "ram_gb": 32, "harddisk_gb": 1024, "screen_size": 15.6},

    {"brand": "asus", "ram_gb": 8,  "harddisk_gb": 256,  "screen_size": 15.6},
    {"brand": "asus", "ram_gb": 16, "harddisk_gb": 512,  "screen_size": 15.6},
    {"brand": "asus", "ram_gb": 32, "harddisk_gb": 1024, "screen_size": 15.6},
])

what_if_cases["predicted_price"] = rf_model.predict(what_if_cases)
what_if_cases["price_band"] = what_if_cases["predicted_price"].apply(assign_price_band)

base_prices = (
    what_if_cases
    .sort_values(["brand", "ram_gb", "harddisk_gb"])
    .groupby("brand")["predicted_price"]
    .transform("first")
)

what_if_cases["price_change_vs_base"] = what_if_cases["predicted_price"] - base_prices
what_if_cases["price_change_pct_vs_base"] = what_if_cases["price_change_vs_base"] / base_prices * 100

what_if_cases

spark.createDataFrame(what_if_cases).write.mode("overwrite").format("delta").saveAsTable(
    "databricks0501.gold.what_if_prediction_results"
)

# COMMAND ----------

# MAGIC %sql
# MAGIC --CREATE SCHEMA IF NOT EXISTS databricks0501.bronze;
# MAGIC CREATE SCHEMA IF NOT EXISTS databricks0501.models;

# COMMAND ----------

import joblib
import os

# --- 1. 定義 Volume 的完整路徑 ---
# 根據你的截圖，路徑結構為: /Volumes/<catalog>/<schema>/<volume>
volume_path = "/Volumes/databricks0501/models/volume_laptop_models"

# 如果你想在 Volume 裡面再分資料夾，可以這樣做：
model_folder = f"{volume_path}/v1"
os.makedirs(model_folder, exist_ok=True)

# --- 2. 儲存模型檔案 (.pkl) ---
model_filename = f"{model_folder}/laptop_price_rf_model.pkl"
joblib.dump(rf_model, model_filename)

# --- 3. 儲存特徵元數據 (這對 Streamlit 預測時非常重要，用來對齊欄位) ---
feature_filename = f"{model_folder}/model_features.pkl"
model_features = {
    "numeric_features": ["ram_gb", "harddisk_gb", "screen_size"],
    "categorical_features": ["brand"],
    "target": "price"
}
joblib.dump(model_features, feature_filename)

print(f"✅ 成功！模型與特徵資訊已存入：{model_folder}")

# --- 4. 驗證檔案是否真的在那裡 ---
files = os.listdir(model_folder)
print(f"資料夾內容: {files}")

# COMMAND ----------


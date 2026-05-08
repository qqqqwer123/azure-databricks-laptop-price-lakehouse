# Databricks notebook source
# MAGIC %sql
# MAGIC --CREATE SCHEMA IF NOT EXISTS databricks0501.bronze;
# MAGIC CREATE SCHEMA IF NOT EXISTS databricks0501.silver;
# MAGIC CREATE SCHEMA IF NOT EXISTS databricks0501.gold;

# COMMAND ----------

from pyspark.sql.functions import (
    col, regexp_extract, regexp_replace, when,
    lower, trim, current_timestamp
)

# COMMAND ----------

from pyspark.sql.functions import (
    col, regexp_extract, regexp_replace, when,
    lower, trim, current_timestamp
)

df = spark.table("databricks0501.bronze.laptop_raw_landing")

# brand standardization
df = df.withColumn(
    "brand",
    lower(trim(col("brand")))
)

# price: remove currency symbols and comma, then cast to double
df = df.withColumn(
    "price_str",
    regexp_replace(col("price").cast("string"), r"[$,]", "")
)

df = df.withColumn(
    "price",
    when(col("price_str") != "", col("price_str").cast("double")).otherwise(None)
)

# ram: extract number and convert to GB
df = df.withColumn(
    "ram_str",
    regexp_extract(col("ram").cast("string"), r"(\d+)", 1)
)

df = df.withColumn(
    "ram_gb",
    when(col("ram_str") != "", col("ram_str").cast("int")).otherwise(None)
)

# harddisk: extract number and convert TB to GB
df = df.withColumn(
    "harddisk_value_str",
    regexp_extract(col("harddisk").cast("string"), r"(\d+)", 1)
)

df = df.withColumn(
    "harddisk_value",
    when(col("harddisk_value_str") != "", col("harddisk_value_str").cast("double")).otherwise(None)
)

df = df.withColumn(
    "harddisk_gb",
    when(
        lower(col("harddisk")).contains("tb"),
        col("harddisk_value") * 1024
    ).otherwise(col("harddisk_value"))
)

# screen_size: extract number and convert to double
df = df.withColumn(
    "screen_size_str",
    regexp_extract(col("screen_size").cast("string"), r"(\d+(\.\d+)?)", 1)
)

df = df.withColumn(
    "screen_size",
    when(col("screen_size_str") != "", col("screen_size_str").cast("double")).otherwise(None)
)

# remove helper columns
df = df.drop(
    "price_str",
    "ram_str",
    "harddisk_value_str",
    "harddisk_value",
    "screen_size_str"
)

# drop rows with missing key fields
df = df.dropna(subset=["price", "harddisk_gb", "screen_size"])

# remove duplicate rows
df = df.dropDuplicates()

# add Silver metadata
df = df.withColumn("_silver_updated_at", current_timestamp())

display(df)

# COMMAND ----------

df.write.mode("overwrite").format("delta").saveAsTable(
    "databricks0501.silver.laptop_cleaned"
)

# COMMAND ----------


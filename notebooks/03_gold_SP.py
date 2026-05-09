# Databricks notebook source
df=spark.table("databricks0501.silver.laptop_cleaned")
display(df)

# COMMAND ----------

#table 1 品牌價格摘要
from pyspark.sql.functions import count, avg, min, max, round, col

brand_price_summary = (
    df.groupBy("brand")
      .agg(
          count("*").alias("product_count"),
          round(avg("price"), 2).alias("avg_price"),
          round(min("price"), 2).alias("min_price"),
          round(max("price"), 2).alias("max_price")
      )
      .filter(col("product_count") >= 5) #過濾樣本太少的品牌，有發現一個叫做bestnotebook應該是誤植
      .orderBy("avg_price", ascending=False)
)

brand_price_summary.write.mode("overwrite").format("delta").saveAsTable(
    "databricks0501.gold.brand_price_summary"
)

display(brand_price_summary)

# COMMAND ----------

#table2價格帶分布
from pyspark.sql.functions import when, col

df_price_band = df.withColumn(
    "price_band",
    when(col("price") < 500, "Low")
    .when(col("price") < 1000, "Mid")
    .when(col("price") < 1500, "High")
    .otherwise("Premium")
)

price_band_summary = (
    df_price_band.groupBy("brand", "price_band")
    .agg(count("*").alias("product_count"))
    .filter(col("product_count") >= 5)
    .orderBy("brand", "price_band")
)

price_band_summary.write.mode("overwrite").format("delta").saveAsTable(
    "databricks0501.gold.price_band_summary"
)

display(price_band_summary)

# COMMAND ----------

#table3 規格與價格摘要
spec_price_summary = (
    df.groupBy("ram_gb", "harddisk_gb")
      .agg(
          count("*").alias("product_count"),
          round(avg("price"), 2).alias("avg_price")
      )
      .orderBy("ram_gb", "harddisk_gb")
)

spec_price_summary.write.mode("overwrite").format("delta").saveAsTable(
    "databricks0501.gold.spec_price_summary"
)

display(spec_price_summary)


# COMMAND ----------

#table 4 看CP值
from pyspark.sql.functions import col, round

cp_value_ranking = (
    df
    .withColumn("rating_num", col("rating").cast("double"))
    .where(col("price").isNotNull())
    .where(col("price") > 0)
    .where(col("rating_num").isNotNull())
    .where(col("ram_gb").isin([4, 8, 16, 32, 64]))
    .where(col("harddisk_gb").isin([128, 256, 512, 1024, 2048]))
    .withColumn(
        "spec_score",
        col("rating_num") * 0.4
        + (col("ram_gb") / 16) * 0.3
        + (col("harddisk_gb") / 512) * 0.3
    )
    .withColumn(
        "cp_score",
        round(col("spec_score") / col("price"), 6)
    )
    .select(
        "brand",
        "model",
        "price",
        "rating_num",
        "ram_gb",
        "harddisk_gb",
        "screen_size",
        "spec_score",
        "cp_score"
    )
    .orderBy("cp_score", ascending=False)
)

cp_value_ranking.write.mode("overwrite").format("delta").saveAsTable(
    "databricks0501.gold.cp_value_ranking"
)

display(cp_value_ranking)


# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS databricks0501.gold.cp_value_ranking;

# COMMAND ----------


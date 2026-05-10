# Databricks notebook source
# MAGIC %sql
# MAGIC -- Data Engineer
# MAGIC GRANT USE CATALOG ON CATALOG databricks0501 TO `data engineer`;
# MAGIC
# MAGIC GRANT SELECT, MODIFY ON SCHEMA databricks0501.bronze TO `data engineer`;
# MAGIC GRANT SELECT, MODIFY ON SCHEMA databricks0501.silver TO `data engineer`;
# MAGIC GRANT SELECT, MODIFY ON SCHEMA databricks0501.gold TO `data engineer`;
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Data Analyst
# MAGIC GRANT USE CATALOG ON CATALOG databricks0501 TO `data analyst`;
# MAGIC
# MAGIC GRANT USE SCHEMA ON SCHEMA databricks0501.silver TO `data analyst`;
# MAGIC GRANT USE SCHEMA ON SCHEMA databricks0501.gold TO `data analyst`;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Business User
# MAGIC GRANT USE CATALOG ON CATALOG databricks0501 TO `business user`;
# MAGIC GRANT USE SCHEMA ON SCHEMA databricks0501.gold TO `business user`;
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW GRANTS ON SCHEMA databricks0501.gold;

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW GRANTS ON SCHEMA databricks0501.silver;

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW GRANTS ON SCHEMA databricks0501.bronze;
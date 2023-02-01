# Databricks notebook source
# In three ways, do the following: 
# Print the total number of the flights counts grouped by the origin country, show only the three minimum counts of these flights. - Hosam Mahmoud

flightData2015 = spark\
  .read\
  .option("inferSchema", "true")\
  .option("header", "true")\
  .csv("/FileStore/tables/flight_data.csv")
flightData2015.createOrReplaceTempView("flight_data_2015")

print(flightData2015.head(5))

# first way
flightData2015\
  .groupBy("ORIGIN_COUNTRY_NAME")\
  .sum("count")\
  .withColumnRenamed("sum(count)", "Origin_total")\
  .sort("Origin_total")\
  .limit(3)\
  .show()

# second way
maxSql = spark.sql("""
SELECT ORIGIN_COUNTRY_NAME, sum(count) as Origin_total
FROM flight_data_2015
GROUP BY ORIGIN_COUNTRY_NAME
ORDER BY sum(count)
LIMIT 3
""")

maxSql.show()

# Print the execution plan for one of these ways.
print(maxSql.explain())
print(flightData2015.explain())

# COMMAND ----------

# the third way using normal sql

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT ORIGIN_COUNTRY_NAME, sum(count) as Origin_total FROM flight_data_2015
# MAGIC GROUP BY ORIGIN_COUNTRY_NAME
# MAGIC ORDER BY sum(count)
# MAGIC LIMIT 3

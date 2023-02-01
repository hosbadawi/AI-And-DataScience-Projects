// Databricks notebook source
// Print all the partitions in this path "/FileStore/tables/Retail" through a file system command. ---------- Hosam Mahmoud

// COMMAND ----------

// MAGIC %fs
// MAGIC ls /FileStore/tables/Retail

// COMMAND ----------

//For the second partition, calculate the sum, min, max and avg of the "UnitPrice" column. ----------- Hosam

val ds = spark.read.format("csv")
  .option("header", "true")
  .option("inferSchema", "true")
  .load("/FileStore/tables/Retail/2010_12_02.csv")
  .coalesce(5)
ds.cache()
ds.createOrReplaceTempView("dfTable")

import org.apache.spark.sql.functions.{min, max, sum, avg}
ds.select(sum("UnitPrice"),min("UnitPrice"), max("UnitPrice"), avg("UnitPrice")).show(false)

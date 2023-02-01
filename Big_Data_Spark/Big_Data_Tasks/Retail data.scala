// Databricks notebook source
val path = "/FileStore/tables/retail-data/daily/*.csv"
val Retail_Data = spark.read.option("header","true").csv(path)
Retail_Data.show()

// COMMAND ----------

val Retail_Data = 

// COMMAND ----------



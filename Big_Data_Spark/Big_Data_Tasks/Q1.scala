// Databricks notebook source
// Print the first 3 records of sales data. - Hosam Mahmoud
val salesData = spark
  .read
  .option("inferSchema", "true")
  .option("header", "true")
  .csv("/FileStore/tables/Sales.csv")

salesData.show(3)

// COMMAND ----------

 // Print the minimum count of flights in 2 ways from flight data. - Hosam Mahmoud
val flightData = spark
  .read
  .option("inferSchema", "true")
  .option("header", "true")
  .csv("/FileStore/tables/flight_data.csv")

// first way
import org.apache.spark.sql.functions.{min,max}
flightData.agg(max($"count"), min($"count")).show()

// second way
flightData.agg(max(flightData(flightData.columns(2))),min(flightData(flightData.columns(2)))).show()

// Databricks notebook source
// Print the first 5 records that have "Quantity" of 12 and their "unitPrice" > 2 ------ Hosam Mahmoud

val df = spark.read.format("csv")
  .option("header", "true")
  .option("inferSchema", "true")
  .load("/FileStore/tables/2010_12_01.csv")

// Show the inferred schema
df.printSchema()
// Create a view from the data frame.
df.createOrReplaceTempView("dfTable")

import org.apache.spark.sql.functions.col
df.where(col("Quantity").equalTo(12) && col("UnitPrice").gt(2))
  .select("*")
  .show(5, false)

// Split the "Description" column and re-name it as "Detailed description" ----- Hosam Mahmoud
import org.apache.spark.sql.functions.split
df.select(split(col("Description"), " ").alias("Detailed description")).show(false)

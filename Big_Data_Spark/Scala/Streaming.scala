// Databricks notebook source

val Static_Data = spark
  .read
  .option("inferSchema", "true")
  .option("header", "true")
  .csv("/FileStore/tables/retail-data/by-day/*.csv")


val dataSchema = Static_Data.schema


// COMMAND ----------

val streaming = spark
                  .readStream.schema(dataSchema)
                  .option("maxFilesPerTrigger", 20)
                  .csv("/FileStore/tables/retail-data/by-day/*.csv")


// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC # B

// COMMAND ----------


import org.apache.spark.sql.functions._

val activityCounts = streaming.withColumn("Stockesvalue",col("UnitPrice")*col("Quantity")).groupBy("CustomerID").agg(
      sum("Quantity").as("total stocks"),
      sum("Stockesvalue").as("total value"))


// COMMAND ----------

val Customer_Stock_Query = activityCounts.writeStream.queryName("Stock_Query")
  .format("memory").outputMode("complete")
  .start()


// COMMAND ----------

spark.streams.active

Thread.sleep(16000)


// COMMAND ----------

for( i <- 1 to 10 ) {
    spark.sql("SELECT * FROM Stock_Query").show()
    Thread.sleep(1000)
}


// COMMAND ----------

// MAGIC %md
// MAGIC # C

// COMMAND ----------

val Stream_C = streaming
                              .withColumn("Sales_Value", col("Quantity")*col("UnitPrice"))
                              .withColumn("Records_Imported", lit(1))
                              .withColumn("Trigger_Time", current_timestamp())
                              .groupBy("Trigger_Time")
                              .agg(count("Records_Imported").as("RecordsImported"), sum("Sales_Value").as("SaleValue")) 



// COMMAND ----------

val stream_c_query = Stream_C.writeStream.queryName("stream_c")
  .format("memory").outputMode("complete")
  .start()


// COMMAND ----------

spark.streams.active

Thread.sleep(10000)


// COMMAND ----------

for( i <- 1 to 10 ) {
    spark.sql("SELECT * FROM stream_c").show()
    Thread.sleep(1000)
}


// COMMAND ----------

// MAGIC %md
// MAGIC # D

// COMMAND ----------

display(Stream_C)

// COMMAND ----------



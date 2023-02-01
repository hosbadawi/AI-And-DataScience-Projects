// Databricks notebook source
// For the data in this path "/FileStore/tables/retail-data/by-day/2010_12_01.csv", do the following: ------ Hosam Mahmoud

val RetailDF = spark.read.format("csv")
  .option("header", "true")
  .option("inferSchema", "true")
  .load("/FileStore/tables/Retail/2010_12_01.csv")

RetailDF.printSchema()
RetailDF.createOrReplaceTempView("dfTable")


// COMMAND ----------

RetailDF.show(5)

// COMMAND ----------

// Print the first 5 records of the "Description" column such that every word starts with a capital letter. ------- Hosam Mahmoud
val Capletters = RetailDF.filter(x => {val totalStrings = x.getString(2).split(" ")
                                   var condition = true
                                   for (x <- totalStrings){
                                     if (!x(0).isUpper){
                                       condition=false
                                     }
                                   }
                                   condition})

Capletters.show(5,false)

// COMMAND ----------

// What are the two cases to drop records that contain any null values.  ------- Hosam Mahmoud.

// first case
RetailDF.na.drop("all")

// second case
RetailDF.na.drop("any")

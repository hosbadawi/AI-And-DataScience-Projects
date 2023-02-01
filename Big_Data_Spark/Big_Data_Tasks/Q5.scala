// Databricks notebook source
// Show the five words that have the least number of occurences in Adult data. ---- Hosam Mahmoud

val textFile = spark.sparkContext.textFile("/FileStore/tables/Adult.csv")

// print the text lines
textFile.collect.foreach(println)

// COMMAND ----------

// print the occurences of each word  ---- Hosam Mahmoud

val counts = textFile.flatMap(line => line.split(","))
                    .map(word => (word, 1))
                    .reduceByKey(_+_)
counts.collect.foreach(println)

// COMMAND ----------

// print the five words that have the least number of occurences in Adult data. ---- Hosam Mahmoud
val allwords = textFile.flatMap(_.split("\\W+"))
val words = allwords.filter(!_.isEmpty)
val pairs = words.map((_,1))
val reducedByKey = pairs.reduceByKey(_+_)
val top5words = reducedByKey.takeOrdered(5)(Ordering[Int].on(_._2))
top5words.foreach(println) 

// COMMAND ----------

// the number of jobs is = 1 ---> (  (1) Spark Jobs  )

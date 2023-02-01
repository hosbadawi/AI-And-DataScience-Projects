// Databricks notebook source
// b)Use Spark Scala to load your data into an RDD. 
val textFile = spark.sparkContext.textFile("/FileStore/tables/Text Corpus")
display(dbutils.fs.ls("/FileStore/tables/Text Corpus"))

// COMMAND ----------

// c) count the number of lines across all the files. 
println("the number of lines across all the files are " +textFile.count())

// COMMAND ----------

// d) Find the number of occurrences of the word “antibiotics” 
val word_count = textFile.flatMap(line => line.split(","))
                    .map(word => (word, 1))
                    .filter(word=> word._1.contains("antibiotics"))
println("word_count : "+ word_count.count())

// COMMAND ----------

/*e)	Count the occurrence of the word “patient” and “admitted” on the same line of text. 
Please ensure that your code contains at least 2 transformation functions in a pipeline. */

val word_count = textFile.flatMap(line => line.split(","))
                    .map(word => (word, 2))
                    .filter(word=> word._1.contains("patient") &&  word._1.contains("admitted"))
println("word_count : "+ word_count.count())

// COMMAND ----------



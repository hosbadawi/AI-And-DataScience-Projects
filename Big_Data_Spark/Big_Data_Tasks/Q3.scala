// Databricks notebook source
// Create a structured data set with the name "Flights_info" using the map function, this data set includes two columns, "random" column that holds a random variable with the type integer, and "count" column 
// with range of (500). ------- Hosam Mahmoud

case class Flight(DEST_COUNTRY_NAME: String,
                  ORIGIN_COUNTRY_NAME: String, count: BigInt)
val flightsDF = spark.read
  .parquet("/FileStore/tables/part_r_00000_1a9822ba_b8fb_4d8e_844a_ea30d0801b9e_gz.parquet")
val flights = flightsDF.as[Flight]

flights.show(5)

case class FlightMetadata(count: BigInt, random: BigInt)

val flightsMeta = spark.range(500)
  .map(x => (x, scala.util.Random.nextInt(1000)))
  .withColumnRenamed("_1", "count")
  .withColumnRenamed("_2", "random")
  .as[FlightMetadata]

flightsMeta.show(5)

// Using "count" column, join the “Flights_info” data set to the Flight data coming from this path "/FileStore/tables/flightdata/parquet/part_r_00000_1a9822ba_b8fb_4d8e_844a_ea30d0801b9e_gz.parquet"
// ------- Hosam Mahmoud

val flights2 = flights
  .joinWith(flightsMeta, flights.col("count") === flightsMeta.col("count"))
  .withColumnRenamed("_1", "count")
  .withColumnRenamed("_2", "random")

flights2.show(10, false)

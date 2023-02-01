-- Databricks notebook source
-- Create a table that contains all the information in the flights data. (1 pt)
-- Note that, flights data contains all the json files in this path'/FileStore/tables/json/*.json 
-- ########################## Hosam Mahmoud

DROP TABLE IF EXISTS flights;
CREATE TABLE flights (
  DEST_COUNTRY_NAME STRING, ORIGIN_COUNTRY_NAME STRING, count LONG)
USING JSON OPTIONS (path '/FileStore/tables/*.json')

-- COMMAND ----------

-- MAGIC %sql
-- MAGIC select * from flights

-- COMMAND ----------

-- i've read the data from 6 json files (partitions)

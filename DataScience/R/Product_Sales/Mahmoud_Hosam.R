rm(list=ls())

CUSTOMER <- read.csv(file = 'CUSTOMER.csv')
PRODUCT <- read.csv(file = 'PRODUCT.csv')
PRODUCT_SALES <- read.csv(file = 'PRODUCT_SALES.csv')
TIMELINE <- read.csv(file = 'TIMELINE.csv')

Quan_cube <- 
  tapply(PRODUCT_SALES$Quantity, 
         PRODUCT_SALES[,c("TimeID","CustomerID","ProductNumber")], 
         FUN=function(x){return(sum(x))})

Quan_cube
dimnames(Quan_cube)
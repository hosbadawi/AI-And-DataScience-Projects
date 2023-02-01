# to delete all the variables in the environment
rm(list=ls())

############## install packages
install.packages("pacman")
install.packages("superml")
install.packages("lattice")
install.packages("reshape2")
install.packages("caTools")
install.packages("party")
install.packages("rpart.plot")
install.packages("caret")
install.packages('xgboost')
install.packages('pROC')

install.packages('Rtools')
install.packages("devtools")
devtools::install_github("rstudio/keras")
devtools::install_github("rstudio/reticulate")
install.packages("tensorflow")
install_tensorflow(method = "auto")

require(pacman)
library(pacman)
library(superml)
library(plyr)
library(GGally)
library(lattice)
library(reshape2)
library(caTools)
library(rpart)
library(rpart.plot)
library(party)
library(caret)
library(xgboost)
library(pROC)

library(keras)
library(tensorflow)

pacman::p_load(pacman, dplyr, ggplot2)


############################# Part A #############################
# Read the Chum Dataset
FullData = read.csv('Churn Dataset.csv')

# Drop the first column 'CustomerID' hence It will negatively affect the training process
FullData = subset(FullData, select = -c(customerID))

# Data Exploration
str(FullData)
summary(FullData)

# Display unique values of categorical columns
unique(FullData$gender)
unique(FullData$SeniorCitizen)
unique(FullData$Partner)
unique(FullData$Dependents)
unique(FullData$PhoneService)
unique(FullData$MultipleLines)
unique(FullData$InternetService)
unique(FullData$OnlineSecurity)
unique(FullData$OnlineBackup)
unique(FullData$DeviceProtection)
unique(FullData$TechSupport)
unique(FullData$StreamingTV)
unique(FullData$StreamingMovies)
unique(FullData$Contract)
unique(FullData$PaperlessBilling)
unique(FullData$PaymentMethod)

# Delete Duplicated rows
FullData <- subset(FullData, !duplicated(FullData))

# Print which columns contain Missing Data 
sapply(FullData, function(x) sum(is.na(x)))

# Print the 11's Null values that in TotalCharges Column
print(subset(FullData, is.na(FullData$TotalCharges)))

# Check the Distribution of TotalCharges Column
hist(FullData$TotalCharges, breaks=50, col="red") # the distribution is Right Skewed

# Check the effect of these 11 customers in comparison to the total customers
sum(is.na(FullData$TotalCharges))/nrow(FullData)

# This 11 customers is 0.16% of our data which is too small, so i will drop these 11 rows.
FullData <- FullData[complete.cases(FullData),]

# first I'll convert categorical column to string to be able in future to changes all NO to 0 and all YES to 1
FullData$SeniorCitizen <- as.factor(plyr::mapvalues(FullData$SeniorCitizen,
                                                    from=c("0","1"),
                                                    to=c("No", "Yes")))

# the column MultipleLines is related to PhoneService column so if  PhoneService == NO
# MultipleLines will be for sure = NO (no need to be named NO phone service)
FullData$MultipleLines <- as.factor(plyr::mapvalues(FullData$MultipleLines, 
                                                    from=c("No phone service"),
                                                    to=c("No")))

# the same idea here if InternetService == NO, the other features that depend on it will be NO
# (no need to be named NO Internet Service)

for(column in 9:14){
  FullData[,column] <- as.factor(plyr::mapvalues(FullData[,column],
                                                 from= c("No internet service"), to= c("No")))
}

# Categorical to numerical values
for(column in 1:19){
  FullData[,column] <- as.factor(plyr::mapvalues(FullData[,column],
                                                 from= c("No","Yes"), to= c(0,1)))
}
FullData$gender <- as.factor(plyr::mapvalues(FullData$gender,
                                             from= c("Female","Male"), to= c(0,1)))
FullData$InternetService <- as.factor(plyr::mapvalues(FullData$InternetService,
                                                      from= c("DSL","Fiber optic"), to= c(1,2)))
FullData$Contract <- as.factor(plyr::mapvalues(FullData$Contract,
                                               from= c("Month-to-month","One year","Two year"), to= c(0,1,2)))
FullData$PaymentMethod <- as.factor(plyr::mapvalues(FullData$PaymentMethod,
                                                    from= c("Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"), to= c(0,1,2,3)))
FullData$Churn <- as.factor(plyr::mapvalues(FullData$Churn,
                                            from= c("No","Yes"), to= c(0,1)))


# change from factor data type to numeric
indx <- sapply(FullData, is.factor)
FullData[indx] <- lapply(FullData[indx], function(x) as.numeric(as.character(x)))

sapply(FullData, class)

############## Plot Scatter plot Matrix for numeric columns
Churn.Colors = c("#F2EDD7FF","#E94B3CFF")

# Correlation panel (Lower Panel)
Correlation <- function(x, y){
  usr <- par("usr"); on.exit(par(usr))
  par(usr = c(0, 1, 0, 1))
  r <- round(cor(x, y), digits=2)
  txt <- paste0("Corr = ", r)
  cex.cor <- 0.8/strwidth(txt)
  text(0.5, 0.5, txt, cex = cex.cor * r)
}

# Customize upper panel
FullData$Churn = as.factor(FullData$Churn)
Upper <-function(x, y){
  points(x,y, pch = 10, cex = 0.5, col = Churn.Colors[FullData$Churn])
}

# Create the plots
pairs(FullData[c("tenure","MonthlyCharges","TotalCharges")], 
      lower.panel = Correlation,
      upper.panel = Upper)

FullData$Churn = as.numeric(as.character(FullData$Churn))

############## Plot Heatmap Correlation for numeric values only
corr <- round(cor(FullData[c("tenure","MonthlyCharges","TotalCharges","Churn")]),2)
CorrDF <- melt(corr)

ggplot(data = CorrDF, aes(x=Var1, y=Var2,
                          fill=value)) +
  geom_tile() +
  geom_text(aes(Var2, Var1, label = value),
            color = "white", size = 4)

############## Plot Heatmap Correlation for all columns
corr <- round(cor(FullData),2)
CorrDF <- melt(corr)

ggplot(data = CorrDF, aes(x=Var1, y=Var2,
                          fill=value)) +
  geom_tile() +
  geom_text(aes(Var2, Var1, label = value),
            color = "white", size = 4)

# change data types
indx <- sapply(FullData, is.numeric)
FullData[indx] <- lapply(FullData[indx], function(x) as.factor(as.character(x)))
FullData$tenure = as.numeric(as.character(FullData$tenure))
FullData$MonthlyCharges = as.numeric(as.character(FullData$MonthlyCharges))
FullData$TotalCharges = as.numeric(as.character(FullData$TotalCharges))

sapply(FullData, class)
row.names(FullData) <- NULL

# Feature Scaling
FullData[c("tenure","MonthlyCharges","TotalCharges")] = scale(FullData[c("tenure","MonthlyCharges","TotalCharges")])

# Split the dataset
set.seed(123)
sample <- sample(c(TRUE, FALSE), nrow(FullData), replace=TRUE, prob=c(0.8,0.2))
Train  <- FullData[sample, ]
Test   <- FullData[!sample, ]

row.names(Train) <- NULL
row.names(Test) <- NULL

XTest = Test[,1:19]
YTest = Test$Churn

XTrain = Train[,1:19]
YTrain = Train$Churn

row.names(XTest) <- NULL
row.names(YTest) <- NULL
row.names(XTrain) <- NULL
row.names(YTrain) <- NULL


################################ Decision Tree
DT.model.rpart <- rpart(Churn ~., data = Train, method = "class")
rpart.plot(DT.model.rpart)

DT_model_ctree<- party::ctree(Churn ~ ., Train)
plot(DT_model_ctree)

# Decision Tree Confusion matrix and accuracy rpart
DT_Ypred_prob <- predict(DT.model.rpart, Test)
DT_Ypred <- ifelse(DT_Ypred_prob[,2] > 0.5,"1","0")
DT_Ypred = as.factor(DT_Ypred)
confusionMatrix(DT_Ypred, Test$Churn, mode = "everything")

# Decision Tree Confusion matrix and accuracy ctree
DT_Ypred_ctree <- predict(DT_model_ctree, Test)
confusionMatrix(DT_Ypred_ctree, Test$Churn, mode = "everything")

# try Different splitting strategies and cross-validation
ctrl <- trainControl(method = "cv", number = 10)
dtree_fit_gini <- caret::train(Churn~., data = Train, method = "rpart", parms = list(split = "gini"), trControl = ctrl, tuneLength = 10)
dtree_fit_information <- caret::train(Churn~., data = Train, method = "rpart", parms = list(split = "information"), trControl = ctrl, tuneLength = 10)

print(dtree_fit_gini)
print(dtree_fit_information)

dtree_fit_gini$finalModel
prp(dtree_fit_gini$finalModel, box.palette = "Reds", tweak = 1.5)

dtree_fit_information$finalModel
prp(dtree_fit_information$finalModel, box.palette = "Reds", tweak = 1.5)

dtree_fit_gini$resample
dtree_fit_information$resample

# Check accuracy for gini
test_pred_gini <- predict(dtree_fit_gini, newdata = Test)
confusionMatrix(test_pred_gini, Test$Churn , mode = "everything")

# Check accuracy for information gain
test_pred_info <- predict(dtree_fit_information, newdata = Test)
confusionMatrix(test_pred_info, Test$Churn, mode = "everything") 

######## prune
printcp(DT.model.rpart)
Best_cp <- DT.model.rpart$cptable[which.min(DT.model.rpart$cptable[,"xerror"]),"CP"]
pruned <- prune(DT.model.rpart, cp = Best_cp)

# Plot pruned tree
prp(pruned, faclen = 0, cex = 0.8, extra = 1)

# Check accuracy for pruned tree
test_pred_pruned_prob <- predict(pruned, newdata = Test)
test_pred_pruned <- ifelse(test_pred_pruned_prob[,2] > 0.5,"1","0")
test_pred_pruned = as.factor(test_pred_pruned)
confusionMatrix(test_pred_pruned, Test$Churn , mode = "everything")

################ XG_Boost
XG_model <- xgboost(data = as.matrix(sapply(Train[,1:19], as.numeric)), label = as.numeric(as.character(Train$Churn)), max.depth = 3, nrounds = 70, objective = "binary:logistic")

# Accuracy and confusion matrix
pred_XG_prob = predict(XG_model, as.matrix(sapply(Test[,1:19], as.numeric)))
XG_Ypred <- as.numeric(pred_XG_prob > 0.5)
XG_Ypred = as.factor(XG_Ypred)
confusionMatrix(XG_Ypred, Test$Churn, mode = "everything") 

############### Deep Neural network
tensorflow::set_random_seed(42)
DNN_model1 <- keras_model_sequential()

DNN_model1 %>%
  layer_dense(units = 50, input_shape = 19) %>%
  layer_dropout(rate=0.7)%>%
  layer_activation(activation = 'relu') %>%
  
  layer_dense(units = 30) %>%
  layer_dropout(rate=0.5)%>%
  layer_activation(activation = 'relu') %>%
  
  layer_dense(units = 1) %>%
  layer_activation(activation = 'sigmoid')

summary(DNN_model1)

get_config(DNN_model1)
get_layer(DNN_model1, index = 1)

DNN_model1$layers

DNN_model1$inputs
DNN_model1$outputs

# Compile the model
DNN_model1 %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = c('accuracy')
)

DNN_model1 %>% fit(as.matrix(sapply(Train[,1:19], as.numeric)), as.numeric(as.character(Train$Churn)), epochs = 150, batch_size = 100)

# Accuracy and confusion matrix
pred_DNN_prob1 = predict(DNN_model1, as.matrix(sapply(Test[,1:19], as.numeric)))
DNN_Ypred1 <- as.numeric(pred_DNN_prob1 > 0.5)
DNN_Ypred1 = as.factor(DNN_Ypred1)
confusionMatrix(DNN_Ypred1, Test$Churn, mode = "everything")


############### Deep Neural network model 2 
tensorflow::set_random_seed(42)
DNN_model2 <- keras_model_sequential()

DNN_model2 %>%
  layer_dense(units = 600, input_shape = 19) %>%
  layer_dropout(rate=0.5)%>%
  layer_activation(activation = 'relu') %>%
  
  layer_dense(units = 300) %>%
  layer_dropout(rate=0.4)%>%
  layer_activation(activation = 'relu') %>%
  
  layer_dense(units = 1) %>%
  layer_activation(activation = 'sigmoid')

summary(DNN_model2)

get_config(DNN_model2)
get_layer(DNN_model2, index = 1)

DNN_model2$layers

DNN_model2$inputs
DNN_model2$outputs

# Compile the model
DNN_model2 %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = c('accuracy')
)

DNN_model2 %>% fit(as.matrix(sapply(Train[,1:19], as.numeric)), as.numeric(as.character(Train$Churn)), epochs = 150, batch_size = 80)

# Accuracy and confusion matrix
pred_DNN_prob2 = predict(DNN_model2, as.matrix(sapply(Test[,1:19], as.numeric)))
DNN_Ypred2 <- as.numeric(pred_DNN_prob2 > 0.5)
DNN_Ypred2 = as.factor(DNN_Ypred2)
confusionMatrix(DNN_Ypred2, Test$Churn, mode = "everything")

############### Deep Neural network model 3
tensorflow::set_random_seed(42)
DNN_model3 <- keras_model_sequential()

DNN_model3 %>%
  layer_dense(units = 768, input_shape = 19) %>%
  layer_dropout(rate=0.3)%>%
  layer_activation(activation = 'tanh') %>%
  
  layer_dense(units = 640) %>%
  layer_dropout(rate=0.3)%>%
  layer_activation(activation = 'tanh') %>%
  
  layer_dense(units = 1) %>%
  layer_activation(activation = 'sigmoid')

summary(DNN_model3)

get_config(DNN_model3)
get_layer(DNN_model3, index = 1)
S
DNN_model3$layers

DNN_model3$inputs
DNN_model3$outputs

# Compile the model
DNN_model3 %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = c('accuracy')
)

DNN_model3 %>% fit(as.matrix(sapply(Train[,1:19], as.numeric)), as.numeric(as.character(Train$Churn)), epochs = 150, batch_size = 128)

# Accuracy and confusion matrix
pred_DNN_prob3 = predict(DNN_model3, as.matrix(sapply(Test[,1:19], as.numeric)))
DNN_Ypred3 <- as.numeric(pred_DNN_prob3 > 0.5)
DNN_Ypred3 = as.factor(DNN_Ypred3)
confusionMatrix(DNN_Ypred3, Test$Churn, mode = "everything")

######################### ROC
par(pty="s")

# Decision tree models
roc(Test$Churn, as.numeric(as.character(DT_Ypred)), plot=TRUE, legacy.axes=TRUE, percent=TRUE, xlab="False Positive Percentage", ylab="True Postive Percentage", col="#0083D8", lwd=4, print.auc=TRUE)
legend("bottomright", legend=c("DT_rpart"), col=c("#0083D8"), lwd=4)

roc(Test$Churn, as.numeric(as.character(DT_Ypred_ctree)), plot=TRUE, legacy.axes=TRUE, percent=TRUE, xlab="False Positive Percentage", ylab="True Postive Percentage", col="#048F58", lwd=4, print.auc=TRUE)
legend("bottomright", legend=c("DT_cTree"), col=c("#048F58"), lwd=4)

roc(Test$Churn, as.numeric(as.character(test_pred_gini)), plot=TRUE, legacy.axes=TRUE, percent=TRUE, xlab="False Positive Percentage", ylab="True Postive Percentage", col="#C700C7", lwd=4, print.auc=TRUE)
legend("bottomright", legend=c("DT_Gini"), col=c("#C700C7"), lwd=4)

roc(Test$Churn, as.numeric(as.character(test_pred_info)), plot=TRUE, legacy.axes=TRUE, percent=TRUE, xlab="False Positive Percentage", ylab="True Postive Percentage", col="#C73600", lwd=4, print.auc=TRUE)
legend("bottomright", legend=c("DT_Info"), col=c("#C73600"), lwd=4)

roc(Test$Churn, as.numeric(as.character(test_pred_pruned)), plot=TRUE, legacy.axes=TRUE, percent=TRUE, xlab="False Positive Percentage", ylab="True Postive Percentage", col="#FF5E76", lwd=4, print.auc=TRUE)
legend("bottomright", legend=c("DT_rpart_pruned"), col=c("#FF5E76"), lwd=4)


# XG_Boost Model
roc(Test$Churn, as.numeric(as.character(XG_Ypred)), plot=TRUE, legacy.axes=TRUE, percent=TRUE, xlab="False Positive Percentage", ylab="True Postive Percentage", col="#4B0071", lwd=4, print.auc=TRUE)
legend("bottomright", legend=c("XG_Boost"), col=c("#4B0071"), lwd=4)


# DNN Models
roc(Test$Churn, as.numeric(as.character(DNN_Ypred1)), plot=TRUE, legacy.axes=TRUE, percent=TRUE, xlab="False Positive Percentage", ylab="True Postive Percentage", col="#5CC27E", lwd=4, print.auc=TRUE)
legend("bottomright", legend=c("DNN_model 1"), col=c("#5CC27E"), lwd=4)

roc(Test$Churn, as.numeric(as.character(DNN_Ypred2)), plot=TRUE, legacy.axes=TRUE, percent=TRUE, xlab="False Positive Percentage", ylab="True Postive Percentage", col="#484EFF", lwd=4, print.auc=TRUE)
legend("bottomright", legend=c("DNN_model 2"), col=c("#484EFF"), lwd=4)

roc(Test$Churn, as.numeric(as.character(DNN_Ypred3)), plot=TRUE, legacy.axes=TRUE, percent=TRUE, xlab="False Positive Percentage", ylab="True Postive Percentage", col="#C3B15E", lwd=4, print.auc=TRUE)
legend("bottomright", legend=c("DNN_model 3"), col=c("#C3B15E"), lwd=4)




############################# Part B #############################
install.packages("arules")
library(arules)

# Read the transactions.csv Dataset
Transactions_Data = read.csv('transactions.csv', header = FALSE)

# Create the sparse matrix
Transactions_Data = read.transactions('transactions.csv', sep=',', rm.duplicates = TRUE)

# information about the data
summary(Transactions_Data)

# Plot of top 10 transactions
itemFrequencyPlot(Transactions_Data ,topN = 10)

# Create the model
model = apriori(data = Transactions_Data , parameter = list(support = 0.002 ,confidence = 0.20, maxlen = 3))
model1 = apriori(data = Transactions_Data , parameter = list(support = 0.002 ,confidence = 0.20, maxlen = 2))

# Displaying rules sorted by descending lift value
inspect(sort(model, by = 'lift')[1:10])
inspect(sort(model1, by = 'lift')[1:10])

# Displaying rules sorted by descending support value
inspect(sort(model, by = 'support')[1:10])
inspect(sort(model1, by = 'support')[1:10])

# Displaying rules sorted by descending confidence value
inspect(sort(model, by = 'confidence')[1:10])
inspect(sort(model1, by = 'confidence')[1:10])
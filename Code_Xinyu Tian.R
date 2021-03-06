# APS Failure in Scania Trucks
# Xinyu Tian | xt5@illinois.edu

source("https://bioconductor.org/biocLite.R")
biocLite("pcaMethods")
library(pcaMethods)

library(data.table)
library(dplyr)
library(ranger)
library(caret)
library(unbalanced)
library(naivebayes)
library(rpart)
library(mice)
library(ggplot2)
library(VIM)
library(pROC)

### Preparation ###
getwd()
#setwd("D:/Datasets")

# read the dataset
tra <- fread("aps_failure_training_set.csv", na.strings = "NA", strip.white = TRUE)
tes <- fread("aps_failure_test_set.csv", na.strings = "NA", strip.white = TRUE)

# map class value ["neg", "pos"] to [0,1]
tra$class[tra$class == "neg"] <- 0
tra$class[tra$class == "pos"] <- 1

tes$class[tes$class == "neg"] <- 0
tes$class[tes$class == "pos"] <- 1

# visualize class value distributions
qplot(as.factor(tra$class), xlab = "class")

# convert every attribute into numeric type
tra <- tra %>% mutate_if(sapply(tra, is.character), as.numeric)
tes <- tes %>% mutate_if(sapply(tes, is.character), as.numeric)

# slice the features/predictors
tra_fea <- tra[,2:ncol(tra)]
tes_fea <- tes[,2:ncol(tes)]

### Heuristic Feature Selection ###
# remove columns where over 60% of instances are missing
col_miss <- lapply(tra_fea, function(col){sum(is.na(col))/length(col)})
tra_fea <- tra_fea[, !(names(tra_fea) %in% names(col_miss[lapply(col_miss, function(x) x) > 0.6]))]

# remove columns where over 90% of instance values are zero
col_zero <- lapply(tra_fea, function(col){length(which(col==0))/length(col)})
tra_fea <- tra_fea[, !(names(tra_fea) %in% names(col_zero[lapply(col_zero, function(x) x) > 0.9]))]

# remove columns where the standard derivation is zero
col_std <- lapply(tra_fea, function(col){sd(col, na.rm = TRUE)})
tra_fea <- tra_fea[, !(names(tra_fea) %in% names(col_std[lapply(col_std, function(x) x) == 0]))]

# also apply to test set
tes_fea <- tes_fea[, names(tes_fea) %in% names(tra_fea)]

### Multicollinearity Detection ###
# PCA attempts - select Bayesian PCA which tolerates missing data
tra_pca <- pca(tra_fea, method="bpca")
tes_pca <- pca(tes_fea, method="bpca")

# visualize PCA
plotPcs(tra_pca)

### Missing Data Imputation ###
# visualize the missing value patterns
miss <- aggr(tra_fea)
attach(tra_fea)
plot(tra_fea, sortVars = TRUE, sortCombs = TRUE, axes = TRUE,  combined = FALSE, labels= TRUE)

# fill NA by Bayesian PCA
tra_mi <- as.data.frame(cbind(tra$class, completeObs(tra_pca)))
colnames(tra_mi)[1] <- 'class'
summary(tra_mi)

tes_mi <- as.data.frame(cbind(tes$class, completeObs(tes_pca)))
colnames(tes_mi)[1] <- 'class'
summary(tes_mi)

### Model Fitting ###
# Model Fitting and Classification Prediction - Naive Bayes
nb_model <- naive_bayes(as.logical(class) ~ ., data = tra_mi)
nb_prediction <- predict(nb_model, tes_mi)
nb_prediction <- ifelse(nb_prediction == TRUE, 1, 0)

# Model Fitting and Classification Prediction - Random Forest with SMOTE Resampling
# try Random Forest on the original training set
rfgrid <- expand.grid(
  .mtry = 3:15,
  .splitrule = "gini",
  .min.node.size = 1
)

rf_model <- train(as.factor(class)~., data = tra_mi, method = "ranger", 
                     trControl = trainControl(method="cv", number = 5, verboseIter = TRUE),
                     tuneGrid = rfgrid, metric = "Kappa", 
                     num.trees = 500,
                     importance = "impurity")
rf_prediction <- predict(rf_model, newdata = tes_mi, type = "raw")

# SMOTE boosting
tra_SMOTE <- ubSMOTE(tra_mi[,2:ncol(tra_mi)], as.factor(tra_mi$class), 
                     perc.over = 200, k = 5, perc.under = 200, verbose = TRUE)
tra_bl <- cbind(tra_SMOTE$Y, tra_SMOTE$X)
colnames(tra_bl)[1] <- 'class'

# Random Forest with SMOTE
rfsgrid <- expand.grid(
  .mtry = 3:12,
  .splitrule = "gini",
  .min.node.size = 1
)

rfs_model <- train(as.factor(class)~., data = tra_bl, method = "ranger", 
                     trControl = trainControl(method="cv", number = 5, verboseIter = TRUE),
                     tuneGrid = rfsgrid, metric = "Kappa", 
                     num.trees = 500,
                     importance = "impurity")
rfs_prediction <- predict(rfs_model, tes_mi)

# feature selection by importance
rfs_imp <- varImp(rfs_model)

# visualize variable importance
plot(rfs_imp, top = 12)

rfs4grid <- expand.grid(
  .mtry = 4,
  .splitrule = "gini",
  .min.node.size = 1
)

rfs4_model <- train(as.factor(class)~., data = tra_bl, method = "ranger", 
                   trControl = trainControl(method="cv", number = 5, verboseIter = TRUE),
                   tuneGrid = rfs4grid, metric = "Kappa", 
                   num.trees = 500)
rfs4_prediction <- predict(rfs4_model,tes_mi)

# Model Fitting and Classification Prediction - Cost Sensitive CART
cc_model <- rpart(as.factor(class)~., data = tra_mi, method="class",  
                  parms = list(split = 'gini', loss = matrix(c(0,500,10,0))))
cc_prediction <- predict(cc_model, tes_mi, type = "class")

### Result Comparison ###
# get all confusion matrixes
nb_result <- confusionMatrix(nb_prediction, tes_mi$class, positive = "1")
rf_result <- confusionMatrix(rf_prediction, tes_mi$class, positive = "1")
rfs_result <- confusionMatrix(rfs_prediction, tes_mi$class, positive = "1")
rfs4_result <- confusionMatrix(rfs4_prediction, tes_mi$class, positive = "1")
cc_result <- confusionMatrix(cc_prediction, tes_mi$class, positive = "1")

# visualize ROC curve
nb_ROC <- roc(tes_mi$class, as.numeric(nb_prediction))
nb_ROC <- smooth(nb_ROC, method = "fitdistr")
plot(nb_ROC, add = FALSE, col = "green", xlim=c(1, 0), ylim = c(0,1))
rf_ROC <- roc(tes_mi$class, as.numeric(as.vector(rfs4_prediction)))
rf_ROC <- smooth(rf_ROC, method = "fitdistr")
plot(rf_ROC, add = TRUE, col = "blue")
cc_ROC <- roc(tes_mi$class, as.numeric(as.vector(cc_prediction)))
cc_ROC <- smooth(cc_ROC, method = "fitdistr")
plot(cc_ROC, add = TRUE, col = "magenta")
legend("bottomright", legend=c("NB", "RF", "CC"),
       col=c("green", "blue", "magenta"), lwd = 0)

# calculate total cost
total_cost <- function(result){
  costs <- matrix(c(0, 10, 500, 0), 2)
  return(sum(result$table * costs))
}

nb_tc <- total_cost(nb_result)
rfs_tc <- total_cost(rfs_result)
cc_tc <- total_cost(cc_result)

tc <- as.data.frame(cbind(c("Naive Bayes", "Random Forest with SMOTE Boosting (dim = 4)", "Cost Sensitive CART"), 
             c(nb_tc, rfs4_tc, cc_tc),
             c(as.numeric(nb_result$overall['Accuracy']), as.numeric(rfs4_result$overall['Accuracy']), as.numeric(cc_result$overall['Accuracy']))))
colnames(tc) <- c("Model","Total_Cost","Accuracy")
arrange(tc, Total_Cost)
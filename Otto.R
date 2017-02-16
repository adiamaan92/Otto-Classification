library(Hmisc)
library(caret)
library(magrittr)
library(doParallel)
library(randomForest)
library(Boruta)
library(MASS)


# 93 features and a target variable with 9 classess
# seems no missing values. Al predictors appears to be mostly categorical
# All 9 classes available in the train data set

train <- read.csv("train.csv", header = TRUE)
test <- read.csv("test.csv", header = TRUE)
# removing id column
train <- train[,-c(1)]

# log loss evaluation metrics to be used on all models

LogLoss <- function(actual, predicted, eps=0.00001) {
  predicted <- pmin(pmax(predicted, eps), 1-eps)
  -1/length(actual)*(sum(actual*log(predicted)+(1-actual)*log(1-predicted)))
}

#
# correlation between predictors

cr <- train
correlation.matrix <- data.frame("attribute1" = character(),"attribute2" = character(),"correlation"=numeric())
for (i in 1:ncol(cr))
{
  for (j in 1:ncol(cr))
  {
    correlation.matrix <- rbind(correlation.matrix,data.frame("attribute1" = colnames(cr)[i],"attribute2" = colnames(cr)[j],"correlation" = rcorr(cr[[i]],cr[[j]])$r[2]))
  }
}



# negatively correlated top 5  - 14,40,25,15 and 88. All these with the target variable
# positively correlated top 5 - 24,36,20,69,8. All these with the target variable

# highly correlated pairs other than with the target variables. top - 5 - 39,45 | 3,46 | 15,72 | 30,84 | 9,64

# Data preprocessing

summary(train)

# most of the features appears to be unbalanced

levels(train$target)

# initial modelling
# lets try logistic regression.

# intial thought to make use of the entire data with cross validation instead of splitting the test data set again.

nzv <- nearZeroVar(train)

# removing near zero variance predictors

train_nzv <- train[,-c(nzv)]
levels(train_nzv$target) <- c("Class_1","Class_2","Class_3","Class_4","Class_5","Class_6","Class_7","Class_8","Class_9")
# train_target <- train_nzv[,81]
# train_nzv <- train_nzv[,-c(81)]

# resampling - bootstrap since we are more focused in choosing between models as opposed to getting the best indicator of preformance, boot strap can be used since it has  avery low variance
# Logistic regression

# Penalized Multinomial Regression

bc <- train_nzv
transform.data <- data.frame("attribute"= character(),"skewness" = numeric(),"lambda" = numeric())
for (i in 1:ncol(bc))
{
  temp <- BoxCoxTrans(bc[[i]])
  transform.data <- rbind(transform.data,data.frame("attribute"=colnames(bc)[i],"skewness"=temp$skewness,"lambda"=temp$lambda))
}

# data appears to be highly skewed

Prep <- preProcess(train_nzv[,-c(ncol(train_nzv))],method = c("center","scale"))
train_nzv <- predict(Prep,train_nzv[,-c(ncol(train_nzv))])


nameLastCol <- names(train)[ncol(train)]
y <- train[, nameLastCol][[1]] %>% gsub('Class_','',.) %>% {as.integer(.) -1}
y <- gsub('Class_','',train$target) %>% as.numeric
y <- as.factor(y)

train_nzv$target <- y
hist(train_nzv[[1]])

BoxCoxTrans(train[[1]])$skewness


Partition_train <- createDataPartition(y=train_nzv$target,p=0.75,list = FALSE)

true_train <- train_nzv[Partition_train,]
true_test <- train_nzv[-Partition_train,]

levels(true_train$target) <- c("Class_1","Class_2","Class_3","Class_4","Class_5","Class_6","Class_7","Class_8","Class_9")

prc <- prcomp(true_train[,-c(81)])

summary(prc$rotation[,1])

# parallel random forest

# call parallel processing
cl <- makeCluster(3)
registerDoParallel(cl)

# train a random forest on the internal training set
set.seed(12631)

# with 100 trees runs 5 to 6 minutes while run on a 7 core in parallel

rf = foreach(ntree=rep(100,7), .combine=combine,
             .multicombine=TRUE, .packages="randomForest") %dopar% {
               randomForest(x=true_train[ ,-c(81)], y=true_train[ ,c(81)],
                            data=true_train, ntree=ntree, do.trace=1, importance=TRUE,
                            replace=TRUE, forest=TRUE)
             }

summary(rf)

# stop 7-core parallel processing
stopCluster(cl)
registerDoSEQ()

rf_pred <- predict(rf,true_test[,-c(81)],type = "prob")

# let's now see the performance

test_output <- dummyVars(~ target,data = true_test, levelsOnly = TRUE)
test_target_dv <- predict(test_output,true_test)

LogLoss(test_target_dv,rf_pred)

rf_notprob <- predict(rf,true_test[,-c(81)])
levels(rf_notprob) <- c("Class_1","Class_2","Class_3","Class_4","Class_5","Class_6","Class_7","Class_8","Class_9")
confusionMatrix(rf_notprob,true_test$target)

# we will use this log loss to finalize our best model.

# lets try another model

# Gradient Boosted trees

cl <- makeCluster(7)
registerDoParallel(cl)

# runs for 10 minutes for 10 fold cross validation

levels(true_train$target) <- c("Class_1","Class_2","Class_3","Class_4","Class_5","Class_6","Class_7","Class_8","Class_9")
ctrl <- trainControl(method = "cv", number = 10, classProbs = TRUE, summaryFunction = multiClassSummary)
bt_fit <- train(target~., true_train, trControl = ctrl, method = "gbm", metric = "logLoss")

bt_fit

# The final values used for the model were n.trees = 150, interaction.depth = 3, shrinkage = 0.1 and n.minobsinnode = 10.
stopCluster(cl)
registerDoSEQ()

bt_predict <- predict(bt_fit,true_test[,-c(81)])
bt_prob <- predict(bt_fit,true_test[,-c(81)], type= "prob")

levels(true_test$target) <-  c("Class_1","Class_2","Class_3","Class_4","Class_5","Class_6","Class_7","Class_8","Class_9")
levels(true_train$target) <-  c("Class_1","Class_2","Class_3","Class_4","Class_5","Class_6","Class_7","Class_8","Class_9")
confusionMatrix(bt_predict,true_test$target)
LogLoss(test_target_dv,as.matrix(bt_prob))


# needs 37 PC Axes to represent 75 percent of the variation

# neural network
# Runs for about 5 minutes in a neural network model with 10 fold cross validation

cl <- makeCluster(7)
registerDoParallel(cl)

ctrl <- trainControl(method = "cv", number = 10, classProbs = TRUE, summaryFunction = multiClassSummary)
nn_fit <- train(target~.,true_train, trControl = ctrl, method = "nnet", metric = "logLoss")


stopCluster(cl)
registerDoSEQ()

sa <- stepAIC(nn_fit,direction = "both")

nn_predict <- predict(nn_fit,true_test[,-c(81)] )
nn_prob <- predict(nn_fit,true_test[,-c(81)], type = "prob")

after <- confusionMatrix(nn_predict, true_test$target)
after_ll <- LogLoss(test_target_dv, as.matrix(nn_prob))


# taking too long, more than 1 hour for 10 fold cross validation

cl <- makeCluster(7)
registerDoParallel(cl)


ctrl <- trainControl(method = "cv", number = 10, classProbs = TRUE, summaryFunction = multiClassSummary)
svm_fit <- train(target~.,true_train, trControl = ctrl, method = "svmRadial", metric = "logLoss")


stopCluster(cl)
registerDoSEQ()


# lets train our model in random forest with no feature engineering to set a bench mark
# repeated cv with 3 repeats and 10 fold cross validation taking too long.

cl <- makeCluster(7)
registerDoParallel(cl)

ctrl <- trainControl(method = "cv", number = 10, classProbs = TRUE, summaryFunction = multiClassSummary)
rf_full_fit <- train(target~.,train_nzv, trControl = ctrl, method = "rf", metric = "logLoss",ntree = 100)


stopCluster(cl)
registerDoSEQ()

# feature construction
# a feature for row sum
row_sum <- list(0)
for (i in 1:nrow(train_nzv))
{
  row_sum[i] <-  sum(as.numeric(train_nzv[i,-c(ncol(train_nzv))]))
}

# a feature for row variance
row_var <- list(0)
for (i in 1:nrow(train_nzv))
{
  row_var[i] <-  var(as.integer(train_nzv[i,-c(ncol(train_nzv))]))
}

# a feature for counting non zero predictors
row_nonzero <- list(0)
for (i in 1:nrow(train_nzv))
{
  cnt = 0
  for (j in 1:(ncol(train_nzv)-1))
  {
    if (train_nzv[i,j] == 0)
    {
      cnt = cnt+1
    }
  }
  row_nonzero[i] <- cnt
}

# adding them all to the data frame

train_nzv$row_sum <- as.numeric(row_sum)
train_nzv$row_var <- as.numeric(row_var)
train_nzv$row_nonzero <- as.numeric(row_nonzero)

set.seed(123)


cl <- makeCluster(7)
registerDoParallel(cl)
rfProfile <- rfe(train_nzv[,-81],train_nzv[,81], rfeControl = rfeControl(functions = rfFuncs,method= "boot", number = 10))

# Boruta wrapper around rf for feature selection

br <- Boruta(target~.,data = true_train, doTrace = 2, maxRuns = 2)





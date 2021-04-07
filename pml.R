# load data to memory
traindata <- read.csv('https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv', na.strings = c("", "NA"))
testdata <-read.csv('https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv', na.strings = c("", "NA"))
head(traindata)
head(testdata)
names(traindata) == names(testdata)

traindata = traindata[,-(1:7)]
testdata = testdata[,-(1:7)]
testdata = testdata[,-ncol(testdata)]
sum(colSums(is.na(traindata)) == 0)

badcol = c()
for (i in names(traindata)){
    if(mean(is.na(traindata[,i])) > .5){
        badcol = c(badcol,i)
    }
}
badcol
library(dplyr)
traindata = select(traindata, -c(badcol))
testdata = select(testdata, -badcol)

# split data into Training and testing data
library(caret)
inTrain = createDataPartition(y = traindata$classe, p = .7, list = FALSE)
training = traindata[inTrain,]
testing = traindata[-inTrain,]
dim(training)
dim(testing)

library(parallel)
library(doParallel)
cl <- makeCluster(detectCores() - 2)
registerDoParallel(cl)

ctrl = trainControl(method="cv", number=5, verboseIter=FALSE, allowParallel = T)

## Random Forest 
modrf = train(classe ~ ., data = training, method = 'rf', trControl = ctrl)
predrf = predict(modrf, testing)
confusionMatrix(predrf, factor(testing$classe))$overall["Accuracy"]
## Linear Discriminant Analysis
modlda = train(classe ~. , data = training, method = 'lda', trControl = ctrl)
predlda = predict(modlda, testing)
confusionMatrix(predlda, factor(testing$classe))$overall["Accuracy"]

## Recurisive Partitioning and Regression Trees
modrpart = train(classe ~., data = training, method = 'rpart', trControl = ctrl)
predrpart = predict(modrpart, testing)
confusionMatrix(predrpart, factor(testing$classe))$overall["Accuracy"]

## Generalized Boosted Model (GBM)
modgbm = train(classe ~., data = training, method = 'gbm', trControl = ctrl, verbose = FALSE)
predgbm = predict(modgbm, testing)
confusionMatrix(predgbm, factor(testing$classe))$overall["Accuracy"]

# final predict
finalpred = predict(modrf, newdata = testdata)
finalpred

# close multicores
stopCluster(cl)
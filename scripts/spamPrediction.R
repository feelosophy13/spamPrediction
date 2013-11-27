###### Setting up working directory
getwd()
setwd('/Users/hawooksong/Desktop/programming_projects/spamPrediction')

###### Updating packages
update.packages()

###### Loading data
# install.packages('psych')
# install.packages('kernlab')
library(psych)
library(kernlab)
data(spam)

###### Brief overview of data
?spam
names(spam)
head(spam)
dim(spam)
nRow <- dim(spam)[1]
nCol <- dim(spam)[2]

###### Adding a new column (for later)
table(spam$type)
table(as.integer(spam$type))
### nonspam is coded as 1
### spam is coded as 2
spam$typeNumeric <- as.integer(spam$type) - 1
### nonspam as 0
### spam as 1

###### Splitting training and testing set
set.seed(13)
trainingSetCond <- rbinom(nRow, 1, 0.5)
trainingSet <- spam[trainingSetCond == 1, ]
testSet <- spam[trainingSetCond == 0, ]

###### Making sure the above split was successful
table(trainingSet$type)
table(testSet$type)
table(trainingSet$type)[1] + table(trainingSet$type)[2] + table(testSet$type)[1] + table(testSet$type)[2]

###### Exploratory analysis on training set
names(trainingSet)
describe(trainingSet)
plot(trainingSet$type, trainingSet$make)
plot(trainingSet$type, trainingSet$money)
plot(trainingSet$type, trainingSet$remove)
plot(trainingSet$type, trainingSet$credit)
plot(trainingSet$type, trainingSet$business)
plot(trainingSet$type, trainingSet$charDollar)

###### Creating models
model0 <- glm(type ~ charDollar, data=trainingSet, family='binomial')
summary(model0)
confint(model0)

model1 <- glm(type ~ money, data=trainingSet, family='binomial')
summary(model1)

model2 <- glm(type ~ charDollar + money, data=trainingSet, family='binomial')
summary(model2)

###### Testing accuracy on the training set
modelPredictionPercentage <- predict(model2, type='response')
head(modelPredictionPercentage)

###### Choosing a cutoff (re-substitution)
cutoff <- seq(0, 1, length=10) 
errorsByCutoff <- rep(NA, 10)
for (i in 1:length(cutoff)) {
  spamPrediction <- as.integer(modelPredictionPercentage > cutoff[i])
  error <- sum(spamPrediction != trainingSet$typeNumeric)
  errorsByCutoff[i] <- error
}
plot(cutoff, errorsByCutoff, xlab='Cutoff', ylab='Error count')
### when cutoff is 0.3 (NOT 0.5), the prediction is most accurate

modelPrediction <- modelPredictionPercentage > 0.3
for (i in 1:length(modelPrediction)) {
  if (modelPrediction[i] == T) {
    modelPrediction[i] = 'spam'
  }
  else {
    modelPrediction[i] = 'nonspam'
  }
}
table(modelPrediction)

###### Creating a "confusion matrix"
confusionMatrix <- table(observed = trainingSet$type, predict = modelPrediction)  
confusionMatrix

###### Calculating the error rate and total number of errors
##### Type I error rate (failing to detect spam)
##### Type II error rate (miscategorizing nonspam as spam)
totalCases <- confusionMatrix[1] + confusionMatrix[2] + confusionMatrix[3] + confusionMatrix[4]
totalCases
totalErrors <- confusionMatrix[2, 1] + confusionMatrix[1, 2]
totalErrors 
type1ErrorRate <- confusionMatrix[2, 1] / totalCases
type1ErrorRate
type2ErrorRate <- confusionMatrix[1, 2] / totalCases
type2ErrorRate
totalErrorRate <- totalErrors / totalCases
totalErrorRate

###### Too large error rate above
###### Looking for a better model
model3 <- glm(type ~ charDollar + money + remove + free 
              + charExclamation + receive + order + will + capitalTotal, 
              data=trainingSet, family='binomial')
summary(model3)  # the word 'will' does not create stat. sig. reg. coef.

model4 <- glm(type ~ charDollar + money + remove + free 
              + charExclamation + receive + order + capitalTotal, 
              data=trainingSet, family='binomial')
summary(model4)  ## lower residual deviance than model2 (good!)

###### Testing accuracy on the training set
modelPredictionPercentage <- predict(model4, type='response')
head(modelPredictionPercentage)

###### Choosing a cutoff (re-substitution)
cutoff <- seq(0, 1, length=10) 
errorsByCutoff <- rep(NA, 10)
for (i in 1:length(cutoff)) {
  spamPrediction <- as.integer(modelPredictionPercentage > cutoff[i])
  error <- sum(spamPrediction != trainingSet$typeNumeric)
  errorsByCutoff[i] <- error
}
plot(cutoff, errorsByCutoff, xlab='Cutoff', ylab='Error count')
### when cutoff is 0.35 (again, NOT 0.5), the prediction is most accurate

modelPrediction <- modelPredictionPercentage > 0.35
for (i in 1:length(modelPrediction)) {
  if (modelPrediction[i] == T) {
    modelPrediction[i] = 'spam'
  }
  else {
    modelPrediction[i] = 'nonspam'
  }
}
table(modelPrediction)

###### Creating a "confusion matrix"
confusionMatrix <- table(observed = trainingSet$type, predict = modelPrediction)  
confusionMatrix

###### Calculating the error rate and total number of errors
##### Type I error rate (failing to detect spam)
##### Type II error rate (miscategorizing nonspam as spam)
totalCases <- confusionMatrix[1] + confusionMatrix[2] + confusionMatrix[3] + confusionMatrix[4]
totalCases
totalErrors <- confusionMatrix[2, 1] + confusionMatrix[1, 2]
totalErrors 
type1ErrorRate <- confusionMatrix[2, 1] / totalCases
type1ErrorRate
type2ErrorRate <- confusionMatrix[1, 2] / totalCases
type2ErrorRate
totalErrorRate <- totalErrors / totalCases
totalErrorRate
### error rate of 12% on training set
### any better way to reduce the error rate?

###### Letting R pick its own predictor variables 
###### Automatic variable selection
model5 <- glm(type ~ . - type - typeNumeric, 
              data=trainingSet, family='binomial')
summary(model5)  # thank you; you make my life easy!

model6 <- glm(type ~ our + over + remove + internet + order + addresses
              + free + business + your + num000 + hp + george + num650
              + meeting + re + edu + charExclamation + charDollar 
              + capitalAve + capitalTotal,
              data = trainingSet, family = 'binomial')
summary(model6)

###### How do we know which class our model is predicting?
###### That is, how do we know our model is predicting spam and not non-spam?

head(trainingSet$type)
head(as.numeric(trainingSet$type))
tail(trainingSet$type)
tail(as.numeric(trainingSet$type))

# 'spam' encodes 2 and 'nonspam' encodes 1
# glm(), or generalized linear model, predict 'spam' and not 'nonspam' because 'spam (2)' > 'nonspam (1)


###### Understanding our prediction model and its regression coefficients
# In generalized linear model, the regression coefficient for each variable signifies the 
# change in logit--ln(P / [1 - P])--for 1 unit change in the said variable across the average 
# values of other variables, where p is our probability of an event we are predicting. 
#
# P: probability of spam 
# 1 - P: probability of nonspam


###### Testing accuracy on the training set
modelPredictionPercentage <- predict(model6, type='response')
head(modelPredictionPercentage)


###### Choosing a cutoff (re-substitution)
cutoff <- seq(0, 1, length=10) 
errorsByCutoff <- rep(NA, 10)
for (i in 1:length(cutoff)) {
  spamPrediction <- as.integer(modelPredictionPercentage > cutoff[i])
  error <- sum(spamPrediction != trainingSet$typeNumeric)
  errorsByCutoff[i] <- error
}
plot(cutoff, errorsByCutoff, xlab='Cutoff', ylab='Error count',
     main='Number of prediction errors\n under various cutoff values', pch=20)
### when cutoff is 0.4 (again, NOT 0.5), the prediction is most accurate

# creating a png image of the plot
dev.copy(png, 'images/prediction_errors_vs_probability_cutoff_values.png')
dev.off()

# classifying anything above 0.4 as spam and less than 0.4 as nonspam
modelPrediction <- modelPredictionPercentage > 0.4
for (i in 1:length(modelPrediction)) {
  if (modelPrediction[i] == T) {
    modelPrediction[i] = 'spam'
  }
  else {
    modelPrediction[i] = 'nonspam'
  }
}
table(modelPrediction)

###### Creating a "confusion matrix"
confusionMatrix <- table(observed = trainingSet$type, predict = modelPrediction)  
confusionMatrix

###### Calculating the error rate and total number of errors
##### Type I error rate (failing to detect spam)
##### Type II error rate (miscategorizing nonspam as spam)
totalCases <- confusionMatrix[1] + confusionMatrix[2] + confusionMatrix[3] + confusionMatrix[4]
totalCases
totalErrors <- confusionMatrix[2, 1] + confusionMatrix[1, 2]
totalErrors 
type1ErrorRate <- confusionMatrix[1, 2] / totalCases
type1ErrorRate
type2ErrorRate <- confusionMatrix[2, 1] / totalCases
type2ErrorRate
totalErrorRate <- totalErrors / totalCases
totalErrorRate
### Error rate dropped from around 12% to 7.45%!

###### Testing the model on test data set
###### Testing accuracy on the training set
modelPredictionPercentage <- predict(model6, type='response', testSet)
head(modelPredictionPercentage)

###### Ideal cutoff was determined to be 0.38 from testing on the training set
modelPrediction <- modelPredictionPercentage > 0.4
for (i in 1:length(modelPrediction)) {
  if (modelPrediction[i] == T) {
    modelPrediction[i] = 'spam'
  }
  else {
    modelPrediction[i] = 'nonspam'
  }
}
head(modelPrediction)
table(modelPrediction)

###### Creating a "confusion matrix"
confusionMatrix <- table(observed = testSet$type, predict = modelPrediction)  
confusionMatrix

###### Calculating the error rate and total number of errors
##### Type I error rate (failing to detect spam)
##### Type II error rate (miscategorizing nonspam as spam)
totalCases <- confusionMatrix[1] + confusionMatrix[2] + confusionMatrix[3] + confusionMatrix[4]
totalCases
totalErrors <- confusionMatrix[2, 1] + confusionMatrix[1, 2]
totalErrors 
type1ErrorRate <- confusionMatrix[2, 1] / totalCases
type1ErrorRate
type2ErrorRate <- confusionMatrix[1, 2] / totalCases
type2ErrorRate
totalErrorRate <- totalErrors / totalCases
totalErrorRate
### Wow! Even lower, only 7.2% error rate on the testing set!


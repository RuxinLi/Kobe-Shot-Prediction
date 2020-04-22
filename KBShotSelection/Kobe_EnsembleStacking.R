# Wade Strain
# Machine Learning 2 - TP2
# April 19, 2020

rm(list=ls())
library(mlbench)
library(caret)
library(caretEnsemble)

# Load in data
data <- read.csv('data.csv')

# Data Cleaning
data$time_remaining <- data$minutes_remaining*60 + data$seconds_remaining

remove <- c('action_type','game_event_id','game_id','lat','loc_x','loc_y','lon','minutes_remaining','playoffs','season',
            'seconds_remaining','shot_zone_basic','shot_zone_range','team_id','team_name','game_date','matchup','opponent')
data <- data[ , -which(names(data) %in% remove)]

# convert factors to numeric
data$combined_shot_type <- as.numeric(data$combined_shot_type)
data$shot_zone_area <- as.numeric(data$shot_zone_area)
data$shot_type <- as.numeric(data$shot_type)

# Split Data
Train <- data[!is.na(data$shot_made_flag),]
test <- data[is.na(data$shot_made_flag),]

# remove shot_id feature; only need for submission
testID <- test$shot_id
Train <- Train[, -which(names(Train) %in% 'shot_id')]
test <- test[, -which(names(test) %in% 'shot_id')]

# Stacking Algorithm
control <- trainControl(method='cv', number=3, savePredictions='final', classProbs=TRUE)
algList <- c('glm','knn','rpart','svmRadial')
set.seed(2)
models <- caretList(shot_made_flag~., data=Train, trControl=control, methodList=algList)

results <- resamples(models)
summary(results)
modelCor(results)

stack.rf <- caretStack(models, method='rf', metric='LogLoss', trControl=control)
print(stack.rf)

test.x = test[, -which(names(test) %in% 'shot_made_flag')]
pred.y <- predict(stack.rf, newdata=test.x)

submission <- data.frame(shot_id=testID, shot_made_flag=pred.y)
write.csv(submission, 'Ensemble_Submission.csv', row.names=FALSE)

# Team 8
# ML2 - TP2
# XGBoost
# April 21, 2020

rm(list=ls())

# Load in data
data <- read.csv('data.csv')

# Data Cleaning
data$time_remaining <- data$minutes_remaining*60 + data$seconds_remaining

# Will not be using all the features
remove <- c('action_type','game_event_id','game_id','lat','loc_x','loc_y','lon','minutes_remaining','playoffs','season',
            'seconds_remaining','shot_zone_basic','shot_zone_range','team_id','team_name','game_date','matchup','opponent')
data <- data[ , -which(names(data) %in% remove)]

# convert factors to numeric XGBoost needs numeric dependant variable
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

# Spilt the dependant variable from the data
x_train <- Train[, -which(names(Train) %in% 'shot_made_flag')]
x_test <- test[, -which(names(test) %in% 'shot_made_flag')]
y_train <- Train$shot_made_flag

# XGBoost
library(xgboost)

# convert shot flag to numeric for classification
y_train <- as.numeric(y_train)


# prepare the matrices
xgb.train <- xgb.DMatrix(data=as.matrix(x_train), label=y_train)

# Set the parameters
params <- list(booster = 'gbtree',
               objective = 'binary:logistic',
               eval_metric = 'logloss',
               eta = 0.3,
               gamma = 0,
               max_depth = 6,
               min_child_weight = 1,
               subsample = 1,
               colsample_bytree = 1)

# calculate the best nround
xgbcv <- xgb.cv(params = params,
                data = xgb.train,
                nrounds = 100,
                nfold = 5,
                verbose = 1)

(best.nrounds <- which.min(as.matrix(xgbcv$evaluation_log[,4])))

# train xgboost model
xgb <- xgb.train(params = params,
                 data = xgb.train,
                 nrounds = best.nrounds,
                 watchlist = list(train = xgb.train),
                 verbose = 1)

# predict
xgb.test <- xgb.DMatrix(data=as.matrix(x_test))
pred.y <- predict(xgb, xgb.test)

# submission file
submission <- data.frame(shot_id=testID, shot_made_flag=pred.y)
write.csv(submission, 'XGBoost_Submission.csv', row.names=FALSE)

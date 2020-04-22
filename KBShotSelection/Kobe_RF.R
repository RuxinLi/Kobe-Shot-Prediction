rm(list=ls())

####################################################
####           Load required packages           ####
####################################################
installIfAbsentAndLoad <- function(neededVector) {
  for(thispackage in neededVector) {
    if( ! require(thispackage, character.only = T) )
    { install.packages(thispackage)}
    require(thispackage, character.only = T)
  }
}

needed <- c('xgboost',"ggplot",'mlbench','caret','caretEnsemble')  
installIfAbsentAndLoad(needed)

library(mlbench)
library(caret)
library(caretEnsemble)
## Load Data
data <- read.csv('data.csv')


# Data Cleaning
## Create Time Remaining
data$time_remaining <- data$minutes_remaining*60 + data$seconds_remaining


# convert factors to numeric
data$combined_shot_type <- as.numeric(data$combined_shot_type)
data$shot_zone_area <- as.numeric(data$shot_zone_area)
data$shot_type <- as.numeric(data$shot_type)
data$playoffs <- as.factor(data$playoffs)


## Preprocess Data
data$shot_made_flag <- as.factor(data$shot_made_flag)
data$shot_made_flag <- factor(data$shot_made_flag, levels = c('0','1'))

##show shot_made_flag distributiion
install.packages("ggplot2")
library("ggplot2")

ggplot(data = data[!is.na(data$shot_made_flag),], aes(x = loc_x, y = loc_y)) +
  geom_point(aes(color = shot_made_flag), alpha = 0.6, size = 0.5) + 
  theme(legend.position="none") +
  scale_color_brewer(palette = "Set1") +
  facet_grid(~ shot_made_flag) +
  coord_fixed() +
  labs(title = "            Shots Made Flag 0 (misses) vs. 1 (makes) ")

## Columnns to drop (cant take more than 53 factors and some are useles)
remove <- c('action_type','game_date', 'matchup','game_id','season','team_id','team_name', 'minutes_remaining', 'seconds_remaining','lat','lon','loc_x','loc_y')
data <- data[ , -which(names(data) %in% remove)]

# remove shot_id feature; only need for submission
train <- data[!is.na(data$shot_made_flag),]
test <- data[is.na(data$shot_made_flag),]
testID <- test$shot_id
train <- train[, -which(names(train) %in% 'shot_id')]
test <- test[, -which(names(test) %in% 'shot_id')]

#test <- test[, -which(names(test) %in% 'shot_made_flag')]


## Split Data

train.index <- sample(1:nrow(train), nrow(train)* .70)

x_train <- train[train.index,]
y_train <- x_train$shot_made_flag
x_test <- train[-train.index,]
y_test <- x_test$shot_made_flag
x_test <- x_test[, -which(names(x_test) %in% 'shot_made_flag')]


##### Random Forest Classifier Training #####
library(randomForest)
set.seed(2)


#Random Forest
rf.shots <- randomForest(shot_made_flag ~., data = x_train, mtry = 3, ntree = 500, importance = TRUE)
yhat.rf <- predict(rf.shots, newdata = x_test)

rf.table <- table(y_test, yhat.rf)
(rf.table)
rf.errorRate <- (rf.table[1,2] + rf.table[2,1]) / sum(rf.table)
(rf.errorRate)

## Importance
importance(rf.shots)
varImpPlot(rf.shots)

#Random Forest Loop
features <- ncol(test)
mtry.num = seq(3,features)
trees.num = seq(100,600,100)
error.rate = rep(0, length(mtry.num) * length(trees.num))
mt <- rep(0,length(mtry.num) * length(trees.num))
tr <- rep(0,length(mtry.num) * length(trees.num))

k <- 1
for (i in 1:length(mtry.num)){
  for (j in 1:length(trees.num)){
    rf.shots <- randomForest(shot_made_flag ~., data = x_train, mtry = i, ntree = j, importance = FALSE)
    yhat.rf <- predict(rf.shots, newdata = x_test)
    rf.table <- table(y_test, yhat.rf)
    rf.errorRate <- (rf.table[1,2] + rf.table[2,1]) / sum(rf.table)
    error.rate[k] <- rf.errorRate
    mt[k] <- mtry.num[i]
    tr[k] <- trees.num[j]
    print(k)
    k <- k + 1
  }
}

best.mod <- cbind(mt, tr, error.rate)
index <- which.min(error.rate)
best.mod[index,]
final_mtry <- best.mod[index,1]
final_trees <- best.mod[index,2]

########## Final Test ##########

rf.shots.final <- randomForest(shot_made_flag ~., data = train, mtry = final_mtry, ntree = final_trees, importance = TRUE)
yhat.rf.final <- predict(rf.shots.final, newdata = test, type = 'prob')

submission <- data.frame(shot_id = testID, shot_made_flag = yhat.rf.final[,1])

write.csv(submission, file ='submission.csv', row.names = FALSE)

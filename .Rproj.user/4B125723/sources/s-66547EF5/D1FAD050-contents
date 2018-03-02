library(here)
library(tidyverse)
library(caret)

here()

train <- read.csv(here("data","train.csv"), stringsAsFactors = F)
train <- train %>% select(-Id)

#Look at missing values for each column
map_df(train, ~sum(is.na(.))) %>% gather() %>% 
  filter(value > 0) %>%
  ggplot(.,aes(x = reorder(key,-value), y = value)) + 
  geom_bar(stat="identity") +
  labs(title = "Missing Values",x="") +
  theme(axis.text.x=element_text(angle=90,hjust=1,vjust=0.5))

#Replace missing values with "none" or "0" based on categorical vs numeric and combine back to create an imputed train dataframe
train_num <- train %>% select_if(is.numeric) 
train_cat <- train %>% select_if(is.character) 

train_num[is.na(train_num)] <- 0
train_cat[is.na(train_cat)] <- "None"
train_cat <- map_df(train_cat,as.factor)

train_pp <- cbind(train_cat,train_num)
rm(train_cat,train_num)

#define training folds and steps for modeling

set.seed(1108)
myFolds <- createFolds(train_pp$SalePrice,10)

myControl <- trainControl(
  verboseIter = T,
  savePredictions = T,
  index = myFolds
)

## Multiple Regression Models
lm <- train(
  SalePrice~.,
  data = train_pp,
  preProcess = c("nzv","center","scale"),
  metric = "RMSE",
  method = "lm",
  trControl = myControl
)

lm_pca <- train(
  SalePrice~.,
  data = train_pp,
  preProcess = c("zv","center","scale","pca"),
  metric = "RMSE",
  method = "lm",
  trControl = myControl
)


##GLMNET Model
glmnet <- train(
  SalePrice~.,
  data = train_pp,
  metric = "RMSE",
  method = "glmnet",
  trControl = myControl,
  tuneGrid = expand.grid(
    alpha = seq(0,1,.1),
    lambda = seq(1000,50000,100)
  )
)

##rpart model
rpart <- train(
  SalePrice~.,
  data = train_pp,
  metric = "RMSE",
  method = "rpart",
  trControl = myControl
)


##RF Models
rf <- train(
  SalePrice~.,
  data = train_pp,
  metric = "RMSE",
  method = "rf",
  trControl = myControl,
  tuneGrid = expand.grid(
    mtry = seq(100, 150, 5)
  )
)
min(rf$results$RMSE)

##XGBoost model
xgbTree <- train(
  SalePrice~.,
  data = train_pp,
  metric = "RMSE",
  method = "xgbTree",
  trControl = myControl,
  tuneGrid = expand.grid(
    nrounds= 500,
    max_depth = c(4,6,8),
    eta = c(0.1,0.2,0.3),
    gamma = 0,
    colsample_bytree = c(.5,.7),
    subsample = c(0.5,0.8),
    min_child_weight = 0
  )
)
min(xgbTree$results$RMSE)
##Compare all models to choose the best one
mod_list <- list(lm = lm,pca = lm_pca,glmnet = glmnet,rf = rf, XGBTree = xgbTree, rpart = rpart)

resamp <- resamples(mod_list)

bwplot(resamp,metric = "RMSE")
dotplot(resamp, metric = "RMSE")
parallelplot(resamp, metric = "RMSE")


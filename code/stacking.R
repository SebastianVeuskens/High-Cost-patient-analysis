################################
#### STACK MODEL ###############
################################
# Experimental file to see if stacking leads to reasonable results 

#### MODIFY ####
# Your working directory 
setwd('C:/Users/s.veuskens/Documents/Sebastian/Projekt Sebastian/modelling')
# Indicates whether to include High-Cost patients from the last year into analysis 
filter_hc <- FALSE 
# Whether you want to save your results (and overwrite the old results) or not
overwrite <- TRUE
# Number of folds to be used for cross-validation 
nfolds <- 5
#### MODIFY END ####

#######################
#### PRELIMINARIES ####
#######################

#### LIBRARIES & SOURCES ####

# INSTALL LIBRARIES 


# LOAD LIBRARIES & SOURCES


#### LOAD DATA ####
# Load the train & validate data sets for training & evaluating the model, respectively 
load(paste0('data/', ifelse(filter_hc, 'filtered/', 'original/'), 'train', '.Rdata'))
load(paste0('data/', ifelse(filter_hc, 'filtered/', 'original/'), 'validate', '.Rdata'))

# Start H2O package
h2o.init()

# Load train & validate data sets into H2O framework
train <- as.h2o(train)
validate <- as.h2o(validate)

# Load the model predictions on the train data set. These will be used for training. 
lr_predictions_train <- as.numeric(as.data.frame(h2o.predict(lr_model, train))$p1)
nn_predictions_train <- as.numeric(as.data.frame(h2o.predict(nn_model, train))$p1)
rf_predictions_train <- as.numeric(as.data.frame(h2o.predict(rf_model, train))$p1)
gbm_predictions_train <- as.numeric(as.data.frame(h2o.predict(gbm_model, train))$p1)

label_train <- as.factor(as.data.frame(train)$HC_Patient_Next_Year)

model_predictions_train <- data.frame(lr = lr_predictions_train
                                    , nn = nn_predictions_train
                                    , rf = rf_predictions_train
                                    , gbm = gbm_predictions_train
                                    , HC_Patient_Next_Year = label_train)
                                    
# Load model_predictions_train data set (training data set) into H2O framework
model_predictions_train <- as.h2o(model_predictions_train)

# Load the model predictions on the validate data set. These will be used for validation.  
lr_predictions_validate <- as.numeric(as.data.frame(h2o.predict(lr_model, validate))$p1)
nn_predictions_validate <- as.numeric(as.data.frame(h2o.predict(nn_model, validate))$p1)
rf_predictions_validate <- as.numeric(as.data.frame(h2o.predict(rf_model, validate))$p1)
gbm_predictions_validate <- as.numeric(as.data.frame(h2o.predict(gbm_model, validate))$p1)

label_validate <- as.factor(as.data.frame(validate)$HC_Patient_Next_Year)

model_predictions_validate <- data.frame(lr = lr_predictions_validate
                                       , nn = nn_predictions_validate
                                       , rf = rf_predictions_validate
                                       , gbm = gbm_predictions_validate
                                       , HC_Patient_Next_Year = label_validate)

# Load model_predictions_validate data set (validation data set) into H2O framework
model_predictions_validate <- as.h2o(model_predictions_validate)

#### STACK MODELS ####

# Train the logistic regression stacked model
st_lr_model <- h2o.glm(x               = 1:4, 
                       y               = 5,
                       training_frame  = model_predictions_train, 
                       seed            = 12345,
                       balance_classes = TRUE)

# Train the logistic regression stacked model
st_dt_model <- h2o.decision_tree(x               = 1:4, 
                                 y               = 5,
                                 training_frame  = model_predictions_train, 
                                 seed            = 12345)

# Evaluate the models
# TODO: Add automatic folder creation for stacking 
st_lr_filepath <- paste0('results/', ifelse(filter_hc, 'filtered/', 'original/'), 'stacking/logistic_regression')
st_lr_performance <- evaluate_model(st_lr_model, st_lr_filepath, overwrite, newdata=model_predictions_validate)

st_dt_filepath <- paste0('results/', ifelse(filter_hc, 'filtered/', 'original/'), 'stacking/decision_tree')
st_dt_performance <- evaluate_model(st_dt_model, st_dt_filepath, overwrite, newdata=model_predictions_validate)

index <- 18467
predictions[index, ]
model_predictions_validate[1, ]
length(validate)
dim(validate)
dim(train)
dim(predictions)
str(predictions)
tail(predictions)
str(probabilities)
any(is.na(probabilities))
which(is.na(probabilities))
model_predictions_validate[4793,]
h2o.predict(model, model_predictions_validate[4793,])

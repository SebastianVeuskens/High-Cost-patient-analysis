###################################
#### COMPARE BEST MODELS ##########
###################################
# File: 3_model_performance.R
# Author: Sebastian Benno Veuskens 
# Date: 2024-05-10
# Data: Use the train or train_validate data set for training and the validate or test data set for evaluation. 
#
# This script performs model training and evaluation.
# The best models for each model type are used according to the grid parameter search (see 2_model_tuning.R).
# Standard evaluation measures are used. 

#### MODIFY ####
# Your working directory 
setwd("C:/Users/s.veuskens/Documents/Sebastian/Projekt Sebastian/modelling")
# Indicates whether to include High-Cost patients from the last year into analysis 
filter_hc <- FALSE 
# Indicates whether to include as many High-Cost patients as not-High-Cost patients 
balance_hc <- FALSE 
# Number of folds to be used for cross-validation 
nfolds <- 5 
# The measure to order the models by, default is the area under the curve (AUC)
measure <- 'auc'
# Whether you want to save your results (and overwrite the old results) or not
overwrite <- TRUE
# Indicate whether to retrieve performance or validation (one of 'validation' or 'performance')
scheme <- 'performance'
#### MODIFY END ####

#######################
#### PRELIMINARIES ####
#######################

#### LIBRARIES & SOURCES ####

# INSTALL LIBRARIES 
# install.packages('h2o')
# install.packages('cvAUC')
# install.packages('rlist')

# LOAD LIBRARIES & SOURCES
library(h2o)            # The modelling framework 
library(cvAUC)          # For the Area Under the Curve (AUC) computation
source('code/utils.R')  # Auxiliary functions for simplicity and concise code 

#### LOAD DATA ####

# Indicate from which relative location to load & save from. Depends on user input. 
relative_dir <- paste0(ifelse(filter_hc, 'filtered/', 'complete/'), ifelse(balance_hc, 'balanced/', 'unbalanced/'))

if (scheme == 'performance') {
    load(paste0('data/', relative_dir, 'train',    '.Rdata'))
    load(paste0('data/', relative_dir, 'validate', '.Rdata'))
    train_data <- train
    test_data <- validate
} else if (scheme == 'validation') {
    load(paste0('data/', relative_dir, 'train_validate',    '.Rdata'))
    load(paste0('data/', relative_dir, 'test', '.Rdata'))
    train_data <- train_validate
    test_data <- test
}

# Start H2O package
h2o.init()

# Load data frames into H2O framework
train_data <- as.h2o(train_data)
test_data <- as.h2o(test_data)

#### CREATE FOLDER STRUCTURE ####
if (overwrite) {
    dir.create('results', showWarnings=FALSE)
    dir.create(paste0('results/', ifelse(filter_hc, 'filtered', 'complete')), showWarnings=FALSE)
    dir.create(paste0('results/', relative_dir), showWarnings=FALSE)
    if (scheme == 'performance') {
        dir.create(paste0('results/', relative_dir, 'model_performance'), showWarnings=FALSE)
    } else if (scheme == 'validation') {
        dir.create(paste0('results/', relative_dir, 'model_validation'), showWarnings=FALSE)
    }
}

##################################################################
###################### MODEL SELECTION ###########################
##################################################################

# Position of label and variables. Indicate where the features for prediction should start and end in the data.
label_pos <- 1 
first_val <- 3
last_val <- ncol(train_data)


#############################
#### LOGISTIC REGRESSION ####
#############################

# Measure the time for the whole training & validation process 
start_time <- Sys.time()

# Load the best parameters
lr_filename <- 'logistic_regression_best_parameters.RData'
lr_params <- list.load(paste0('results/', relative_dir, 'model_tuning/', lr_filename))
lr_best_params <- lr_params[[1]]

lr_indices <- match(strsplit(lr_best_params[['predictors']], ', ')[[1]], colnames(train_data))

# Train the model
lr_model <- h2o.glm(x                = lr_indices, 
                    y                = label_pos,
                    training_frame   = train_data,  
                    nfolds           = nfolds,
                    seed             = 12345,
                    calc_like        = TRUE,
                    compute_p_values = TRUE
                    )

# Evaluate the trained model
lr_filepath <- paste0('results/', relative_dir, 'model_performance/logistic_regression')
lr_performance <- evaluate_model(lr_model, lr_filepath, overwrite, newdata=test_data)

# Save the coefficients of the model 
save_lr_coefs(lr_model, lr_filepath)

# Get predictions of the model
lr_predictions <- h2o.predict(lr_model, newdata=test_data)

# Display the runtime
end_time <- Sys.time() 
lr_runtime <- difftime(end_time, start_time, units='mins')
print(paste0('Runtime logistic regression: ', round(lr_runtime, 2), ' minutes'))


########################
#### NEURAL NETWORK ####
########################

# Measure the time for the whole training & validation process 
start_time <- Sys.time()

# Load the best parameters
nn_filename <- 'neural_network_best_parameters.RData'
nn_params <- list.load(paste0('results/', relative_dir, 'model_tuning/', nn_filename))
nn_best_params <- nn_params[[1]]

# Train the model
nn_model <- h2o.deeplearning(x              = first_val:last_val, 
                             y              = label_pos,
                             training_frame = train_data, 
                             nfolds         = nfolds,
                             seed           = 12345,
                             activation     = nn_best_params[['activation']],
                             hidden         = as.numeric(nn_best_params[['hidden']]),
                             rate           = as.numeric(nn_best_params[['rate']]))

# Evaluate the trained model
nn_filepath <- paste0('results/', relative_dir, 'model_performance/neural_network')
nn_performance <- evaluate_model(nn_model, nn_filepath, overwrite, newdata=test_data)

# Get predictions of the model
nn_predictions <- h2o.predict(nn_model, newdata=test_data)

# Display the runtime
end_time <- Sys.time() 
nn_runtime <- difftime(end_time, start_time, units='mins')
print(paste0('Runtime neural network: ', round(nn_runtime, 2), ' minutes'))


#######################
#### RANDOM FOREST ####
#######################

# Measure the time for the whole training & validation process 
start_time <- Sys.time()

# Load the best parameters
rf_filename <- 'random_forest_best_parameters.RData'
rf_params <- list.load(paste0('results/', relative_dir, 'model_tuning/', rf_filename))
rf_best_params <- rf_params[[1]]

# Train the model
rf_model <- h2o.randomForest(x              = first_val:last_val, 
                             y              = label_pos,
                             training_frame = train_data, 
                             nfolds         = nfolds,
                             seed           = 12345,
                             ntrees         = as.numeric(rf_best_params[['ntrees']]),
                             mtries         = as.numeric(rf_best_params[['mtries']]))

# Evaluate the trained model
rf_filepath <- paste0('results/', relative_dir, 'model_performance/random_forest')
rf_performance <- evaluate_model(rf_model, rf_filepath, overwrite, newdata=test_data)

# Get predictions of the model
rf_predictions <- h2o.predict(rf_model, newdata=test_data)

# Display the runtime
end_time <- Sys.time() 
rf_runtime <- difftime(end_time, start_time, units='mins')
print(paste0('Runtime random forest: ', round(rf_runtime, 2), ' minutes'))


###################################
#### GRADIENT BOOSTING MACHINE #### 
###################################

# Measure the time for the whole training & validation process 
start_time <- Sys.time()

# Load the best parameters
gbm_filename <- 'gradient_boosting_machine_best_parameters.RData'
gbm_params <- list.load(paste0('results/', relative_dir, 'model_tuning/', gbm_filename))
gbm_best_params <- gbm_params[[1]]

# Train the model
gbm_model <- h2o.gbm(x              = first_val:last_val, 
                     y              = label_pos,
                     training_frame = train_data, 
                     nfolds         = nfolds,
                     seed           = 12345,
                     ntrees         = as.numeric(gbm_best_params[['ntrees']]),
                     max_depth      = as.numeric(gbm_best_params[['max_depth']]))

# Evaluate the trained model
gbm_filepath <- paste0('results/', relative_dir, 'model_performance/gradient_boosting_machine')
gbm_performance <- evaluate_model(gbm_model, gbm_filepath, overwrite, newdata=test_data)

# Get predictions of the model
gbm_predictions <- h2o.predict(gbm_model, newdata=test_data)

# Display the runtime
end_time <- Sys.time() 
gbm_runtime <- difftime(end_time, start_time, units='mins')
print(paste0('Runtime gradient boosting machine: ', round(gbm_runtime, 2), ' minutes'))

#  Save the runtimes
models <- c('logistic regression', 'neural network', 'random forest', 'gradient boosting machine')
runtimes_in_minutes <- round(c(lr_runtime, nn_runtime, rf_runtime, gbm_runtime), 4)
results <- data.frame(models, runtimes_in_minutes)
rt_filepath <- paste0('results/', relative_dir, 'model_performance/runtimes_model_performance.csv')
if (overwrite) write.csv(results, rt_filepath)

#################################
#### DECISION CURVE ANALYSIS ####
#################################
models_predictions <- list(lr_predictions$p1, nn_predictions$p1, rf_predictions$p1, gbm_predictions$p1)
model_names <- list('Logistic Regression', 'Neural Network', 'Random Forest', 'Gradient Boosting Machine')
filepath <- paste0('results/', relative_dir, 'model_performance')

decision_curves(models_predictions, model_names, newdata=test_data, filepath=filepath,, x=seq(0, 0.3, 0.001))

###########################
#### SELECT BEST MODEL ####
###########################

# Combine all model results and parameters 
model_names <- c('logistic regression', 'neural network', 'random forest', 'gradient boosting machine')
model_results <- setNames(list(lr_performance, nn_performance, rf_performance, gbm_performance), model_names)
model_parameters <- setNames(list(NULL, unlist(nn_best_params), unlist(rf_best_params), unlist(gbm_best_params)), model_names)
model_predictions <- setNames(list(lr_predictions, nn_predictions, rf_predictions, gbm_predictions), model_names)

# Determine the order of the models, according to their auc
model_aucs <- lapply(model_results, function(result) {result$values[result$measures == measure]})
model_order <- order(unlist(model_aucs), decreasing=TRUE)

model_names_ordered <- model_names[model_order]
model_results_ordered <- model_results[model_order]
model_parameters_ordered <- model_parameters[model_order]
model_predictions_ordered <- model_predictions[model_order]

# Order the models according to their auc 
ordered_models <- list(model_names_ordered, 
                    model_results_ordered, 
                    model_parameters_ordered,
                    model_predictions_ordered)

# Select and display the best model
best_model <- lapply(ordered_models, function(model) {model[[1]]})
print('Best model (according to ROC-AUC):')
print(best_model)
                   
# Save the ordered models and best model along with its parameters and results. 

# Use the save_list function from utils.R file 
if (overwrite) {
    om_filepath <- paste0('results/', relative_dir, 'model_', scheme ,'/ordered_models')
    bm_filepath <- paste0('results/', relative_dir, 'model_', scheme, '/best_model')
    save_list(ordered_models, om_filepath)
    save_list(best_model, bm_filepath)
}

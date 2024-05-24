################################################
#### Determine model hyperparameters ###########
################################################
# File: 2_model_tuning.R
# Author: Sebastian Benno Veuskens 
# Date: 2024-05-09
# Data: Use the train data set for training and the cross-validation predictions on that data set for evaluation. 
#
# This script performs hyperparameter tuning to the neural network, random Forest and gradient boosting machine algorithms.
# A grid search for the potential parameters is applied.

#### MODIFY ####
# Your working directory 
setwd('C:/Users/s.veuskens/Documents/Sebastian/Projekt Sebastian/modelling')
# Indicates whether to include High-Cost patients from the last year into analysis 
filter_hc <- FALSE 
# Indicates whether to include as many High-Cost patients as not-High-Cost patients 
balance_hc <- TRUE 
# Number of folds to be used for cross-validation 
nfolds <- 5 
# Number of the best models to save in the best parameters folder 
num_models <- 2 
# Whether you want to save your results (and overwrite the old results) or not
overwrite <- TRUE

## Specifiy the grid search space
# Neural Network
nn_hidden <- c(10, 20, 100, c(10, 10), c(20, 20), c(100, 100))
nn_activation <- c('Tanh', 'TanhWithDropout', 'Rectifier', 'RectifierWithDropout', 'Maxout', 'MaxoutWithDropout')
nn_rate <-  c(0.003, 0.005, 0.007) 
# Random forest 
rf_ntrees <- c(250, 500, 1000) 
rf_mtries <- c(10, sqrt(last_val - first_val), 20, 30)
# Gradient Boosting Machine 
gbm_ntrees <- c(250, 500, 1000, 2000)
gbm_max_depth <- c(1, 3, 5, 10)
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

load(paste0('data/', relative_dir, 'train', '.Rdata'))

# Start H2O package
h2o.init()

# Load data frames into H2O framework
train <- as.h2o(train)

#### CREATE FOLDER STRUCTURE ####
if (overwrite) {
    dir.create('results', showWarnings=FALSE)
    dir.create(paste0('results/', ifelse(filter_hc, 'filtered', 'complete')), showWarnings=FALSE)
    dir.create(paste0('results/', relative_dir), showWarnings=FALSE)
    dir.create(paste0('results/', relative_dir, 'model_tuning'), showWarnings=FALSE)
}

##################################################################
###################### HYPERPARAMETER TUNING #####################
##################################################################

# Position (column number) of label and variables. Indicate where the features for prediction should start and end in the data.
label_pos <- 1 
first_val <- 2
last_val <- ncol(train)

########################
#### NEURAL NETWORK ####
########################

# Measure the time for the whole training & validation process 
start_time <- Sys.time()

# Assign the parameters to perform a grid search about
nn_grid_params <- list(hidden = nn_hidden,
                       activation = nn_activation,
                       rate = nn_rate) 

# Perform the grid search 
nn_grid <- h2o.grid('deeplearning', 
                    x = first_val:last_val, 
                    y = label_pos,
                    grid_id = 'nn_grid',
                    training_frame = train,
                    nfolds = nfolds,
                    seed = 12345,
                    hyper_params = nn_grid_params)

# Show the grid search results, ordered by the corrisponding AUC 
nn_gridperf <- h2o.getGrid(grid_id = 'nn_grid',
                           sort_by = 'auc',
                           decreasing = TRUE)

print(nn_gridperf)  

# TODO: Check why the cross validation auc is bigger than the training auc -> Should be the other way round 
# Save the parameters for the best num_models (default=2) models
nn_best_ids         <- nn_gridperf@model_ids[1:num_models]
nn_best_models      <- lapply(nn_best_ids, function(id) {h2o.getModel(id)})
nn_best_params      <- lapply(nn_best_models, function(model) {c(activation = model@parameters$activation,
                                                                 hidden     = model@parameters$hidden,
                                                                 rate       = model@parameters$rate,
                                                                 auc        = model@model$cross_validation_metrics@metrics$AUC)})
nn_filepath <- paste0('results/', relative_dir, 'model_tuning/neural_network_best_parameters')
# Use the save_list function from utils.R file 
if (overwrite) save_list(nn_best_params, nn_filepath)

# Display the runtime 
end_time <- Sys.time() 
nn_runtime <- difftime(end_time, start_time, units='mins')
print(paste0('Runtime neural network: ', round(nn_runtime, 2), ' minutes'))


#######################
#### RANDOM FOREST ####
#######################

# Measure the time for the whole training & validation process 
start_time <- Sys.time()

# Assign the parameters to perform a grid search about
rf_grid_params <- list(ntrees = rf_ntrees,
                       mtries = rf_mtries)

# Perform the grid search 
rf_grid <- h2o.grid('randomForest', x = first_val:last_val, y = label_pos,
                    grid_id = 'rf_grid',
                    training_frame = train,
                    nfolds = nfolds,
                    seed = 12345,
                    hyper_params = rf_grid_params) 

# Show the grid search results, ordered by the corrisponding AUC 
rf_gridperf <- h2o.getGrid(grid_id = 'rf_grid',
                            sort_by = 'auc',
                            decreasing = TRUE)

print(rf_gridperf)

# Save the parameters for the best num_models (default=2) models
rf_best_ids         <- rf_gridperf@model_ids[1:num_models]
rf_best_models      <- lapply(rf_best_ids, function(id) {h2o.getModel(id)})
rf_best_params      <- lapply(rf_best_models, function(model) {c(ntrees = model@parameters$ntrees,
                                                                 mtries = model@parameters$mtries,
                                                                 auc    = model@model$cross_validation_metrics@metrics$AUC)})
rf_filepath <- paste0('results/', relative_dir, 'model_tuning/random_forest_best_parameters')
# Use the save_list function from utils.R file 
if (overwrite) save_list(rf_best_params, rf_filepath)

# Display the runtime 
end_time <- Sys.time() 
rf_runtime <- difftime(end_time, start_time, units='mins')
print(paste0('Runtime random forest: ', round(rf_runtime, 2), ' minutes'))


###################################
#### GRADIENT BOOSTING MACHINE #### 
###################################

# Measure the time for the whole training & validation process 
start_time <- Sys.time()

# Assign the parameters to perform a grid search about
gbm_grid_params <- list(ntrees = gbm_ntrees,
                        max_depth = gbm_max_depth)

# Perform the grid search 
gbm_grid <- h2o.grid('gbm', x = first_val:last_val, y = label_pos,
                    grid_id = 'gbm_grid',
                    training_frame = train,
                    nfolds = nfolds,
                    seed = 12345,
                    hyper_params = gbm_grid_params) 

# Show the grid search results, ordered by the corrisponding AUC 
gbm_gridperf <- h2o.getGrid(grid_id = 'gbm_grid',
                            sort_by = 'auc',
                            decreasing = TRUE)

print(gbm_gridperf)

# Save the parameters for the best num_models (default=2) models
gbm_best_ids         <- gbm_gridperf@model_ids[1:num_models]
gbm_best_models      <- lapply(gbm_best_ids, function(id) {h2o.getModel(id)})
gbm_best_params      <- lapply(gbm_best_models, function(model) {c(ntrees    = model@parameters$ntrees,
                                                                   max_depth = model@parameters$max_depth,
                                                                   auc       = model@model$cross_validation_metrics@metrics$AUC)})
gbm_filepath <- paste0('results/', relative_dir, 'model_tuning/gradient_boosting_machine_best_parameters')
# Use the save_list function from utils.R file 
if (overwrite) save_list(gbm_best_params, gbm_filepath)

# Display the runtime 
end_time <- Sys.time() 
gbm_runtime <- difftime(end_time, start_time, units='mins')
print(paste0('Runtime gradient boosting machine: ', round(gbm_runtime, 2), ' minutes'))

#  Save the runtimes
models <- c('neural network', 'random forest', 'gradient boosting machine')
runtimes_in_minutes <- round(c(nn_runtime, rf_runtime, gbm_runtime), 4)
results <- data.frame(models, runtimes_in_minutes)
rt_filepath <- paste0('results/', relative_dir, 'model_tuning/runtimes_model_tuning.csv')
if (overwrite) write.csv(results, rt_filepath)

################################################
#### DETERMINE MODEL HYPERPARAMETERS ###########
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
setwd("C:/Users/Sebastian's work/OneDrive - OptiMedis AG/Dokumente/Coding/High-Cost-patient-analysis")
# Indicates whether to include High-Cost patients from the last year into analysis 
filter_hc <- FALSE 
# Indicates whether to include as many High-Cost patients as not-High-Cost patients 
balance_hc <- FALSE 
# Number of folds to be used for cross-validation 
nfolds <- 5 
# Number of the best models to save in the best parameters folder 
num_models <- 3 
# Whether you want to save your results (and overwrite the old results) or not
overwrite <- TRUE

## Specifiy the grid search space
# Logistic regression
lr_sig_level <- 0.05
lr_min_pred_num <- 10
# Neural Network
nn_hidden <- c(10, 20, 100, c(10, 10), c(20, 20), c(100, 100))
nn_activation <- c('Tanh', 'TanhWithDropout', 'Rectifier', 'RectifierWithDropout', 'Maxout', 'MaxoutWithDropout')
nn_rate <-  c(0.003, 0.005, 0.007) 
# Random forest 
rf_ntrees <- c(250, 500, 1000) 
rf_mtries <- c(10, 20, 30)
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
# install.packages(dplyr)
# install.packages('rlist')

# LOAD LIBRARIES & SOURCES
library(h2o)            # The modelling framework 
library(cvAUC)          # For the Area Under the Curve (AUC) computation
library(dplyr)          # For the pipe operator
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
first_val <- 3
last_val <- ncol(train)

#############################
#### LOGISTIC REGRESSION ####
#############################

# Measure the time for the whole training & validation process 
start_time <- Sys.time()

# Fit the full logistic regression model on all variables. Remove collinear columns for p-value estimation 
lr_full_model <- h2o.glm(x                        = first_val:last_val, 
                         y                        = label_pos,
                         training_frame           = train,  
                         nfolds                   = nfolds,
                         seed                     = 12345,
                         remove_collinear_columns = TRUE,
                         calc_like                = TRUE 
                         )

# Perform backward variable selection
# Stopping criterion: All variable p-values are below significance level 
lr_backward_pval_model <- h2o.modelSelection(x                          = first_val:last_val,
                                             y                          = label_pos,
                                             training_frame             = train,
                                             seed                       = 12345,
                                             mode                       = 'backward',
                                             p_values_threshold         = lr_sig_level,
                                             remove_collinear_columns   = TRUE)
lr_bpval_idx <- h2o.result(lr_backward_pval_model)[1,'predictor_names'] %>% 
                strsplit(', ')  %>% 
                unlist()        %>% 
                match(colnames(train))
lr_bpval_best_model <- train_lr_model(lr_bpval_idx, label_pos, train)

# Perform backward variable selection
# Stopping criterion: Only the minimum specified number of predictors is     included 
lr_backward_min_num_model <- h2o.modelSelection(x                           = first_val:last_val,
                                                y                           = label_pos,
                                                training_frame              = train,
                                                seed                        = 12345,
                                                mode                        = 'backward',
                                                # TODO: Check why the number of variables reduces so much
                                                min_predictor_number        = lr_min_pred_num + 7, 
                                                remove_collinear_columns    = TRUE)                                             
lr_bminn_idx <- h2o.result(lr_backward_min_num_model)[1,'predictor_names'] %>%
                strsplit(', ')  %>% 
                unlist()        %>% 
                match(colnames(train))
lr_bminn_best_model <- train_lr_model(lr_bminn_idx, label_pos, train)                

# Likelihood-ratio test 
ndiff_fbpval <- length(lr_full_model@parameters$x)-length(lr_bpval_best_model@parameters$x)      
ndiff_fbminn <- length(lr_full_model@parameters$x)-length(lr_bminn_best_model@parameters$x)      
ndiff_fbminn <- length(lr_bpval_best_model@parameters$x)-length(lr_bminn_best_model@parameters$x)
lr_nll_full             <- -2 * h2o.loglikelihood(lr_full_model)
lr_nll_backward_pval    <- -2 * h2o.loglikelihood(lr_bpval_best_model)
lr_nll_backward_min_num <- -2 * h2o.loglikelihood(lr_bminn_best_model)
lr_likelihood_ratio_fbpval      <- pchisq(lr_nll_backward_pval    - lr_nll_full, ndiff_fbpval, lower.tail=FALSE)                                      # Chi Square test of: -2 * (log-likelihood of reduced model -log-likelihood of full model)
lr_likelihood_ratio_fbminn      <- pchisq(lr_nll_backward_min_num - lr_nll_full, ndiff_fbminn, lower.tail=FALSE)                                      # Chi Square test of: -2 * (log-likelihood of reduced model -log-likelihood of full model)
lr_likelihood_ratio_bpval_bminn <- pchisq(lr_nll_backward_min_num - lr_nll_backward_pval, ndiff_fbminn, lower.tail=FALSE)                                      # Chi Square test of: -2 * (log-likelihood of reduced model -log-likelihood of full model)
print(paste0('P-value of full model and backward p-value model: ', round(lr_likelihood_ratio_fbpval, 3)))
print(paste0('P-value of full model and backward minimum number of predictors model: ', round(lr_likelihood_ratio_fbminn, 3)))
if (length(setdiff(lr_bminn_idx, lr_bpval_idx)) == 0) {
    print(paste0('P-value of backward p-value model and backward minimum number of predictors model: ', round(lr_likelihood_ratio_fbminn, 3)))
} else {
    warning('WARNING: NOT NESTED MODELS USED IN LIKELIHOOD-RATIO TEST!') 
}

# Save the parameters for the best num_models (default=2) models
lr_all_models  <- c(lr_full_model, lr_bpval_best_model, lr_bminn_best_model)
lr_all_params  <- lapply(lr_all_models, function(model) {c(lambda       = model@parameters$lambda,
                                                           predictors   = do.call(paste, c(as.list(model@parameters$x), sep=', ')),
                                                           auc          = model@model$cross_validation_metrics@metrics$AUC,
                                                           aic          = model@model$cross_validation_metrics@metrics$AIC)})

# Order results by AIC value (increasing)  
lr_model_order <- order(sapply(lr_all_params, function(x) {x['aic']}))
lr_best_params <- lr_all_params[lr_model_order][1:num_models]

lr_filepath <- paste0('results/', relative_dir, 'model_tuning/logistic_regression_best_parameters')
# Use the save_list function from utils.R file 
if (overwrite) save_list(lr_best_params, lr_filepath)                                                                 

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

# Save the parameters for the best num_models (default=2) models
nn_best_ids         <- nn_gridperf@model_ids[1:num_models]
nn_best_models      <- lapply(nn_best_ids, function(id) {h2o.getModel(id)})
nn_best_params      <- lapply(nn_best_models, function(model) {c(activation = model@parameters$activation,
                                                                 hidden     = model@parameters$hidden,
                                                                 rate       = model@parameters$rate,
                                                                 aic        = model@model$cross_validation_metrics@metrics$AIC,
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
                                                                 aic    = model@model$cross_validation_metrics@metrics$AIC,
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
                                                                   aic       = model@model$cross_validation_metrics@metrics$AIC,
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

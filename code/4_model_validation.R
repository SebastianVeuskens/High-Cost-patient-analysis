#####################################
#### VALIDATE FINAL MODEL ###########
#####################################
# File: 4_model_validation.R
# Author: Sebastian Benno Veuskens 
# Date: 2024-05-13
# Data: Use the train_validate data set for training and the test data set for validation. 
#
# This script trains the best model from the model selection (see 3_model_performance.R) or user-specified model.
# For validation, the previously unseen test data set is used. 

#### MODIFY ####
# Your working directory 
setwd("C:/Users/Sebastian's work/OneDrive - OptiMedis AG/Dokumente/Coding/High-Cost-patient-analysis")
# Indicates whether to include High-Cost patients from the last year into analysis 
filter_hc <- FALSE 
# Indicates whether to include as many High-Cost patients as not-High-Cost patients 
balance_hc <- FALSE 
# Indicate the model to evaluate. Default (NULL) selects the best model from the model selection (see 3_model_selection.R).
# TODO: Change this to NULL at the end 
user_model_name <- 'random forest'
# Whether you want to save your results (and overwrite the old results) or not
overwrite <- TRUE
# Whether to use a patient that was actually is positive or negative
use_pos <- TRUE 
# Whether to use a patient that is predicted positive or negative
use_true <- TRUE
# Number of variables to display in variable importance plot
num_vars <- 5
# Number of features to display in SHAP analysis plot
num_features <- 5
# Specify which feature to investigate better 
feature_of_interest <- 'Age'
#### MODIFY END ####


#######################
#### PRELIMINARIES ####
#######################

#### LIBRARIES & SOURCES ####

# INSTALL LIBRARIES 
# install.packages('h2o')
# install.packages('cvAUC')
# install.packages('rlist')
# install.packages('PRROC')
# install.packages('dplyr')
# install.packages('caret')

# LOAD LIBRARIES & SOURCES
library(h2o)            # The modelling framework 
library(cvAUC)          # For the Area Under the Curve (AUC) computation
library(PRROC)          # For the ROC and Precision-Recall curve 
library(dplyr)          # To bind two data frames (predictions and labels) together 
library(caret)          # To compute the confusion matrix
source('code/utils.R')  # Auxiliary functions for simplicity and concise code 

#### LOAD DATA ####

# Indicate from which relative location to load & save from. Depends on user input. 
relative_dir <- paste0(ifelse(filter_hc, 'filtered/', 'complete/'), ifelse(balance_hc, 'balanced/', 'unbalanced/'))

load(paste0('data/', relative_dir, 'test', '.Rdata'))

# Start H2O package
h2o.init()

# Load test data frame into H2O framework
test_df <- test
test <- as.h2o(test)

# # Load the best model identified in the model selection (see 3_model_performance.R) or the user-specified model
# filename_params <- paste0('random_forest_best_parameters.RData')
# params <- list.load(paste0('results/', relative_dir, 'model_tuning/', filename_params))
# best_params <- params[[1]]

#### CREATE FOLDER STRUCTURE ####
if (overwrite) {
    dir.create('results', showWarnings=FALSE)
    dir.create(paste0('results/', ifelse(filter_hc, 'filtered', 'complete')), showWarnings=FALSE)
    dir.create(paste0('results/', relative_dir), showWarnings=FALSE)
    dir.create(paste0('results/', relative_dir, 'model_explanation'), showWarnings=FALSE)
}


##################################################################
###################### MODEL EVALUATION ##########################
##################################################################

# Position of label and variables. Indicate where the features for prediction should start and end in the data.
label_pos <- 1 
first_val <- 3
last_val <- ncol(test)

#### LOAD MODEL ####
model_name <- gsub(' ', '_', user_model_name) 
model_filepath <- paste0('results/', relative_dir, 'model_validation/model_files_', model_name)
# TODO: Delete this after I ran model_validation 
# model_filepath <- paste0('results/', relative_dir, 'model_explanation/model_files_', model_name)
model <- h2o.loadModel(model_filepath)

# # Train the model. Use the train_model function from the utils.R file
# model <- train_model(model_params, train_validate, first_val, last_val, label_pos)

# # Save the model
# # Exchange tabs with underscore for consistent file naming (to correct user input).  
# h2o.saveModel(model, paste0('results/', relative_dir, 'model_validation'), filename='random_forest', force=TRUE)

# Find model threshold
performance     <- h2o.performance(model, newdata = test)
sensitivities   <- h2o.sensitivity(performance)
specificities   <- h2o.specificity(performance)
gmeans          <- sqrt(sensitivities$tpr * specificities$tnr)
threshold_index <- which.max(gmeans)
threshold       <- sensitivities[threshold_index, 'threshold']

prediction_matrix <- h2o.predict(model, newdata=test)
predictions <- as.numeric(prediction_matrix$p1 >= threshold)

#####################
#### XAI METHODS ####
#####################

true_hcp_status <- as.numeric(test_df$HC_Patient_Next_Year) - 1 == use_true
predicted_hcp_status <- as.data.frame(predictions) == use_pos
sample_indices <- which(true_hcp_status & predicted_hcp_status)
sample_idx <- sample_indices[1]
sample <- test[sample_idx,]
predictions[sample_idx]

# Specifiy where to save the results
result_filepath <- paste0('results/', relative_dir, 'model_explanation/h2o')

# exp_global_h2o <- h2o.explain(model, test[1:1000,])
# exp_local_h2o <- h2o.explain(model, test, row_index=sample_idx)

# Variable importance
# Display and save the variable importance plot 
if (overwrite) {png(paste0(result_filepath, '_variable_importance.png'))}
h2o.varimp_plot(model, num_of_features=num_vars)
if (overwrite) {dev.off()}

# SHAP analysis 
# Display and save the SHAP summary analysis plot
if (overwrite) {png(paste0(result_filepath, 'shap_summary_analysis.png'))}
h2o.shap_summary_plot(model, newdata = test, top_n_features = num_features)
if (overwrite) {dev.off()}

# Display and save the SHAP local explanation plot 
if (overwrite) {png(paste0(result_filepath, '_shap_local_analysis.png'))}
h2o.shap_explain_row_plot(model, newdata = test, row_index = sample_idx)
if (overwrite) {dev.off()}


# Partial Dependence Plot 
if (overwrite) {png(paste0(result_filepath, '_pdp.png'))}
h2o.pd_plot(model, newdata = test)
if (overwrite) {dev.off()}


# Individual Conditional Expectations Plot 
if (overwrite) {png(paste0(result_filepath, '_ice.png'))}
h2o.ice_plot(model, newdata = test, feature_of_interest)
if (overwrite) {dev.off()}

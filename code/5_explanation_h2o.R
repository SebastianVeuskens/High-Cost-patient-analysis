#########################################
#### EXPLAIN RANDOM FOREST H2O ##########
#########################################
# File: 5_explanation_h20.R
# Author: Sebastian Benno Veuskens 
# Date: 2024-08-21
# Data: Generate explanations with the h2o-inherent environment
#
# The purpose of this script is to show the extend of explanations in h2o. 

#### MODIFY ####
# Your working directory 
setwd("C:/Users/s.veuskens/Documents/Sebastian/Projekt Sebastian/modelling")
# Indicates whether to include High-Cost patients from the last year into analysis 
filter_hc <- FALSE 
# Indicates whether to include as many High-Cost patients as not-High-Cost patients 
balance_hc <- FALSE 
# Whether you want to save your results (and overwrite the old results) or not
overwrite <- TRUE
# Target variable
target <- 'HC_Patient_Next_Year'
# Variables to exclude 
excluded <- 'Total_Costs_Next_Year'
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
# TODO: Change this in later evaluations 
sample_idx <- 1
#### MODIFY END ####

#######################
#### PRELIMINARIES ####
#######################

#### LIBRARIES & SOURCES ####
# TODO: Check later which libraries I really need here. 
# INSTALL LIBRARIES 
# install.packages("dplyr")
# install.packages("h2o")
# install.packages('randomForest')
# install.packages("pdp")               
# install.packages("shapper")  
# install.packages("cvAUC")
# install.packages("PRROC") 
# install.packages("caret")
# install.packages('randomForest')
# install.packages('DALEX')
# install.packages('DALEXtra')
# install.packages('lime')
# install.packages('localModel')

# LOAD LIBRARIES & SOURCES
library(dplyr)          # To bind two data frames (predictions and labels) together 
library(h2o)            # The modelling framework 
# library(randomForest)   # For the random Forest in R
# library(pdp)            # PDP 
# library(shapper)        # SHAP 
# library(cvAUC)          # For the Area Under the Curve (AUC) computation
# library(PRROC)          # For the ROC and Precision-Recall curve
# library(caret)          # To compute the confusion matrix
# library(randomForest)   # To model the R native Random Forest 
# library(DALEX)          # For model explanations 
# library(DALEXtra)       # Needed for lime functionality of DALEX 
# library(lime)           # Needed for lime calculation in DALEX 
# library(localModel)     # Used for additional local explanation 
source('code/utils.R')  # Auxiliary functions for simplicity and concise code 

#### LOAD DATA ####

# Indicate from which relative location to load & save from. Depends on user input. 
relative_dir <- paste0(ifelse(filter_hc, 'filtered/', 'complete/'), ifelse(balance_hc, 'balanced/', 'unbalanced/'))

load(paste0('data/', relative_dir, 'train_validate',    '.Rdata'))
load(paste0('data/', relative_dir, 'test', '.Rdata'))

# train_validate$Sex <- as.factor(train_validate$Sex)
# test$Sex <- as.factor(test$Sex)

# Start H2O package
h2o.init()

# Load data frames into H2O framework
train_validate <- as.h2o(train_validate)
test <- as.h2o(test)

#### CREATE FOLDER STRUCTURE ####
if (overwrite) {
    dir.create('results', showWarnings=FALSE)
    dir.create(paste0('results/', ifelse(filter_hc, 'filtered', 'complete')), showWarnings=FALSE)
    dir.create(paste0('results/', relative_dir), showWarnings=FALSE)
    dir.create(paste0('results/', relative_dir, 'model_explanation'), showWarnings=FALSE)
}

#############################################################
###################### LOAD MODEL ###########################
#############################################################

# Position of label and variables. Indicate where the features for prediction should start and end in the data.
label_pos <- 1 
first_val <- 3
last_val <- ncol(train_validate)

# Specify location to save the model 
model_filepath <- paste0('results/', relative_dir, 'model_validation/model_files_random_forest')

if (file.exists(model_filepath)) {
    model <- h2o.loadModel(model_filepath)
} else {
    # Load the best parameters from hyperparameter tuning for the specified model 
    filename_params <- paste0('random_forest_best_parameters.RData')
    params <- list.load(paste0('results/', relative_dir, 'model_tuning/', filename_params))
    best_params <- params[[1]]

    # Train the model
    model <- h2o.randomForest(x            = first_val:last_val, 
                            y              = label_pos,
                            training_frame = train_validate, 
                            nfolds         = nfolds,
                            seed           = 12345,
                            ntrees         = as.numeric(rf_best_params[['ntrees']]),
                            mtries         = as.numeric(rf_best_params[['mtries']]))

    predictions <- h2o.predict(model, newdata=test)
}


#####################
#### XAI METHODS ####
#####################

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
if (overwrite) {png(paste0(result_filepath, '_', feature_of_interest, '_pdp.png'))}
h2o.pd_plot(model, newdata = test, column=feature_of_interest)
if (overwrite) {dev.off()}


# Individual Conditional Expectations Plot 
if (overwrite) {png(paste0(result_filepath, '_', feature_of_interest, '_ice.png'))}
h2o.ice_plot(model, newdata = test, column=feature_of_interest)
if (overwrite) {dev.off()}
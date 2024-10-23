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
# Number of features to display in SHAP summary analysis plot
num_features <- 5
# Number of features to display in SHAP local analysis plot
num_features_local <- 10
# Specify which feature to investigate better 
feature_of_interest <- 'Age'
# The index of the sample to investigate (different indexing in R an Python)
sample_idx <- 1467 + 1
#### MODIFY END ####

#######################
#### PRELIMINARIES ####
#######################

#### LIBRARIES & SOURCES ####
# INSTALL LIBRARIES 
# install.packages("dplyr")
# install.packages("h2o")

# LOAD LIBRARIES & SOURCES
library(dplyr)          # To bind two data frames (predictions and labels) together 
library(h2o)            # The modelling framework 
source('code/utils.R')  # Auxiliary functions for simplicity and concise code 

#### LOAD DATA ####

load('data/train_validate.Rdata')
load('data/test.Rdata')

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
    dir.create('results/model_explanation', showWarnings=FALSE)
    dir.create('results/model_explanation/h2o', showWarnings=FALSE)
}

#############################################################
###################### LOAD MODEL ###########################
#############################################################

# Position of label and variables. Indicate where the features for prediction should start and end in the data.
label_pos <- 1 
first_val <- 3
last_val <- ncol(train_validate)

# Specify location to save the model 
model_filepath <- 'results/model_validation/model_files_random_forest'

model <- h2o.loadModel(model_filepath)


#####################
#### XAI METHODS ####
#####################

# Specifiy where to save the results
result_filepath <- 'results/model_explanation/h2o/'

# exp_global_h2o <- h2o.explain(model, test[1:1000,])
# exp_local_h2o <- h2o.explain(model, test, row_index=sample_idx)

# VARIABLE IMPORTANCE

# Start time measurement 
varimp_start <- Sys.time()

# Display and save the variable importance plot 
if (overwrite) {png(paste0(result_filepath, 'variable_importance.png'))}
h2o.varimp_plot(model, num_of_features=num_vars)
if (overwrite) {dev.off()}

# Stop and report time
varimp_end <- Sys.time()
varimp_runtime <- difftime(varimp_end, varimp_start, units='mins')
print(paste0('Runtime for variable importance plots: ', round(varimp_runtime, 2), ' minutes'))

# SHAP ANALYSIS

# Start time measurement 
shap_sum_start <- Sys.time()

# Display and save the SHAP summary analysis plot
if (overwrite) {png(paste0(result_filepath, 'shap_summary_analysis.png'))}
h2o.shap_summary_plot(model, newdata = test, top_n_features = num_features)
if (overwrite) {dev.off()}

# Stop and report time
shap_sum_end <- Sys.time()
shap_sum_runtime <- difftime(shap_sum_end, shap_sum_start, units='mins')
print(paste0('Runtime for SHAP summary plots: ', round(shap_sum_runtime, 2), ' minutes'))

# Start time measurement 
shap_local_start <- Sys.time()

# Display and save the SHAP local explanation plot 
if (overwrite) {png(paste0(result_filepath, 'shap_local_analysis.png'))}
h2o.shap_explain_row_plot(model, newdata = test, row_index = sample_idx, top_n_features=num_features_local / 2)
if (overwrite) {dev.off()}

# Stop and report time
shap_local_end <- Sys.time()
shap_local_runtime <- difftime(shap_local_end, shap_local_start, units='mins')
print(paste0('Runtime for SHAP local plots: ', round(shap_local_runtime, 2), ' minutes'))


# PARTIAL DEPENDENCE PLOT 

# Start time measurement 
pdp_start <- Sys.time()

if (overwrite) {png(paste0(result_filepath, feature_of_interest, '_pdp.png'))}
h2o.pd_plot(model, newdata = test, column=feature_of_interest)
if (overwrite) {dev.off()}

# Stop and report time
pdp_end <- Sys.time()
pdp_runtime <- difftime(pdp_end, pdp_start, units='mins')
print(paste0('Runtime for PDP plots: ', round(pdp_runtime, 2), ' minutes'))


# INDIVIDUAL CONDITIONAL EXPECTATIONS PLOT 

# Start time measurement 
ice_start <- Sys.time()

if (overwrite) {png(paste0(result_filepath, feature_of_interest, '_ice.png'))}
h2o.ice_plot(model, newdata = test, column=feature_of_interest)
if (overwrite) {dev.off()}

# Stop and report time
ice_end <- Sys.time()
ice_runtime <- difftime(ice_end, ice_start, units='mins')
print(paste0('Runtime for grouped ICE plots: ', round(ice_runtime, 2), ' minutes'))

#  Save the runtimes
methods <- c('Variable importance', 'SHAP summary', 'SHAP local', 'PDP', 'ICE')
runtimes_in_minutes <- round(c(varimp_runtime, shap_sum_runtime, shap_local_runtime, pdp_runtime, ice_runtime), 4)
results <- data.frame(methods, runtimes_in_minutes)
rt_filepath <- 'results/model_explanation/h2o/runtimes_methods.csv'
if (overwrite) write.csv(results, rt_filepath)
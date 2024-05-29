#####################################
#### EVALUATION EXPERIMENTS #########
#####################################
# File: 4_model_evaluation.R
# Author: Sebastian Benno Veuskens 
# Date: 2024-05-24
# Data: 
#
# 
# 

#### MODIFY ####
# Your working directory 
setwd('C:/Users/s.veuskens/Documents/Sebastian/Projekt Sebastian/modelling')
# Indicates whether to include High-Cost patients from the last year into analysis 
filter_hc <- FALSE 
# Indicates whether to include as many High-Cost patients as not-High-Cost patients 
balance_hc <- TRUE 
# Whether you want to save your results (and overwrite the old results) or not
overwrite <- TRUE
#### MODIFY END ####


#######################
#### PRELIMINARIES ####
#######################

#### LIBRARIES & SOURCES ####

# INSTALL LIBRARIES 
# install.packages('h2o')


# LOAD LIBRARIES & SOURCES
library(h2o)            # The modelling framework 
source('code/utils.R')  # Auxiliary functions for simplicity and concise code 

#### LOAD DATA ####

# Indicate from which relative location to load & save from. Depends on user input. 
relative_dir <- paste0(ifelse(filter_hc, 'filtered/', 'complete/'), ifelse(balance_hc, 'balanced/', 'unbalanced/'))

load(paste0('data/', relative_dir, 'train',    '.Rdata'))
load(paste0('data/', relative_dir, 'validate', '.Rdata'))
load(paste0('data/', relative_dir, 'train_validate', '.Rdata'))
load(paste0('data/', relative_dir, 'test', '.Rdata'))

# Load true labels
label <- validate$HC_Patient_Next_Year

# Load predictions 
ordered_models <- list.load(paste0('results/', relative_dir, 'model_selection/ordered_models.RData'))
lr_predictions <- as.data.frame(ordered_models[['predictions']][['logistic regression']])
nn_predictions <- as.data.frame(ordered_models[['predictions']][['neural network']])
rf_predictions <- as.data.frame(ordered_models[['predictions']][['random forest']])
gbm_predictions <- as.data.frame(ordered_models[['predictions']][['gradient boosting machine']])


##########################################################
################### EVALUATIONS ##########################
##########################################################
table(lr_predictions$predict, label)
table(nn_predictions$predict, label)
table(rf_predictions$predict, label)
table(gbm_predictions$predict, label)

table(lr_predictions$predict, nn_predictions$predict)
table(lr_predictions$predict, rf_predictions$predict)
table(lr_predictions$predict, gbm_predictions$predict)

table(nn_predictions$predict, lr_predictions$predict)
table(nn_predictions$predict, rf_predictions$predict)
table(nn_predictions$predict, gbm_predictions$predict)

# TODO: Check why I have this number of High-Cost patients 
sum(lr_predictions$predict == 1)
sum(nn_predictions$predict == 1)
sum(rf_predictions$predict == 1)
sum(gbm_predictions$predict == 1)

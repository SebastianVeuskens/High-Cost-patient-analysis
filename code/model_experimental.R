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
balance_hc <- FALSE 
# Whether you want to save your results (and overwrite the old results) or not
overwrite <- TRUE
#### MODIFY END ####


#######################
#### PRELIMINARIES ####
#######################

#### LIBRARIES & SOURCES ####

# INSTALL LIBRARIES 
# install.packages('mgcv')


# LOAD LIBRARIES & SOURCES
library(mgcv)           # For Generalized additive models 
source('code/utils.R')  # Auxiliary functions for simplicity and concise code 

#### LOAD DATA ####

# Indicate from which relative location to load & save from. Depends on user input. 
relative_dir <- paste0(ifelse(filter_hc, 'filtered/', 'complete/'), ifelse(balance_hc, 'balanced/', 'unbalanced/'))

load(paste0('data/', relative_dir, 'train_validate',    '.Rdata'))
load(paste0('data/', relative_dir, 'test', '.Rdata'))

##########################################################
################### MODELLING ############################
##########################################################
gam_model <- gam(HC_Patient_Next_Year ~ s(Age) + ), family=binomial(), data=train_validate)

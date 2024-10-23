################################
#### VALIDATE DATA #############
################################
# File: data_validation.R
# Author: Sebastian Benno Veuskens 
# Date: 2024-05-06
# Data: #TODO: 
#
# 
# 
#  

#### MODIFY ####
# Your working directory 
setwd('C:/Users/s.veuskens/Documents/Sebastian/Projekt Sebastian/modelling')
# Indicates whether to include High-Cost patients from the last year into analysis 
filter_hc <- FALSE 
#### MODIFY END ####


#######################
#### PRELIMINARIES ####
#######################

#### LIBRARIES & SOURCES ####

# INSTALL LIBRARIES 
# install.packages('testit')
# install.packages('dyplr')

# LOAD LIBRARIES & SOURCES
library(testit)
library(dplyr)

#### LOAD DATA ####

load(paste0('data/', ifelse(filter_hc, 'filtered/', 'original/'), 'train_validate', '.Rdata'))
load(paste0('data/', ifelse(filter_hc, 'filtered/', 'original/'), 'train',          '.Rdata'))
load(paste0('data/', ifelse(filter_hc, 'filtered/', 'original/'), 'validate',       '.Rdata'))
load(paste0('data/', ifelse(filter_hc, 'filtered/', 'original/'), 'test',           '.Rdata'))


########################################################
####################### CHECKS #########################
########################################################


#### HIGH-COST PATIENT PROPORTION #### 
# This function checks the correct proportion of High-Cost patients 
check_hc_proportion <- function(data, data_name, tolerance=0.01) {
    hc_proportion <- summary(data$HC_Patient)['Mean']
    message <-paste('Data set', data_name, 'has right amount of High-Cost patients')
    # A boolean indicator whether the proportion of High-Cost patients is correct within a tolerance
    correct_proportion <- abs(hc_proportion - 0.05) <= tolerance
    assert(message, correct_proportion)
}

check_hc_proportion(train_validate, 'train_validate')
check_hc_proportion(train, 'train')
check_hc_proportion(validate, 'validate')
check_hc_proportion(test, 'test')



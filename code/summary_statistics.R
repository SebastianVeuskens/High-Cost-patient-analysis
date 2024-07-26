################################################
#### SUMMARY STATISTIC GENERATION ##############
################################################
# File: summary_statistics.R
# Author: Sebastian Benno Veuskens 
# Date: 2024-07-25
# Data: HC_Patient_Data for years 2019-2021 (including HC-indicators for following year, respectively)
#
# 
# This script creates the tables as in the first results section.
# Main demographic information is given as well as HC-patient overviews. 

#### MODIFY ####
# Your working directory 
setwd('C:/Users/s.veuskens/Documents/Sebastian/Projekt Sebastian/modelling')
# Indicates whether to include High-Cost patients from the last year into analysis 
filter_hc <- FALSE 
# Indicates whether to include as many High-Cost patients as not-High-Cost patients 
balance_hc <- FALSE 
# Number of the best models to save in the best parameters folder 
num_models <- 2 
#### MODIFY END ####

#######################
#### PRELIMINARIES ####
#######################

#### LIBRARIES & SOURCES ####

# INSTALL LIBRARIES 
# install.packages('arsenal')

# LOAD LIBRARIES & SOURCES
source('code/utils.R')  # Auxiliary functions for simplicity and concise code 
library('arsenal')      # Package for summary statistics 

#### LOAD DATA ####

# Indicate from which relative location to load & save from. Depends on user input. 
relative_dir <- paste0(ifelse(filter_hc, 'filtered/', 'complete/'), ifelse(balance_hc, 'balanced/', 'unbalanced/'))

load(paste0('data/', relative_dir, 'data_2019.Rdata'))
load(paste0('data/', relative_dir, 'data_2020.Rdata'))
load(paste0('data/', relative_dir, 'data_2021.Rdata'))
load(paste0('data/', relative_dir, 'data.Rdata'))
load(paste0('data/', relative_dir, 'train.Rdata'))
load(paste0('data/', relative_dir, 'validate.Rdata'))
load(paste0('data/', relative_dir, 'train_validate.Rdata'))
load(paste0('data/', relative_dir, 'test.Rdata'))

# Position of label and variables. Indicate where the features for prediction should start and end in the data.
label_pos <- 1 
first_val <- 2
last_val <- ncol(train)

#######################
#### SUMMARIZE ########
#######################
cur_data <- data_2019

n <- nrow(cur_data)
str(cur_data)

variables <- c('HC_Patient_Next_Year'
             , 'Total_Costs_Next_Year',
             , 'HC_Patient'
             , 'Age'
             , 'Need_of_Care_Duration'
             , 'DMP_Duration'
             , 'Total_Costs'
             , 'Inpatient_Num_Diagnoses'
             , 'Outpatient_Num_Diagnoses'
             , 'Prescription_Num_Prescriptions'
                )

gather(hospdata[,variables], "var", "value", -race) %>%
  count(var, value) %>%
  group_by(var) %>%       
  mutate(prop = prop.table(n))
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
setwd("C:/Users/Sebastian's work/OneDrive - OptiMedis AG/Dokumente/Coding/High-Cost-patient-analysis")
# Indicates whether to include High-Cost patients from the last year into analysis 
filter_hc <- FALSE 
# Indicates whether to include as many High-Cost patients as not-High-Cost patients 
balance_hc <- FALSE 
# Number of the best models to save in the best parameters folder 
num_models <- 2 
# Way to divide the age into groups
age_split <- c(30, 65, 80)
#### MODIFY END ####

#######################
#### PRELIMINARIES ####
#######################

#### LIBRARIES & SOURCES ####

# INSTALL LIBRARIES 
# install.packages('tidyr')
# install.packages('dplyr')
# install.packages('arsenal')

# LOAD LIBRARIES & SOURCES
source('code/utils.R')  # Auxiliary functions for simplicity and concise code 
library('tidyr')        # Auxiliary tools for frequency tables 
library('dplyr')        # Auxiliary tools for frequency tables  
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

#### CREATE FOLDER STRUCTURE ####
if (overwrite) {
    dir.create('results', showWarnings=FALSE)
    dir.create(paste0('results/', ifelse(filter_hc, 'filtered', 'complete')), showWarnings=FALSE)
    dir.create(paste0('results/', relative_dir), showWarnings=FALSE)
    dir.create(paste0('results/', relative_dir, 'summary_statistics'), showWarnings=FALSE)
}

#######################
#### SUMMARIZE ########
#######################
cur_data <- data_2019
cur_data$Age_groups_num <- findInterval(cur_data$Age, age_split)
# TODO: Make this dynamic so that it changes when the group change in the modify part 
age_code <- c('0-30' = 0, '30-65' = 1, '65-80' = 2, '>80' = 3)
cur_data$Age_groups <- names(age_code)[match(cur_data$Age_groups_num, age_code)]
# cur_data <- as.data.frame(cbind(c(1, 2, 3), c(4, 5, 6), c(7, 8, 9), c(10, 11, 12)))

variables <- c('HC_Patient_Next_Year'
             , 'Total_Costs_Next_Year'
             , 'HC_Patient'
             , 'Sex'
             , 'Age_groups'
             , 'Need_of_Care_Duration'
             , 'DMP_Duration'
             , 'Total_Costs'
             , 'Inpatient_Num_Diagnoses'
             , 'Outpatient_Num_Diagnoses'
             , 'Prescription_Num_Prescriptions'
                )

cur_data <- cur_data[,variables]        
n <- nrow(cur_data)
str(cur_data)



variable_frequency <- function(data, variable) {
  return(
    gather(data, 'var', 'value', variable) %>%
      count(var, value) %>%
      group_by(var) %>%       
      mutate(prop = prop.table(n))
  )
}
frequency_table <- list(
  'Sex' = variable_frequency(cur_data, 'Sex'),
  'Age' = variable_frequency(cur_data, 'Age_groups'),
  'HC_Patient' = variable_frequency(cur_data, 'HC_Patient')
)
print(frequency_table)

## Save frequency table 
rf_filepath <- paste0('results/', relative_dir, 'summary_statistics/frequency_table')
if (overwrite) save_list(frequency_table, rf_filepath)


################################################
#### EXPLORATORY DATA ANALYSIS #################
################################################
# File: eda.R
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
# Number of the best models to save in the best parameters folder 
num_models <- 2 
#### MODIFY END ####

#######################
#### PRELIMINARIES ####
#######################

#### LIBRARIES & SOURCES ####

# INSTALL LIBRARIES 

# LOAD LIBRARIES & SOURCES
source('code/utils.R')  # Auxiliary functions for simplicity and concise code 

#### LOAD DATA ####

# Indicate from which relative location to load & save from. Depends on user input. 
relative_dir <- paste0(ifelse(filter_hc, 'filtered/', 'complete/'), ifelse(balance_hc, 'balanced/', 'unbalanced/'))

load(paste0('data/', relative_dir, 'train', '.Rdata'))
load(paste0('data/', relative_dir, 'validate', '.Rdata'))
load(paste0('data/', relative_dir, 'train_validate', '.Rdata'))
load(paste0('data/', relative_dir, 'test', '.Rdata'))
load(paste0('data/', relative_dir, 'data_2019.Rdata'))
load(paste0('data/', relative_dir, 'data_2020.Rdata'))
load(paste0('data/', relative_dir, 'data_2021.Rdata'))

# Position of label and variables. Indicate where the features for prediction should start and end in the data.
label_pos <- 1 
first_val <- 3
last_val <- ncol(train)


#########################################################################
################## EXPLORATION ##########################################
#########################################################################
data <- train_validate

column_names <- colnames(data)
sub('Inpatient_', '', column_names[grep('Inpatient_', column_names)])
length(column_names[grep('Inpatient_', column_names)])
length(column_names[grep('Outpatient_', column_names)])
length(column_names[grep('Prescription_', column_names)])

nrow(data_2019)
nrow(data_2020)
nrow(data_2021)
nrow(train)
nrow(validate)
nrow(train_validate)
nrow(test)
nrow(train_validate) + nrow(test)
nrow(data_2019) + nrow(data_2020) + nrow(data_2021)

sum(data_2019$HC_Patient_Next_Year == 1)
sum(data_2020$HC_Patient_Next_Year == 1)
sum(data_2021$HC_Patient_Next_Year == 1)
sum(data_2019$HC_Patient_Next_Year == 0)
sum(data_2020$HC_Patient_Next_Year == 0)
sum(data_2021$HC_Patient_Next_Year == 0)

dim(data)
str(data)
data <- data[, colSums(data != 0) > 10]

dim(data[data$year < 2022, ])
# Print the structure of the dataset
print(str(data))

# Summary statistics
summary(data)

# Check for missing values
sum(is.na(data))

# Plotting histograms for all numeric columns
num_cols <- sapply(data, is.numeric)
hist_data <- data[, num_cols]
length(num_cols)

heatmap(cor(hist_data), scale="none")
str(cor(hist_data))

par(mfrow=c(2,2))
lapply(hist_data, function(x) hist(x, main = names(hist_data)[which(hist_data == x)], xlab = NULL))

# Boxplots for all numeric columns
par(mfrow=c(2,2))
lapply(hist_data, function(x) boxplot(x, main = names(hist_data)[which(hist_data == x)], xlab = NULL))

# Pairwise scatterplots for all numeric columns
pairs(hist_data)

#### FEATURE STATISTICS ####
# TODO: Add data overviews like age distribution plot (by sex?), number of total patients & number of high-cost patients, inpatient/outpatient/prescription total number distributions (by year?)  

#### EXAMPLES ####
# TODO: Add some examples of how the data looks like, e.g. patients that have 0 costs, patients with/without care, 

# TODO: Add parts from the end of 5_best_models.R file from Reproduction folder 


################
#### T-TEST ####
################

# Perform t-tests to determine significant differences in means between High-Cost and not High-Cost patients.
t.test(Need_of_Care_Duration ~ HC_Patient, data = data)
t.test(Age ~ HC_Patient, data = data)
t.test(DMP_Duration ~ HC_Patient, data = data)
t.test(Total_Costs ~ HC_Patient, data = data)
t.test(Inpatient_Num_Diagnoses ~ HC_Patient, data = data)
t.test(Outpatient_Num_Diagnoses ~ HC_Patient, data = data)
t.test(Prescription_Num_Prescriptions ~ HC_Patient, data = data)

#######################
#### TEST FUNCTION ####
#######################
# With this function, I can fast & reliably check if data transformation improves performance. 

test_data <- function(train_data, test_data, label_pos, first_val, last_val) {
    library(h2o)
    train_data[,label_pos] <- as.factor(train_data[,label_pos]) 
    test_data[,label_pos] <- as.factor(test_data[,label_pos]) 
    h2o.init()
    train_data <- as.h2o(train_data)
    test_data <- as.h2o(test_data)
    lr_model <- h2o.glm(x                               = first_val:last_val, 
                    y                                   = label_pos,
                    training_frame                      = train_data, 
                    seed                                = 12345)

    return(evaluate_model(lr_model, '', FALSE, test_data))
}

test_data(train, train, 1, 2, ncol(data))
test_data(train, validate, 1, 2, ncol(data))

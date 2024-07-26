################################
#### PREPARE DATA ##############
################################
# File: 1_data_preparation.R
# Author: Sebastian Benno Veuskens 
# Date: 2024-05-06
# Data: The raw data set 'HC_Patient_Data.csv'
#
# This script create train_validate, train, validate and test data sets from the raw data set 'HC_Patient_Data.csv'.
# train_validate is the data set that consists of train and validate data sets. 
# The data sets are saved as '.RData' files for better performance (short loading times). 


### MODIFY ####
# Your working directory
setwd('C:/Users/s.veuskens/Documents/Sebastian/Projekt Sebastian/modelling')
# Indicates whether to include High-Cost patients from the last year into analysis 
filter_hc <- FALSE 
# Indicates whether to include as many High-Cost patients as not-High-Cost patients 
balance_hc <- FALSE 
# The training dataset is split into train and validate. Validate is used to compare different hyperparameters 
proportion_train <- 0.75
# The years that are used for training the models
year_training <- c(2019, 2020)
#The year that is used to test the models 
year_test <- c(2021)
# Columns to exclude from data sets
exclude_cols <- c('year', 'ID')
# Whether you want to save your results (and overwrite the old results) or not
overwrite <- TRUE
#### MODIFY END ####

# Indicate from which relative location to load & save from. Depends on user input. 
relative_dir <- paste0(ifelse(filter_hc, 'filtered/', 'complete/'), ifelse(balance_hc, 'balanced/', 'unbalanced/'))

set.seed(12345)

data_2019 <- read.csv2('data/raw/HC_Patient_Data_19.csv', header=TRUE, sep=',')
data_2020 <- read.csv2('data/raw/HC_Patient_Data_20.csv', header=TRUE, sep=',')
data_2021 <- read.csv2('data/raw/HC_Patient_Data_21.csv', header=TRUE, sep=',')
data <- rbind(data_2019, data_2020, data_2021)

# Make label factor 
data$HC_Patient_Next_Year <- as.factor(data$HC_Patient_Next_Year)

# Filter out the (past) High-Cost patients, if specified 
if (filter_hc) {
    data <- data[data$HC_Patient == 0, ]
}

# Balance the data set, if specified
# TODO: Either delete or adapt this part with data_2019... instead of just data 
if (balance_hc) {
    # The samples that will be in the final data set 
    samples_included <- c()
    # Indices of the High-Cost patients 
    indices_hc <- data$HC_Patient_Next_Year == 1
    # For each year, filter out all High-Cost patients and the same amount of not-High-Cost patients 
    for (year in c(year_training, year_test)) {
        indices_year        <- data$year == year
        samples_hc          <- data[indices_hc & indices_year,]
        all_samples_not_hc  <- data[(!indices_hc) & indices_year,]
        indices_not_hc      <- sample(1:nrow(all_samples_not_hc), nrow(samples_hc))
        samples_not_hc      <- all_samples_not_hc[indices_not_hc,]
        # Add the High-Cost patients to the final balanced data set
        samples_included    <- rbind(samples_included, samples_hc)
        # Add the not-High-Cost patients to the final balanced data set
        samples_included    <- rbind(samples_included, samples_not_hc)
    }
    # Continue working with the balanced data set 
    data <- as.data.frame(samples_included)
}

# Assign train_validate, train, validate and test datasets 
train_validate <- data[data$year %in% year_training, ]
sample_size <- floor(proportion_train * nrow(train_validate))
sample_indices <- sample(seq_len(nrow(train_validate)), size=sample_size)
train <- train_validate[sample_indices, ]
validate <- train_validate[-sample_indices, ]
test <- data[data$year %in% year_test, ]

# Exclude year and ID column 
exclude_indices <- which(names(data) %in% exclude_cols)
data_2019 <- data_2019[,-exclude_indices]
data_2020 <- data_2020[,-exclude_indices]
data_2021 <- data_2021[,-exclude_indices]
data <- data[,-exclude_indices]
train_validate <- train_validate[,-exclude_indices]
train <- train[,-exclude_indices]
validate <- validate[,-exclude_indices]
test <- test[,-exclude_indices]

# Overview over samples per each dataset 
dim(data)
dim(train_validate)
dim(train)
dim(validate)
dim(test)

# Save the datasets 
if (overwrite) {
    dir.create('data', showWarning=FALSE)
    dir.create(paste0('data/', ifelse(filter_hc, 'filtered', 'complete')), showWarnings=FALSE)
    dir.create(paste0('data/', relative_dir), showWarnings=FALSE)
    save(data_2019,      file=paste0('data/', relative_dir, 'data_2019.Rdata'))
    save(data_2020,      file=paste0('data/', relative_dir, 'data_2020.Rdata'))
    save(data_2021,      file=paste0('data/', relative_dir, 'data_2021.Rdata'))
    save(data,           file=paste0('data/', relative_dir, 'data.Rdata'))
    save(train_validate, file=paste0('data/', relative_dir, 'train_validate.Rdata'))
    save(train,          file=paste0('data/', relative_dir, 'train.Rdata'))
    save(validate,       file=paste0('data/', relative_dir, 'validate.Rdata'))
    save(test,           file=paste0('data/', relative_dir, 'test.Rdata'))
} 

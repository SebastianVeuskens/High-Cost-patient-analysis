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
setwd("C:/Users/s.veuskens/Documents/Sebastian/Projekt Sebastian/modelling")
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

set.seed(12345)

data_2019 <- read.csv2('data/raw/HC_Patient_Data_19.csv', header=TRUE, sep=',')
data_2020 <- read.csv2('data/raw/HC_Patient_Data_20.csv', header=TRUE, sep=',')
data_2021 <- read.csv2('data/raw/HC_Patient_Data_21.csv', header=TRUE, sep=',')
data <- rbind(data_2019, data_2020, data_2021)

# Make label factor 
data_2019$HC_Patient_Next_Year <- as.factor(data_2019$HC_Patient_Next_Year)
data_2020$HC_Patient_Next_Year <- as.factor(data_2020$HC_Patient_Next_Year)
data_2021$HC_Patient_Next_Year <- as.factor(data_2021$HC_Patient_Next_Year)
data$HC_Patient_Next_Year <- as.factor(data$HC_Patient_Next_Year)

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
    save(data_2019,      file='data/data_2019.Rdata')
    save(data_2020,      file='data/data_2020.Rdata')
    save(data_2021,      file='data/data_2021.Rdata')
    save(data,           file='data/data.Rdata')
    save(train_validate, file='data/train_validate.Rdata')
    save(train,          file='data/train.Rdata')
    save(validate,       file='data/validate.Rdata')
    save(test,           file='data/test.Rdata')

    write.csv(as.data.frame(data_2019),      file='data/data_2019.csv', row.names=FALSE)
    write.csv(as.data.frame(data_2020),      file='data/data_2020.csv', row.names=FALSE)
    write.csv(as.data.frame(data_2021),      file='data/data_2021.csv', row.names=FALSE)
    write.csv(as.data.frame(data),           file='data/data.csv', row.names=FALSE)
    write.csv(as.data.frame(train_validate), file='data/train_validate.csv', row.names=FALSE)
    write.csv(as.data.frame(train),          file='data/train.csv', row.names=FALSE)
    write.csv(as.data.frame(validate),       file='data/validate.csv', row.names=FALSE)
    write.csv(as.data.frame(test),           file='data/test.csv', row.names=FALSE)
} 

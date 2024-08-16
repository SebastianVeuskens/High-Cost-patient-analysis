#####################################
#### VALIDATE FINAL MODEL ###########
#####################################
# File: 4_model_validation.R
# Author: Sebastian Benno Veuskens 
# Date: 2024-05-13
# Data: Use the train_validate data set for training and the test data set for validation. 
#
# This script trains the best model from the model selection (see 3_model_performance.R) or user-specified model.
# For validation, the previously unseen test data set is used. 

#### MODIFY ####
# Your working directory 
setwd("C:/Users/Sebastian's work/OneDrive - OptiMedis AG/Dokumente/Coding/High-Cost-patient-analysis")
# Indicates whether to include High-Cost patients from the last year into analysis 
filter_hc <- FALSE 
# Indicates whether to include as many High-Cost patients as not-High-Cost patients 
balance_hc <- FALSE 
# Whether you want to save your results (and overwrite the old results) or not
overwrite <- TRUE
# Indicate the model to evaluate. Default (NULL) selects the best model from the model selection (see 3_model_performance.R).
user_model_name <- 'neural network'
# Number of variables to display in variable importance plot
num_vars <- 5
# Number of features to display in SHAP analysis plot
num_features <- 5
#### MODIFY END ####


#######################
#### PRELIMINARIES ####
#######################

#### LIBRARIES & SOURCES ####

# INSTALL LIBRARIES 
# install.packages('h2o')
# install.packages('cvAUC')
# install.packages('rlist')
# install.packages('PRROC')
# install.packages('dplyr')
# install.packages('caret')

# LOAD LIBRARIES & SOURCES
library(h2o)            # The modelling framework 
library(cvAUC)          # For the Area Under the Curve (AUC) computation
library(PRROC)          # For the ROC and Precision-Recall curve 
library(dplyr)          # To bind two data frames (predictions and labels) together 
library(caret)          # To compute the confusion matrix
source('code/utils.R')  # Auxiliary functions for simplicity and concise code 

#### LOAD DATA ####

# Indicate from which relative location to load & save from. Depends on user input. 
relative_dir <- paste0(ifelse(filter_hc, 'filtered/', 'complete/'), ifelse(balance_hc, 'balanced/', 'unbalanced/'))

load(paste0('data/', relative_dir, 'train_validate',    '.Rdata'))
load(paste0('data/', relative_dir, 'test', '.Rdata'))

# Start H2O package
h2o.init()

# Load data frames into H2O framework
train_validate <- as.h2o(train_validate)
test <- as.h2o(test)

# Load the best model identified in the model selection (see 3_model_performance.R) or the user-specified model
if (is.null(user_model_name)) {
    model_params <- list.load(paste0('results/', relative_dir, 'model_performance/best_model.RData'))
} else {
    ordered_models <- list.load(paste0('results/', relative_dir, 'model_performance/ordered_models.RData'))
    # Select the model as specified by the user. 
    # First, identify the index of the specified model in the ordered_models object.
    model_ind <- which(ordered_models[[1]] == user_model_name) 
    # Second, extract the model at the index model_ind from the ordered_models object that contains all model parameters. 
    model_params <- lapply(ordered_models, function(model) {model[[model_ind]]})
}

#### CREATE FOLDER STRUCTURE ####
if (overwrite) {
    dir.create('results', showWarnings=FALSE)
    dir.create(paste0('results/', ifelse(filter_hc, 'filtered', 'complete')), showWarnings=FALSE)
    dir.create(paste0('results/', relative_dir), showWarnings=FALSE)
    dir.create(paste0('results/', relative_dir, 'model_validation'), showWarnings=FALSE)
}


##################################################################
###################### MODEL EVALUATION ##########################
##################################################################

# Position of label and variables. Indicate where the features for prediction should start and end in the data.
label_pos <- 1 
first_val <- 3
last_val <- ncol(train_validate)

# Train the model. Use the train_model function from the utils.R file
model <- train_model(model_params, train_validate, first_val, last_val, label_pos)

# Save the model
# Exchange tabs with underscore for consistent file naming (to correct user input).  
model_name <- gsub(' ', '_', model_params[[1]])
h2o.saveModel(model, paste0('results/', relative_dir, 'model_validation'), filename=model_name, force=TRUE)


#####################################
#### STANDARD EVALUATION METRICS ####
#####################################

# Evaluate the model using the standard metrics. Use the evaluate_model function from the utils.R file
filepath <- paste0('results/', relative_dir, 'model_validation/', model_name)
results <- evaluate_model(model, filepath, overwrite, newdata=test)


############################
#### PERFORMANCE CURVES ####
############################

# Bind the predicted and true labels together 
predictions <- as.data.frame(h2o.predict(model, test))
label <- as.data.frame(test)$HC_Patient_Next_Year
predicted_vs_true <- bind_cols(predictions, label=label)
# Get the prediction probabilities for each class 
class_1_probabilities <- predicted_vs_true$p1[predicted_vs_true$label == 1]
class_0_probabilities <- predicted_vs_true$p1[predicted_vs_true$label == 0]

# Receiver Operating Characteristic (ROC) curve 
roc <- roc.curve(class_1_probabilities, class_0_probabilities, curve=TRUE)
aucroc <- roc$auc
# Display and save the ROC curve 
print(paste('ROC Area under the curve (AUC):', round(aucroc, 4)))
plot(roc)
if (overwrite) {
    png(paste0(filepath, '_roc_curve.png'))
    plot(roc)
    dev.off() 
}

# Precision-Recall curve
# TODO: Check if this gives the same results as the pr-curve in Benedikts script
# TODO: Run h2o.aucpr in Benedikt script to see if he gets such low results there too 
pr <- pr.curve(class_1_probabilities, class_0_probabilities, curve=TRUE)
aucpr <- pr$auc.integral
# Display and save the Precision-Recall curve 
print(paste('Precision-Recall Area under the curve (AUC):', round(aucpr, 4)))
plot(pr)
if (overwrite) {
    png(paste0(filepath, '_precision_recall_curve.png'))
    plot(pr)
    dev.off() 
}

# Validate results by comparing the AUC with the Precision-Recall AUC of the h2o built-in function. 
aucpr_h2o <- h2o.aucpr(h2o.performance(model, test))
aucpr_match <- isTRUE(all.equal(aucpr, aucpr_h2o, tolerance=0.1))
if (!aucpr_match) {
    print('EXCEPTON: CHECK PRECISION-RECALL COMPUTATION!')
}

# Confusion matrix
conf_matrix <- confusionMatrix(predicted_vs_true$predict, predicted_vs_true$label)
print(conf_matrix$table)


#####################
#### XAI METHODS ####
##################### 

# Variable importance
# Display and save the variable importance plot 
varimp_plot <- h2o.varimp_plot(model, num_of_features=num_vars)
if (overwrite) {
    png(paste0(filepath, '_variable_importance.png'))
    h2o.varimp_plot(model, num_of_features=num_vars)
    dev.off()
}

# SHAP analysis 
# Display and save the SHAP analysis plot
shap <- h2o.shap_summary_plot(model, newdata = test, top_n_features = num_features)
if (overwrite) {
    png(paste0(filepath, '_shap_analysis.png'))
    shap <- h2o.shap_summary_plot(model, newdata = test, top_n_features = num_features)
    dev.off()
}

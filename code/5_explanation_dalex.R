#########################################
#### CREATE SIMPLE TEST MODEL ###########
#########################################
# File: 5_explanation_dalex.R
# Author: Sebastian Benno Veuskens 
# Date: 2024-07-10
# Data: Create a small data set with simple relations. 
#
# The purpose of this script is to create a model that is easy to debug.
# The model will be used for proof-of-concepts of the explanation methods. 

#### MODIFY ####
# Your working directory 
setwd("C:/Users/s.veuskens/Documents/Sebastian/Projekt Sebastian/modelling")
# Indicates whether to include High-Cost patients from the last year into analysis 
filter_hc <- FALSE 
# Indicates whether to include as many High-Cost patients as not-High-Cost patients 
balance_hc <- FALSE 
# Indicate the model to evaluate. Default (NULL) selects the best model from the model selection (see 3_model_selection.R).
# TODO: Change this to NULL at the end 
user_model_name <- 'random forest'
# Whether to use H2O package or standard R packages
use_h2o <- FALSE
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
# Specify the number of features you want to include in the LIME surrogate model  
n_features <- 2
#### MODIFY END ####

#######################
#### PRELIMINARIES ####
#######################

#### LIBRARIES & SOURCES ####

# INSTALL LIBRARIES 
# install.packages("dplyr")
# install.packages("h2o")
# install.packages('randomForest')
# install.packages("pdp")               
# install.packages("shapper")  
# install.packages("cvAUC")
# install.packages("PRROC") 
# install.packages("caret")
# install.packages('randomForest')
# install.packages('DALEX')
# install.packages('DALEXtra')
# install.packages('lime')
# install.packages('localModel')

# LOAD LIBRARIES & SOURCES
library(dplyr)          # To bind two data frames (predictions and labels) together 
library(h2o)            # The modelling framework 
library(randomForest)   # For the random Forest in R
library(pdp)            # PDP 
library(shapper)        # SHAP 
library(cvAUC)          # For the Area Under the Curve (AUC) computation
library(PRROC)          # For the ROC and Precision-Recall curve
library(caret)          # To compute the confusion matrix
library(randomForest)   # To model the R native Random Forest 
library(DALEX)          # For model explanations 
library(DALEXtra)       # Needed for lime functionality of DALEX 
library(lime)           # Needed for lime calculation in DALEX 
library(localModel)     # Used for additional local explanation 
source('code/utils.R')  # Auxiliary functions for simplicity and concise code 

#### LOAD DATA ####

# Indicate from which relative location to load & save from. Depends on user input. 
relative_dir <- paste0(ifelse(filter_hc, 'filtered/', 'complete/'), ifelse(balance_hc, 'balanced/', 'unbalanced/'))

load(paste0('data/', relative_dir, 'train_validate',    '.Rdata'))
load(paste0('data/', relative_dir, 'test', '.Rdata'))

# train_validate$Sex <- as.factor(train_validate$Sex)
# test$Sex <- as.factor(test$Sex)

#### CREATE FOLDER STRUCTURE ####
if (overwrite) {
    dir.create('results', showWarnings=FALSE)
    dir.create(paste0('results/', ifelse(filter_hc, 'filtered', 'complete')), showWarnings=FALSE)
    dir.create(paste0('results/', relative_dir), showWarnings=FALSE)
    dir.create(paste0('results/', relative_dir, 'model_explanation'), showWarnings=FALSE)
}


#### LOAD MODEL #### 
# Specify location to save the model 
model_name <- gsub(' ', '_', user_model_name) 
model_filepath <- paste0('results/', relative_dir, 'model_explanation/', model_name)

# Load the best parameters from hyperparameter tuning for the specified model 
filename_params <- paste0(model_name, '_best_parameters.RData')
params <- list.load(paste0('results/', relative_dir, 'model_tuning/', filename_params))
best_params <- params[[1]]

if (use_h2o) {
    # Start H2O package
    h2o.init()

    # Load data frames into H2O framework
    train_validate <- as.h2o(train_validate)
    test <- as.h2o(test)

    # Load the model 
    # TODO: Check where I exactly load this here from 
    model <- h2o.loadModel(model_filepath)
    predictions <- as.data.frame(h2o.predict(model, test))
    
} else {
    if (file.exists(paste0(model_filepath, '.RData'))) {
        model <- readRDS(paste0(model_filepath, '.RData'))
    } else {
        if (user_model_name == 'random forest') {
            ntrees <- as.numeric(best_params[['ntrees']])
            mtries <- as.numeric(best_params[['mtries']])
            model <- randomForest(formula = HC_Patient_Next_Year ~ ., data=train_validate[,-2], ntree=ntrees, mtry=mtries)
            model$cutoff <- evaluate_r_model(model, model_filepath, overwrite, newdata=test[,-2])[[1]]
            saveRDS(model, paste0(model_filepath, '.RData'))
        } else {
            warning('ONLY RANDOM FOREST IS IMPLEMENTED SO FAR')
        }
    }
    prediction_probs <- predict(model, test[,-2], type='prob')
    predictions <- as.numeric(prediction_probs >= model$cutoff)
}

###############################################################
##################### XAI METHODS #############################
###############################################################

#######################
#### PRELIMINARIES ####
#######################

# Select samples
test_df <- as.data.frame(test)
train_validate_df <- as.data.frame(train_validate)
test_df$HC_Patient_Next_Year <- as.numeric(test_df$HC_Patient_Next_Year)
train_validate_df$HC_Patient_Next_Year <- as.numeric(train_validate_df$HC_Patient_Next_Year)

true_pos <- test_df[test_df$HC_Patient_Next_Year == 1 & predictions == 1,]
false_pos <- test_df[test_df$HC_Patient_Next_Year == 1 & predictions == 0,]
true_neg <- test_df[test_df$HC_Patient_Next_Year == 0 & predictions == 0,]
false_neg <- test_df[test_df$HC_Patient_Next_Year == 0 & predictions == 1,]

nrow(true_pos)
nrow(false_pos)
nrow(true_neg)
nrow(false_neg)

if (use_pos) {
    if (use_true) {
        sample <- true_pos[1,]
    } else {
        sample <- false_pos[1,]
    }
} else {
    if (use_true) {
        sample <- true_neg[1,]
    } else {
        sample <- false_neg[1,]
    }
}

# TODO: Add part for logistic regresion later 
if (user_model_name == 'logistic regression') {
    warning('NOT THE RIGHT PREDICTORS USED') 
} else {
    predictors <- setdiff(names(test), c(target, excluded))
}
feature_of_interest <- 'Age'

if (use_h2o) {
    exp_dalex <- DALEXtra::explain_h2o(model, data = train_validate_df[predictors], y=train_validate_df[target])
} else {
    exp_dalex <- DALEX::explain(model, data=train_validate_df[predictors], y=train_validate_df[target])
}

##########################
#### BREAK-DOWN PLOTS ####
##########################
# CHARACTERISTIC: Uses different order of explanatory covariates to calculate feature attribution

# Start time measurement 
bd_start <- Sys.time()

bd_dalex <- predict_parts(explainer=exp_dalex,
                          new_observation=sample,
                          type='break_down_interactions')

plot(bd_dalex)

# Stop and report time
bd_end <- Sys.time()
bd_runtim <- difftime(bd_end, bd_start, units='mins')
print(paste0('Runtime for break-down plots: ', round(bd_runtime, 2), ' minutes'))

##############
#### SHAP ####
##############

# Start time measurement 
shap_start <- Sys.time()

shap_dalex <- predict_parts(explainer=exp_dalex, 
                            new_observation=sample, 
                            type='shap',
                            B=25 # Number of orderings of explanatory variables to compute the shapley values 
                           )

# EXPLANATION: With boxplots, one can see whether effects are reliable.
# If, for example, boxplot goes over zero line, effect could also be negative with statistical tolerance 
plot(shap_dalex, show_boxplots=TRUE)   

# Stop and report time
shap_end <- Sys.time()
shap_runtim <- difftime(shap_end, shap_start, units='mins')
print(paste0('Runtime for SHAP plots: ', round(shap_runtime, 2), ' minutes'))


##############
#### LIME ####
##############
# CHARACTERISTIC: Useful for explaining when a lot of explanatory variables exist, 
# in opposition to SHAP (and other XAI methods) -> Here the case. 

# Start time measurement 
lime_start <- Sys.time()

# Assert correct function naming between DALEX, DALEXtra and lime
model_type.dalex_explainer <- DALEXtra::model_type.dalex_explainer
predict_model.dalex_explainer <- DALEXtra::predict_model.dalex_explainer

lime_dalex <- predict_surrogate(explainer=exp_dalex,
                                new_observation=sample,
                                n_features=n_features_lime,
                                n_permutations=1000,
                                type='lime')
plot(lime_dalex)

# Stop and report time
lime_end <- Sys.time()
lime_runtim <- difftime(lime_end, lime_start, units='mins')
print(paste0('Runtime for LIME plots: ', round(lime_runtime, 2), ' minutes'))


#####################
#### LOCAL MODEL ####
#####################
# CHARACTERISTIC: Can be understood and used as method to see what-if,
# somehow similar to Counterfactual explanations 

# Start time measurement 
locModel_start <- Sys.time()

# Also uses LIME. 
locModel_dalex <- predict_surrogate(explainer=exp_dalex,
                                    new_observation=sample,
                                    size=1000,
                                    seed=1,
                                    type='localModel')

plot_interpretable_feature(locModel_dalex, feature_of_interest)                                    

# Stop and report time
locModel_end <- Sys.time()
locModel_runtim <- difftime(locModel_end, locModel_start, units='mins')
print(paste0('Runtime for local model plots: ', round(locModel_runtime, 2), ' minutes'))


################################
#### LOCAL-DIAGNOSTIC PLOTS ####
################################
# CHARACTERISTIC: Not established in literature -> Have to think about whether to use it or not 
# Helpful to compare to other users and determine model reliability in that certain region. 

# Start time measurement 
locDiag_start <- Sys.time()

neighbors <- 100
locDiag_dalex_gen <- DALEX::predict_diagnostics(explainer=exp_dalex,
                                         new_observation=sample,
                                         neighbors=neighbors)

plot(locDiag_dalex_gen)

neighbors <- 10
locDiag_dalex_ind <- predict_diagnostics(explainer=exp_dalex,
                                         new_observation=sample,
                                         neighbors=neighbors,
                                         variables=feature_of_interest)

plot(locDiag_dalex_ind)

# Stop and report time
locDiag_end <- Sys.time()
locDiag_runtim <- difftime(locDiag_end, locDiag_start, units='mins')
print(paste0('Runtime for local diagnostic plots: ', round(locDiag_runtime, 2), ' minutes'))


#############################
#### VARIABLE IMPORTANCE ####
#############################
# CHARACTERISTICS: Depends on the ordering of the variables (can be changed with the variables argument)

# Start time measurement 
vImp_start <- Sys.time()

# Variable importance based on RMSE loss 
set.seed(12345)
vImp_dalex <- model_parts(explainer=exp_dalex,
                          loss_function=DALEX::loss_one_minus_auc,
                          B=1,
                          type='difference')

plot(vImp_dalex)    

# Stop and report time
vImp_end <- Sys.time()
vImp_runtim <- difftime(vImp_end, vImp_start, units='mins')
print(paste0('Runtime variable importence plots: ', round(vImp_runtime, 2), ' minutes'))


#############
#### PDP ####
#############
# CHARACTERISTICS: Can be split up into different categories and show different variables in one plot
# -> not exactly 2-variable plot, only for categorical + numerical variable pair 

#### Standard PDP plot

# Start time measurement 
pdp_start <- Sys.time()

pdp_dalex <- model_profile(explainer=exp_dalex, 
                           variables=feature_of_interest)

plot(pdp_dalex)                           

# Stop and report time
pdp_end <- Sys.time()
pdp_runtim <- difftime(pdp_end, pdp_start, units='mins')
print(paste0('Runtime for PDP plots: ', round(pdp_runtime, 2), ' minutes'))


#### Grouped PDP plot 

# Start time measurement 
gPdp_start <- Sys.time()

feature_to_group <- 'x1' # Must be categorical variable 

gPdp_dalex <- model_profile(explainer=exp_dalex, 
                           variables=feature_of_interest,
                           groups=feature_to_group)

plot(gPdp_dalex)    

# Stop and report time
gPdp_end <- Sys.time()
gPdp_runtim <- difftime(gPdp_end, gPdp_start, units='mins')
print(paste0('Runtime for grouped PDP plots: ', round(gPdp_runtime, 2), ' minutes'))

#  Save the runtimes
methods <- c('Break-down plot', 'SHAP', 'LIME', 'Local Model', 
             'Local Diagnostics', 'Variable Importance', 'PDP', 'Grouped PDP')
runtimes_in_minutes <- round(c(bd_runtime, shap_runtime, lime_runtime, locModel_runtime,
                               locDiag_runtime, vImp_runtime, pdp_runtime, gPdp_runtime), 4)
results <- data.frame(methods, runtimes_in_minutes)
rt_filepath <- paste0('results/', relative_dir, 'model_explanation/runtimes_methods.csv')
if (overwrite) write.csv(results, rt_filepath)
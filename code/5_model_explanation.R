#########################################
#### CREATE SIMPLE TEST MODEL ###########
#########################################
# File: test_model.R
# Author: Sebastian Benno Veuskens 
# Date: 2024-07-10
# Data: Create a small data set with simple relations. 
#
# The purpose of this script is to create a model that is easy to debug.
# The model will be used for proof-of-concepts of the explanation methods. 

#### MODIFY ####
# Your working directory 
# .libPaths("C:/Users/Sebastian's work/OneDrive - OptiMedis AG/Dokumente/Coding/Settings/R_libraries")
setwd("C:/Users/Sebastian's work/OneDrive - OptiMedis AG/Dokumente/Coding/High-Cost-patient-analysis")
# Indicates whether to include High-Cost patients from the last year into analysis 
filter_hc <- FALSE 
# Indicates whether to include as many High-Cost patients as not-High-Cost patients 
balance_hc <- FALSE 
# Indicate the model to evaluate. Default (NULL) selects the best model from the model selection (see 3_model_selection.R).
user_model_name <- 'random forest'
# Whether to use H2O package or standard R packages
use_h2o <- FALSE
# Target variable
target <- 'HC_Patient_Next_Year'
# Variables to exclude 
excluded <- 'Total_Costs_Next_Year'
#### MODIFY END ####

#######################
#### PRELIMINARIES ####
#######################

#### LIBRARIES & SOURCES ####

# INSTALL LIBRARIES 
# install.packages("dplyr")
# install.packages("h2o")
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
library(pdp)            # PDP 
library(shapper)        # SHAP 
library(cvAUC)          # For the Area Under the Curve (AUC) computation
library(PRROC)          # For the ROC and Precision-Recall curve
library(caret)          # To compute the confusion matrix
library (randomForest)  # To model the R native Random Forest 
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

# Select sample
# TODO: Delete this part later 
num_samples <- 3000
train_validate <- train_validate[sample(nrow(train_validate), num_samples),]
test <- test[sample(nrow(test), num_samples),]
if (use_h2o) {
    # Start H2O package
    h2o.init()

    # Load data frames into H2O framework
    train_validate <- as.h2o(train_validate)
    test <- as.h2o(test)
    # Load the best model identified in the model selection (see 3_model_selection.R) or the user-specified model
    if (is.null(user_model_name)) {
        model_params <- list.load(paste0('results/', relative_dir, 'model_selection/best_model.RData'))
    } else {
        ordered_models <- list.load(paste0('results/', relative_dir, 'model_selection/ordered_models.RData'))
        # Select the model as specified by the user. 
        # First, identify the index of the specified model in the ordered_models object.
        model_ind <- which(ordered_models[[1]] == user_model_name) 
        # Second, extract the model at the index model_ind from the ordered_models object that contains all model parameters. 
        model_params <- lapply(ordered_models, function(model) {model[[model_ind]]})
    }

    # Position of label and variables. Indicate where the features for prediction should start and end in the data.
    label_pos <- 1 
    first_val <- 3
    last_val <- ncol(train_validate)

    # Train the model. Use the train_model function from the utils.R file
    model <- train_model(model_params, train_validate, first_val, last_val, label_pos)
} else {
    model <- randomForest(formula = HC_Patient_Next_Year ~ ., data=train_validate[,-2], ntree=1000, mtry=30)
}

###############################################################
##################### XAI METHODS #############################
###############################################################

#######################
#### PRELIMINARIES ####
#######################

predictors <- setdiff(names(test), c(target, excluded))
sample <- test[predictors][1,]
feature_of_interest <- 'Age'

if (use_h2o) {
    train_validate_df <- as.data.frame(train_validate)
    sample_df <- as.data.frame(sample)
    predict_fn <- function(model, newdata) {
        results <- as.data.frame(h2o.predict(model, as.h2o(newdata)))
        return(results[[3L]])
    exp_dalex <- DALEX::explain(model, data=train_validate_df[predictors], y=train_validate_df[target], predict_function=predict_fn)
    }
} else {
    exp_dalex <- DALEX::explain(model, data=train_validate_df[predictors], y=train_validate_df[target])
}

##########################
#### BREAK-DOWN PLOTS ####
##########################
# CHARACTERISTIC: Uses different order of explanatory covariates to calculate feature attribution
bd_dalex <- predict_parts(explainer=exp_dalex,
                          new_observation=sample,
                          type='break_down_interactions')

plot(bd_dalex)

##############
#### SHAP ####
##############

shap_dalex <- predict_parts(explainer=exp_dalex, 
                            new_observation=sample, 
                            type='shap',
                            B=25 # Number of orderings of explanatory variables to compute the shapley values 
                           )

# EXPLANATION: With boxplots, one can see whether effects are reliable.
# If, for example, boxplot goes over zero line, effect could also be negative with statistical tolerance 
plot(shap_dalex, show_boxplots=TRUE)   

##############
#### LIME ####
##############
# CHARACTERISTIC: Useful for explaining when a lot of explanatory variables exist, 
# in opposition to SHAP (and other XAI methods) -> Here the case. 

# Specify the number of features you want to include in the LIME surrogate model  
n_features <- 2
# Assert correct function naming between DALEX, DALEXtra and lime
model_type.dalex_explainer <- DALEXtra::model_type.dalex_explainer
predict_model.dalex_explainer <- DALEXtra::predict_model.dalex_explainer

lime_dalex <- predict_surrogate(explainer=exp_dalex,
                                new_observation=sample,
                                n_features=n_features,
                                n_permutations=1000,
                                type='lime')
plot(lime_dalex)

#####################
#### LOCAL MODEL ####
#####################
# CHARACTERISTIC: Can be understood and used as method to see what-if,
# somehow similar to Counterfactual explanations 
# Also uses LIME. 
locModel_dalex <- predict_surrogate(explainer=exp_dalex,
                                    new_observation=sample,
                                    size=1000,
                                    seed=1,
                                    type='localModel')

plot_interpretable_feature(locModel_dalex, feature_of_interest)                                    

################################
#### LOCAL-DIAGNOSTIC PLOTS ####
################################
# CHARACTERISTIC: Not established in literature -> Have to think about whether to use it or not 
# Helpful to compare to other users and determine model reliability in that certain region. 
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

#############################
#### VARIABLE IMPORTANCE ####
#############################
# CHARACTERISTICS: Depends on the ordering of the variables (can be changed with the variables argument)
# Variable importance based on RMSE loss 
set.seed(12345)
vImp_dalex <- model_parts(explainer=exp_dalex,
                          loss_function=DALEX::loss_one_minus_auc,
                          B=1,
                          type='difference')

plot(vImp_dalex)    

#############
#### PDP ####
#############
# CHARACTERISTICS: Can be split up into different categories and show different variables in one plot
# -> not exactly 2-variable plot, only for categorical + numerical variable pair 

#### Standard PDP plot
pdp_dalex <- model_profile(explainer=exp_dalex, 
                           variables=feature_of_interest)

plot(pdp_dalex)                           

#### Grouped PDP plot 
feature_to_group <- 'x1' # Must be categorical variable 

pdp_dalex <- model_profile(explainer=exp_dalex, 
                           variables=feature_of_interest,
                           groups=feature_to_group)

plot(pdp_dalex)    
#########################################
#### CREATE SIMPLE TEST MODEL ###########
#########################################
# File: 5_explanation_dalex.R
# Author: Sebastian Benno Veuskens 
# Date: 2024-07-10
# Data: 
#
# The purpose of this script is to 

#### MODIFY ####
# Your working directory 
setwd("C:/Users/s.veuskens/Documents/Sebastian/Projekt Sebastian/modelling")
# Indicate the model to evaluate. Default (NULL) selects the best model from the model selection (see 3_model_selection.R).
# TODO: Change this to NULL at the end 
user_model_name <- 'random forest'
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
n_features_lime <- 5
# Specify which feature to investigate better 
feature_of_interest <- 'Age'
# The index of the sample I want to investigate (different indexing in R an Python)
sample_indices <- 1467 + 1
# Variable to be used for grouping by grouped PDP plot
feature_to_group <- 'Total_Costs' # Must be categorical variable 
#### MODIFY END ####

#######################
#### PRELIMINARIES ####
#######################

#### LIBRARIES & SOURCES ####

# INSTALL LIBRARIES 
# install.packages("dplyr")
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

load('data/train_validate.Rdata')
load('data/test.Rdata')

# train_validate$Sex <- as.factor(train_validate$Sex)
# test$Sex <- as.factor(test$Sex)

#### CREATE FOLDER STRUCTURE ####
if (overwrite) {
    dir.create('results', showWarnings=FALSE)
    dir.create('results/model_explanation', showWarnings=FALSE)
    dir.create('results/model_explanation/dalex', showWarnings=FALSE)
}


#### LOAD MODEL #### 
# Specify location to save the model 
model_name <- gsub(' ', '_', user_model_name) 
model_filepath <- paste0('results/model_explanation/dalex/', model_name)

# Load the best parameters from hyperparameter tuning for the specified model 
filename_params <- paste0(model_name, '_best_parameters.RData')
params <- list.load(paste0('results/model_tuning/', filename_params))
best_params <- params[[1]]

excluded_idx <- which(names(test) == excluded)
if (file.exists(paste0(model_filepath, '.RData'))) {
    model <- readRDS(paste0(model_filepath, '.RData'))
} else {
    if (user_model_name == 'random forest') {
        ntrees <- as.numeric(best_params[['ntrees']])
        mtries <- as.numeric(best_params[['mtries']])
        model <- randomForest(formula = HC_Patient_Next_Year ~ ., data=train_validate[,-excluded_idx], ntree=ntrees, mtry=mtries)
        model$cutoff <- evaluate_r_model(model, model_filepath, overwrite, newdata=test[,-excluded_idx])[[1]]
        saveRDS(model, paste0(model_filepath, '.RData'))
    } else {
        warning('ONLY RANDOM FOREST IS IMPLEMENTED SO FAR')
    }
}
prediction_probs <- predict(model, test[,-excluded_idx], type='prob')
predictions <- as.numeric(prediction_probs >= model$cutoff)

###############################################################
##################### XAI METHODS #############################
###############################################################

#######################
#### PRELIMINARIES ####
#######################

# Specifiy where to save the results
result_filepath <- paste0('results/model_explanation/dalex/')

# Select samples
test_df <- as.data.frame(test)
train_validate_df <- as.data.frame(train_validate)
test_df$HC_Patient_Next_Year <- as.numeric(test_df$HC_Patient_Next_Year) - 1
train_validate_df$HC_Patient_Next_Year <- as.numeric(train_validate_df$HC_Patient_Next_Year) - 1

# true_pos <- test_df[test_df$HC_Patient_Next_Year == 1 & predictions == 1,]
# false_pos <- test_df[test_df$HC_Patient_Next_Year == 1 & predictions == 0,]
# true_neg <- test_df[test_df$HC_Patient_Next_Year == 0 & predictions == 0,]
# false_neg <- test_df[test_df$HC_Patient_Next_Year == 0 & predictions == 1,]

# nrow(true_pos)
# nrow(false_pos)
# nrow(true_neg)
# nrow(false_neg)

# if (use_pos) {
#     if (use_true) {
#         sample <- true_pos[1,]
#     } else {
#         sample <- false_pos[1,]
#     }
# } else {
#     if (use_true) {
#         sample <- true_neg[1,]
#     } else {
#         sample <- false_neg[1,]
#     }
# }

sample <- as.data.frame(test[sample_indices,])

# TODO: Add part for logistic regresion later 
if (user_model_name == 'logistic regression') {
    warning('NOT THE RIGHT PREDICTORS USED') 
} else {
    predictors <- setdiff(names(test), c(target, excluded))
}

exp_dalex <- DALEX::explain(model, data=train_validate_df[,predictors], y=train_validate_df[target])

#############################
#### VARIABLE IMPORTANCE ####
#############################
# CHARACTERISTIC: Ranking depends on the order of the variables, therefore permutations
#                 Loss function has to be specified: Here cross_entropy (binary classification problem)

# Start time measurement 
vi_start <- Sys.time()

vi_dalex <- model_parts(explainer=exp_dalex,
                        # loss_function=loss_cross_entropy,
                        B=1,
                        type='difference')

if (overwrite) {png(paste0(result_filepath, 'variable_importance.png'))}
plot(vi_dalex)
if (overwrite) {dev.off()}

# Stop and report time
vi_end <- Sys.time()
vi_runtime <- difftime(vi_end, vi_start, units='mins')
print(paste0('Runtime for variable importance plots: ', round(vi_runtime, 2), ' minutes'))

##############
#### SHAP ####
##############

# Start time measurement 
shap_start <- Sys.time()

shap_dalex <- predict_parts(explainer=exp_dalex, 
                            new_observation=sample, 
                            type='shap',
                            B=10 # Number of orderings of explanatory variables to compute the shapley values 
                           )

# EXPLANATION: With boxplots, one can see whether effects are reliable.
# If, for example, boxplot goes over zero line, effect could also be negative with statistical tolerance 
if (overwrite) {png(paste0(result_filepath, 'shap_local_analysis.png'))}
plot(shap_dalex, show_boxplots=TRUE)   
if (overwrite) {dev.off()}

# Stop and report time
shap_end <- Sys.time()
shap_runtime <- difftime(shap_end, shap_start, units='mins')
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

if (overwrite) {png(paste0(result_filepath, n_features_lime, '_features_lime_plot.png'), width=700, height=375)}
plot(lime_dalex)
if (overwrite) {dev.off()}

# Stop and report time
lime_end <- Sys.time()
lime_runtime <- difftime(lime_end, lime_start, units='mins')
print(paste0('Runtime for LIME plots: ', round(lime_runtime, 2), ' minutes'))


#####################
#### LOCAL MODEL ####
#####################
# CHARACTERISTIC: Can be understood and used as method to see what-if,
# somehow similar to Counterfactual explanations 

# Start time measurement 
# locModel_start <- Sys.time()

# # Also uses LIME. 
# locModel_dalex <- predict_surrogate(explainer=exp_dalex,
#                                     new_observation=sample,
#                                     size=1000,
#                                     seed=1,
#                                     type='localModel')

# plot_interpretable_feature(locModel_dalex, feature_of_interest)                                    

# # Stop and report time
# locModel_end <- Sys.time()
# locModel_runtime <- difftime(locModel_end, locModel_start, units='mins')
# print(paste0('Runtime for local model plots: ', round(locModel_runtime, 2), ' minutes'))


################################
#### LOCAL-DIAGNOSTIC PLOTS ####
################################
# CHARACTERISTIC: Not established in literature -> Have to think about whether to use it or not 
# Helpful to compare to other users and determine model reliability in that certain region. 

# Start time measurement 
# locDiag_start <- Sys.time()

# neighbors <- 100
# locDiag_dalex_gen <- DALEX::predict_diagnostics(explainer=exp_dalex,
#                                          new_observation=sample,
#                                          neighbors=neighbors)

# plot(locDiag_dalex_gen)

# neighbors <- 10
# locDiag_dalex_ind <- predict_diagnostics(explainer=exp_dalex,
#                                          new_observation=sample,
#                                          neighbors=neighbors,
#                                          variables=feature_of_interest)

# plot(locDiag_dalex_ind)

# # Stop and report time
# locDiag_end <- Sys.time()
# locDiag_runtime <- difftime(locDiag_end, locDiag_start, units='mins')
# print(paste0('Runtime for local diagnostic plots: ', round(locDiag_runtime, 2), ' minutes'))


#############################
#### VARIABLE IMPORTANCE ####
#############################
# CHARACTERISTICS: Depends on the ordering of the variables (can be changed with the variables argument)

# Start time measurement 
# vImp_start <- Sys.time()

# # Variable importance based on RMSE loss 
# set.seed(12345)
# vImp_dalex <- model_parts(explainer=exp_dalex,
#                           loss_function=DALEX::loss_one_minus_auc,
#                           B=1,
#                           type='difference')

# plot(vImp_dalex)    

# # Stop and report time
# vImp_end <- Sys.time()
# vImp_runtime <- difftime(vImp_end, vImp_start, units='mins')
# print(paste0('Runtime variable importence plots: ', round(vImp_runtime, 2), ' minutes'))


#############
#### PDP ####
#############
# CHARACTERISTICS: Can be split up into different categories and show different variables in one plot
# -> not exactly 2-variable plot, only for categorical + numerical variable pair 

#### Standard PDP plot

# Start time measurement 
pdp_start <- Sys.time()

pdp_dalex <- model_profile(explainer=exp_dalex, 
                        type='partial', 
                        variables=feature_of_interest)

if (overwrite) {png(paste0(result_filepath, 'pdp_', feature_of_interest, '.png'))}
plot(pdp_dalex)                 
if (overwrite) {dev.off()}

# Stop and report time
pdp_end <- Sys.time()
pdp_runtime <- difftime(pdp_end, pdp_start, units='mins')
print(paste0('Runtime for PDP plots: ', round(pdp_runtime, 2), ' minutes'))


#### ALE plot 

# Start time measurement 
ale_start <- Sys.time()

ale_dalex <- model_profile(explainer=exp_dalex, 
                           type='accumulated', 
                           variables=feature_of_interest,
                           center=TRUE)

if (overwrite) {png(paste0(result_filepath, 'ale_', feature_of_interest, '.png'))}
plot(ale_dalex)
if (overwrite) {dev.off()}

# Stop and report time
ale_end <- Sys.time()
ale_runtime <- difftime(ale_end, ale_start, units='mins')
print(paste0('Runtime for ALE plots: ', round(ale_runtime, 2), ' minutes'))

#### Combined PDP & ALE plot
pdp_dalex$agr_profiles$'_label_' = 'partial_dependence'
ale_dalex$agr_profiles$'_label_' = 'accumulated_local'

if (overwrite) {png(paste0(result_filepath, 'pdp_ale_', feature_of_interest, '.png'))}
plot(pdp_dalex, ale_dalex) +
# scale_x_continuous(trans='log10', 
#                     breaks=c(0, 10, 100, 1000, 10000, 100000, 1000000),
#                     labels=scales::label_currency(suffix='€', prefix='')) + 
ylim(0, 0.4)
if (overwrite) {dev.off()}


#### Grouped PDP plot 

# Start time measurement 
gPdp_start <- Sys.time()

gPdp_dalex <- model_profile(explainer=exp_dalex, 
                           variables=feature_of_interest,
                           groups=feature_to_group)

if (overwrite) {png(paste0(result_filepath, 'pdp_', feature_of_interest, '_grouped_by_', feature_to_group, '.png'))}
plot(gPdp_dalex)  
if (overwrite) {dev.off()}

# Stop and report time
gPdp_end <- Sys.time()
gPdp_runtime <- difftime(gPdp_end, gPdp_start, units='mins')
print(paste0('Runtime for grouped PDP plots: ', round(gPdp_runtime, 2), ' minutes'))

#  Save the runtimes
methods <- c('SHAP', 'LIME', 'Variable Importance', 'PDP', 'ALE', 'Grouped PDP')
# methods <- c('SHAP', 'LIME', 'Local Model', 
#              'Local Diagnostics', 'Variable Importance', 'PDP', 'Grouped PDP')
runtimes_in_minutes <- round(c(shap_runtime, lime_runtime, vi_runtime, pdp_runtime, ale_runtime, gPdp_runtime), 4)
# runtimes_in_minutes <- round(c(shap_runtime, lime_runtime, locModel_runtime,
#                                locDiag_runtime, vImp_runtime, pdp_runtime, gPdp_runtime), 4)
results <- data.frame(methods, runtimes_in_minutes)
rt_filepath <- 'results/model_explanation/dalex/runtimes_methods.csv'
if (overwrite) write.csv(results, rt_filepath)

# TODO: Add methods from inTrees
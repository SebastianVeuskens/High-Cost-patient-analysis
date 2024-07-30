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
# Number of samples 
n <- 500
# Percentage of positive outcomes in label y
percentage <- 0.5
# Type of output to use
binary <- TRUE  
# Type of model to use 
model_name <- "rf"
# Whether to use H2O package or standard R packages
use_h2o <- FALSE
# Target variable
target <- 'y'
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

#### LOAD DATA ####
x1 <- rnorm(n, 1, 5)
x2 <- rnorm(n, -1, 2)
err <- rnorm(n, 0, 20)
y <- x1 ** 2 + x2 ** 4 + x1 * x2 + err
outcome <- ifelse(y < quantile(y, percentage), 0, 1)
data_con <- data.frame(x1, x2, y)
data_bin <- data.frame(x1, x2, y = as.factor(outcome))
# TODO: I only write this here because of the local-diagnostic plots 
data_bin <- data.frame(x1, x2, y = outcome == 1)
summary(data_con)
summary(data_bin)

# Divide into train and test
sample_ind <- sample(1:n, floor(n * 3 / 4))
train_con <- data_con[sample_ind,]
train_bin <- data_bin[sample_ind,]
test_con <- data_con[-sample_ind,]
test_bin <- data_bin[-sample_ind,]
str(train_con)
str(train_bin)
str(test_con)
str(test_bin)

if (binary) {
    train <- train_bin
    test <- test_bin
} else {
    train <- train_con
    test <- test_con
}


#######################
#### RANDOM FOREST ####
#######################

## Train the model

if (use_h2o) {
    # Start H2O package
    h2o.init()
    # Load data frames into H2O framework
    train <- as.h2o(train)
    test <- as.h2o(test)

    rf_model <- h2o.randomForest(x              = 1:2, 
                                 y              = 3,
                                 training_frame = train,
                                 seed           = 12345,
                                 ntrees         = 1000,
                                 mtries         = 2)

} else {
    rf_model <- randomForest(formula = y ~ ., data=train)
}

############################
#### PERFORMANCE CURVES ####
############################
if (model_name == "rf") {
    model <- rf_model 
} else if (model_name == "lr") {
    model <- lr_model
}

# # Bind the predicted and true labels together 
# predictions <- if (use_h2o) as.data.frame(h2o.predict(model, test)) else predict(test, model)
# label <- as.data.frame(test)$y
# predicted_vs_true <- bind_cols(predictions, label=label)
# # Get the prediction probabilities for each class 
# class_1_probabilities <- predicted_vs_true$p1[predicted_vs_true$label == 1]
# class_0_probabilities <- predicted_vs_true$p1[predicted_vs_true$label == 0]

# # Receiver Operating Characteristic (ROC) curve 
# roc <- roc.curve(class_1_probabilities, class_0_probabilities, curve=TRUE)
# aucroc <- roc$auc
# # Display and save the ROC curve 
# print(paste('ROC Area under the curve (AUC):', round(aucroc, 4)))
# plot(roc)

# # Precision-Recall curve
# # TODO: Check if this gives the same results as the pr-curve in Benedikts script
# # TODO: Run h2o.aucpr in Benedikt script to see if he gets such low results there too 
# pr <- pr.curve(class_1_probabilities, class_0_probabilities, curve=TRUE)
# aucpr <- pr$auc.integral
# # Display and save the Precision-Recall curve 
# print(paste('Precision-Recall Area under the curve (AUC):', round(aucpr, 4)))
# plot(pr)

# # Validate results by comparing the AUC with the Precision-Recall AUC of the h2o built-in function. 
# aucpr_h2o <- h2o.aucpr(h2o.performance(model, test))
# aucpr_match <- isTRUE(all.equal(aucpr, aucpr_h2o, tolerance=0.1))
# if (!aucpr_match) {
#     print('EXCEPTON: CHECK PRECISION-RECALL COMPUTATION!')
# }

# # Confusion matrix
# conf_matrix <- confusionMatrix(predicted_vs_true$predict, predicted_vs_true$label)
# print(conf_matrix$table)

###############################################################
##################### XAI METHODS #############################
###############################################################

#######################
#### PRELIMINARIES ####
#######################

cov_names <- setdiff(names(test), target) 
sample <- test[1,cov_names] 
feature_of_interest <- 'x1'

if (use_h2o) {
    # predict_fn <- function(object, newdata) {as.data.frame(h2o.predict(object, as.h2o(newdata)))[,'p1']}
    # exp_dalex <- explain(model, data=as.data.frame(train[,cov_names]), 
    #                      y=as.data.frame(train[,target]), predict_function=predict_fn)
    # predict_fn <- function(object, newdata) {h2o.predict(object, newdata)}
    # exp_dalex <- explain(model, data=train[,cov_names], y=train[,target], predict_function=predict_fn)
    exp_dalex <- explain(rf_model_h2o, data=train[,cov_names], y=train[,target])
} else {
    exp_dalex <- DALEX::explain(model, data=train[,cov_names], y=train[,target])
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

###################################
#### RESIDUAL-DIAGNOSTIC PLOTS ####
###################################
# CHARACTERISTICS: Not really XAI, but model validation method
# NOT RELEVANT -> Only for regression, not (binary) classification -> Residuals very limited
# Only relevant might be ids and maybe if I figure out how to predict probabilities and not classes (0 or 1) 
mp_dalex <- DALEX::model_performance(exp_dalex)
plot(mp_dalex, geom='histogram')
plot(mp_dalex, geom='boxplot')

md_dalex <- model_diagnostics(exp_dalex)
plot(md_dalex, variable=target, yvariable='residuals')
plot(md_dalex, variable=target, yvariable='y_hat')
plot(md_dalex, variable='ids', yvariable='y_hat')
plot(md_dalex, variable='y_hat', yvariable='abs_residuals')

value = 0.3
n = 100
confidence=0.95
    alpha <- 1 - confidence
    z <- abs(qnorm(1 - alpha / 2))
    interval <- z * sqrt(value * (1-value) / n) 
    lower <- value - interval
    upper <- value + interval
    c(lower, upper)


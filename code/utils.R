##############################################
#### AUXILIARY FUNCTIONS #####################
##############################################

#######################
#### PRELIMINARIES ####
#######################

#### LIBRARIES & SOURCES ####

# INSTALL LIBRARIES 
# install.packages('h2o')
# install.packages('cvAUC')
# install.packages('rlist')

# LOAD LIBRARIES & SOURCES
library(h2o)    # The modelling framework 
library(cvAUC)  # For the Area Under the Curve (AUC) computation
library(rlist)  # To save inhomogeneous lists and load them again 

# TODO: Check Interal calculation 
# Compute the confidence interval
#
# This function takes a statistics value along with a sample size. Based on  the normal approximation 
# of the binomial distribution, the confidence interval is computed and returned. 
# The formula is given by: value +- z_{1 - \alpha / 2} \sqrt{value (1 - value) / n}
#
# @params value The statistic for which the confidence interval is calculated 
# @params n The sample size
# @params confidence The level of confidence 
confidence_interval <- function(value, n, confidence = 0.95) {
    alpha <- 1 - confidence
    z <- abs(qnorm(1 - alpha / 2))
    interval <- z * sqrt(value * (1-value) / n) 
    lower <- value - interval
    upper <- value + interval
    return(c(lower, upper))
}

# Compute the AUC 
# TODO: Add documentation here 
compute_auc <- function(predictions, newdata, label, confidence = 0.95) {
    labels <- as.data.frame(newdata[, label])[[1]]
    probabilities <- predictions$p1
    auc_info <- ci.cvAUC(probabilities, labels, confidence=confidence)
    return(auc_info)
}

# Evaluate a trained h2o model 
#
# A fully automated evaluation of all different metrics of the specified model
# Results are saved in designated file.
#
# @params model The trained model
# @params filepath The filepath and name to save the results 
# @params overwrite Indicator whether to save the results 
# @params newdata The test data to evaluate the model on. If NULL, cross validation results are used
evaluate_model <- function(model, filepath, overwrite, newdata = NULL, target_label='HC_Patient_Next_Year', cost_label='Total_Costs_Next_Year'){
    # Use the G-Mean score of sensitivity and specificity to determine the optimal cutoff threshold 
    # xval indicates that we want to receive the results of cross-validation -> Use only if no newdata is provided
    xval = is.null(newdata)
    performance     <- h2o.performance(model, newdata = newdata, xval = xval)
    sensitivities   <- h2o.sensitivity(performance)
    specificities   <- h2o.specificity(performance)
    gmeans          <- sqrt(sensitivities$tpr * specificities$tnr)
    threshold_index <- which.max(gmeans)
    threshold       <- sensitivities[threshold_index, 'threshold'] # Could also use specificities, the thresholds are the same 

    # Compute predictions 
    if (is.null(newdata)) {
        predictions <- h2o.cross_validation_holdout_predictions(model)
        newdata <- h2o.getFrame(model@parameters$training_frame)        # Gets the training data that was used for training the model 
    } else {
        predictions <- h2o.predict(model, newdata=newdata)
    }
    predictions <- as.data.frame(predictions)
    n <- nrow(newdata) 

    # Compute the important measures with this threshold 
    confusion_matrix <- h2o.confusionMatrix(performance, threshold=threshold)
    auc_h2o          <-  as.numeric(h2o.auc(performance))
    aucpr_h2o       <-  as.numeric(h2o.aucpr(performance))          
    accuracy         <-  as.numeric(h2o.accuracy (performance, threshold))       
    sensitivity      <-  as.numeric(h2o.sensitivity (performance, threshold)) 
    specificity      <-  as.numeric(h2o.specificity (performance, threshold)) 
    gmean            <- gmeans[threshold_index]      

    # TODO: Check if floor is really appropriate here (changes results only very slightly)
    idx_cc_pred      <- which(predictions$p1 >= quantile(predictions$p1, 0.95))[1:floor(n * 0.05)]
    idx_cc_true      <- which(as.data.frame(newdata)[target_label] == 1)
    cost_capture     <- 100 * sum(newdata[idx_cc_pred, cost_label]) / sum(newdata[idx_cc_true, cost_label])  

    # Confidence intervals 
    interval_auc_h2o        <- confidence_interval(auc_h2o, n)
    interval_aucpr_h2o     <- confidence_interval(aucpr_h2o, n)
    interval_accuracy       <- confidence_interval(accuracy, n)
    interval_sensitivity    <- confidence_interval(sensitivity, n)
    interval_specificity    <- confidence_interval(specificity, n)
    interval_gmean          <- confidence_interval(gmean, n)
    interval_cost_capture   <- confidence_interval(cost_capture * 0.01, n) * 100

    # TODO: Check why I calculated the AUC like this and not with H2O native functions 
    # AUC 
    auc_info     <- compute_auc(predictions, newdata=newdata, label=target_label)
    auc          <- auc_info[['cvAUC']]
    interval_auc <- auc_info[['ci']]

    # Combine the results 
    measures <- c('auc h2o', 'auc pr', 'accuracy', 'sensitivity', 'specificity', 'gmean', 'auc', 'cost_capture')
    values   <- c(auc_h2o, aucpr_h2o, accuracy, sensitivity, specificity, gmean, auc, cost_capture) 
    lower    <- c(interval_auc_h2o[1], interval_aucpr_h2o[1], interval_accuracy[1], interval_sensitivity[1], interval_specificity[1], interval_gmean[1], interval_auc[1], interval_cost_capture[1])
    upper    <- c(interval_auc_h2o[2], interval_aucpr_h2o[2], interval_accuracy[2], interval_sensitivity[2], interval_specificity[2], interval_gmean[2], interval_auc[2], interval_cost_capture[2])
    results  <- data.frame(measures, values, lower, upper)

    # Save the results 
    if (overwrite) write.csv(results, paste0(filepath, '.csv'))

    # Return the results
    return(results)
}

# Evaluate a trained R model 
#
# A fully automated evaluation of all different metrics of the specified model
# Results are saved in designated file.
#
# @params model The trained model
# @params filepath The filepath and name to save the results 
# @params overwrite Indicator whether to save the results 
# @params newdata The test data to evaluate the model on. If NULL, cross validation results are used
evalutate_r_model <- function(model, filepath, overwrite, newdata, target_label='HC_Patient_Next_Year', cost_label='Total_Costs_Next_Year') {
    # AUC 
    auc_info     <- compute_auc(predictions, newdata=newdata, label=target_label)
    auc          <- auc_info[['cvAUC']]
    interval_auc <- auc_info[['ci']]
    
    # Confidence intervals 
    interval_accuracy       <- confidence_interval(accuracy, n)
    interval_sensitivity    <- confidence_interval(sensitivity, n)
    interval_specificity    <- confidence_interval(specificity, n)
    interval_gmean          <- confidence_interval(gmean, n)
    interval_cost_capture   <- confidence_interval(cost_capture * 0.01, n) * 100
    
    # Combine the results 
    measures <- c('accuracy', 'sensitivity', 'specificity', 'gmean', 'auc', 'cost_capture')
    values   <- c(accuracy, sensitivity, specificity, gmean, auc, cost_capture) 
    lower    <- c(interval_accuracy[1], interval_sensitivity[1], interval_specificity[1], interval_gmean[1], interval_auc[1], interval_cost_capture[1])
    upper    <- c(interval_accuracy[2], interval_sensitivity[2], interval_specificity[2], interval_gmean[2], interval_auc[2], interval_cost_capture[2])
    results  <- data.frame(measures, values, lower, upper)

    
    # Save the results 
    if (overwrite) write.csv(results, paste0(filepath, '.csv'))

    # Return the results
    return(results)
}


# Save a list 
#
# Simplify the repetitive process of saving a list in R into 2 file formats
#
# @params list The list that is saved
# @params filepath The location and name of the file 
save_list <- function(list, filepath) {
    # Save list to human-readable file
    sink(paste0(filepath, '.txt'))
    print('HUMAN READABLE FORMAT: FOR INFORMATION PURPOSES ONLY.')
    print(list)
    sink()
    # Save list in R format to reload for subsequent processing 
    list.save(list, file=paste0(filepath, '.RData'))
}


# Train a model of unknown type
#
# This function generically trains a model as specified in the model_params argument.
# The type of the model is derived from the model_params agrument. 
#
# @params model_params The information about the model specifications.
# @params train The training data set
# @params first_val Column number of first feature
# @params last_val Column number of last feature 
# @params label_pos Column number of label 
train_model <- function(model_params, train, first_val, last_val, label_pos) {
    name <- model_params[[1]] 
    params <- model_params[[3]]
    if (name == 'logistic regression') {
        # TODO: Check if this is the right way to index the predictors 
        indices <- match(strsplit(params[['predictors']], ', ')[[1]], colnames(train))
        model <- h2o.glm(x                                   = indices, 
                         y                                   = label_pos,
                         training_frame                      = train, 
                         seed                                = 12345)
    } else if (name == 'neural network') {
        activation <- params[['activation']]
        hidden <- as.numeric(params[['hidden']])
        rate <- as.numeric(params[['rate']])

        model <- h2o.deeplearning(x              = first_val:last_val, 
                                  y              = label_pos,
                                  training_frame = train, 
                                  seed           = 12345,
                                  activation     = activation,
                                  hidden         = hidden,
                                  rate           = rate)
    } else if (name == 'random forest') {
        ntrees <- as.numeric(params[['ntrees']])
        mtries <- as.numeric(params[['mtries']])

        model <- h2o.randomForest(x              = first_val:last_val, 
                                  y              = label_pos,
                                  training_frame = train, 
                                  seed           = 12345,
                                  ntrees         = ntrees,
                                  mtries         = mtries)
    } else if (name == 'gradient boosting machine') {
        ntrees <- as.numeric(params[['ntrees']])
        max_depth <- as.numeric(params[['max_depth']])
        
        model <- h2o.gbm(x              = first_val:last_val, 
                         y              = label_pos,
                         training_frame = train, 
                         seed           = 12345,
                         ntrees         = ntrees,
                         max_depth      = max_depth)
    } else {
        print('EXCEPTION: UNKNOWN MODEL TYPE!')
    }
    return(model)
}


# Train a reduced logistic regression model 
#
# This function trains a logistic regression model.
# Only the specified indices are included. 
#
# @params indices A list of indices to include as predictors. 
# @params label_pos The index of the outcome or label (response variable)
# @params train_data Data set used for training.
# @nfolds Number of folds for cross validation metrics 
train_lr_model <- function(indices, label_pos, train_data, nfolds) {
    return(h2o.glm(x                  = indices,
                   y                  = label_pos,
                   training_frame     = train_data,  
                   nfolds             = nfolds,
                   seed               = 12345,
                   calc_like          = TRUE
                   )
    )
}


# Save the coefficients of a (reduced) logistic regression model 
#
# This function saves the coefficients of a logistic regression model.
# Only the specified indices are included. P-values are reported too. 
#
# @params model the logistic regression model that contains the coefficients. 
# @params filepath The location and name of the file 
save_lr_coefs <- function(model, filepath) {
    all_coefs <- h2o.coef_with_p_values(model)
    sig_coefs <- all_coefs[all_coefs$p_value < 0.05,]
    all_num_coefs <- nrow(all_coefs)
    sig_num_coefs <- nrow(sig_coefs)
    coefs <- list(
        'Number of significant coefficients' = sig_num_coefs, 
        'Number of all coefficients' = all_num_coefs, 
        'Significant coefficients' = sig_coefs, 
        'All coefficients' = all_coefs
    )
    save_list(coefs, paste0(filepath, '_coefficients'))
}

# TODO: Add documetation 
net_benefit <- function(prediction_true, x, n, group) {
    nb <- c()
    for (p_t in x) {
        tpr <- sum(prediction_true > p_t) / n 
        fnr <- sum(prediction_true <= p_t) / n 
        nb <- c(nb, tpr - fnr * p_t / (1 - p_t))
    }
    return(data.frame(x=x, net_benefit=nb, group=group))
}


# Create a decision curve analysis plot
#
# This function creates the decision curves for all specified models
# Default decision strategies (all and none receives treatment) are included 
#
# @params models_predictions_list List of model predictions which to include a decision curve  
# @params model_names_list List of names for each model 
# @params newdata Data to make the decision curve on 
decision_curves <- function(models_predictions_list, model_names_list, newdata, x=seq(0, 0.3, 0.0001), target='HC_Patient_Next_Year') {
    # Assert that a name exists for each model and vice versa 
    stopifnot(length(models_predictions_list) == length(model_names_list))

    n <- nrow(newdata)
    none_nb <- numeric(length(x))
    none <- data.frame(x=x, net_benefit=none_nb, group='None')
    all_nb <- sum(newdata[target] == 1) / n - sum(newdata[target] == 0) / n * x / (1 - x) 
    all <- data.frame(x=x, net_benefit=all_nb, group='All')
    dca <- rbind(none, all)
    for (model_idx in seq(length(models_predictions_list))) {
        predictions <- models_predictions_list[[model_idx]]
        predictions_true <- predictions[newdata[target] == 1]
        nb <- net_benefit(predictions_true, x, n, model_names_list[[model_idx]])
        dca <- rbind(dca, nb)
    }

    ggplot(dca, aes(x=x, y=net_benefit, col=group)) + 
        geom_line() 
}
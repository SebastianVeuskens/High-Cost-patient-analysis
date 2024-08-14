## DATA SOURCE 
**File:** HC_Patient_Data.csv

# Purpose

# Structure
## Code
| FILE | DESCRIPTION |
|------|-------------|
| **eda.R** | Perform some primary data analysis. Summarize and visualize the data for first insights.|
**1_data_extraction.R** | Load the whole dataset. Divide it into training, validation and test set and store the data. |
**2_model_tuning.R** | Find the best model hyperparameters for neural network, random forest and gradient boosting machine. (Logistic regression does not have hyperparameters). Use cross-validation on the training set to chooose the best hyperparameter configuration for each model.|
**3_model_selection.R** | Compare the different algorithms. Use the hyperparameters determined in *2_model_tuning.R*. Use the validation dataset for comparison. Store the algorithms, ranked by their performance.|
**4_model_evaluation.R** | Evaluate the best algorithm (or a user-defined one) on the test dataset. In addition, apply some XAI methods for model insights. |
**utils.R** | Store functions for concise and non-repetitive code. These functions are used in the other files to make the files more readable. |

## Data 
- Nested structure for filtered/balanced 
results, data, code -> 

## Algorithms
For threshold definition, the F1-score is used. 

## Reproduction
In each R-file in the *code* folder is a section called **MODIFY**. Inside that section, feel free to change the values of the variables to adapt them to your needs. However, do NOT change anything outside that section except you know what you are doing.  
If you for example change the parameter *filter_hc* to TRUE. This will remove all the..... You have to run 1-4 and change it in all of the files. 
The best models are saved in *model_evaluation* repository. For better reproduction, all model parameters for all models are also saved in both R-native and txt-file format in the *model_selection* repository. 

## TODO
Figure out why my training auc is lower than my test/evaluation auc. 
Investigate why the Prescription_L04 feature is so important. 
Nice2Have: Add roc.tests (compare two models) like Benedikt in 5_best_models.R. 

## Limitations
In some cases, the trained h2o neural-network model does not contain the rate as a parameter anymore. This might cause problems in the case when the neural network would be the best model and would have to be reloaded with its best performing parameters. However, this is not the case with the data used in this analysis. 
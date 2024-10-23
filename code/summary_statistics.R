################################################
#### SUMMARY STATISTIC GENERATION ##############
################################################
# File: summary_statistics.R
# Author: Sebastian Benno Veuskens 
# Date: 2024-07-25
# Data: HC_Patient_Data for years 2019-2021 (including HC-indicators for following year, respectively)
#
# 
# This script creates the tables as in the first results section.
# Main demographic information is given as well as HC-patient overviews. 

#### MODIFY ####
# Your working directory 
setwd("C:/Users/Sebastian's work/OneDrive - OptiMedis AG/Dokumente/Coding/High-Cost-patient-analysis")
# Number of the best models to save in the best parameters folder 
num_models <- 2 
# Whether you want to save your results (and overwrite the old results) or not
overwrite <- FALSE
# Which center function to use for the plot 
center_func_label <- 'mean'
#### MODIFY END ####

#######################
#### PRELIMINARIES ####
#######################

#### LIBRARIES & SOURCES ####

# INSTALL LIBRARIES 
# install.packages('tidyr')
# install.packages('dplyr')
# install.packages('ggplot2')
# install.packages('ggthemes')
# install.packages('scales')
# install.packages('glue')

# LOAD LIBRARIES & SOURCES
library(tidyr)          # Auxiliary tools for frequency tables 
library(dplyr)          # Auxiliary tools for frequency tables  
library(ggplot2)        # To visualize the plot 
library(ggthemes)       # A better style for the plot visualization
library(scales)         # For logarithmic scaling of the plot
library(glue)           # For dynamic plot labels 
source('code/utils.R')  # Auxiliary functions for simplicity and concise code 

#### LOAD DATA ####

# Use for frequency table creation 
load('data/data_2019.Rdata')
load('data/data_2020.Rdata')
load('data/data_2021.Rdata')

# Use for validation purposes only
load('data/data.Rdata')
load('data/train.Rdata')
load('data/validate.Rdata')
load('data/train_validate.Rdata')
load('data/test.Rdata')

## Data manipulation
all_data <- data 
data_2019$year <- 2019
data_2020$year <- 2020
data_2021$year <- 2021
data <- rbind(data_2019, data_2020, data_2021)
data$year <- as.factor(data$year)

# Add Age group column
age_split = c(30, 65, 80)
Age_groups_num <- findInterval(data$Age, age_split)
age_code <- c('0-30' = 0, '30-65' = 1, '65-80' = 2, '>80' = 3)
data$Age_groups <- names(age_code)[match(Age_groups_num, age_code)]


#### CREATE FOLDER STRUCTURE ####
if (overwrite) {
    dir.create('results', showWarnings=FALSE)
    dir.create('results/summary_statistics', showWarnings=FALSE)
}

#######################
#### SUMMARIZE ########
#######################

#### BASIC STATISTICS ####

nrow(data)
nrow(data_2019)
nrow(data_2020)
nrow(data_2021)
nrow(train)
nrow(validate)
nrow(train_validate)
nrow(test)

#### DESCRIPTIVE STATISTICS ####
## HELPER FUNCTIONS  

# GENERAL 
# Count number of variables for each value of the split variable 
num <- function(data, split_var) {
  split_var <- sym(split_var)
  return(
    data %>% 
    group_by(!!split_var) %>% 
    summarise(count = n()) %>%
    pivot_wider(names_from = !!split_var, values_from=count)
  )
}

# NUMERICAL VARIABLES 
# Calculate the midpoint for a given variable for each value of the split variable
# The method for determining the midpoint must be specified by the user.
mid <- function(data, variable, split_var, sum_func, decimals) {
  variable <- sym(variable)
  split_var <- sym(split_var)
  return(
    data %>%  
    group_by(!!split_var) %>%
    summarize(midpoint = sum_func(!!variable)) %>% 
    mutate(midpoint = round(midpoint, decimals)) %>%
    pivot_wider(names_from = !!split_var, values_from=midpoint) %>% 
    as.data.frame()
  )
}

# Wrapper for midpoint calculation function that caculates the mean
avg <- function(data, variable, split_var, decimals) {return(mid(data, variable, split_var, mean, decimals))}
# Wrapper for midpoint calculation function that caculates the median 
med <- function(data, variable, split_var, decimals) {return(mid(data, variable, split_var, median, decimals))}

# Perform a t-test or ANOVA to test whether the mean of a variable differs across the split variable.
test_num <- function(data, variable, split_var) {
  distinct_vals <- unique(data[split_var]) 
  formula <- as.formula(paste(variable, '~', split_var))
  if (nrow(distinct_vals) == 2) {
    return(t.test(formula, data)$p.value)
  } else if (nrow(distinct_vals) > 2) {
    # P-value is not easy to access in ANOVA test
    test_aov <- summary(aov(formula, data))[[1]]
    return(test_aov[["Pr(>F)"]][1])
  } else {
    warning('SELECTED VARIABLE HAS INCORRECT NUMBER OF UNIQUE VALUES.')
  }
}

avg_and_test <- function(data, variable, split_var, decimals) {
  var_test <- round(test_num(data, variable, split_var), decimals)
  return(
    avg(data, variable, split_var, decimals) %>%
      mutate(p_value = var_test)
  )
}

# CATEGORICAL VARIABLES   
# Determines the frequency of each value for a certain variable for each value of the split variable 
variable_frequency <- function(data, variable, split_var) {
  variable <- sym(variable)
  split_var <- sym(split_var)
  return(
    data %>% 
      count(!!variable, !!split_var) %>% 
      pivot_wider(id_cols=!!variable, names_from=!!split_var, values_from=n)
  )
}

test_cat <- function(data, variable, split_var) {
  variable <- sym(variable)
  split_var <- sym(split_var)
  return(data %>%
          count(!!variable, !!split_var) %>% 
          pivot_wider(id_cols=!!variable, names_from=!!split_var, values_from=n) %>%
          select(-!!variable) %>%
          as.data.frame() %>%
          chisq.test() %>% 
          with(p.value)
  )        
}

freq_and_test <- function(data, variable, split_var, decimals) {
  var_test <- round(test_cat(data, variable, split_var), decimals)
  return(
    variable_frequency(data, variable, split_var) %>%
      mutate(p_value = var_test)
  )
}

# Combine all variables for the final descriptive summary table 
frequency_table <- function(data, split_var, decimals=3) {
  return(list(
    'N = ' = num(data, split_var)
    , 'Sex' = freq_and_test(data, 'Sex', split_var, decimals)
    , 'Age groups' = freq_and_test(data, 'Age_groups', split_var, decimals)
    , 'Average age (years)' = avg_and_test(data, 'Age', split_var, decimals)
    , 'Current HC patient' = freq_and_test(data, 'HC_Patient', split_var, decimals)
    , rbind(
      c('Average need of care duration ', avg_and_test(data, 'Need_of_Care_Duration', split_var, decimals))
      , c('Average DMP duration ', avg_and_test(data, 'DMP_Duration', split_var, decimals))
      , c('Average total costs in the current year', avg_and_test(data, 'Total_Costs', split_var, decimals))
      , c('Average total costs in the following year', avg_and_test(data, 'Total_Costs_Next_Year', split_var, decimals))
      , c('Average inpatient number of diagnoses (years) ', avg_and_test(data, 'Inpatient_Num_Diagnoses', split_var, decimals))
      , c('Average outpatient number of diagnoses (years) ', avg_and_test(data, 'Outpatient_Num_Diagnoses', split_var, decimals))
      , c('Average number of prescriptions (years) ', avg_and_test(data, 'Prescription_Num_Prescriptions', split_var, decimals))
    )
  ))
}                

freq_tab_year <- frequency_table(data, 'year')
freq_tab_hcp <- frequency_table(data, 'HC_Patient_Next_Year')

# Save the frequency tables
filepath_year <- 'results/summary_statistics/frequencies_table_year'
filepath_hcp <- 'results/summary_statistics/frequencies_table_high_cost_patient'
if (overwrite) save_list(freq_tab_year, filepath_year)
if (overwrite) save_list(freq_tab_hcp, filepath_hcp)

# Assert correctness
nrow(data_2019) + nrow(data_2020) == nrow(train_validate)
nrow(train_validate) == nrow(train) + nrow(validate)
nrow(test) == nrow(data_2021)
nrow(train_validate) + nrow(test) == nrow(data_2019) + nrow(data_2020) + nrow(data_2021)


#######################
#### VISUALIZATION  ###
#######################

# Compute centers for vertical lines
if (center_func_label == 'mean') {
  center_func <- mean
} else if (center_func_label == 'median') {
  center_func <- median
} else {
  warning('UNKNOWN CENTER FUNCTION LABEL USED.')
}

centers <- data %>% 
             group_by(HC_Patient_Next_Year) %>% 
             summarize(center = round(center_func(Total_Costs)), 0) %>%
             as.data.frame()

# Function to set numbers with marks and without scientific notation
marks_no_sci <- function(x) format(x, big.mark = ",", decimal.mark = ".", scientific = FALSE)

# Create density plot for different Total costs in HC and non-HC patients 
ggplot(data, aes(x=Total_Costs, fill=HC_Patient_Next_Year)) +
  geom_density(position='identity', alpha=0.5) +
  geom_vline(data=centers, aes(xintercept=center, color=HC_Patient_Next_Year), 
             linetype='dashed', linewidth=1, show.legend=FALSE) +
  geom_text(data=centers, aes(x=center * 0.7, y=0.55, col=HC_Patient_Next_Year, 
            label=glue('{center_func_label} = {format(center, big.mark=",")}€')),
            angle=90, size=4, show.legend=FALSE, hjust=0)  +           
  scale_x_continuous(trans=scales::pseudo_log_trans(base=10), 
                     breaks=c(0, 10, 100, 1000, 10000, 100000, 1000000),
                     labels=scales::label_currency(suffix='€', prefix='')) + 
  scale_fill_discrete(name='Prospective HC-patient',
                      breaks=c(0, 1), 
                      labels=c('no', 'yes')) + 
  labs(x='Total Costs', y='Density') + 
  ggtitle('Comparison in current Total Costs for prospective HC-patients') + 
  theme_fivethirtyeight() + 
  theme(axis.title=element_text(), legend.position='top')             

# Save plot
ggsave(filename='results/summary_statistics/total_costs_comparison.png', 
       width=640/72, height=450/72, dpi=300)

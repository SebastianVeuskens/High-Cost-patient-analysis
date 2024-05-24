# Load necessary libraries
library(ggplot2)
library(dplyr)

data <- read.csv('data/HC_Patient_Data.csv')

data <- data[, colSums(data != 0) > 0]

dim(data[data$year < 2022, ])
# Print the structure of the dataset
print(str(data))

# Summary statistics
summary(data)

# Check for missing values
sum(is.na(data))

# Plotting histograms for all numeric columns
num_cols <- sapply(data, is.numeric)
hist_data <- data[, num_cols]
length(num_cols)

heatmap(cor(hist_data), scale="none")
str(cor(hist_data))

par(mfrow=c(2,2))
lapply(hist_data, function(x) hist(x, main = names(hist_data)[which(hist_data == x)], xlab = NULL))

# Boxplots for all numeric columns
par(mfrow=c(2,2))
lapply(hist_data, function(x) boxplot(x, main = names(hist_data)[which(hist_data == x)], xlab = NULL))

# Pairwise scatterplots for all numeric columns
pairs(hist_data)

#### FEATURE STATISTICS ####
# TODO: Add data overviews like age distribution plot (by sex?), number of total patients & number of high-cost patients, inpatient/outpatient/prescription total number distributions (by year?)  

#### EXAMPLES ####
# TODO: Add some examples of how the data looks like, e.g. patients that have 0 costs, patients with/without care, 

# TODO: Add parts from the end of 5_best_models.R file from Reproduction folder 


################
#### T-TEST ####
################

# Perform t-tests to determine significant differences in means between High-Cost and not High-Cost patients.
t.test(Need_of_Care_Duration ~ HC_Patient, data = data)
t.test(Age ~ HC_Patient, data = data)
t.test(DMP_Duration ~ HC_Patient, data = data)
t.test(Total_Costs ~ HC_Patient, data = data)
t.test(Inpatient_Num_Diagnoses ~ HC_Patient, data = data)
t.test(Outpatient_Num_Diagnoses ~ HC_Patient, data = data)
t.test(Prescription_Num_Prescriptions ~ HC_Patient, data = data)
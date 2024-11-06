# Load necessary libraries
library(mice)
library(datasets)

# Load the airquality dataset
data("airquality")

# Check for missing values
summary(airquality)

# Handle missing values using MICE
# Specify the method for each variable if needed; here we use defaults
imputed_data <- mice(airquality, method = 'pmm', m = 5, seed = 123)

# Complete the imputed dataset
complete_data <- complete(imputed_data, 1)  # Using the first imputed dataset

# Add a log transformation of Ozone
complete_data$logOzone <- log(complete_data$Ozone)

# Fit a linear model to predict log(Ozone) from Solar.R, Wind, and Temp
model <- lm(logOzone ~ Solar.R + Wind + Temp, data = complete_data)

# Summarize the model
summary(model)

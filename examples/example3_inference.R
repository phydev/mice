# In this example we will perform inference with the complete data,
# the imputed data, and the imputed data using the mice package.
# We compare the adjusted r-squared between the models and the
# estimates of the coefficients.

library(mice)
library(dplyr)

# load complete dataset
diabetes <- read.csv("../data/diabetes.csv")
diabetes <- diabetes %>% select(c("age", "sex", "bmi", "target"))

# load amputed dataset with 30% missing values
diabetes_amp <- read.csv("../data/diabetes_amputed.csv")

# load imputed dataset
diabetes_imp <- read.csv("../data/diabetes_imputed.csv")

# model for the complete data
model_complete <- lm(target ~ age + sex + bmi, data = diabetes)

# models on the imputed data using our python implementation
models <- diabetes_imp %>%
    group_by(imputation) %>%
    do(model = lm(target ~ age + sex + bmi, data = .))

# create a vector with the adjusted r-squared for each imputation
r_squared <- c(seq_along(models$model))
for (m in seq_along(models$model)) {
    r_squared[m] <- summary(models$model[[m]])$adj.r.squared
}

# models using the mice package
imp <- mice(diabetes_amp, m = 10, maxit = 20)

fit <- with(imp, lm(target ~ age + sex + bmi, diabetes))

pool_fit <- pool(fit)

print("Adjusted R-Squared and 95% CI")
print("Complete data:")
print(summary(model_complete)$adj.r.squared)
print("Imputed data:")
print(c(mean(r_squared),
        1.96 * sd(r_squared) / sqrt(length(r_squared)))
        )
print("Imputed with MICE:")
print(summary(pool_fit)$adj.r.squared)



# create empty list to store the estimates
estimates <- list()

# append the estimates to the list
for (m in seq_along(models$model)) {
    estimates[[m]] <- summary(models$model[[m]])$coefficients[, "Estimate"]
}

# compute mean and standard deviation of the estimates
estimates <- do.call(rbind, estimates)
estimates <- as.data.frame(estimates)

# convert estimates to long format and compute mean and standard error
estimates %>%
  tidyr::pivot_longer(
    cols = `(Intercept)`:`bmi`,
    names_to = "estimate",
    values_to = "value") %>%
    group_by(estimate) %>%
    summarise(mean = mean(value),
              std.error = sd(value) / sqrt(length(models$model)))


# The adjusted r-squareds are similar, but the coefficient for the
# sex variable is very different from the complete data model and
# the mice model.
# The reason behind this is that we are using a linear model to
# impute the missing values, but the sex variable is a categorical
# variable that requires a different imputation model.
# This is a limitation of our implementation, which was not designed
# to deal with categorical variables. We can expand our implementation
# by using logistic regression when the variable is categorical.
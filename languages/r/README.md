# R Programming for Data Science

## Overview

Comprehensive guide to R programming for data science, statistics, and analytics. Covers everything from R fundamentals to advanced statistical modeling, data visualization, and package development.

## Table of Contents

- [R Fundamentals](#r-fundamentals)
- [Data Manipulation](#data-manipulation)
- [Statistical Analysis](#statistical-analysis)
- [Data Visualization](#data-visualization)
- [Machine Learning in R](#machine-learning-in-r)
- [Package Development](#package-development)
- [Advanced Topics](#advanced-topics)
- [Best Practices](#best-practices)

## R Fundamentals

### Basic Data Types and Structures

```r
# Vectors
numeric_vector <- c(1, 2, 3, 4, 5)
character_vector <- c("apple", "banana", "orange")
logical_vector <- c(TRUE, FALSE, TRUE)
factor_vector <- factor(c("low", "medium", "high"), 
                       levels = c("low", "medium", "high"))

# Lists
my_list <- list(
  numbers = numeric_vector,
  fruits = character_vector,
  flags = logical_vector
)

# Data frames
df <- data.frame(
  id = 1:5,
  name = c("Alice", "Bob", "Carol", "David", "Eve"),
  age = c(25, 30, 35, 28, 32),
  salary = c(50000, 60000, 70000, 55000, 65000),
  stringsAsFactors = FALSE
)

# Matrices
matrix_data <- matrix(1:12, nrow = 3, ncol = 4)
colnames(matrix_data) <- paste0("Col", 1:4)
rownames(matrix_data) <- paste0("Row", 1:3)

# Arrays
array_data <- array(1:24, dim = c(2, 3, 4))
```

### Control Structures

```r
# Conditional statements
check_grade <- function(score) {
  if (score >= 90) {
    return("A")
  } else if (score >= 80) {
    return("B")
  } else if (score >= 70) {
    return("C")
  } else if (score >= 60) {
    return("D")
  } else {
    return("F")
  }
}

# Loops
# For loop
for (i in 1:5) {
  print(paste("Iteration:", i))
}

# While loop
counter <- 1
while (counter <= 5) {
  print(paste("Counter:", counter))
  counter <- counter + 1
}

# Apply family functions (vectorized operations)
numbers <- 1:10

# lapply - returns list
squared_list <- lapply(numbers, function(x) x^2)

# sapply - returns vector
squared_vector <- sapply(numbers, function(x) x^2)

# mapply - multiple arguments
mapply(function(x, y) x + y, 1:5, 6:10)
```

### Functions and Programming

```r
# Basic function
calculate_bmi <- function(weight, height) {
  bmi <- weight / (height^2)
  return(bmi)
}

# Function with default parameters
describe_data <- function(data, na.rm = TRUE, digits = 2) {
  summary_stats <- list(
    mean = mean(data, na.rm = na.rm),
    median = median(data, na.rm = na.rm),
    sd = sd(data, na.rm = na.rm),
    min = min(data, na.rm = na.rm),
    max = max(data, na.rm = na.rm)
  )
  
  return(lapply(summary_stats, round, digits))
}

# Function with error handling
safe_divide <- function(x, y) {
  tryCatch({
    if (y == 0) {
      stop("Division by zero is not allowed")
    }
    result <- x / y
    return(result)
  }, error = function(e) {
    message("Error: ", e$message)
    return(NA)
  })
}

# Closure example
make_multiplier <- function(n) {
  function(x) {
    x * n
  }
}

double <- make_multiplier(2)
triple <- make_multiplier(3)
```

## Data Manipulation

### Base R Data Manipulation

```r
# Subsetting
df[1:3, ]                    # First 3 rows
df[, c("name", "age")]       # Specific columns
df[df$age > 30, ]            # Conditional subsetting
df$high_earner <- df$salary > 60000  # Add new column

# Aggregation
aggregate(salary ~ age > 30, data = df, FUN = mean)
by(df$salary, df$age > 30, summary)

# Merging data frames
df1 <- data.frame(id = 1:3, value1 = c(10, 20, 30))
df2 <- data.frame(id = 2:4, value2 = c(100, 200, 300))

# Inner join
merge(df1, df2, by = "id")

# Left join
merge(df1, df2, by = "id", all.x = TRUE)

# Reshaping data
wide_data <- data.frame(
  id = 1:3,
  measure1 = c(10, 20, 30),
  measure2 = c(15, 25, 35)
)

# Wide to long
long_data <- reshape(wide_data, 
                    varying = c("measure1", "measure2"),
                    v.names = "value",
                    timevar = "measure",
                    times = c("measure1", "measure2"),
                    direction = "long")
```

### dplyr for Data Manipulation

```r
library(dplyr)
library(magrittr)

# Sample dataset
employees <- data.frame(
  id = 1:100,
  department = sample(c("Sales", "Marketing", "IT", "HR"), 100, replace = TRUE),
  salary = rnorm(100, 60000, 15000),
  years_experience = sample(1:20, 100, replace = TRUE),
  performance_rating = sample(1:5, 100, replace = TRUE)
)

# Basic dplyr operations
result <- employees %>%
  filter(department %in% c("Sales", "IT")) %>%
  select(id, department, salary, years_experience) %>%
  mutate(
    salary_category = case_when(
      salary < 50000 ~ "Low",
      salary < 70000 ~ "Medium",
      TRUE ~ "High"
    ),
    experience_level = cut(years_experience, 
                          breaks = c(0, 5, 10, 20),
                          labels = c("Junior", "Mid", "Senior"))
  ) %>%
  arrange(desc(salary))

# Grouping and summarizing
summary_stats <- employees %>%
  group_by(department) %>%
  summarise(
    count = n(),
    avg_salary = mean(salary),
    median_salary = median(salary),
    sd_salary = sd(salary),
    avg_experience = mean(years_experience),
    .groups = "drop"
  )

# Window functions
employees_ranked <- employees %>%
  group_by(department) %>%
  mutate(
    salary_rank = row_number(desc(salary)),
    salary_percentile = percent_rank(salary),
    salary_lag = lag(salary),
    salary_lead = lead(salary),
    cumulative_salary = cumsum(salary)
  ) %>%
  ungroup()

# Complex transformations
performance_analysis <- employees %>%
  group_by(department, performance_rating) %>%
  summarise(
    count = n(),
    avg_salary = mean(salary),
    .groups = "drop"
  ) %>%
  pivot_wider(
    names_from = performance_rating,
    values_from = c(count, avg_salary),
    names_sep = "_rating_"
  )
```

### tidyr for Data Reshaping

```r
library(tidyr)

# Sample data
sales_data <- data.frame(
  product = rep(c("A", "B", "C"), each = 4),
  quarter = rep(paste0("Q", 1:4), 3),
  sales = rnorm(12, 1000, 200)
)

# Pivot operations
wide_sales <- sales_data %>%
  pivot_wider(names_from = quarter, values_from = sales)

long_sales <- wide_sales %>%
  pivot_longer(cols = starts_with("Q"), 
               names_to = "quarter", 
               values_to = "sales")

# Separating and uniting columns
messy_data <- data.frame(
  id = 1:5,
  name_age = c("John_25", "Jane_30", "Bob_35", "Alice_28", "Charlie_32")
)

clean_data <- messy_data %>%
  separate(name_age, into = c("name", "age"), sep = "_") %>%
  mutate(age = as.numeric(age))

# Handling missing data
data_with_na <- data.frame(
  id = 1:10,
  value1 = c(1, 2, NA, 4, 5, NA, 7, 8, 9, 10),
  value2 = c(NA, 2, 3, 4, NA, 6, 7, NA, 9, 10)
)

# Fill missing values
filled_data <- data_with_na %>%
  fill(value1, .direction = "down") %>%
  replace_na(list(value2 = 0))
```

## Statistical Analysis

### Descriptive Statistics

```r
# Generate sample data
set.seed(123)
data <- data.frame(
  group = rep(c("A", "B", "C"), each = 100),
  value = c(rnorm(100, 10, 2), rnorm(100, 12, 3), rnorm(100, 8, 1.5)),
  category = sample(c("X", "Y"), 300, replace = TRUE)
)

# Basic descriptive statistics
summary(data$value)
describe_data(data$value)

# Group-wise statistics
data %>%
  group_by(group) %>%
  summarise(
    n = n(),
    mean = mean(value),
    median = median(value),
    sd = sd(value),
    q25 = quantile(value, 0.25),
    q75 = quantile(value, 0.75),
    iqr = IQR(value),
    .groups = "drop"
  )

# Correlation analysis
cor_matrix <- cor(mtcars[, sapply(mtcars, is.numeric)])
print(cor_matrix)

# Covariance
cov(mtcars$mpg, mtcars$wt)
```

### Hypothesis Testing

```r
# T-tests
# One-sample t-test
t.test(data$value, mu = 10)

# Two-sample t-test
group_a <- data$value[data$group == "A"]
group_b <- data$value[data$group == "B"]
t.test(group_a, group_b)

# Paired t-test
before <- rnorm(20, 100, 15)
after <- before + rnorm(20, 5, 3)
t.test(before, after, paired = TRUE)

# ANOVA
anova_result <- aov(value ~ group, data = data)
summary(anova_result)

# Post-hoc tests
library(TukeyHSD)
TukeyHSD(anova_result)

# Chi-square test
chi_test <- chisq.test(table(data$group, data$category))
print(chi_test)

# Shapiro-Wilk normality test
shapiro.test(data$value[1:50])  # Max 5000 observations

# Kolmogorov-Smirnov test
ks.test(group_a, group_b)

# Wilcoxon tests (non-parametric)
wilcox.test(group_a, group_b)  # Mann-Whitney U test
wilcox.test(before, after, paired = TRUE)  # Wilcoxon signed-rank test
```

### Regression Analysis

```r
# Linear regression
model <- lm(mpg ~ wt + hp + cyl, data = mtcars)
summary(model)

# Model diagnostics
par(mfrow = c(2, 2))
plot(model)

# Residual analysis
residuals <- residuals(model)
fitted_values <- fitted(model)

# Check assumptions
# Linearity
plot(fitted_values, residuals)
abline(h = 0, col = "red")

# Normality of residuals
qqnorm(residuals)
qqline(residuals)

# Homoscedasticity
library(lmtest)
bptest(model)  # Breusch-Pagan test

# Autocorrelation
dwtest(model)  # Durbin-Watson test

# Multiple regression with interactions
model_interaction <- lm(mpg ~ wt * hp + cyl, data = mtcars)
summary(model_interaction)

# Polynomial regression
model_poly <- lm(mpg ~ poly(wt, 2) + hp, data = mtcars)
summary(model_poly)

# Logistic regression
# Create binary outcome
mtcars$high_mpg <- ifelse(mtcars$mpg > median(mtcars$mpg), 1, 0)

logistic_model <- glm(high_mpg ~ wt + hp + cyl, 
                     data = mtcars, 
                     family = binomial)
summary(logistic_model)

# Predictions
predictions <- predict(logistic_model, type = "response")
binary_predictions <- ifelse(predictions > 0.5, 1, 0)

# Model evaluation
confusion_matrix <- table(mtcars$high_mpg, binary_predictions)
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
```

### Time Series Analysis

```r
library(forecast)
library(tseries)

# Create time series data
ts_data <- ts(rnorm(100, 100, 10) + 1:100 * 0.5, 
              start = c(2020, 1), 
              frequency = 12)

# Time series decomposition
decomp <- decompose(ts_data)
plot(decomp)

# Stationarity tests
adf.test(ts_data)  # Augmented Dickey-Fuller test
kpss.test(ts_data)  # KPSS test

# ARIMA modeling
# Auto ARIMA
auto_model <- auto.arima(ts_data)
summary(auto_model)

# Manual ARIMA
manual_model <- arima(ts_data, order = c(1, 1, 1))
summary(manual_model)

# Forecasting
forecast_result <- forecast(auto_model, h = 12)
plot(forecast_result)

# Model diagnostics
checkresiduals(auto_model)

# Exponential smoothing
ets_model <- ets(ts_data)
ets_forecast <- forecast(ets_model, h = 12)
plot(ets_forecast)
```

## Data Visualization

### Base R Graphics

```r
# Scatter plots
plot(mtcars$wt, mtcars$mpg, 
     main = "MPG vs Weight",
     xlab = "Weight (1000 lbs)",
     ylab = "Miles per Gallon",
     col = mtcars$cyl,
     pch = 19)

# Add regression line
abline(lm(mpg ~ wt, data = mtcars), col = "red", lwd = 2)

# Add legend
legend("topright", 
       legend = unique(mtcars$cyl),
       col = unique(mtcars$cyl),
       pch = 19,
       title = "Cylinders")

# Histograms
hist(mtcars$mpg, 
     main = "Distribution of MPG",
     xlab = "Miles per Gallon",
     breaks = 10,
     col = "lightblue",
     border = "black")

# Box plots
boxplot(mpg ~ cyl, 
        data = mtcars,
        main = "MPG by Number of Cylinders",
        xlab = "Cylinders",
        ylab = "MPG",
        col = c("red", "green", "blue"))

# Bar plots
counts <- table(mtcars$cyl)
barplot(counts,
        main = "Number of Cars by Cylinder",
        xlab = "Cylinders",
        ylab = "Count",
        col = "orange")
```

### ggplot2 Visualizations

```r
library(ggplot2)
library(scales)

# Basic scatter plot
p1 <- ggplot(mtcars, aes(x = wt, y = mpg)) +
  geom_point(aes(color = factor(cyl), size = hp)) +
  geom_smooth(method = "lm", se = TRUE) +
  labs(title = "MPG vs Weight",
       subtitle = "Colored by number of cylinders",
       x = "Weight (1000 lbs)",
       y = "Miles per Gallon",
       color = "Cylinders",
       size = "Horsepower") +
  theme_minimal()

print(p1)

# Faceted plots
p2 <- ggplot(mtcars, aes(x = wt, y = mpg)) +
  geom_point() +
  geom_smooth(method = "lm") +
  facet_wrap(~cyl, ncol = 3) +
  labs(title = "MPG vs Weight by Cylinder Count") +
  theme_bw()

print(p2)

# Box plots with jitter
p3 <- ggplot(mtcars, aes(x = factor(cyl), y = mpg)) +
  geom_boxplot(aes(fill = factor(cyl)), alpha = 0.7) +
  geom_jitter(width = 0.2, alpha = 0.5) +
  stat_summary(fun = mean, geom = "point", 
               shape = 23, size = 3, fill = "red") +
  labs(title = "MPG Distribution by Cylinder Count",
       x = "Number of Cylinders",
       y = "Miles per Gallon") +
  theme_classic() +
  guides(fill = "none")

print(p3)

# Heatmap
library(reshape2)
cor_data <- melt(cor(mtcars))

p4 <- ggplot(cor_data, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", mid = "white", high = "red",
                       midpoint = 0, limit = c(-1, 1)) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Correlation Matrix Heatmap",
       fill = "Correlation")

print(p4)

# Time series plot
ts_df <- data.frame(
  date = seq.Date(from = as.Date("2020-01-01"), 
                  by = "month", 
                  length.out = 100),
  value = as.numeric(ts_data)
)

p5 <- ggplot(ts_df, aes(x = date, y = value)) +
  geom_line(color = "blue", size = 1) +
  geom_smooth(method = "loess", color = "red", se = TRUE) +
  scale_x_date(labels = date_format("%Y-%m"), 
               breaks = date_breaks("1 year")) +
  labs(title = "Time Series Data",
       x = "Date",
       y = "Value") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

print(p5)
```

### Interactive Visualizations

```r
library(plotly)
library(DT)

# Interactive scatter plot
p_interactive <- plot_ly(mtcars, 
                        x = ~wt, 
                        y = ~mpg, 
                        color = ~factor(cyl),
                        size = ~hp,
                        text = ~paste("Car:", rownames(mtcars),
                                     "<br>Weight:", wt,
                                     "<br>MPG:", mpg,
                                     "<br>HP:", hp),
                        hovertemplate = "%{text}<extra></extra>") %>%
  add_markers() %>%
  layout(title = "Interactive MPG vs Weight",
         xaxis = list(title = "Weight (1000 lbs)"),
         yaxis = list(title = "Miles per Gallon"))

p_interactive

# Interactive data table
datatable(mtcars, 
          options = list(scrollX = TRUE,
                        pageLength = 10,
                        searchHighlight = TRUE),
          filter = "top")
```

## Machine Learning in R

### Supervised Learning

```r
library(caret)
library(randomForest)
library(e1071)

# Prepare data
set.seed(123)
data(iris)
trainIndex <- createDataPartition(iris$Species, p = 0.8, list = FALSE)
train_data <- iris[trainIndex, ]
test_data <- iris[-trainIndex, ]

# Random Forest
rf_model <- randomForest(Species ~ ., data = train_data, 
                        ntree = 100, importance = TRUE)

# Predictions
rf_predictions <- predict(rf_model, test_data)

# Confusion matrix
confusionMatrix(rf_predictions, test_data$Species)

# Feature importance
importance(rf_model)
varImpPlot(rf_model)

# SVM
svm_model <- svm(Species ~ ., data = train_data, 
                kernel = "radial", cost = 1, gamma = 0.1)

svm_predictions <- predict(svm_model, test_data)
confusionMatrix(svm_predictions, test_data$Species)

# Cross-validation
train_control <- trainControl(method = "cv", number = 10)

# Train multiple models
models <- list(
  rf = train(Species ~ ., data = train_data, method = "rf",
            trControl = train_control),
  svm = train(Species ~ ., data = train_data, method = "svmRadial",
             trControl = train_control),
  knn = train(Species ~ ., data = train_data, method = "knn",
             trControl = train_control)
)

# Compare models
resamples_result <- resamples(models)
summary(resamples_result)
bwplot(resamples_result)
```

### Unsupervised Learning

```r
# K-means clustering
set.seed(123)
iris_scaled <- scale(iris[, 1:4])

# Determine optimal number of clusters
wss <- sapply(1:10, function(k) {
  kmeans(iris_scaled, k, nstart = 10)$tot.withinss
})

plot(1:10, wss, type = "b", pch = 19, frame = FALSE,
     xlab = "Number of clusters K",
     ylab = "Total within-clusters sum of squares")

# K-means with optimal clusters
kmeans_result <- kmeans(iris_scaled, centers = 3, nstart = 25)

# Visualize clusters
library(cluster)
clusplot(iris_scaled, kmeans_result$cluster, 
         color = TRUE, shade = TRUE, lines = 0)

# Hierarchical clustering
dist_matrix <- dist(iris_scaled)
hclust_result <- hclust(dist_matrix, method = "ward.D2")

plot(hclust_result, main = "Hierarchical Clustering Dendrogram")
rect.hclust(hclust_result, k = 3, border = "red")

# Principal Component Analysis
pca_result <- prcomp(iris[, 1:4], scale. = TRUE)
summary(pca_result)

# PCA visualization
biplot(pca_result, scale = 0)

# Scree plot
plot(pca_result, type = "l", main = "Scree Plot")

# PCA scores
pca_scores <- as.data.frame(pca_result$x)
pca_scores$Species <- iris$Species

ggplot(pca_scores, aes(x = PC1, y = PC2, color = Species)) +
  geom_point(size = 3) +
  labs(title = "PCA - First Two Components",
       x = paste0("PC1 (", round(summary(pca_result)$importance[2,1]*100, 1), "%)"),
       y = paste0("PC2 (", round(summary(pca_result)$importance[2,2]*100, 1), "%)")) +
  theme_minimal()
```

## Package Development

### Creating an R Package

```r
# Install required packages
if (!require(devtools)) install.packages("devtools")
if (!require(roxygen2)) install.packages("roxygen2")
if (!require(testthat)) install.packages("testthat")

# Create package structure
# devtools::create("mypackage")

# Example function with documentation
#' Calculate BMI
#'
#' This function calculates Body Mass Index (BMI) from weight and height.
#'
#' @param weight Numeric vector of weights in kilograms
#' @param height Numeric vector of heights in meters
#' @return Numeric vector of BMI values
#' @examples
#' calculate_bmi(70, 1.75)
#' calculate_bmi(c(60, 70, 80), c(1.6, 1.7, 1.8))
#' @export
calculate_bmi <- function(weight, height) {
  if (!is.numeric(weight) || !is.numeric(height)) {
    stop("Weight and height must be numeric")
  }
  
  if (any(weight <= 0) || any(height <= 0)) {
    stop("Weight and height must be positive")
  }
  
  bmi <- weight / (height^2)
  return(bmi)
}

# Testing framework
library(testthat)

test_that("BMI calculation works correctly", {
  expect_equal(calculate_bmi(70, 1.75), 22.86, tolerance = 0.01)
  expect_error(calculate_bmi("70", 1.75))
  expect_error(calculate_bmi(-70, 1.75))
})

# Package documentation
# DESCRIPTION file content:
Package: mypackage
Title: My Data Science Package
Version: 0.1.0
Author: Your Name <your.email@domain.com>
Maintainer: Your Name <your.email@domain.com>
Description: A collection of useful functions for data science.
License: MIT + file LICENSE
Encoding: UTF-8
LazyData: true
RoxygenNote: 7.1.0
Imports:
    dplyr,
    ggplot2
Suggests:
    testthat

# Build and check package
# devtools::document()
# devtools::check()
# devtools::install()
```

### Package Management

```r
# Package installation and management
install.packages("package_name")
library(package_name)

# Install from GitHub
devtools::install_github("username/repository")

# Package dependencies
renv::init()    # Initialize project environment
renv::snapshot() # Save current package versions
renv::restore()  # Restore saved environment

# Check package status
sessionInfo()
installed.packages()[, c("Package", "Version")]
```

## Advanced Topics

### Functional Programming

```r
# Higher-order functions
apply_to_columns <- function(data, columns, func, ...) {
  data[columns] <- lapply(data[columns], func, ...)
  return(data)
}

# Example usage
mtcars_scaled <- apply_to_columns(mtcars, 
                                 c("mpg", "hp", "wt"), 
                                 scale)

# Purrr for functional programming
library(purrr)

# Map functions
numbers <- 1:10
map_dbl(numbers, ~ .x^2)
map_chr(numbers, ~ paste("Number:", .x))

# Working with lists
nested_list <- list(
  a = 1:5,
  b = 6:10,
  c = 11:15
)

map_dbl(nested_list, mean)
map_dbl(nested_list, ~ sum(.x > 8))

# Safely handle errors
safe_log <- safely(log)
map(c(1, -1, 2), safe_log)

# Reduce operations
reduce(1:10, `+`)  # Sum
reduce(c("a", "b", "c"), paste, sep = "-")
```

### Metaprogramming

```r
# Non-standard evaluation
library(rlang)

# Creating functions that use NSE
summarize_var <- function(data, var) {
  var <- enquo(var)
  
  data %>%
    summarise(
      mean = mean(!!var, na.rm = TRUE),
      median = median(!!var, na.rm = TRUE),
      sd = sd(!!var, na.rm = TRUE)
    )
}

# Usage
mtcars %>% summarize_var(mpg)

# Dynamic column selection
select_numeric <- function(data) {
  numeric_vars <- map_lgl(data, is.numeric)
  select(data, which(numeric_vars))
}

# Expression evaluation
my_expr <- expr(x + y)
eval_tidy(my_expr, data = list(x = 1, y = 2))
```

### Parallel Computing

```r
library(parallel)
library(foreach)
library(doParallel)

# Detect number of cores
num_cores <- detectCores()
print(paste("Available cores:", num_cores))

# Parallel apply
data_list <- split(mtcars, mtcars$cyl)
results <- mclapply(data_list, function(x) mean(x$mpg), mc.cores = 2)

# Parallel for loop
registerDoParallel(cores = 2)

results <- foreach(i = 1:10, .combine = c) %dopar% {
  sum(rnorm(1000000))
}

# Benchmark
system.time({
  sequential <- sapply(1:10, function(i) sum(rnorm(1000000)))
})

system.time({
  parallel_result <- foreach(i = 1:10, .combine = c) %dopar% {
    sum(rnorm(1000000))
  }
})

stopImplicitCluster()
```

### Database Connections

```r
library(DBI)
library(RSQLite)
library(odbc)

# SQLite connection
con <- dbConnect(RSQLite::SQLite(), ":memory:")

# Write data to database
dbWriteTable(con, "mtcars", mtcars)

# Query data
result <- dbGetQuery(con, "SELECT * FROM mtcars WHERE mpg > 20")

# Use dplyr with databases
library(dbplyr)
mtcars_db <- tbl(con, "mtcars")

mtcars_db %>%
  filter(mpg > 20) %>%
  select(mpg, cyl, hp) %>%
  arrange(desc(mpg)) %>%
  collect()

# Close connection
dbDisconnect(con)

# PostgreSQL connection (example)
# con_pg <- dbConnect(RPostgres::Postgres(),
#                     dbname = "mydb",
#                     host = "localhost",
#                     port = 5432,
#                     user = "username",
#                     password = "password")
```

## Best Practices

### Code Style and Organization

```r
# Good function naming and structure
calculate_customer_lifetime_value <- function(customer_data, 
                                            discount_rate = 0.1,
                                            time_horizon = 5) {
  # Validate inputs
  stopifnot(
    is.data.frame(customer_data),
    is.numeric(discount_rate),
    discount_rate >= 0 && discount_rate <= 1,
    is.numeric(time_horizon),
    time_horizon > 0
  )
  
  # Main calculation
  clv <- customer_data %>%
    mutate(
      annual_value = revenue / years_active,
      discounted_value = annual_value / (1 + discount_rate),
      lifetime_value = discounted_value * time_horizon
    )
  
  return(clv)
}

# Error handling
robust_mean <- function(x, na.rm = TRUE, trim = 0) {
  tryCatch({
    if (!is.numeric(x)) {
      stop("Input must be numeric")
    }
    
    if (length(x) == 0) {
      warning("Input vector is empty")
      return(NA)
    }
    
    result <- mean(x, na.rm = na.rm, trim = trim)
    return(result)
    
  }, error = function(e) {
    message("Error in robust_mean: ", e$message)
    return(NA)
  })
}

# Logging
library(logging)
basicConfig()

analyze_data <- function(data) {
  loginfo("Starting data analysis")
  
  # Analysis steps
  summary_stats <- summary(data)
  loginfo("Computed summary statistics")
  
  # More analysis...
  
  loginfo("Analysis completed successfully")
  return(summary_stats)
}
```

### Performance Optimization

```r
# Vectorization
# Bad: Using loops
slow_function <- function(x) {
  result <- numeric(length(x))
  for (i in seq_along(x)) {
    result[i] <- x[i]^2 + 2*x[i] + 1
  }
  return(result)
}

# Good: Vectorized
fast_function <- function(x) {
  return(x^2 + 2*x + 1)
}

# Benchmark
library(microbenchmark)
x <- 1:10000
microbenchmark(
  slow = slow_function(x),
  fast = fast_function(x),
  times = 100
)

# Memory management
# Pre-allocate vectors
n <- 10000
result <- numeric(n)  # Pre-allocate
for (i in 1:n) {
  result[i] <- i^2
}

# Use appropriate data types
# Bad: Everything as character
bad_data <- data.frame(
  id = as.character(1:1000),
  category = as.character(sample(letters[1:5], 1000, replace = TRUE)),
  stringsAsFactors = FALSE
)

# Good: Appropriate types
good_data <- data.frame(
  id = 1:1000,
  category = factor(sample(letters[1:5], 1000, replace = TRUE))
)

object.size(bad_data)
object.size(good_data)
```

### Testing and Validation

```r
# Unit testing with testthat
library(testthat)

# Test file: test_functions.R
test_that("calculate_bmi works correctly", {
  # Test normal case
  expect_equal(calculate_bmi(70, 1.75), 22.86, tolerance = 0.01)
  
  # Test edge cases
  expect_error(calculate_bmi("70", 1.75))
  expect_error(calculate_bmi(-70, 1.75))
  expect_error(calculate_bmi(70, 0))
  
  # Test vectorization
  weights <- c(60, 70, 80)
  heights <- c(1.6, 1.7, 1.8)
  expected <- weights / (heights^2)
  expect_equal(calculate_bmi(weights, heights), expected)
})

# Data validation
validate_data <- function(data) {
  tests <- list()
  
  # Check for required columns
  required_cols <- c("id", "value", "category")
  missing_cols <- setdiff(required_cols, names(data))
  tests$missing_columns <- length(missing_cols) == 0
  
  # Check data types
  tests$id_numeric <- is.numeric(data$id)
  tests$value_numeric <- is.numeric(data$value)
  
  # Check for missing values
  tests$no_missing_ids <- !any(is.na(data$id))
  tests$reasonable_na_rate <- mean(is.na(data$value)) < 0.1
  
  # Return validation results
  all_passed <- all(unlist(tests))
  
  return(list(
    passed = all_passed,
    tests = tests,
    issues = names(tests)[!unlist(tests)]
  ))
}

# Example usage
sample_data <- data.frame(
  id = 1:100,
  value = c(rnorm(90), rep(NA, 10)),
  category = sample(c("A", "B", "C"), 100, replace = TRUE)
)

validation_result <- validate_data(sample_data)
print(validation_result)
```

---

*This comprehensive R guide covers essential concepts for data science. Practice these techniques and adapt them to your specific analytical needs.* 
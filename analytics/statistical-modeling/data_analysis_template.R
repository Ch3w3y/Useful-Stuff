# Data Analysis Template for R
# =============================
# 
# A comprehensive template for data analysis projects including:
# - Data loading and preprocessing
# - Exploratory Data Analysis (EDA)
# - Statistical modeling
# - Visualization with ggplot2
# - Report generation
#
# Author: Data Science Team
# Date: 2024

# Load Required Libraries
# -----------------------
library(tidyverse)      # Data manipulation and visualization
library(readxl)         # Excel file reading
library(janitor)        # Data cleaning
library(skimr)          # Summary statistics
library(corrplot)       # Correlation plots
library(GGally)         # Extended ggplot2
library(plotly)         # Interactive plots
library(DT)             # Interactive tables
library(knitr)          # Report generation
library(rmarkdown)      # Report generation
library(broom)          # Model tidying
library(modelr)         # Model utilities
library(car)            # Statistical tests
library(psych)          # Psychological statistics
library(VIM)            # Visualization of missing values
library(mice)           # Multiple imputation
library(caret)          # Machine learning
library(randomForest)   # Random forest
library(glmnet)         # Regularized regression
library(xgboost)        # Gradient boosting
library(pROC)           # ROC curves

# Configuration
# -------------
options(scipen = 999)   # Disable scientific notation
theme_set(theme_minimal()) # Set default ggplot theme

# Data Analysis Functions
# -----------------------

#' Load Data from Various Formats
#' @param file_path Path to the data file
#' @return data frame
load_data <- function(file_path) {
  ext <- tools::file_ext(file_path)
  
  data <- switch(tolower(ext),
    "csv" = read_csv(file_path),
    "xlsx" = read_excel(file_path),
    "xls" = read_excel(file_path),
    "rds" = readRDS(file_path),
    "txt" = read_delim(file_path, delim = "\t"),
    stop("Unsupported file format")
  )
  
  return(data)
}

#' Comprehensive Data Overview
#' @param df Data frame to analyze
#' @return List of summary information
data_overview <- function(df) {
  cat("=== DATASET OVERVIEW ===\n")
  cat("Dimensions:", dim(df), "\n")
  cat("Memory usage:", format(object.size(df), units = "MB"), "\n\n")
  
  # Column types
  cat("=== COLUMN TYPES ===\n")
  print(df %>% summarise_all(class) %>% gather(key = "Column", value = "Type"))
  
  # Missing values
  cat("\n=== MISSING VALUES ===\n")
  missing_summary <- df %>%
    summarise_all(~sum(is.na(.))) %>%
    gather(key = "Column", value = "Missing_Count") %>%
    mutate(Missing_Percent = round(Missing_Count / nrow(df) * 100, 2)) %>%
    filter(Missing_Count > 0) %>%
    arrange(desc(Missing_Percent))
  
  if (nrow(missing_summary) > 0) {
    print(missing_summary)
  } else {
    cat("No missing values found!\n")
  }
  
  # Basic statistics
  cat("\n=== BASIC STATISTICS ===\n")
  print(skim(df))
  
  return(list(
    dimensions = dim(df),
    missing_summary = missing_summary,
    column_types = df %>% summarise_all(class)
  ))
}

#' Identify Column Types
#' @param df Data frame
#' @return List of numeric and categorical columns
identify_column_types <- function(df) {
  numeric_cols <- df %>% select_if(is.numeric) %>% colnames()
  categorical_cols <- df %>% select_if(function(x) is.character(x) | is.factor(x)) %>% colnames()
  
  cat("Numeric columns (", length(numeric_cols), "):", paste(numeric_cols, collapse = ", "), "\n")
  cat("Categorical columns (", length(categorical_cols), "):", paste(categorical_cols, collapse = ", "), "\n")
  
  return(list(
    numeric = numeric_cols,
    categorical = categorical_cols
  ))
}

#' Detect Outliers using IQR Method
#' @param df Data frame
#' @param columns Columns to check for outliers
#' @return Data frame with outlier information
detect_outliers <- function(df, columns = NULL) {
  if (is.null(columns)) {
    columns <- df %>% select_if(is.numeric) %>% colnames()
  }
  
  outlier_summary <- df %>%
    select(all_of(columns)) %>%
    gather(key = "Variable", value = "Value") %>%
    group_by(Variable) %>%
    summarise(
      Q1 = quantile(Value, 0.25, na.rm = TRUE),
      Q3 = quantile(Value, 0.75, na.rm = TRUE),
      IQR = Q3 - Q1,
      Lower_Bound = Q1 - 1.5 * IQR,
      Upper_Bound = Q3 + 1.5 * IQR,
      Outliers = sum(Value < Lower_Bound | Value > Upper_Bound, na.rm = TRUE),
      Outlier_Percent = round(Outliers / n() * 100, 2)
    ) %>%
    filter(Outliers > 0) %>%
    arrange(desc(Outlier_Percent))
  
  return(outlier_summary)
}

#' Create Distribution Plots
#' @param df Data frame
#' @param columns Columns to plot
#' @return ggplot object
create_distribution_plots <- function(df, columns = NULL) {
  if (is.null(columns)) {
    columns <- df %>% select_if(is.numeric) %>% colnames()
  }
  
  plot_data <- df %>%
    select(all_of(columns)) %>%
    gather(key = "Variable", value = "Value")
  
  p <- plot_data %>%
    ggplot(aes(x = Value)) +
    geom_histogram(aes(y = ..density..), bins = 30, alpha = 0.7, color = "white") +
    geom_density(color = "red", size = 1) +
    facet_wrap(~Variable, scales = "free") +
    labs(title = "Distribution of Numeric Variables",
         x = "Value", y = "Density") +
    theme_minimal()
  
  return(p)
}

#' Create Correlation Analysis
#' @param df Data frame
#' @param method Correlation method
#' @return Correlation matrix and plot
correlation_analysis <- function(df, method = "pearson") {
  numeric_data <- df %>% select_if(is.numeric)
  
  if (ncol(numeric_data) < 2) {
    stop("Need at least 2 numeric variables for correlation analysis")
  }
  
  # Calculate correlation matrix
  cor_matrix <- cor(numeric_data, use = "complete.obs", method = method)
  
  # Create correlation plot
  p1 <- corrplot(cor_matrix, method = "color", type = "upper", 
                 order = "hclust", tl.cex = 0.8, tl.col = "black")
  
  # Create ggplot version
  cor_df <- cor_matrix %>%
    as.data.frame() %>%
    rownames_to_column("Variable1") %>%
    gather(key = "Variable2", value = "Correlation", -Variable1)
  
  p2 <- cor_df %>%
    ggplot(aes(x = Variable1, y = Variable2, fill = Correlation)) +
    geom_tile() +
    scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                         midpoint = 0, limit = c(-1, 1)) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    labs(title = paste("Correlation Matrix (", method, ")", sep = ""))
  
  return(list(matrix = cor_matrix, plot = p2))
}

#' Create Categorical Variable Plots
#' @param df Data frame
#' @param max_categories Maximum categories to show
#' @return ggplot object
create_categorical_plots <- function(df, max_categories = 10) {
  categorical_cols <- df %>% select_if(function(x) is.character(x) | is.factor(x)) %>% colnames()
  
  if (length(categorical_cols) == 0) {
    stop("No categorical variables found")
  }
  
  plot_data <- df %>%
    select(all_of(categorical_cols)) %>%
    gather(key = "Variable", value = "Value") %>%
    group_by(Variable, Value) %>%
    summarise(Count = n(), .groups = "drop") %>%
    group_by(Variable) %>%
    top_n(max_categories, Count) %>%
    ungroup()
  
  p <- plot_data %>%
    ggplot(aes(x = reorder(Value, Count), y = Count)) +
    geom_bar(stat = "identity", fill = "steelblue", alpha = 0.7) +
    facet_wrap(~Variable, scales = "free") +
    coord_flip() +
    labs(title = "Distribution of Categorical Variables",
         x = "Category", y = "Count") +
    theme_minimal()
  
  return(p)
}

#' Perform Statistical Tests
#' @param df Data frame
#' @param target_var Target variable name
#' @return List of test results
statistical_tests <- function(df, target_var) {
  if (!target_var %in% colnames(df)) {
    stop("Target variable not found in data frame")
  }
  
  results <- list()
  
  # Identify variable types
  col_types <- identify_column_types(df)
  
  # If target is numeric, test against categorical variables
  if (target_var %in% col_types$numeric) {
    for (cat_var in col_types$categorical) {
      if (cat_var != target_var) {
        # ANOVA test
        formula_str <- paste(target_var, "~", cat_var)
        tryCatch({
          aov_result <- aov(as.formula(formula_str), data = df)
          results[[paste(cat_var, "vs", target_var, sep = "_")]] <- list(
            test = "ANOVA",
            f_statistic = summary(aov_result)[[1]][1, "F value"],
            p_value = summary(aov_result)[[1]][1, "Pr(>F)"],
            significant = summary(aov_result)[[1]][1, "Pr(>F)"] < 0.05
          )
        }, error = function(e) {
          cat("Error in ANOVA for", cat_var, ":", e$message, "\n")
        })
      }
    }
  }
  
  # Chi-square tests for categorical vs categorical
  if (target_var %in% col_types$categorical) {
    for (cat_var in col_types$categorical) {
      if (cat_var != target_var) {
        tryCatch({
          chi_result <- chisq.test(table(df[[target_var]], df[[cat_var]]))
          results[[paste(cat_var, "vs", target_var, "chi2", sep = "_")]] <- list(
            test = "Chi-square",
            chi_square = chi_result$statistic,
            p_value = chi_result$p.value,
            significant = chi_result$p.value < 0.05
          )
        }, error = function(e) {
          cat("Error in Chi-square for", cat_var, ":", e$message, "\n")
        })
      }
    }
  }
  
  return(results)
}

#' Build Predictive Model
#' @param df Data frame
#' @param target_var Target variable
#' @param method Model method (lm, glm, rf, xgb)
#' @return Model object and performance metrics
build_model <- function(df, target_var, method = "lm") {
  # Prepare data
  df_clean <- df %>% 
    select_if(function(x) !all(is.na(x))) %>%  # Remove all-NA columns
    na.omit()  # Remove rows with any NA (simple approach)
  
  # Identify predictors
  predictors <- setdiff(colnames(df_clean), target_var)
  
  # Create formula
  formula_str <- paste(target_var, "~", paste(predictors, collapse = " + "))
  
  # Split data
  set.seed(42)
  train_indices <- sample(1:nrow(df_clean), 0.8 * nrow(df_clean))
  train_data <- df_clean[train_indices, ]
  test_data <- df_clean[-train_indices, ]
  
  # Build model based on method
  if (method == "lm") {
    model <- lm(as.formula(formula_str), data = train_data)
    predictions <- predict(model, test_data)
    
    # Calculate metrics for regression
    mse <- mean((test_data[[target_var]] - predictions)^2)
    rmse <- sqrt(mse)
    mae <- mean(abs(test_data[[target_var]] - predictions))
    r2 <- cor(test_data[[target_var]], predictions)^2
    
    metrics <- list(MSE = mse, RMSE = rmse, MAE = mae, R2 = r2)
    
  } else if (method == "rf") {
    library(randomForest)
    model <- randomForest(as.formula(formula_str), data = train_data)
    predictions <- predict(model, test_data)
    
    # Calculate metrics
    if (is.numeric(test_data[[target_var]])) {
      mse <- mean((test_data[[target_var]] - predictions)^2)
      rmse <- sqrt(mse)
      mae <- mean(abs(test_data[[target_var]] - predictions))
      r2 <- cor(test_data[[target_var]], predictions)^2
      metrics <- list(MSE = mse, RMSE = rmse, MAE = mae, R2 = r2)
    } else {
      # Classification metrics
      conf_matrix <- table(test_data[[target_var]], predictions)
      accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
      metrics <- list(Accuracy = accuracy, Confusion_Matrix = conf_matrix)
    }
  }
  
  return(list(model = model, metrics = metrics, formula = formula_str))
}

#' Generate Analysis Report
#' @param df Data frame
#' @param target_var Target variable (optional)
#' @param output_file Output file name
generate_report <- function(df, target_var = NULL, output_file = "analysis_report.html") {
  # Create a simple report
  report_content <- paste(
    "# Data Analysis Report",
    "## Dataset Overview",
    paste("- **Dimensions**: ", nrow(df), " rows x ", ncol(df), " columns"),
    paste("- **Memory Usage**: ", format(object.size(df), units = "MB")),
    "",
    "## Column Information",
    "```",
    capture.output(str(df)),
    "```",
    "",
    "## Missing Values Summary",
    "```",
    capture.output(df %>% summarise_all(~sum(is.na(.)))),
    "```",
    "",
    "## Basic Statistics",
    "```",
    capture.output(summary(df)),
    "```",
    sep = "\n"
  )
  
  writeLines(report_content, output_file)
  cat("Report saved to:", output_file, "\n")
}

# Example Usage
# -------------
if (FALSE) {  # Set to TRUE to run examples
  # Create sample data
  set.seed(42)
  n <- 1000
  
  sample_data <- data.frame(
    age = rnorm(n, 35, 10),
    income = rlnorm(n, 10, 1),
    education = sample(c("High School", "Bachelor", "Master", "PhD"), n, replace = TRUE),
    city = sample(c("New York", "Los Angeles", "Chicago", "Houston"), n, replace = TRUE),
    satisfaction = sample(1:10, n, replace = TRUE),
    stringsAsFactors = FALSE
  )
  
  # Run analysis
  overview <- data_overview(sample_data)
  col_types <- identify_column_types(sample_data)
  
  # Visualizations
  dist_plot <- create_distribution_plots(sample_data)
  print(dist_plot)
  
  cat_plot <- create_categorical_plots(sample_data)
  print(cat_plot)
  
  cor_analysis <- correlation_analysis(sample_data)
  print(cor_analysis$plot)
  
  # Statistical tests
  test_results <- statistical_tests(sample_data, "satisfaction")
  print(test_results)
  
  # Build model
  model_results <- build_model(sample_data, "satisfaction", method = "rf")
  print(model_results$metrics)
  
  # Generate report
  generate_report(sample_data, target_var = "satisfaction")
  
  cat("\nAnalysis complete! Check the generated plots and report.\n")
}

# Utility Functions
# ----------------

#' Quick EDA Function
#' Performs a quick exploratory data analysis
#' @param df Data frame
#' @param target_var Optional target variable
quick_eda <- function(df, target_var = NULL) {
  cat("Starting Quick EDA...\n\n")
  
  # Basic overview
  overview <- data_overview(df)
  
  # Column types
  col_types <- identify_column_types(df)
  
  # Outliers
  if (length(col_types$numeric) > 0) {
    outliers <- detect_outliers(df, col_types$numeric)
    if (nrow(outliers) > 0) {
      cat("\n=== OUTLIERS DETECTED ===\n")
      print(outliers)
    }
  }
  
  # Correlation analysis
  if (length(col_types$numeric) > 1) {
    cor_result <- correlation_analysis(df)
    print(cor_result$plot)
  }
  
  # Statistical tests if target provided
  if (!is.null(target_var)) {
    test_results <- statistical_tests(df, target_var)
    if (length(test_results) > 0) {
      cat("\n=== STATISTICAL TESTS ===\n")
      print(test_results)
    }
  }
  
  cat("\nQuick EDA complete!\n")
}

# Print package versions for reproducibility
cat("=== PACKAGE VERSIONS ===\n")
cat("R version:", R.version.string, "\n")
cat("tidyverse:", as.character(packageVersion("tidyverse")), "\n")
cat("caret:", as.character(packageVersion("caret")), "\n")
cat("========================\n\n")

cat("Data Analysis Template loaded successfully!\n")
cat("Use quick_eda(your_data) for a fast analysis overview.\n") 
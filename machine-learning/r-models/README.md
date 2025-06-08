# R Machine Learning & Statistical Modeling

## Overview

Comprehensive guide to machine learning and statistical modeling in R, covering traditional statistical methods, modern ML algorithms, and production deployment strategies.

## Table of Contents

- [Statistical Foundations](#statistical-foundations)
- [Classical ML Algorithms](#classical-ml-algorithms)
- [Advanced Statistical Models](#advanced-statistical-models)
- [Time Series Modeling](#time-series-modeling)
- [Bayesian Methods](#bayesian-methods)
- [Model Validation & Selection](#model-validation--selection)
- [Production Deployment](#production-deployment)

## Statistical Foundations

### Essential Libraries and Setup

```r
# Core libraries
library(tidyverse)
library(caret)
library(randomForest)
library(xgboost)
library(glmnet)
library(e1071)
library(rpart)
library(party)
library(MASS)
library(car)
library(broom)
library(modelr)
library(yardstick)

# Advanced modeling
library(mgcv)
library(nlme)
library(lme4)
library(survival)
library(forecast)
library(prophet)
library(brms)
library(rstanarm)

# Visualization
library(ggplot2)
library(plotly)
library(corrplot)
library(VIM)
library(mice)

# Set global options
options(scipen = 999)
set.seed(42)

# Helper function for model evaluation
evaluate_model <- function(model, test_data, target_var) {
  predictions <- predict(model, test_data)
  actual <- test_data[[target_var]]
  
  if (is.numeric(actual)) {
    # Regression metrics
    rmse <- sqrt(mean((actual - predictions)^2))
    mae <- mean(abs(actual - predictions))
    r_squared <- cor(actual, predictions)^2
    
    return(list(
      RMSE = rmse,
      MAE = mae,
      R_squared = r_squared
    ))
  } else {
    # Classification metrics
    confusion <- table(actual, predictions)
    accuracy <- sum(diag(confusion)) / sum(confusion)
    
    return(list(
      Confusion_Matrix = confusion,
      Accuracy = accuracy
    ))
  }
}
```

### Data Preprocessing Pipeline

```r
# Comprehensive data preprocessing class
DataPreprocessor <- R6::R6Class("DataPreprocessor",
  public = list(
    data = NULL,
    target_var = NULL,
    numeric_vars = NULL,
    categorical_vars = NULL,
    preprocessing_steps = list(),
    
    initialize = function(data, target_var) {
      self$data <- data
      self$target_var <- target_var
      self$identify_variable_types()
    },
    
    identify_variable_types = function() {
      self$numeric_vars <- names(select_if(self$data, is.numeric))
      self$categorical_vars <- names(select_if(self$data, function(x) is.factor(x) || is.character(x)))
      
      # Remove target variable from predictors
      self$numeric_vars <- setdiff(self$numeric_vars, self$target_var)
      self$categorical_vars <- setdiff(self$categorical_vars, self$target_var)
    },
    
    handle_missing_values = function(method = "mice", m = 5) {
      if (method == "mice") {
        # Multiple imputation
        mice_result <- mice(self$data, m = m, method = 'pmm', printFlag = FALSE)
        self$data <- complete(mice_result)
        self$preprocessing_steps <- append(self$preprocessing_steps, "MICE imputation")
      } else if (method == "median_mode") {
        # Simple imputation
        for (var in self$numeric_vars) {
          self$data[[var]][is.na(self$data[[var]])] <- median(self$data[[var]], na.rm = TRUE)
        }
        for (var in self$categorical_vars) {
          mode_val <- names(sort(table(self$data[[var]]), decreasing = TRUE))[1]
          self$data[[var]][is.na(self$data[[var]])] <- mode_val
        }
        self$preprocessing_steps <- append(self$preprocessing_steps, "Median/Mode imputation")
      }
      invisible(self)
    },
    
    scale_numeric_variables = function(method = "standardize") {
      if (method == "standardize") {
        self$data[self$numeric_vars] <- scale(self$data[self$numeric_vars])
        self$preprocessing_steps <- append(self$preprocessing_steps, "Standardization")
      } else if (method == "normalize") {
        normalize <- function(x) (x - min(x)) / (max(x) - min(x))
        self$data[self$numeric_vars] <- lapply(self$data[self$numeric_vars], normalize)
        self$preprocessing_steps <- append(self$preprocessing_steps, "Min-Max normalization")
      }
      invisible(self)
    },
    
    encode_categorical_variables = function(method = "dummy") {
      if (method == "dummy") {
        # Create dummy variables
        dummy_vars <- model.matrix(~ . - 1, data = self$data[self$categorical_vars])
        self$data <- cbind(self$data[setdiff(names(self$data), self$categorical_vars)], dummy_vars)
        self$preprocessing_steps <- append(self$preprocessing_steps, "Dummy encoding")
      }
      invisible(self)
    },
    
    remove_outliers = function(method = "iqr", threshold = 1.5) {
      if (method == "iqr") {
        for (var in self$numeric_vars) {
          Q1 <- quantile(self$data[[var]], 0.25, na.rm = TRUE)
          Q3 <- quantile(self$data[[var]], 0.75, na.rm = TRUE)
          IQR <- Q3 - Q1
          
          lower_bound <- Q1 - threshold * IQR
          upper_bound <- Q3 + threshold * IQR
          
          self$data <- self$data[self$data[[var]] >= lower_bound & self$data[[var]] <= upper_bound, ]
        }
        self$preprocessing_steps <- append(self$preprocessing_steps, paste("IQR outlier removal (threshold:", threshold, ")"))
      }
      invisible(self)
    },
    
    feature_selection = function(method = "correlation", threshold = 0.8) {
      if (method == "correlation") {
        # Remove highly correlated features
        cor_matrix <- cor(self$data[self$numeric_vars], use = "complete.obs")
        high_cor <- findCorrelation(cor_matrix, cutoff = threshold)
        
        if (length(high_cor) > 0) {
          vars_to_remove <- self$numeric_vars[high_cor]
          self$data <- self$data[, !names(self$data) %in% vars_to_remove]
          self$numeric_vars <- setdiff(self$numeric_vars, vars_to_remove)
          self$preprocessing_steps <- append(self$preprocessing_steps, 
                                           paste("Correlation-based feature removal (threshold:", threshold, ")"))
        }
      }
      invisible(self)
    },
    
    get_processed_data = function() {
      return(self$data)
    },
    
    get_preprocessing_summary = function() {
      cat("Preprocessing Steps Applied:\n")
      for (i in seq_along(self$preprocessing_steps)) {
        cat(paste(i, ".", self$preprocessing_steps[i], "\n"))
      }
      
      cat("\nFinal Dataset Summary:\n")
      cat("Dimensions:", dim(self$data), "\n")
      cat("Numeric variables:", length(self$numeric_vars), "\n")
      cat("Categorical variables:", length(self$categorical_vars), "\n")
    }
  )
)
```

## Classical ML Algorithms

### Comprehensive Model Suite

```r
# Machine Learning Model Suite
MLModelSuite <- R6::R6Class("MLModelSuite",
  public = list(
    data = NULL,
    target_var = NULL,
    train_data = NULL,
    test_data = NULL,
    models = list(),
    results = list(),
    
    initialize = function(data, target_var, test_split = 0.2) {
      self$data <- data
      self$target_var <- target_var
      self$split_data(test_split)
    },
    
    split_data = function(test_split) {
      train_indices <- createDataPartition(self$data[[self$target_var]], p = 1 - test_split, list = FALSE)
      self$train_data <- self$data[train_indices, ]
      self$test_data <- self$data[-train_indices, ]
    },
    
    # Linear Models
    fit_linear_regression = function() {
      formula <- as.formula(paste(self$target_var, "~ ."))
      model <- lm(formula, data = self$train_data)
      self$models[["linear_regression"]] <- model
      
      # Diagnostics
      diagnostics <- list(
        summary = summary(model),
        anova = anova(model),
        residuals_vs_fitted = plot(model, which = 1),
        qq_plot = plot(model, which = 2),
        vif = vif(model)
      )
      
      self$results[["linear_regression"]] <- list(
        model = model,
        diagnostics = diagnostics,
        predictions = predict(model, self$test_data)
      )
      
      invisible(self)
    },
    
    fit_logistic_regression = function() {
      formula <- as.formula(paste(self$target_var, "~ ."))
      model <- glm(formula, data = self$train_data, family = binomial())
      self$models[["logistic_regression"]] <- model
      
      # Model evaluation
      predictions <- predict(model, self$test_data, type = "response")
      predicted_classes <- ifelse(predictions > 0.5, 1, 0)
      
      self$results[["logistic_regression"]] <- list(
        model = model,
        predictions = predictions,
        predicted_classes = predicted_classes,
        auc = pROC::auc(self$test_data[[self$target_var]], predictions)
      )
      
      invisible(self)
    },
    
    # Regularized Models
    fit_ridge_regression = function(alpha = 0) {
      x_train <- model.matrix(as.formula(paste("~", paste(setdiff(names(self$train_data), self$target_var), collapse = " + "))), 
                             data = self$train_data)[, -1]
      y_train <- self$train_data[[self$target_var]]
      
      cv_model <- cv.glmnet(x_train, y_train, alpha = alpha)
      model <- glmnet(x_train, y_train, alpha = alpha, lambda = cv_model$lambda.min)
      
      x_test <- model.matrix(as.formula(paste("~", paste(setdiff(names(self$test_data), self$target_var), collapse = " + "))), 
                            data = self$test_data)[, -1]
      predictions <- predict(model, x_test)
      
      self$models[["ridge_regression"]] <- model
      self$results[["ridge_regression"]] <- list(
        model = model,
        cv_model = cv_model,
        predictions = predictions,
        lambda_min = cv_model$lambda.min
      )
      
      invisible(self)
    },
    
    fit_lasso_regression = function(alpha = 1) {
      x_train <- model.matrix(as.formula(paste("~", paste(setdiff(names(self$train_data), self$target_var), collapse = " + "))), 
                             data = self$train_data)[, -1]
      y_train <- self$train_data[[self$target_var]]
      
      cv_model <- cv.glmnet(x_train, y_train, alpha = alpha)
      model <- glmnet(x_train, y_train, alpha = alpha, lambda = cv_model$lambda.min)
      
      x_test <- model.matrix(as.formula(paste("~", paste(setdiff(names(self$test_data), self$target_var), collapse = " + "))), 
                            data = self$test_data)[, -1]
      predictions <- predict(model, x_test)
      
      # Feature selection (non-zero coefficients)
      selected_features <- rownames(coef(model))[which(coef(model) != 0)]
      
      self$models[["lasso_regression"]] <- model
      self$results[["lasso_regression"]] <- list(
        model = model,
        cv_model = cv_model,
        predictions = predictions,
        selected_features = selected_features,
        lambda_min = cv_model$lambda.min
      )
      
      invisible(self)
    },
    
    # Tree-based Models
    fit_random_forest = function(ntree = 500, mtry = NULL) {
      if (is.null(mtry)) {
        mtry <- floor(sqrt(ncol(self$train_data) - 1))
      }
      
      formula <- as.formula(paste(self$target_var, "~ ."))
      model <- randomForest(formula, data = self$train_data, ntree = ntree, mtry = mtry, importance = TRUE)
      
      predictions <- predict(model, self$test_data)
      feature_importance <- importance(model)
      
      self$models[["random_forest"]] <- model
      self$results[["random_forest"]] <- list(
        model = model,
        predictions = predictions,
        feature_importance = feature_importance,
        oob_error = model$err.rate[ntree, "OOB"]
      )
      
      invisible(self)
    },
    
    fit_xgboost = function(nrounds = 100, max_depth = 6, eta = 0.3) {
      # Prepare data for XGBoost
      train_matrix <- model.matrix(as.formula(paste("~", paste(setdiff(names(self$train_data), self$target_var), collapse = " + "))), 
                                  data = self$train_data)[, -1]
      test_matrix <- model.matrix(as.formula(paste("~", paste(setdiff(names(self$test_data), self$target_var), collapse = " + "))), 
                                 data = self$test_data)[, -1]
      
      dtrain <- xgb.DMatrix(data = train_matrix, label = self$train_data[[self$target_var]])
      dtest <- xgb.DMatrix(data = test_matrix, label = self$test_data[[self$target_var]])
      
      # Cross-validation for optimal parameters
      cv_result <- xgb.cv(
        data = dtrain,
        nrounds = nrounds,
        max_depth = max_depth,
        eta = eta,
        nfold = 5,
        early_stopping_rounds = 10,
        verbose = FALSE
      )
      
      optimal_rounds <- cv_result$best_iteration
      
      # Train final model
      model <- xgboost(
        data = dtrain,
        nrounds = optimal_rounds,
        max_depth = max_depth,
        eta = eta,
        verbose = FALSE
      )
      
      predictions <- predict(model, dtest)
      feature_importance <- xgb.importance(model = model)
      
      self$models[["xgboost"]] <- model
      self$results[["xgboost"]] <- list(
        model = model,
        predictions = predictions,
        feature_importance = feature_importance,
        optimal_rounds = optimal_rounds,
        cv_result = cv_result
      )
      
      invisible(self)
    },
    
    # Support Vector Machine
    fit_svm = function(kernel = "radial", cost = 1, gamma = 1) {
      formula <- as.formula(paste(self$target_var, "~ ."))
      
      # Tune parameters
      tune_result <- tune(svm, formula, data = self$train_data,
                         ranges = list(cost = c(0.1, 1, 10, 100),
                                      gamma = c(0.01, 0.1, 1, 10)),
                         kernel = kernel)
      
      best_model <- tune_result$best.model
      predictions <- predict(best_model, self$test_data)
      
      self$models[["svm"]] <- best_model
      self$results[["svm"]] <- list(
        model = best_model,
        predictions = predictions,
        tune_result = tune_result,
        best_parameters = tune_result$best.parameters
      )
      
      invisible(self)
    },
    
    # Model Comparison
    compare_models = function() {
      comparison_results <- data.frame(
        Model = character(),
        RMSE = numeric(),
        MAE = numeric(),
        R_squared = numeric(),
        stringsAsFactors = FALSE
      )
      
      for (model_name in names(self$results)) {
        predictions <- self$results[[model_name]]$predictions
        actual <- self$test_data[[self$target_var]]
        
        rmse <- sqrt(mean((actual - predictions)^2))
        mae <- mean(abs(actual - predictions))
        r_squared <- cor(actual, predictions)^2
        
        comparison_results <- rbind(comparison_results, 
                                  data.frame(Model = model_name, RMSE = rmse, MAE = mae, R_squared = r_squared))
      }
      
      comparison_results <- comparison_results[order(comparison_results$RMSE), ]
      return(comparison_results)
    },
    
    plot_model_comparison = function() {
      comparison <- self$compare_models()
      
      p1 <- ggplot(comparison, aes(x = reorder(Model, -RMSE), y = RMSE)) +
        geom_bar(stat = "identity", fill = "steelblue") +
        theme_minimal() +
        labs(title = "Model Comparison - RMSE", x = "Model", y = "RMSE") +
        theme(axis.text.x = element_text(angle = 45, hjust = 1))
      
      p2 <- ggplot(comparison, aes(x = reorder(Model, -R_squared), y = R_squared)) +
        geom_bar(stat = "identity", fill = "darkgreen") +
        theme_minimal() +
        labs(title = "Model Comparison - R²", x = "Model", y = "R²") +
        theme(axis.text.x = element_text(angle = 45, hjust = 1))
      
      gridExtra::grid.arrange(p1, p2, ncol = 2)
    }
  )
)
```

## Advanced Statistical Models

### Generalized Additive Models (GAMs)

```r
# Advanced GAM modeling
fit_gam_models <- function(data, target_var, smooth_vars = NULL) {
  
  if (is.null(smooth_vars)) {
    numeric_vars <- names(select_if(data, is.numeric))
    smooth_vars <- setdiff(numeric_vars, target_var)
  }
  
  # Build GAM formula with smooth terms
  smooth_terms <- paste0("s(", smooth_vars, ")", collapse = " + ")
  categorical_vars <- names(select_if(data, function(x) is.factor(x) || is.character(x)))
  categorical_vars <- setdiff(categorical_vars, target_var)
  
  if (length(categorical_vars) > 0) {
    categorical_terms <- paste(categorical_vars, collapse = " + ")
    formula_str <- paste(target_var, "~", smooth_terms, "+", categorical_terms)
  } else {
    formula_str <- paste(target_var, "~", smooth_terms)
  }
  
  gam_formula <- as.formula(formula_str)
  
  # Fit GAM model
  gam_model <- gam(gam_formula, data = data, family = gaussian())
  
  # Model diagnostics
  gam_check(gam_model)
  
  # Summary and plots
  summary(gam_model)
  plot(gam_model, pages = 1)
  
  return(gam_model)
}

# Mixed Effects Models
fit_mixed_effects_models <- function(data, target_var, fixed_effects, random_effects) {
  
  # Linear Mixed Effects Model
  lme_formula <- as.formula(paste(target_var, "~", paste(fixed_effects, collapse = " + "), 
                                 "+ (1|", random_effects, ")"))
  
  lme_model <- lmer(lme_formula, data = data)
  
  # Model diagnostics
  plot(lme_model)
  qqnorm(resid(lme_model))
  qqline(resid(lme_model))
  
  # Random effects
  random_effects_plot <- plot(ranef(lme_model))
  
  return(list(
    model = lme_model,
    summary = summary(lme_model),
    random_effects = ranef(lme_model),
    fixed_effects = fixef(lme_model)
  ))
}

# Survival Analysis
perform_survival_analysis <- function(data, time_var, event_var, covariates) {
  
  # Kaplan-Meier estimator
  km_formula <- as.formula(paste("Surv(", time_var, ",", event_var, ") ~ 1"))
  km_fit <- survfit(km_formula, data = data)
  
  # Plot survival curve
  km_plot <- ggsurvplot(km_fit, data = data, pval = TRUE, conf.int = TRUE,
                       risk.table = TRUE, risk.table.col = "strata",
                       linetype = "strata", surv.median.line = "hv",
                       ggtheme = theme_bw())
  
  # Cox Proportional Hazards Model
  cox_formula <- as.formula(paste("Surv(", time_var, ",", event_var, ") ~", 
                                 paste(covariates, collapse = " + ")))
  cox_model <- coxph(cox_formula, data = data)
  
  # Test proportional hazards assumption
  ph_test <- cox.zph(cox_model)
  
  # Forest plot for hazard ratios
  forest_plot <- ggforest(cox_model, data = data)
  
  return(list(
    km_fit = km_fit,
    km_plot = km_plot,
    cox_model = cox_model,
    cox_summary = summary(cox_model),
    ph_test = ph_test,
    forest_plot = forest_plot
  ))
}
```

---

*This comprehensive R machine learning guide provides complete coverage from statistical foundations to advanced modeling techniques and production deployment.* 
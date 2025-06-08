# Advanced Statistical Modeling in R
# ===================================
# 
# Comprehensive template for advanced statistical analysis including:
# - Bayesian inference
# - Time series analysis
# - Causal inference
# - Mixed effects models
# - Survival analysis
# - Hierarchical modeling
#
# Author: Data Science Team
# Date: 2024

# Load Required Libraries
# -----------------------
library(tidyverse)      # Data manipulation and visualization
library(rstanarm)       # Bayesian modeling via Stan
library(brms)           # Bayesian regression models
library(MCMCglmm)       # MCMC generalized linear mixed models
library(forecast)       # Time series forecasting
library(prophet)        # Facebook's Prophet forecasting
library(tseries)        # Time series analysis
library(vars)           # Vector autoregression
library(survival)       # Survival analysis
library(survminer)      # Survival analysis visualization
library(lme4)           # Linear mixed effects models
library(nlme)           # Nonlinear mixed effects models
library(causalweight)   # Causal inference
library(MatchIt)        # Matching for causal inference
library(bcp)            # Bayesian change point analysis
library(BayesFactor)    # Bayesian hypothesis testing
library(rstan)          # R interface to Stan
library(loo)            # Leave-one-out cross-validation
library(bayesplot)      # Bayesian plotting
library(tidybayes)      # Tidy data + geoms for Bayesian models
library(modelr)         # Model utilities
library(broom)          # Tidy model objects
library(broom.mixed)    # Tidy mixed model objects
library(performance)    # Model performance
library(see)            # Visualization for performance
library(report)         # Automated reporting

# Configuration
# -------------
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)
theme_set(theme_minimal())

# =================================================
# 1. BAYESIAN LINEAR REGRESSION
# =================================================

#' Bayesian Linear Regression Analysis
#' @param data Data frame
#' @param formula Model formula
#' @param prior Prior specification
#' @return Bayesian model object
bayesian_linear_regression <- function(data, formula, prior = NULL) {
  
  cat("=== BAYESIAN LINEAR REGRESSION ===\n")
  
  # Set default priors if not specified
  if (is.null(prior)) {
    prior <- normal(location = 0, scale = 2.5, autoscale = TRUE)
  }
  
  # Fit Bayesian linear model
  model <- stan_glm(
    formula = formula,
    data = data,
    family = gaussian(),
    prior = prior,
    prior_intercept = normal(0, 2.5, autoscale = TRUE),
    prior_aux = exponential(1, autoscale = TRUE),
    chains = 4,
    iter = 2000,
    cores = parallel::detectCores()
  )
  
  # Model summary
  cat("\nModel Summary:\n")
  print(model)
  
  # Posterior intervals
  cat("\nPosterior Intervals:\n")
  print(posterior_interval(model, prob = 0.95))
  
  # Model diagnostics
  cat("\nModel Diagnostics:\n")
  
  # R-hat statistics
  rhat_values <- rhat(model)
  cat("R-hat range:", range(rhat_values, na.rm = TRUE), "\n")
  
  # Effective sample size
  ess_values <- neff_ratio(model)
  cat("ESS ratio range:", range(ess_values, na.rm = TRUE), "\n")
  
  # Posterior predictive checks
  pp_check(model) + 
    ggtitle("Posterior Predictive Check") +
    theme_minimal()
  
  return(model)
}

# =================================================
# 2. HIERARCHICAL/MIXED EFFECTS MODELS
# =================================================

#' Hierarchical Bayesian Model
#' @param data Data frame with grouping variables
#' @param formula Model formula with random effects
#' @return Hierarchical model object
hierarchical_model <- function(data, formula) {
  
  cat("=== HIERARCHICAL BAYESIAN MODEL ===\n")
  
  # Fit hierarchical model using brms
  model <- brm(
    formula = formula,
    data = data,
    family = gaussian(),
    prior = c(
      prior(normal(0, 1), class = Intercept),
      prior(normal(0, 1), class = b),
      prior(exponential(1), class = sd),
      prior(exponential(1), class = sigma)
    ),
    chains = 4,
    iter = 2000,
    cores = parallel::detectCores(),
    control = list(adapt_delta = 0.95)
  )
  
  # Model summary
  cat("\nModel Summary:\n")
  print(model)
  
  # Random effects
  cat("\nRandom Effects:\n")
  print(VarCorr(model))
  
  # Plot random effects
  plot(model, ask = FALSE)
  
  # Conditional effects
  plot(conditional_effects(model), ask = FALSE)
  
  return(model)
}

#' Frequentist Mixed Effects Model
#' @param data Data frame
#' @param formula Model formula
#' @return Mixed effects model object
mixed_effects_model <- function(data, formula) {
  
  cat("=== MIXED EFFECTS MODEL ===\n")
  
  # Fit linear mixed effects model
  model <- lmer(formula, data = data)
  
  # Model summary
  cat("\nModel Summary:\n")
  print(summary(model))
  
  # Random effects
  cat("\nRandom Effects:\n")
  print(VarCorr(model))
  
  # Fixed effects
  cat("\nFixed Effects:\n")
  print(fixef(model))
  
  # Model diagnostics
  cat("\nModel Diagnostics:\n")
  
  # Residual plots
  plot(model)
  
  # QQ plot
  qqnorm(resid(model))
  qqline(resid(model))
  
  # Check model assumptions
  check_model(model)
  
  return(model)
}

# =================================================
# 3. TIME SERIES ANALYSIS
# =================================================

#' Comprehensive Time Series Analysis
#' @param ts_data Time series data
#' @param date_col Date column name
#' @param value_col Value column name
#' @return List of analysis results
time_series_analysis <- function(ts_data, date_col, value_col) {
  
  cat("=== TIME SERIES ANALYSIS ===\n")
  
  # Convert to time series object
  ts_object <- ts(ts_data[[value_col]], 
                  start = c(year(min(ts_data[[date_col]])), 1),
                  frequency = 12)  # Assuming monthly data
  
  # Basic time series plots
  autoplot(ts_object) +
    ggtitle("Time Series Plot") +
    theme_minimal()
  
  # Decomposition
  decomp <- decompose(ts_object)
  autoplot(decomp) +
    ggtitle("Time Series Decomposition") +
    theme_minimal()
  
  # Stationarity tests
  cat("\nStationarity Tests:\n")
  adf_test <- adf.test(ts_object)
  cat("ADF Test p-value:", adf_test$p.value, "\n")
  
  kpss_test <- kpss.test(ts_object)
  cat("KPSS Test p-value:", kpss_test$p.value, "\n")
  
  # ACF and PACF
  acf(ts_object, main = "Autocorrelation Function")
  pacf(ts_object, main = "Partial Autocorrelation Function")
  
  # ARIMA modeling
  cat("\nARIMA Modeling:\n")
  arima_model <- auto.arima(ts_object)
  print(arima_model)
  
  # Forecast
  forecast_result <- forecast(arima_model, h = 12)
  autoplot(forecast_result) +
    ggtitle("ARIMA Forecast") +
    theme_minimal()
  
  # Prophet forecasting
  cat("\nProphet Forecasting:\n")
  prophet_data <- data.frame(
    ds = ts_data[[date_col]],
    y = ts_data[[value_col]]
  )
  
  prophet_model <- prophet(prophet_data)
  future <- make_future_dataframe(prophet_model, periods = 12, freq = 'month')
  prophet_forecast <- predict(prophet_model, future)
  
  plot(prophet_model, prophet_forecast)
  prophet_plot_components(prophet_model, prophet_forecast)
  
  return(list(
    ts_object = ts_object,
    decomposition = decomp,
    arima_model = arima_model,
    arima_forecast = forecast_result,
    prophet_model = prophet_model,
    prophet_forecast = prophet_forecast
  ))
}

#' Vector Autoregression (VAR) Analysis
#' @param data Multivariate time series data
#' @return VAR model results
var_analysis <- function(data) {
  
  cat("=== VECTOR AUTOREGRESSION ANALYSIS ===\n")
  
  # Determine optimal lag length
  lag_selection <- VARselect(data, lag.max = 10)
  optimal_lag <- lag_selection$selection["AIC(n)"]
  
  cat("Optimal lag (AIC):", optimal_lag, "\n")
  
  # Fit VAR model
  var_model <- VAR(data, p = optimal_lag)
  
  # Model summary
  cat("\nVAR Model Summary:\n")
  print(summary(var_model))
  
  # Granger causality tests
  cat("\nGranger Causality Tests:\n")
  for (i in 1:ncol(data)) {
    for (j in 1:ncol(data)) {
      if (i != j) {
        causality_test <- causality(var_model, cause = colnames(data)[j])
        cat(paste("Does", colnames(data)[j], "Granger-cause", colnames(data)[i], "?\n"))
        cat("P-value:", causality_test$Granger$p.value, "\n")
      }
    }
  }
  
  # Impulse response functions
  irf_result <- irf(var_model, n.ahead = 10)
  plot(irf_result)
  
  return(var_model)
}

# =================================================
# 4. SURVIVAL ANALYSIS
# =================================================

#' Comprehensive Survival Analysis
#' @param data Data frame with survival data
#' @param time_col Time to event column
#' @param event_col Event indicator column
#' @param group_col Grouping variable
#' @return List of survival analysis results
survival_analysis <- function(data, time_col, event_col, group_col = NULL) {
  
  cat("=== SURVIVAL ANALYSIS ===\n")
  
  # Create survival object
  surv_object <- Surv(data[[time_col]], data[[event_col]])
  
  # Kaplan-Meier estimator
  if (!is.null(group_col)) {
    km_fit <- survfit(surv_object ~ data[[group_col]])
    
    # Plot survival curves
    ggsurvplot(km_fit, 
               data = data,
               pval = TRUE,
               conf.int = TRUE,
               risk.table = TRUE,
               tables.height = 0.2,
               tables.theme = theme_cleantable(),
               title = "Kaplan-Meier Survival Curves")
    
    # Log-rank test
    logrank_test <- survdiff(surv_object ~ data[[group_col]])
    cat("\nLog-rank test:\n")
    print(logrank_test)
    
  } else {
    km_fit <- survfit(surv_object ~ 1)
    
    # Plot overall survival curve
    ggsurvplot(km_fit, 
               data = data,
               conf.int = TRUE,
               title = "Overall Survival Curve")
  }
  
  # Cox proportional hazards model
  if (!is.null(group_col)) {
    cox_formula <- as.formula(paste("surv_object ~", group_col))
  } else {
    # Include all numeric variables
    numeric_vars <- data %>% 
      select_if(is.numeric) %>% 
      select(-all_of(c(time_col, event_col))) %>% 
      names()
    cox_formula <- as.formula(paste("surv_object ~", paste(numeric_vars, collapse = " + ")))
  }
  
  cox_model <- coxph(cox_formula, data = data)
  
  cat("\nCox Proportional Hazards Model:\n")
  print(summary(cox_model))
  
  # Test proportional hazards assumption
  ph_test <- cox.zph(cox_model)
  cat("\nProportional Hazards Test:\n")
  print(ph_test)
  
  # Plot Schoenfeld residuals
  ggcoxzph(ph_test)
  
  return(list(
    km_fit = km_fit,
    cox_model = cox_model,
    ph_test = ph_test
  ))
}

# =================================================
# 5. CAUSAL INFERENCE
# =================================================

#' Propensity Score Matching
#' @param data Data frame
#' @param treatment_col Treatment variable
#' @param outcome_col Outcome variable
#' @param covariates Vector of covariate names
#' @return Matching results and ATE estimate
propensity_matching <- function(data, treatment_col, outcome_col, covariates) {
  
  cat("=== PROPENSITY SCORE MATCHING ===\n")
  
  # Create formula for propensity score model
  ps_formula <- as.formula(paste(treatment_col, "~", paste(covariates, collapse = " + ")))
  
  # Perform matching
  match_result <- matchit(ps_formula, data = data, method = "nearest")
  
  # Summary of matching
  cat("\nMatching Summary:\n")
  print(summary(match_result))
  
  # Plot balance
  plot(match_result, type = "jitter")
  plot(match_result, type = "hist")
  
  # Extract matched data
  matched_data <- match.data(match_result)
  
  # Estimate treatment effect
  ate_model <- lm(as.formula(paste(outcome_col, "~", treatment_col)), 
                  data = matched_data, 
                  weights = weights)
  
  cat("\nAverage Treatment Effect:\n")
  print(summary(ate_model))
  
  # Confidence interval for ATE
  ate_ci <- confint(ate_model)[treatment_col, ]
  cat("ATE 95% CI:", ate_ci, "\n")
  
  return(list(
    match_result = match_result,
    matched_data = matched_data,
    ate_model = ate_model,
    ate_estimate = coef(ate_model)[treatment_col],
    ate_ci = ate_ci
  ))
}

#' Instrumental Variables Analysis
#' @param data Data frame
#' @param outcome_col Outcome variable
#' @param treatment_col Treatment variable
#' @param instrument_col Instrumental variable
#' @param covariates Control variables
#' @return IV analysis results
instrumental_variables <- function(data, outcome_col, treatment_col, instrument_col, covariates = NULL) {
  
  cat("=== INSTRUMENTAL VARIABLES ANALYSIS ===\n")
  
  # First stage: Instrument -> Treatment
  if (!is.null(covariates)) {
    first_stage_formula <- as.formula(paste(treatment_col, "~", instrument_col, "+", paste(covariates, collapse = " + ")))
  } else {
    first_stage_formula <- as.formula(paste(treatment_col, "~", instrument_col))
  }
  
  first_stage <- lm(first_stage_formula, data = data)
  
  cat("\nFirst Stage Results:\n")
  print(summary(first_stage))
  
  # Check instrument strength
  f_stat <- summary(first_stage)$fstatistic[1]
  cat("F-statistic for instrument:", f_stat, "\n")
  
  if (f_stat < 10) {
    warning("Weak instrument detected (F < 10)")
  }
  
  # Second stage: Use predicted treatment values
  predicted_treatment <- predict(first_stage)
  data$predicted_treatment <- predicted_treatment
  
  if (!is.null(covariates)) {
    second_stage_formula <- as.formula(paste(outcome_col, "~ predicted_treatment +", paste(covariates, collapse = " + ")))
  } else {
    second_stage_formula <- as.formula(paste(outcome_col, "~ predicted_treatment"))
  }
  
  second_stage <- lm(second_stage_formula, data = data)
  
  cat("\nSecond Stage Results (2SLS):\n")
  print(summary(second_stage))
  
  # Local Average Treatment Effect
  late <- coef(second_stage)["predicted_treatment"]
  late_se <- summary(second_stage)$coefficients["predicted_treatment", "Std. Error"]
  late_ci <- late + c(-1.96, 1.96) * late_se
  
  cat("Local Average Treatment Effect:", late, "\n")
  cat("95% CI:", late_ci, "\n")
  
  return(list(
    first_stage = first_stage,
    second_stage = second_stage,
    late = late,
    late_ci = late_ci,
    f_statistic = f_stat
  ))
}

# =================================================
# 6. BAYESIAN HYPOTHESIS TESTING
# =================================================

#' Bayesian t-test
#' @param x First group data
#' @param y Second group data (optional)
#' @param alternative Alternative hypothesis
#' @return Bayes factor and posterior
bayesian_t_test <- function(x, y = NULL, alternative = "two.sided") {
  
  cat("=== BAYESIAN T-TEST ===\n")
  
  if (is.null(y)) {
    # One-sample t-test
    bf_result <- ttestBF(x)
  } else {
    # Two-sample t-test
    bf_result <- ttestBF(x, y)
  }
  
  cat("\nBayes Factor:\n")
  print(bf_result)
  
  # Extract and sample from posterior
  posterior_samples <- posterior(bf_result, iterations = 4000)
  
  cat("\nPosterior Summary:\n")
  print(summary(posterior_samples))
  
  # Plot posterior
  plot(posterior_samples[, "delta"])
  
  return(list(
    bayes_factor = bf_result,
    posterior_samples = posterior_samples
  ))
}

#' Bayesian ANOVA
#' @param data Data frame
#' @param formula Model formula
#' @return Bayes factor and model comparison
bayesian_anova <- function(data, formula) {
  
  cat("=== BAYESIAN ANOVA ===\n")
  
  # Fit Bayesian ANOVA
  bf_result <- anovaBF(formula, data = data)
  
  cat("\nBayes Factors:\n")
  print(bf_result)
  
  # Model comparison
  cat("\nModel Comparison:\n")
  plot(bf_result)
  
  return(bf_result)
}

# =================================================
# 7. CHANGE POINT ANALYSIS
# =================================================

#' Bayesian Change Point Analysis
#' @param data Time series data
#' @return Change point analysis results
bayesian_changepoint <- function(data) {
  
  cat("=== BAYESIAN CHANGE POINT ANALYSIS ===\n")
  
  # Bayesian change point analysis
  bcp_result <- bcp(data, mcmc = 5000, burnin = 1000)
  
  cat("\nChange Point Analysis Summary:\n")
  print(summary(bcp_result))
  
  # Plot results
  plot(bcp_result, main = "Bayesian Change Point Analysis")
  
  # Identify probable change points
  prob_changepoints <- which(bcp_result$prob.mean > 0.5)
  cat("Probable change points:", prob_changepoints, "\n")
  
  return(list(
    bcp_result = bcp_result,
    changepoints = prob_changepoints
  ))
}

# =================================================
# 8. MODEL COMPARISON AND SELECTION
# =================================================

#' Comprehensive Model Comparison
#' @param models List of fitted models
#' @param data Data used for fitting
#' @return Model comparison results
model_comparison <- function(models, data) {
  
  cat("=== MODEL COMPARISON ===\n")
  
  # Information criteria comparison
  ic_comparison <- compare_performance(models, metrics = c("AIC", "BIC", "R2"))
  
  cat("\nInformation Criteria Comparison:\n")
  print(ic_comparison)
  
  # Plot comparison
  plot(ic_comparison)
  
  # Cross-validation for Bayesian models
  if (any(sapply(models, function(x) inherits(x, "stanreg")))) {
    
    cat("\nLeave-One-Out Cross-Validation:\n")
    
    loo_results <- list()
    for (i in seq_along(models)) {
      if (inherits(models[[i]], "stanreg")) {
        loo_results[[i]] <- loo(models[[i]])
      }
    }
    
    # Compare LOO
    if (length(loo_results) > 1) {
      loo_compare_result <- loo_compare(loo_results)
      print(loo_compare_result)
    }
  }
  
  return(list(
    ic_comparison = ic_comparison,
    loo_results = if(exists("loo_results")) loo_results else NULL
  ))
}

# =================================================
# 9. AUTOMATED REPORTING
# =================================================

#' Generate Automated Statistical Report
#' @param model Fitted model object
#' @param data Data used for fitting
#' @return Text report
generate_report <- function(model, data) {
  
  cat("=== AUTOMATED STATISTICAL REPORT ===\n")
  
  # Generate report
  model_report <- report(model)
  
  cat("\nModel Report:\n")
  print(model_report)
  
  # Performance metrics
  model_performance <- model_performance(model)
  cat("\nModel Performance:\n")
  print(model_performance)
  
  return(list(
    report = model_report,
    performance = model_performance
  ))
}

# =================================================
# 10. EXAMPLE USAGE FUNCTIONS
# =================================================

#' Example: Complete Bayesian Analysis Workflow
example_bayesian_workflow <- function() {
  
  cat("=== EXAMPLE: BAYESIAN ANALYSIS WORKFLOW ===\n")
  
  # Generate sample data
  set.seed(42)
  n <- 200
  group <- rep(c("A", "B"), each = n/2)
  x1 <- rnorm(n, mean = ifelse(group == "A", 0, 1), sd = 1)
  x2 <- rnorm(n, mean = 0.5, sd = 1)
  y <- 2 + 1.5*x1 + 0.8*x2 + ifelse(group == "A", 0, 1) + rnorm(n, 0, 0.5)
  
  sample_data <- data.frame(y = y, x1 = x1, x2 = x2, group = group)
  
  # Bayesian linear regression
  bayes_model <- bayesian_linear_regression(
    data = sample_data,
    formula = y ~ x1 + x2 + group
  )
  
  # Generate report
  report_results <- generate_report(bayes_model, sample_data)
  
  return(list(
    data = sample_data,
    model = bayes_model,
    report = report_results
  ))
}

#' Example: Time Series Analysis Workflow
example_timeseries_workflow <- function() {
  
  cat("=== EXAMPLE: TIME SERIES ANALYSIS WORKFLOW ===\n")
  
  # Generate sample time series data
  dates <- seq(as.Date("2020-01-01"), as.Date("2023-12-01"), by = "month")
  trend <- 1:length(dates) * 0.1
  seasonal <- sin(2 * pi * (1:length(dates)) / 12) * 2
  noise <- rnorm(length(dates), 0, 0.5)
  values <- 10 + trend + seasonal + noise
  
  ts_data <- data.frame(date = dates, value = values)
  
  # Perform time series analysis
  ts_results <- time_series_analysis(ts_data, "date", "value")
  
  return(list(
    data = ts_data,
    results = ts_results
  ))
}

# Print available functions
if (FALSE) {  # Set to TRUE to run examples
  
  cat("Advanced Statistical Modeling in R - Available Functions:\n")
  cat("=" * 60, "\n")
  cat("1. bayesian_linear_regression() - Bayesian linear models\n")
  cat("2. hierarchical_model() - Hierarchical Bayesian models\n")
  cat("3. mixed_effects_model() - Frequentist mixed effects\n")
  cat("4. time_series_analysis() - Comprehensive time series analysis\n")
  cat("5. var_analysis() - Vector autoregression\n")
  cat("6. survival_analysis() - Survival analysis\n")
  cat("7. propensity_matching() - Causal inference via matching\n")
  cat("8. instrumental_variables() - IV analysis\n")
  cat("9. bayesian_t_test() - Bayesian hypothesis testing\n")
  cat("10. bayesian_changepoint() - Change point analysis\n")
  cat("11. model_comparison() - Compare multiple models\n")
  cat("12. generate_report() - Automated reporting\n")
  cat("\nExample workflows:\n")
  cat("- example_bayesian_workflow()\n")
  cat("- example_timeseries_workflow()\n")
  
  # Run example (uncomment to execute)
  # example_results <- example_bayesian_workflow()
}

cat("Advanced Statistical Modeling template loaded successfully!\n")
cat("Use the functions above for sophisticated statistical analysis.\n") 
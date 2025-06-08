# ===============================================================================
# TIME SERIES ANALYSIS IN R - COMPREHENSIVE TEMPLATE
# ===============================================================================
# A complete guide to time series analysis, forecasting, and financial modeling
# Author: Data Science Toolkit
# Last Updated: 2024
# ===============================================================================

# Required Libraries
suppressMessages({
  library(tidyverse)
  library(lubridate)
  library(forecast)
  library(tseries)
  library(xts)
  library(zoo)
  library(TTR)
  library(quantmod)
  library(rugarch)
  library(vars)
  library(urca)
  library(TSstudio)
  library(prophet)
  library(changepoint)
  library(seasonal)
  library(stl)
  library(bcp)
  library(dynlm)
  library(lmtest)
  library(strucchange)
  library(plotly)
  library(DT)
  library(kableExtra)
})

# ===============================================================================
# 1. DATA PREPARATION AND EXPLORATION
# ===============================================================================

#' Prepare Time Series Data
#' @param data Data frame with date and value columns
#' @param date_col Name of date column
#' @param value_col Name of value column
#' @param frequency Frequency of the time series (daily=365, weekly=52, monthly=12)
prepare_time_series <- function(data, date_col, value_col, frequency = 12) {
  # Convert to proper date format
  data[[date_col]] <- as.Date(data[[date_col]])
  
  # Sort by date
  data <- data[order(data[[date_col]]), ]
  
  # Remove duplicates
  data <- data[!duplicated(data[[date_col]]), ]
  
  # Create time series object
  ts_data <- ts(data[[value_col]], 
                start = c(year(min(data[[date_col]])), 
                         month(min(data[[date_col]]))),
                frequency = frequency)
  
  # Create xts object for advanced analysis
  xts_data <- xts(data[[value_col]], order.by = data[[date_col]])
  
  return(list(
    ts = ts_data,
    xts = xts_data,
    data = data
  ))
}

#' Comprehensive Time Series EDA
#' @param ts_data Time series object
#' @param title Title for plots
perform_ts_eda <- function(ts_data, title = "Time Series Analysis") {
  # Basic statistics
  cat("\n=== TIME SERIES SUMMARY ===\n")
  print(summary(ts_data))
  
  cat("\n=== DESCRIPTIVE STATISTICS ===\n")
  stats <- data.frame(
    Statistic = c("Mean", "Median", "Std Dev", "Min", "Max", "Skewness", "Kurtosis"),
    Value = c(
      mean(ts_data, na.rm = TRUE),
      median(ts_data, na.rm = TRUE),
      sd(ts_data, na.rm = TRUE),
      min(ts_data, na.rm = TRUE),
      max(ts_data, na.rm = TRUE),
      moments::skewness(ts_data, na.rm = TRUE),
      moments::kurtosis(ts_data, na.rm = TRUE)
    )
  )
  print(stats)
  
  # Plots
  par(mfrow = c(2, 2))
  
  # Time series plot
  plot(ts_data, main = paste(title, "- Time Series"), 
       ylab = "Value", xlab = "Time", col = "blue")
  
  # ACF plot
  acf(ts_data, main = paste(title, "- ACF"), na.action = na.pass)
  
  # PACF plot
  pacf(ts_data, main = paste(title, "- PACF"), na.action = na.pass)
  
  # Histogram
  hist(ts_data, main = paste(title, "- Distribution"), 
       xlab = "Value", col = "lightblue", breaks = 30)
  
  par(mfrow = c(1, 1))
  
  return(stats)
}

# ===============================================================================
# 2. STATIONARITY TESTING AND TRANSFORMATION
# ===============================================================================

#' Test for Stationarity
#' @param ts_data Time series object
test_stationarity <- function(ts_data) {
  cat("\n=== STATIONARITY TESTS ===\n")
  
  # Augmented Dickey-Fuller Test
  adf_test <- adf.test(ts_data, alternative = "stationary")
  cat("ADF Test p-value:", adf_test$p.value, "\n")
  
  # KPSS Test
  kpss_test <- kpss.test(ts_data)
  cat("KPSS Test p-value:", kpss_test$p.value, "\n")
  
  # Phillips-Perron Test
  pp_test <- pp.test(ts_data, alternative = "stationary")
  cat("Phillips-Perron Test p-value:", pp_test$p.value, "\n")
  
  # Interpretation
  cat("\n=== INTERPRETATION ===\n")
  if (adf_test$p.value < 0.05) {
    cat("ADF: Series appears to be stationary\n")
  } else {
    cat("ADF: Series appears to be non-stationary\n")
  }
  
  if (kpss_test$p.value > 0.05) {
    cat("KPSS: Series appears to be stationary\n")
  } else {
    cat("KPSS: Series appears to be non-stationary\n")
  }
  
  return(list(
    adf = adf_test,
    kpss = kpss_test,
    pp = pp_test
  ))
}

#' Apply Transformations
#' @param ts_data Time series object
#' @param method Transformation method
apply_transformation <- function(ts_data, method = "log") {
  switch(method,
    "log" = log(ts_data),
    "sqrt" = sqrt(ts_data),
    "diff" = diff(ts_data),
    "box_cox" = forecast::BoxCox(ts_data, lambda = forecast::BoxCox.lambda(ts_data)),
    "seasonal_diff" = diff(ts_data, lag = frequency(ts_data)),
    ts_data
  )
}

# ===============================================================================
# 3. DECOMPOSITION ANALYSIS
# ===============================================================================

#' Comprehensive Time Series Decomposition
#' @param ts_data Time series object
#' @param method Decomposition method
perform_decomposition <- function(ts_data, method = "stl") {
  cat("\n=== TIME SERIES DECOMPOSITION ===\n")
  
  if (method == "stl") {
    # STL Decomposition (Seasonal and Trend decomposition using Loess)
    decomp <- stl(ts_data, s.window = "periodic")
    plot(decomp, main = "STL Decomposition")
    
    # Extract components
    trend <- decomp$time.series[, "trend"]
    seasonal <- decomp$time.series[, "seasonal"]
    remainder <- decomp$time.series[, "remainder"]
    
  } else if (method == "classical") {
    # Classical decomposition
    decomp <- decompose(ts_data)
    plot(decomp, main = "Classical Decomposition")
    
    trend <- decomp$trend
    seasonal <- decomp$seasonal
    remainder <- decomp$random
  }
  
  # Strength of trend and seasonality
  trend_strength <- 1 - var(remainder, na.rm = TRUE) / var(trend + remainder, na.rm = TRUE)
  seasonal_strength <- 1 - var(remainder, na.rm = TRUE) / var(seasonal + remainder, na.rm = TRUE)
  
  cat("Trend Strength:", round(trend_strength, 3), "\n")
  cat("Seasonal Strength:", round(seasonal_strength, 3), "\n")
  
  return(list(
    decomposition = decomp,
    trend_strength = trend_strength,
    seasonal_strength = seasonal_strength
  ))
}

# ===============================================================================
# 4. FORECASTING MODELS
# ===============================================================================

#' ARIMA Model Building and Forecasting
#' @param ts_data Time series object
#' @param h Forecast horizon
#' @param auto_arima Whether to use auto.arima
build_arima_model <- function(ts_data, h = 12, auto_arima = TRUE) {
  cat("\n=== ARIMA MODELING ===\n")
  
  if (auto_arima) {
    # Automatic ARIMA
    model <- auto.arima(ts_data, seasonal = TRUE, stepwise = FALSE, approximation = FALSE)
  } else {
    # Manual ARIMA - you can customize this
    model <- arima(ts_data, order = c(1, 1, 1), seasonal = list(order = c(1, 1, 1)))
  }
  
  cat("Selected Model:", model$arma, "\n")
  print(summary(model))
  
  # Diagnostics
  cat("\n=== MODEL DIAGNOSTICS ===\n")
  
  # Residual tests
  ljung_box <- Box.test(model$residuals, lag = 20, type = "Ljung-Box")
  cat("Ljung-Box Test p-value:", ljung_box$p.value, "\n")
  
  # Plot diagnostics
  par(mfrow = c(2, 2))
  plot(model$residuals, main = "Residuals")
  acf(model$residuals, main = "ACF of Residuals")
  pacf(model$residuals, main = "PACF of Residuals")
  hist(model$residuals, main = "Histogram of Residuals", breaks = 30)
  par(mfrow = c(1, 1))
  
  # Forecast
  forecast_result <- forecast(model, h = h)
  plot(forecast_result, main = "ARIMA Forecast")
  
  return(list(
    model = model,
    forecast = forecast_result,
    diagnostics = ljung_box
  ))
}

#' Exponential Smoothing Models
#' @param ts_data Time series object
#' @param h Forecast horizon
build_ets_model <- function(ts_data, h = 12) {
  cat("\n=== EXPONENTIAL SMOOTHING ===\n")
  
  # Automatic ETS
  model <- ets(ts_data)
  print(summary(model))
  
  # Forecast
  forecast_result <- forecast(model, h = h)
  plot(forecast_result, main = "ETS Forecast")
  
  return(list(
    model = model,
    forecast = forecast_result
  ))
}

#' Prophet Model (Facebook's forecasting tool)
#' @param data Data frame with ds (date) and y (value) columns
#' @param h Forecast horizon in days
build_prophet_model <- function(data, h = 365) {
  cat("\n=== PROPHET MODELING ===\n")
  
  # Prepare data for Prophet
  prophet_data <- data.frame(
    ds = data$date,
    y = data$value
  )
  
  # Build model
  model <- prophet(prophet_data)
  
  # Create future dataframe
  future <- make_future_dataframe(model, periods = h)
  
  # Forecast
  forecast_result <- predict(model, future)
  
  # Plot
  plot(model, forecast_result)
  prophet_plot_components(model, forecast_result)
  
  return(list(
    model = model,
    forecast = forecast_result
  ))
}

# ===============================================================================
# 5. FINANCIAL TIME SERIES ANALYSIS
# ===============================================================================

#' Download and Analyze Stock Data
#' @param symbol Stock symbol
#' @param from Start date
#' @param to End date
analyze_stock_data <- function(symbol, from = "2020-01-01", to = Sys.Date()) {
  cat("\n=== FINANCIAL TIME SERIES ANALYSIS ===\n")
  
  # Download data
  stock_data <- getSymbols(symbol, from = from, to = to, auto.assign = FALSE)
  
  # Calculate returns
  returns <- diff(log(Cl(stock_data)))[-1]
  
  # Basic statistics
  cat("Symbol:", symbol, "\n")
  cat("Mean Return:", mean(returns, na.rm = TRUE), "\n")
  cat("Volatility:", sd(returns, na.rm = TRUE), "\n")
  cat("Sharpe Ratio:", mean(returns, na.rm = TRUE) / sd(returns, na.rm = TRUE), "\n")
  
  # Plot
  par(mfrow = c(2, 2))
  plot(Cl(stock_data), main = paste(symbol, "- Closing Prices"))
  plot(returns, main = paste(symbol, "- Returns"))
  hist(returns, main = "Return Distribution", breaks = 50, col = "lightblue")
  acf(returns^2, main = "ACF of Squared Returns")
  par(mfrow = c(1, 1))
  
  return(list(
    data = stock_data,
    returns = returns
  ))
}

#' GARCH Model for Volatility Modeling
#' @param returns Return series
build_garch_model <- function(returns) {
  cat("\n=== GARCH VOLATILITY MODELING ===\n")
  
  # Specify GARCH model
  spec <- ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
                     mean.model = list(armaOrder = c(1, 1)))
  
  # Fit model
  fit <- ugarchfit(spec, returns)
  
  # Print results
  print(fit)
  
  # Plot
  plot(fit, which = "all")
  
  # Forecast volatility
  forecast_result <- ugarchforecast(fit, n.ahead = 20)
  plot(forecast_result, which = 1)
  
  return(list(
    model = fit,
    forecast = forecast_result
  ))
}

# ===============================================================================
# 6. ANOMALY DETECTION
# ===============================================================================

#' Detect Anomalies in Time Series
#' @param ts_data Time series object
#' @param method Method for anomaly detection
detect_anomalies <- function(ts_data, method = "iqr") {
  cat("\n=== ANOMALY DETECTION ===\n")
  
  if (method == "iqr") {
    # IQR method
    Q1 <- quantile(ts_data, 0.25, na.rm = TRUE)
    Q3 <- quantile(ts_data, 0.75, na.rm = TRUE)
    IQR <- Q3 - Q1
    
    lower_bound <- Q1 - 1.5 * IQR
    upper_bound <- Q3 + 1.5 * IQR
    
    anomalies <- which(ts_data < lower_bound | ts_data > upper_bound)
    
  } else if (method == "zscore") {
    # Z-score method
    z_scores <- abs(scale(ts_data))
    anomalies <- which(z_scores > 3)
    
  } else if (method == "changepoint") {
    # Change point detection
    cpt_result <- cpt.mean(ts_data, method = "PELT")
    anomalies <- cpts(cpt_result)
  }
  
  # Plot
  plot(ts_data, main = "Time Series with Anomalies")
  if (length(anomalies) > 0) {
    points(anomalies, ts_data[anomalies], col = "red", pch = 19, cex = 1.5)
  }
  
  cat("Number of anomalies detected:", length(anomalies), "\n")
  
  return(anomalies)
}

# ===============================================================================
# 7. MULTIVARIATE TIME SERIES (VAR MODELS)
# ===============================================================================

#' Vector Autoregression (VAR) Analysis
#' @param data Multivariate time series data
#' @param p Number of lags
build_var_model <- function(data, p = 2) {
  cat("\n=== VECTOR AUTOREGRESSION (VAR) ===\n")
  
  # Lag selection
  lag_select <- VARselect(data, lag.max = 8)
  print(lag_select)
  
  # Fit VAR model
  var_model <- VAR(data, p = p)
  print(summary(var_model))
  
  # Granger causality tests
  cat("\n=== GRANGER CAUSALITY TESTS ===\n")
  causality_results <- causality(var_model, cause = colnames(data)[1])
  print(causality_results)
  
  # Impulse response functions
  irf_result <- irf(var_model, n.ahead = 10)
  plot(irf_result)
  
  # Forecast error variance decomposition
  fevd_result <- fevd(var_model, n.ahead = 10)
  plot(fevd_result)
  
  return(list(
    model = var_model,
    causality = causality_results,
    irf = irf_result,
    fevd = fevd_result
  ))
}

# ===============================================================================
# 8. MODEL EVALUATION AND COMPARISON
# ===============================================================================

#' Compare Multiple Forecasting Models
#' @param ts_data Time series object
#' @param h Forecast horizon
#' @param split_ratio Train/test split ratio
compare_models <- function(ts_data, h = 12, split_ratio = 0.8) {
  cat("\n=== MODEL COMPARISON ===\n")
  
  # Split data
  n <- length(ts_data)
  train_size <- floor(n * split_ratio)
  train_data <- window(ts_data, end = c(1900 + train_size / frequency(ts_data)))
  test_data <- window(ts_data, start = c(1900 + train_size / frequency(ts_data) + 1))
  
  # Models to compare
  models <- list()
  forecasts <- list()
  
  # ARIMA
  models$arima <- auto.arima(train_data)
  forecasts$arima <- forecast(models$arima, h = length(test_data))
  
  # ETS
  models$ets <- ets(train_data)
  forecasts$ets <- forecast(models$ets, h = length(test_data))
  
  # Naive
  forecasts$naive <- naive(train_data, h = length(test_data))
  
  # Seasonal Naive
  forecasts$snaive <- snaive(train_data, h = length(test_data))
  
  # Calculate accuracy metrics
  accuracy_results <- data.frame(
    Model = character(),
    RMSE = numeric(),
    MAE = numeric(),
    MAPE = numeric(),
    stringsAsFactors = FALSE
  )
  
  for (model_name in names(forecasts)) {
    acc <- accuracy(forecasts[[model_name]], test_data)
    accuracy_results <- rbind(accuracy_results, data.frame(
      Model = model_name,
      RMSE = acc[2, "RMSE"],
      MAE = acc[2, "MAE"],
      MAPE = acc[2, "MAPE"]
    ))
  }
  
  print(accuracy_results)
  
  # Plot comparisons
  plot(ts_data, main = "Model Comparison")
  lines(forecasts$arima$mean, col = "red", lwd = 2)
  lines(forecasts$ets$mean, col = "blue", lwd = 2)
  lines(forecasts$naive$mean, col = "green", lwd = 2)
  legend("topleft", c("ARIMA", "ETS", "Naive"), col = c("red", "blue", "green"), lty = 1)
  
  return(list(
    models = models,
    forecasts = forecasts,
    accuracy = accuracy_results
  ))
}

# ===============================================================================
# 9. PRACTICAL EXAMPLE AND WORKFLOW
# ===============================================================================

#' Complete Time Series Analysis Workflow
#' @param data_path Path to CSV file with time series data
#' @param date_col Name of date column
#' @param value_col Name of value column
complete_ts_workflow <- function(data_path = NULL, date_col = "date", value_col = "value") {
  cat("===============================================================================\n")
  cat("COMPLETE TIME SERIES ANALYSIS WORKFLOW\n")
  cat("===============================================================================\n")
  
  # Load data (or use sample data)
  if (is.null(data_path)) {
    # Generate sample data
    dates <- seq(as.Date("2015-01-01"), as.Date("2023-12-31"), by = "month")
    values <- 100 + cumsum(rnorm(length(dates), 0, 5)) + 
              10 * sin(2 * pi * seq_along(dates) / 12) + 
              rnorm(length(dates), 0, 2)
    
    data <- data.frame(date = dates, value = values)
    cat("Using generated sample data with trend and seasonality\n")
  } else {
    data <- read.csv(data_path)
    cat("Loaded data from:", data_path, "\n")
  }
  
  # 1. Data Preparation
  cat("\n1. PREPARING DATA...\n")
  ts_prepared <- prepare_time_series(data, date_col, value_col, frequency = 12)
  
  # 2. Exploratory Data Analysis
  cat("\n2. EXPLORATORY DATA ANALYSIS...\n")
  eda_results <- perform_ts_eda(ts_prepared$ts, "Sample Time Series")
  
  # 3. Stationarity Testing
  cat("\n3. TESTING STATIONARITY...\n")
  stationarity_results <- test_stationarity(ts_prepared$ts)
  
  # 4. Decomposition
  cat("\n4. TIME SERIES DECOMPOSITION...\n")
  decomp_results <- perform_decomposition(ts_prepared$ts, method = "stl")
  
  # 5. Model Building and Forecasting
  cat("\n5. BUILDING FORECASTING MODELS...\n")
  arima_results <- build_arima_model(ts_prepared$ts, h = 12)
  ets_results <- build_ets_model(ts_prepared$ts, h = 12)
  
  # 6. Model Comparison
  cat("\n6. COMPARING MODELS...\n")
  comparison_results <- compare_models(ts_prepared$ts, h = 12)
  
  # 7. Anomaly Detection
  cat("\n7. DETECTING ANOMALIES...\n")
  anomalies <- detect_anomalies(ts_prepared$ts, method = "iqr")
  
  # Return comprehensive results
  return(list(
    data = data,
    ts_data = ts_prepared,
    eda = eda_results,
    stationarity = stationarity_results,
    decomposition = decomp_results,
    arima = arima_results,
    ets = ets_results,
    comparison = comparison_results,
    anomalies = anomalies
  ))
}

# ===============================================================================
# 10. UTILITY FUNCTIONS
# ===============================================================================

#' Generate Time Series Report
#' @param results Results from complete_ts_workflow
generate_ts_report <- function(results) {
  cat("===============================================================================\n")
  cat("TIME SERIES ANALYSIS REPORT\n")
  cat("===============================================================================\n")
  
  cat("\nDATA SUMMARY:\n")
  cat("- Observations:", nrow(results$data), "\n")
  cat("- Time Range:", min(results$data$date), "to", max(results$data$date), "\n")
  cat("- Frequency:", frequency(results$ts_data$ts), "\n")
  
  cat("\nKEY FINDINGS:\n")
  cat("- Mean Value:", round(mean(results$ts_data$ts, na.rm = TRUE), 2), "\n")
  cat("- Trend Strength:", round(results$decomposition$trend_strength, 3), "\n")
  cat("- Seasonal Strength:", round(results$decomposition$seasonal_strength, 3), "\n")
  cat("- Anomalies Detected:", length(results$anomalies), "\n")
  
  cat("\nBEST MODEL:\n")
  best_model <- results$comparison$accuracy[which.min(results$comparison$accuracy$RMSE), ]
  cat("- Model:", best_model$Model, "\n")
  cat("- RMSE:", round(best_model$RMSE, 3), "\n")
  cat("- MAE:", round(best_model$MAE, 3), "\n")
  cat("- MAPE:", round(best_model$MAPE, 3), "%\n")
  
  # Create summary table
  summary_table <- data.frame(
    Metric = c("Observations", "Mean", "Std Dev", "Min", "Max", "Trend Strength", "Seasonal Strength"),
    Value = c(
      nrow(results$data),
      round(mean(results$ts_data$ts, na.rm = TRUE), 2),
      round(sd(results$ts_data$ts, na.rm = TRUE), 2),
      round(min(results$ts_data$ts, na.rm = TRUE), 2),
      round(max(results$ts_data$ts, na.rm = TRUE), 2),
      round(results$decomposition$trend_strength, 3),
      round(results$decomposition$seasonal_strength, 3)
    )
  )
  
  print(kable(summary_table, caption = "Time Series Summary"))
  
  return(summary_table)
}

#' Save Time Series Results
#' @param results Results from analysis
#' @param output_dir Output directory
save_ts_results <- function(results, output_dir = "time_series_output") {
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }
  
  # Save forecasts
  write.csv(results$arima$forecast$mean, file.path(output_dir, "arima_forecast.csv"))
  write.csv(results$ets$forecast$mean, file.path(output_dir, "ets_forecast.csv"))
  
  # Save model comparison
  write.csv(results$comparison$accuracy, file.path(output_dir, "model_comparison.csv"))
  
  # Save anomalies
  if (length(results$anomalies) > 0) {
    anomaly_data <- data.frame(
      Index = results$anomalies,
      Date = results$data$date[results$anomalies],
      Value = results$ts_data$ts[results$anomalies]
    )
    write.csv(anomaly_data, file.path(output_dir, "anomalies.csv"))
  }
  
  cat("Results saved to:", output_dir, "\n")
}

# ===============================================================================
# EXAMPLE USAGE
# ===============================================================================

# Run complete analysis
# results <- complete_ts_workflow()
# 
# # Generate report
# generate_ts_report(results)
# 
# # Save results
# save_ts_results(results)
# 
# # For stock analysis:
# stock_analysis <- analyze_stock_data("AAPL", from = "2020-01-01")
# garch_results <- build_garch_model(stock_analysis$returns)

cat("Time Series Analysis Template Loaded Successfully!\n")
cat("Key Functions Available:\n")
cat("- complete_ts_workflow(): Run full analysis\n")
cat("- analyze_stock_data(): Financial time series analysis\n")
cat("- build_arima_model(): ARIMA modeling\n")
cat("- build_ets_model(): Exponential smoothing\n")
cat("- detect_anomalies(): Anomaly detection\n")
cat("- compare_models(): Model comparison\n") 
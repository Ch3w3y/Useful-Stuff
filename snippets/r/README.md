# R Code Snippets Collection

## Overview

Comprehensive collection of R code snippets for data manipulation, statistical analysis, visualization, and specialized analytical functions.

## Table of Contents

- [Data Manipulation](#data-manipulation)
- [Statistical Analysis](#statistical-analysis)
- [Data Visualization](#data-visualization)
- [String Operations](#string-operations)
- [Date and Time](#date-and-time)
- [File Operations](#file-operations)
- [Advanced Analytics](#advanced-analytics)
- [Utility Functions](#utility-functions)

## Data Manipulation

### Data Cleaning and Transformation

```r
# Load essential libraries
library(tidyverse)
library(data.table)
library(lubridate)
library(stringr)
library(janitor)

# Quick data overview
quick_summary <- function(df) {
  cat("Dataset Overview\n")
  cat("================\n")
  cat("Dimensions:", dim(df), "\n")
  cat("Column names:", paste(names(df), collapse = ", "), "\n\n")
  
  cat("Data Types:\n")
  print(sapply(df, class))
  cat("\n")
  
  cat("Missing Values:\n")
  missing_counts <- sapply(df, function(x) sum(is.na(x)))
  print(missing_counts[missing_counts > 0])
  cat("\n")
  
  cat("Summary Statistics:\n")
  print(summary(df))
}

# Clean column names
clean_names_custom <- function(df) {
  df %>%
    janitor::clean_names() %>%
    rename_with(~ str_replace_all(.x, "_+", "_")) %>%
    rename_with(~ str_remove(.x, "_$"))
}

# Remove duplicate rows with reporting
remove_duplicates <- function(df, key_cols = NULL) {
  initial_rows <- nrow(df)
  
  if (is.null(key_cols)) {
    df_clean <- df %>% distinct()
  } else {
    df_clean <- df %>% distinct(across(all_of(key_cols)), .keep_all = TRUE)
  }
  
  removed_rows <- initial_rows - nrow(df_clean)
  cat("Removed", removed_rows, "duplicate rows\n")
  cat("Remaining rows:", nrow(df_clean), "\n")
  
  return(df_clean)
}

# Handle missing values with multiple strategies
handle_missing <- function(df, strategy = "remove", threshold = 0.5) {
  missing_summary <- df %>%
    summarise_all(~ sum(is.na(.)) / length(.)) %>%
    gather(variable, missing_prop)
  
  cat("Missing value proportions:\n")
  print(missing_summary %>% filter(missing_prop > 0))
  
  if (strategy == "remove") {
    # Remove columns with high missing proportion
    cols_to_keep <- missing_summary %>%
      filter(missing_prop <= threshold) %>%
      pull(variable)
    
    df_clean <- df %>% select(all_of(cols_to_keep))
    cat("Removed columns with >", threshold * 100, "% missing values\n")
    
  } else if (strategy == "impute") {
    # Simple imputation
    df_clean <- df %>%
      mutate_if(is.numeric, ~ ifelse(is.na(.), median(., na.rm = TRUE), .)) %>%
      mutate_if(is.character, ~ ifelse(is.na(.), "Unknown", .)) %>%
      mutate_if(is.factor, ~ fct_explicit_na(., na_level = "Unknown"))
    
  } else if (strategy == "forward_fill") {
    # Forward fill for time series
    df_clean <- df %>%
      arrange_all() %>%
      fill(everything(), .direction = "down")
  }
  
  return(df_clean)
}

# Outlier detection and treatment
detect_outliers <- function(df, method = "iqr", k = 1.5) {
  numeric_cols <- df %>% select_if(is.numeric) %>% names()
  
  outlier_summary <- map_dfr(numeric_cols, function(col) {
    x <- df[[col]]
    
    if (method == "iqr") {
      Q1 <- quantile(x, 0.25, na.rm = TRUE)
      Q3 <- quantile(x, 0.75, na.rm = TRUE)
      IQR <- Q3 - Q1
      
      lower_bound <- Q1 - k * IQR
      upper_bound <- Q3 + k * IQR
      
      outliers <- which(x < lower_bound | x > upper_bound)
      
    } else if (method == "zscore") {
      z_scores <- abs(scale(x))
      outliers <- which(z_scores > k)
    }
    
    tibble(
      variable = col,
      n_outliers = length(outliers),
      outlier_prop = length(outliers) / length(x),
      outlier_indices = list(outliers)
    )
  })
  
  return(outlier_summary)
}

# Data type conversion utilities
convert_types <- function(df, type_map) {
  for (col in names(type_map)) {
    target_type <- type_map[[col]]
    
    if (target_type == "numeric") {
      df[[col]] <- as.numeric(df[[col]])
    } else if (target_type == "factor") {
      df[[col]] <- as.factor(df[[col]])
    } else if (target_type == "date") {
      df[[col]] <- as.Date(df[[col]])
    } else if (target_type == "datetime") {
      df[[col]] <- as.POSIXct(df[[col]])
    }
  }
  
  return(df)
}

# Create dummy variables
create_dummies <- function(df, cols = NULL, drop_first = TRUE) {
  if (is.null(cols)) {
    cols <- df %>% select_if(is.factor) %>% names()
  }
  
  dummy_df <- df
  
  for (col in cols) {
    # Create dummy variables
    dummies <- model.matrix(~ . - 1, data = df[col])
    
    if (drop_first && ncol(dummies) > 1) {
      dummies <- dummies[, -1, drop = FALSE]
    }
    
    # Add to dataframe
    dummy_names <- paste0(col, "_", colnames(dummies))
    dummy_df <- dummy_df %>%
      bind_cols(as_tibble(dummies, .name_repair = ~ dummy_names)) %>%
      select(-all_of(col))
  }
  
  return(dummy_df)
}
```

### Advanced Data Manipulation

```r
# Sliding window operations
sliding_window <- function(x, window_size, func = mean, ...) {
  n <- length(x)
  if (window_size > n) stop("Window size larger than data length")
  
  result <- numeric(n - window_size + 1)
  
  for (i in 1:(n - window_size + 1)) {
    window_data <- x[i:(i + window_size - 1)]
    result[i] <- func(window_data, ...)
  }
  
  return(result)
}

# Group-wise operations with custom functions
group_apply <- function(df, group_vars, func, ...) {
  df %>%
    group_by(across(all_of(group_vars))) %>%
    group_modify(~ func(.x, ...)) %>%
    ungroup()
}

# Pivot operations with multiple value columns
pivot_multiple <- function(df, id_cols, names_from, values_from) {
  df %>%
    pivot_longer(
      cols = all_of(values_from),
      names_to = "metric",
      values_to = "value"
    ) %>%
    unite("name_metric", all_of(names_from), metric, sep = "_") %>%
    pivot_wider(
      id_cols = all_of(id_cols),
      names_from = name_metric,
      values_from = value
    )
}

# Conditional mutations
conditional_mutate <- function(df, condition, mutations) {
  df %>%
    mutate(
      across(
        all_of(names(mutations)),
        ~ ifelse({{ condition }}, mutations[[cur_column()]], .x)
      )
    )
}

# Lag and lead operations
create_lags <- function(df, vars, lags = 1:3, group_var = NULL) {
  if (!is.null(group_var)) {
    df <- df %>% group_by(across(all_of(group_var)))
  }
  
  for (var in vars) {
    for (lag_n in lags) {
      lag_name <- paste0(var, "_lag", lag_n)
      lead_name <- paste0(var, "_lead", lag_n)
      
      df <- df %>%
        mutate(
          !!lag_name := lag(.data[[var]], lag_n),
          !!lead_name := lead(.data[[var]], lag_n)
        )
    }
  }
  
  if (!is.null(group_var)) {
    df <- df %>% ungroup()
  }
  
  return(df)
}
```

## Statistical Analysis

### Descriptive Statistics

```r
# Comprehensive descriptive statistics
describe_data <- function(df, by_group = NULL) {
  numeric_vars <- df %>% select_if(is.numeric) %>% names()
  
  if (!is.null(by_group)) {
    df <- df %>% group_by(across(all_of(by_group)))
  }
  
  stats_summary <- df %>%
    summarise(
      across(
        all_of(numeric_vars),
        list(
          n = ~ sum(!is.na(.)),
          mean = ~ mean(., na.rm = TRUE),
          median = ~ median(., na.rm = TRUE),
          sd = ~ sd(., na.rm = TRUE),
          min = ~ min(., na.rm = TRUE),
          max = ~ max(., na.rm = TRUE),
          q25 = ~ quantile(., 0.25, na.rm = TRUE),
          q75 = ~ quantile(., 0.75, na.rm = TRUE),
          skewness = ~ moments::skewness(., na.rm = TRUE),
          kurtosis = ~ moments::kurtosis(., na.rm = TRUE)
        ),
        .names = "{.col}_{.fn}"
      ),
      .groups = "drop"
    )
  
  return(stats_summary)
}

# Correlation analysis
correlation_analysis <- function(df, method = "pearson", threshold = 0.7) {
  numeric_df <- df %>% select_if(is.numeric)
  
  # Correlation matrix
  cor_matrix <- cor(numeric_df, use = "complete.obs", method = method)
  
  # High correlations
  high_cor <- which(abs(cor_matrix) > threshold & cor_matrix != 1, arr.ind = TRUE)
  high_cor_pairs <- data.frame(
    var1 = rownames(cor_matrix)[high_cor[, 1]],
    var2 = colnames(cor_matrix)[high_cor[, 2]],
    correlation = cor_matrix[high_cor]
  ) %>%
    filter(var1 != var2) %>%
    arrange(desc(abs(correlation)))
  
  list(
    correlation_matrix = cor_matrix,
    high_correlations = high_cor_pairs
  )
}

# Distribution testing
test_normality <- function(x, tests = c("shapiro", "ks", "ad")) {
  results <- list()
  
  if ("shapiro" %in% tests && length(x) <= 5000) {
    results$shapiro <- shapiro.test(x)
  }
  
  if ("ks" %in% tests) {
    results$ks <- ks.test(x, "pnorm", mean(x, na.rm = TRUE), sd(x, na.rm = TRUE))
  }
  
  if ("ad" %in% tests) {
    results$anderson_darling <- nortest::ad.test(x)
  }
  
  return(results)
}

# Hypothesis testing utilities
compare_groups <- function(df, group_var, value_var, test = "auto") {
  groups <- df[[group_var]] %>% unique() %>% na.omit()
  n_groups <- length(groups)
  
  if (n_groups == 2) {
    group1 <- df %>% filter(.data[[group_var]] == groups[1]) %>% pull(.data[[value_var]])
    group2 <- df %>% filter(.data[[group_var]] == groups[2]) %>% pull(.data[[value_var]])
    
    if (test == "auto") {
      # Check normality
      norm_test1 <- shapiro.test(sample(group1, min(length(group1), 5000)))
      norm_test2 <- shapiro.test(sample(group2, min(length(group2), 5000)))
      
      if (norm_test1$p.value > 0.05 && norm_test2$p.value > 0.05) {
        result <- t.test(group1, group2)
        test_used <- "t-test"
      } else {
        result <- wilcox.test(group1, group2)
        test_used <- "Mann-Whitney U"
      }
    }
    
  } else if (n_groups > 2) {
    if (test == "auto") {
      # Check normality for each group
      norm_tests <- map_lgl(groups, function(g) {
        group_data <- df %>% filter(.data[[group_var]] == g) %>% pull(.data[[value_var]])
        if (length(group_data) > 3) {
          shapiro.test(sample(group_data, min(length(group_data), 5000)))$p.value > 0.05
        } else {
          TRUE
        }
      })
      
      if (all(norm_tests)) {
        result <- aov(as.formula(paste(value_var, "~", group_var)), data = df)
        test_used <- "ANOVA"
      } else {
        result <- kruskal.test(as.formula(paste(value_var, "~", group_var)), data = df)
        test_used <- "Kruskal-Wallis"
      }
    }
  }
  
  list(
    test_used = test_used,
    result = result,
    summary = summary(result)
  )
}
```

### Advanced Statistical Functions

```r
# Bootstrap confidence intervals
bootstrap_ci <- function(x, func = mean, n_bootstrap = 1000, conf_level = 0.95) {
  bootstrap_stats <- replicate(n_bootstrap, {
    sample_data <- sample(x, length(x), replace = TRUE)
    func(sample_data)
  })
  
  alpha <- 1 - conf_level
  ci_lower <- quantile(bootstrap_stats, alpha / 2)
  ci_upper <- quantile(bootstrap_stats, 1 - alpha / 2)
  
  list(
    original_stat = func(x),
    bootstrap_mean = mean(bootstrap_stats),
    ci_lower = ci_lower,
    ci_upper = ci_upper,
    bootstrap_stats = bootstrap_stats
  )
}

# Effect size calculations
effect_size <- function(group1, group2, type = "cohen_d") {
  if (type == "cohen_d") {
    pooled_sd <- sqrt(((length(group1) - 1) * var(group1) + 
                      (length(group2) - 1) * var(group2)) / 
                     (length(group1) + length(group2) - 2))
    
    d <- (mean(group1) - mean(group2)) / pooled_sd
    
    # Interpretation
    interpretation <- case_when(
      abs(d) < 0.2 ~ "negligible",
      abs(d) < 0.5 ~ "small",
      abs(d) < 0.8 ~ "medium",
      TRUE ~ "large"
    )
    
    return(list(effect_size = d, interpretation = interpretation))
  }
}

# Power analysis
power_analysis <- function(effect_size, alpha = 0.05, power = 0.8, type = "two.sample") {
  if (type == "two.sample") {
    result <- pwr::pwr.t.test(d = effect_size, sig.level = alpha, power = power)
  } else if (type == "one.sample") {
    result <- pwr::pwr.t.test(d = effect_size, sig.level = alpha, power = power, type = "one.sample")
  }
  
  return(result)
}
```

## Data Visualization

### Quick Plotting Functions

```r
# Quick exploratory plots
quick_plot <- function(df, x, y = NULL, type = "auto") {
  if (is.null(y)) {
    # Univariate plots
    if (is.numeric(df[[x]])) {
      if (type == "auto" || type == "histogram") {
        p <- ggplot(df, aes_string(x = x)) +
          geom_histogram(bins = 30, fill = "steelblue", alpha = 0.7) +
          theme_minimal() +
          labs(title = paste("Distribution of", x))
      }
    } else {
      if (type == "auto" || type == "bar") {
        p <- ggplot(df, aes_string(x = x)) +
          geom_bar(fill = "steelblue", alpha = 0.7) +
          theme_minimal() +
          labs(title = paste("Frequency of", x)) +
          theme(axis.text.x = element_text(angle = 45, hjust = 1))
      }
    }
  } else {
    # Bivariate plots
    if (is.numeric(df[[x]]) && is.numeric(df[[y]])) {
      p <- ggplot(df, aes_string(x = x, y = y)) +
        geom_point(alpha = 0.6) +
        geom_smooth(method = "lm", se = TRUE) +
        theme_minimal() +
        labs(title = paste(y, "vs", x))
    } else if (is.factor(df[[x]]) || is.character(df[[x]])) {
      p <- ggplot(df, aes_string(x = x, y = y)) +
        geom_boxplot(fill = "steelblue", alpha = 0.7) +
        theme_minimal() +
        labs(title = paste(y, "by", x)) +
        theme(axis.text.x = element_text(angle = 45, hjust = 1))
    }
  }
  
  return(p)
}

# Correlation heatmap
plot_correlation <- function(df, method = "pearson") {
  numeric_df <- df %>% select_if(is.numeric)
  cor_matrix <- cor(numeric_df, use = "complete.obs", method = method)
  
  cor_df <- cor_matrix %>%
    as.data.frame() %>%
    rownames_to_column("var1") %>%
    pivot_longer(-var1, names_to = "var2", values_to = "correlation")
  
  ggplot(cor_df, aes(x = var1, y = var2, fill = correlation)) +
    geom_tile() +
    scale_fill_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    labs(title = paste("Correlation Matrix (", str_to_title(method), ")", sep = ""))
}

# Missing value visualization
plot_missing <- function(df) {
  missing_df <- df %>%
    summarise_all(~ sum(is.na(.))) %>%
    pivot_longer(everything(), names_to = "variable", values_to = "missing_count") %>%
    mutate(missing_prop = missing_count / nrow(df)) %>%
    arrange(desc(missing_count))
  
  ggplot(missing_df, aes(x = reorder(variable, missing_count), y = missing_count)) +
    geom_col(fill = "coral") +
    coord_flip() +
    theme_minimal() +
    labs(
      title = "Missing Values by Variable",
      x = "Variable",
      y = "Number of Missing Values"
    )
}

# Distribution comparison
plot_distributions <- function(df, vars, group_var = NULL) {
  plot_df <- df %>%
    select(all_of(c(vars, group_var))) %>%
    pivot_longer(all_of(vars), names_to = "variable", values_to = "value")
  
  if (!is.null(group_var)) {
    p <- ggplot(plot_df, aes(x = value, fill = .data[[group_var]])) +
      geom_density(alpha = 0.6) +
      facet_wrap(~ variable, scales = "free")
  } else {
    p <- ggplot(plot_df, aes(x = value)) +
      geom_density(fill = "steelblue", alpha = 0.6) +
      facet_wrap(~ variable, scales = "free")
  }
  
  p + theme_minimal() + labs(title = "Distribution Comparison")
}
```

---

*This comprehensive R snippets collection provides practical, reusable code for data analysis, statistical modeling, and visualization workflows.* 
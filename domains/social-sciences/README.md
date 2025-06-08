# Social Sciences Research & Analysis

## Overview

Comprehensive guide for social sciences research covering methodology, statistical analysis, data collection, survey design, and computational social science techniques.

## Table of Contents

- [Research Methodology](#research-methodology)
- [Survey Design & Data Collection](#survey-design--data-collection)
- [Statistical Analysis](#statistical-analysis)
- [Qualitative Research](#qualitative-research)
- [Experimental Design](#experimental-design)
- [Computational Social Science](#computational-social-science)
- [Ethics & Best Practices](#ethics--best-practices)
- [Tools & Software](#tools--software)

## Research Methodology

### Research Design Framework

```r
# Load essential libraries for social science research
library(tidyverse)
library(psych)
library(lavaan)
library(survey)
library(ggplot2)
library(corrplot)
library(car)
library(lme4)
library(broom)
library(knitr)
library(stargazer)

# Research design planning framework
research_design_template <- function() {
  design <- list(
    research_question = "",
    hypotheses = c(),
    variables = list(
      dependent = c(),
      independent = c(),
      control = c(),
      mediating = c(),
      moderating = c()
    ),
    methodology = list(
      approach = "", # quantitative, qualitative, mixed-methods
      design_type = "", # experimental, quasi-experimental, observational
      sampling_method = "",
      data_collection = "",
      analysis_plan = ""
    ),
    ethical_considerations = c(),
    limitations = c(),
    timeline = list(),
    resources_needed = c()
  )
  
  return(design)
}

# Power analysis for sample size determination
calculate_sample_size <- function(effect_size, alpha = 0.05, power = 0.8, test_type = "t.test") {
  library(pwr)
  
  if (test_type == "t.test") {
    result <- pwr.t.test(d = effect_size, sig.level = alpha, power = power)
  } else if (test_type == "anova") {
    result <- pwr.anova.test(f = effect_size, k = 3, sig.level = alpha, power = power)
  } else if (test_type == "correlation") {
    result <- pwr.r.test(r = effect_size, sig.level = alpha, power = power)
  } else if (test_type == "chi.square") {
    result <- pwr.chisq.test(w = effect_size, df = 1, sig.level = alpha, power = power)
  }
  
  cat("Power Analysis Results:\n")
  cat("Effect size:", effect_size, "\n")
  cat("Alpha level:", alpha, "\n")
  cat("Power:", power, "\n")
  cat("Required sample size:", ceiling(result$n), "\n")
  
  return(result)
}

# Literature review synthesis framework
literature_synthesis <- function(studies_df) {
  # studies_df should contain: study_id, author, year, sample_size, effect_size, methodology
  
  synthesis <- list(
    total_studies = nrow(studies_df),
    date_range = paste(min(studies_df$year), "-", max(studies_df$year)),
    total_participants = sum(studies_df$sample_size, na.rm = TRUE),
    methodologies = table(studies_df$methodology),
    effect_sizes = list(
      mean = mean(studies_df$effect_size, na.rm = TRUE),
      median = median(studies_df$effect_size, na.rm = TRUE),
      range = range(studies_df$effect_size, na.rm = TRUE),
      sd = sd(studies_df$effect_size, na.rm = TRUE)
    )
  )
  
  # Meta-analysis if effect sizes available
  if (!all(is.na(studies_df$effect_size))) {
    library(meta)
    meta_result <- metamean(
      n = studies_df$sample_size,
      mean = studies_df$effect_size,
      sd = studies_df$effect_size_sd,
      studlab = paste(studies_df$author, studies_df$year)
    )
    synthesis$meta_analysis <- meta_result
  }
  
  return(synthesis)
}
```

### Theoretical Framework Development

```r
# Conceptual model specification
specify_conceptual_model <- function() {
  model_components <- list(
    constructs = list(
      name = c(),
      definition = c(),
      measurement_type = c(), # latent, observed, composite
      indicators = list()
    ),
    relationships = list(
      from = c(),
      to = c(),
      type = c(), # direct, mediated, moderated
      direction = c(), # positive, negative, bidirectional
      theoretical_basis = c()
    ),
    boundary_conditions = c(),
    assumptions = c()
  )
  
  return(model_components)
}

# Structural equation modeling setup
setup_sem_model <- function(model_syntax, data) {
  # Fit the model
  fit <- sem(model_syntax, data = data, estimator = "MLR")
  
  # Model fit indices
  fit_indices <- fitMeasures(fit, c("chisq", "df", "pvalue", "cfi", "tli", 
                                   "rmsea", "rmsea.ci.lower", "rmsea.ci.upper", 
                                   "srmr", "aic", "bic"))
  
  # Parameter estimates
  parameters <- parameterEstimates(fit, standardized = TRUE)
  
  # Model summary
  model_summary <- list(
    fit_indices = fit_indices,
    parameters = parameters,
    r_squared = inspect(fit, "r2"),
    modification_indices = modificationIndices(fit, sort = TRUE, maximum.number = 10)
  )
  
  return(list(fit = fit, summary = model_summary))
}

# Theory testing framework
test_theory <- function(data, hypotheses_list) {
  results <- list()
  
  for (i in seq_along(hypotheses_list)) {
    hypothesis <- hypotheses_list[[i]]
    
    if (hypothesis$type == "correlation") {
      test_result <- cor.test(data[[hypothesis$var1]], data[[hypothesis$var2]], 
                             method = hypothesis$method)
    } else if (hypothesis$type == "regression") {
      model <- lm(as.formula(hypothesis$formula), data = data)
      test_result <- summary(model)
    } else if (hypothesis$type == "t.test") {
      test_result <- t.test(data[[hypothesis$var1]], data[[hypothesis$var2]])
    } else if (hypothesis$type == "anova") {
      model <- aov(as.formula(hypothesis$formula), data = data)
      test_result <- summary(model)
    }
    
    results[[paste0("H", i)]] <- list(
      hypothesis = hypothesis$statement,
      test_type = hypothesis$type,
      result = test_result,
      supported = ifelse(test_result$p.value < 0.05, "Yes", "No")
    )
  }
  
  return(results)
}
```

## Survey Design & Data Collection

### Survey Construction

```r
# Survey design framework
design_survey <- function() {
  survey_structure <- list(
    objectives = c(),
    target_population = "",
    sampling_frame = "",
    survey_mode = "", # online, phone, face-to-face, mail
    question_types = list(
      demographic = c(),
      substantive = c(),
      behavioral = c(),
      attitudinal = c()
    ),
    scale_types = list(
      likert = list(points = 5, anchors = c()),
      semantic_differential = list(),
      ranking = list(),
      open_ended = c()
    ),
    survey_flow = c(),
    pretesting_plan = "",
    pilot_study_size = 0
  )
  
  return(survey_structure)
}

# Question quality assessment
assess_question_quality <- function(question_text, response_data = NULL) {
  assessment <- list(
    question = question_text,
    clarity_score = NA, # Would need human rating
    bias_indicators = c(),
    response_quality = list()
  )
  
  # Check for common issues
  if (grepl("and|or", question_text, ignore.case = TRUE)) {
    assessment$bias_indicators <- c(assessment$bias_indicators, "Double-barreled question")
  }
  
  if (grepl("don't you|shouldn't you|isn't it", question_text, ignore.case = TRUE)) {
    assessment$bias_indicators <- c(assessment$bias_indicators, "Leading question")
  }
  
  if (nchar(question_text) > 100) {
    assessment$bias_indicators <- c(assessment$bias_indicators, "Question too long")
  }
  
  # Response quality analysis if data provided
  if (!is.null(response_data)) {
    assessment$response_quality <- list(
      response_rate = sum(!is.na(response_data)) / length(response_data),
      variance = var(response_data, na.rm = TRUE),
      skewness = psych::skew(response_data, na.rm = TRUE),
      kurtosis = psych::kurtosi(response_data, na.rm = TRUE)
    )
  }
  
  return(assessment)
}

# Scale reliability analysis
analyze_scale_reliability <- function(data, scale_items) {
  scale_data <- data[scale_items]
  
  # Cronbach's alpha
  alpha_result <- psych::alpha(scale_data)
  
  # Item-total correlations
  item_total_cor <- cor(scale_data, rowSums(scale_data, na.rm = TRUE), use = "complete.obs")
  
  # Factor analysis
  fa_result <- psych::fa(scale_data, nfactors = 1, rotate = "none")
  
  reliability_report <- list(
    cronbach_alpha = alpha_result$total$std.alpha,
    item_statistics = alpha_result$item.stats,
    item_total_correlations = item_total_cor,
    factor_loadings = fa_result$loadings,
    recommendations = c()
  )
  
  # Generate recommendations
  if (reliability_report$cronbach_alpha < 0.7) {
    reliability_report$recommendations <- c(reliability_report$recommendations, 
                                          "Consider adding more items or revising existing items")
  }
  
  low_loading_items <- names(which(abs(fa_result$loadings) < 0.4))
  if (length(low_loading_items) > 0) {
    reliability_report$recommendations <- c(reliability_report$recommendations,
                                          paste("Consider removing items with low loadings:", 
                                                paste(low_loading_items, collapse = ", ")))
  }
  
  return(reliability_report)
}
```

### Data Collection Management

```r
# Survey administration tracking
track_survey_progress <- function(responses_df, target_n) {
  progress <- list(
    current_n = nrow(responses_df),
    target_n = target_n,
    completion_rate = nrow(responses_df) / target_n,
    response_rate_by_day = table(as.Date(responses_df$submission_date)),
    demographic_breakdown = list(),
    data_quality_flags = c()
  )
  
  # Demographic representativeness
  if ("age" %in% names(responses_df)) {
    progress$demographic_breakdown$age <- summary(responses_df$age)
  }
  
  if ("gender" %in% names(responses_df)) {
    progress$demographic_breakdown$gender <- table(responses_df$gender)
  }
  
  # Data quality checks
  if (any(responses_df$completion_time < 60, na.rm = TRUE)) {
    progress$data_quality_flags <- c(progress$data_quality_flags, "Suspiciously fast responses detected")
  }
  
  # Straight-lining detection
  likert_cols <- grep("likert|scale", names(responses_df), value = TRUE)
  if (length(likert_cols) > 3) {
    straight_line_count <- apply(responses_df[likert_cols], 1, function(x) length(unique(x[!is.na(x)])))
    if (any(straight_line_count == 1, na.rm = TRUE)) {
      progress$data_quality_flags <- c(progress$data_quality_flags, "Straight-lining responses detected")
    }
  }
  
  return(progress)
}

# Missing data analysis
analyze_missing_data <- function(data) {
  missing_analysis <- list(
    missing_counts = sapply(data, function(x) sum(is.na(x))),
    missing_proportions = sapply(data, function(x) mean(is.na(x))),
    missing_patterns = VIM::aggr(data, plot = FALSE)$combinations,
    missingness_tests = list()
  )
  
  # Little's MCAR test
  if (require(naniar)) {
    missing_analysis$mcar_test <- naniar::mcar_test(data)
  }
  
  # Missing data visualization
  missing_plot <- VIM::aggr(data, col = c('navyblue', 'red'), 
                           numbers = TRUE, sortVars = TRUE)
  
  missing_analysis$visualization <- missing_plot
  
  return(missing_analysis)
}

# Data cleaning pipeline
clean_survey_data <- function(raw_data, cleaning_rules = list()) {
  cleaned_data <- raw_data
  cleaning_log <- list()
  
  # Remove test responses
  if ("test_response" %in% names(cleaned_data)) {
    test_rows <- which(cleaned_data$test_response == TRUE)
    if (length(test_rows) > 0) {
      cleaned_data <- cleaned_data[-test_rows, ]
      cleaning_log$test_responses_removed <- length(test_rows)
    }
  }
  
  # Remove incomplete responses
  if ("completion_status" %in% names(cleaned_data)) {
    incomplete_rows <- which(cleaned_data$completion_status != "complete")
    if (length(incomplete_rows) > 0) {
      cleaned_data <- cleaned_data[-incomplete_rows, ]
      cleaning_log$incomplete_responses_removed <- length(incomplete_rows)
    }
  }
  
  # Remove speeders (if completion time available)
  if ("completion_time" %in% names(cleaned_data)) {
    speeder_threshold <- cleaning_rules$min_completion_time %||% 120 # 2 minutes default
    speeders <- which(cleaned_data$completion_time < speeder_threshold)
    if (length(speeders) > 0) {
      cleaned_data <- cleaned_data[-speeders, ]
      cleaning_log$speeders_removed <- length(speeders)
    }
  }
  
  # Outlier detection and handling
  numeric_vars <- sapply(cleaned_data, is.numeric)
  for (var in names(cleaned_data)[numeric_vars]) {
    if (var %in% names(cleaning_rules$outlier_handling)) {
      method <- cleaning_rules$outlier_handling[[var]]
      
      if (method == "iqr") {
        Q1 <- quantile(cleaned_data[[var]], 0.25, na.rm = TRUE)
        Q3 <- quantile(cleaned_data[[var]], 0.75, na.rm = TRUE)
        IQR <- Q3 - Q1
        
        outliers <- which(cleaned_data[[var]] < (Q1 - 1.5 * IQR) | 
                         cleaned_data[[var]] > (Q3 + 1.5 * IQR))
        
        if (length(outliers) > 0) {
          cleaned_data[[var]][outliers] <- NA
          cleaning_log[[paste0(var, "_outliers_removed")]] <- length(outliers)
        }
      }
    }
  }
  
  return(list(data = cleaned_data, log = cleaning_log))
}
```

## Statistical Analysis

### Descriptive Statistics

```r
# Comprehensive descriptive analysis
descriptive_analysis <- function(data, group_var = NULL) {
  numeric_vars <- sapply(data, is.numeric)
  categorical_vars <- sapply(data, function(x) is.factor(x) || is.character(x))
  
  results <- list(
    sample_size = nrow(data),
    numeric_summary = list(),
    categorical_summary = list(),
    correlations = list(),
    group_comparisons = list()
  )
  
  # Numeric variables
  if (any(numeric_vars)) {
    numeric_data <- data[numeric_vars]
    
    results$numeric_summary <- data.frame(
      variable = names(numeric_data),
      n = sapply(numeric_data, function(x) sum(!is.na(x))),
      mean = sapply(numeric_data, mean, na.rm = TRUE),
      sd = sapply(numeric_data, sd, na.rm = TRUE),
      median = sapply(numeric_data, median, na.rm = TRUE),
      min = sapply(numeric_data, min, na.rm = TRUE),
      max = sapply(numeric_data, max, na.rm = TRUE),
      skewness = sapply(numeric_data, psych::skew, na.rm = TRUE),
      kurtosis = sapply(numeric_data, psych::kurtosi, na.rm = TRUE)
    )
    
    # Correlation matrix
    results$correlations$matrix <- cor(numeric_data, use = "complete.obs")
    results$correlations$significance <- psych::corr.test(numeric_data)$p
  }
  
  # Categorical variables
  if (any(categorical_vars)) {
    cat_data <- data[categorical_vars]
    results$categorical_summary <- lapply(cat_data, function(x) {
      freq_table <- table(x, useNA = "ifany")
      prop_table <- prop.table(freq_table)
      
      list(
        frequencies = freq_table,
        proportions = prop_table,
        mode = names(freq_table)[which.max(freq_table)]
      )
    })
  }
  
  # Group comparisons
  if (!is.null(group_var) && group_var %in% names(data)) {
    for (var in names(data)[numeric_vars]) {
      if (var != group_var) {
        group_comparison <- data %>%
          group_by(.data[[group_var]]) %>%
          summarise(
            n = sum(!is.na(.data[[var]])),
            mean = mean(.data[[var]], na.rm = TRUE),
            sd = sd(.data[[var]], na.rm = TRUE),
            median = median(.data[[var]], na.rm = TRUE),
            .groups = "drop"
          )
        
        # Statistical test
        if (length(unique(data[[group_var]])) == 2) {
          test_result <- t.test(data[[var]] ~ data[[group_var]])
        } else {
          test_result <- aov(data[[var]] ~ data[[group_var]])
        }
        
        results$group_comparisons[[var]] <- list(
          descriptives = group_comparison,
          test = test_result
        )
      }
    }
  }
  
  return(results)
}

# Effect size calculations
calculate_effect_sizes <- function(data, group_var, outcome_vars) {
  effect_sizes <- list()
  
  for (var in outcome_vars) {
    groups <- split(data[[var]], data[[group_var]])
    
    if (length(groups) == 2) {
      # Cohen's d for two groups
      group1 <- groups[[1]][!is.na(groups[[1]])]
      group2 <- groups[[2]][!is.na(groups[[2]])]
      
      pooled_sd <- sqrt(((length(group1) - 1) * var(group1) + 
                        (length(group2) - 1) * var(group2)) / 
                       (length(group1) + length(group2) - 2))
      
      cohens_d <- (mean(group1) - mean(group2)) / pooled_sd
      
      effect_sizes[[var]] <- list(
        type = "Cohen's d",
        value = cohens_d,
        interpretation = case_when(
          abs(cohens_d) < 0.2 ~ "negligible",
          abs(cohens_d) < 0.5 ~ "small",
          abs(cohens_d) < 0.8 ~ "medium",
          TRUE ~ "large"
        )
      )
    } else {
      # Eta-squared for multiple groups
      model <- aov(data[[var]] ~ data[[group_var]], data = data)
      ss_total <- sum((data[[var]] - mean(data[[var]], na.rm = TRUE))^2, na.rm = TRUE)
      ss_between <- sum(model$residuals^2, na.rm = TRUE)
      eta_squared <- 1 - (ss_between / ss_total)
      
      effect_sizes[[var]] <- list(
        type = "Eta-squared",
        value = eta_squared,
        interpretation = case_when(
          eta_squared < 0.01 ~ "negligible",
          eta_squared < 0.06 ~ "small",
          eta_squared < 0.14 ~ "medium",
          TRUE ~ "large"
        )
      )
    }
  }
  
  return(effect_sizes)
}
```

---

*This comprehensive social sciences research guide provides methodological frameworks, statistical analysis tools, and best practices for conducting rigorous social science research.* 
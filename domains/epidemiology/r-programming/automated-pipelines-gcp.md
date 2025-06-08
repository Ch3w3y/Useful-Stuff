# Automated R Pipelines and GCP Deployment for Epidemiology

## Table of Contents

1. [Pipeline Foundations](#pipeline-foundations)
2. [Local Development with {targets}](#local-development-with-targets)
3. [GCP Services for Epidemiology](#gcp-services-for-epidemiology)
4. [Automated Data Processing](#automated-data-processing)
5. [Real-time Surveillance Systems](#real-time-surveillance-systems)
6. [Cloud Deployment Strategies](#cloud-deployment-strategies)
7. [Monitoring and Alerting](#monitoring-and-alerting)
8. [Security and Compliance](#security-and-compliance)

## Pipeline Foundations

### Why Automate Epidemiological Workflows?

#### Core Benefits
1. **Timeliness**: Real-time analysis and reporting
2. **Consistency**: Standardized methods and quality control
3. **Scalability**: Handle increasing data volumes
4. **Reproducibility**: Eliminate manual errors
5. **Efficiency**: Focus on interpretation, not data processing
6. **Compliance**: Meet regulatory requirements automatically

### Pipeline Design Principles

#### Data Quality Framework
```r
# Comprehensive data quality checking
data_quality_check <- function(data, config) {
  checks <- list(
    completeness = check_completeness(data, config$required_fields),
    validity = check_validity(data, config$validation_rules),
    consistency = check_consistency(data, config$consistency_rules),
    timeliness = check_timeliness(data, config$time_thresholds)
  )
  
  # Generate quality report
  quality_report <- generate_quality_report(checks)
  
  # Fail pipeline if critical issues found
  if (any(sapply(checks, function(x) any(x$critical_failures)))) {
    stop("Critical data quality issues detected")
  }
  
  return(list(data = data, quality = quality_report))
}

# Completeness check
check_completeness <- function(data, required_fields) {
  missing_rates <- sapply(required_fields, function(field) {
    mean(is.na(data[[field]]))
  })
  
  critical_failures <- missing_rates > 0.1  # >10% missing
  
  return(list(
    missing_rates = missing_rates,
    critical_failures = critical_failures
  ))
}
```

#### Fault Tolerance
```r
# Robust execution with retries
execute_with_retry <- function(func, max_retries = 3, backoff = 2) {
  for (attempt in 1:max_retries) {
    tryCatch({
      result <- func()
      return(result)
    }, error = function(e) {
      if (attempt == max_retries) {
        stop(paste("Failed after", max_retries, "attempts:", e$message))
      }
      
      wait_time <- backoff^attempt
      message("Attempt ", attempt, " failed. Retrying in ", wait_time, " seconds...")
      Sys.sleep(wait_time)
    })
  }
}
```

## Local Development with {targets}

### Pipeline Definition
```r
# _targets.R - Complete epidemiology pipeline
library(targets)
library(tarchetypes)

# Source all functions
tar_source("R/")

# Define pipeline
list(
  # Configuration
  tar_target(
    name = config,
    command = load_config("config/pipeline_config.yml")
  ),
  
  # Data import
  tar_target(
    name = raw_surveillance_data,
    command = import_surveillance_data(config),
    cue = tar_cue(mode = "always")
  ),
  
  tar_target(
    name = raw_laboratory_data,
    command = import_laboratory_data(config),
    cue = tar_cue(mode = "always")
  ),
  
  # Data cleaning and validation
  tar_target(
    name = clean_surveillance,
    command = clean_surveillance_data(raw_surveillance_data, config)
  ),
  
  tar_target(
    name = clean_laboratory,
    command = clean_laboratory_data(raw_laboratory_data, config)
  ),
  
  # Data integration
  tar_target(
    name = integrated_dataset,
    command = integrate_surveillance_lab(clean_surveillance, clean_laboratory)
  ),
  
  # Quality control
  tar_target(
    name = quality_report,
    command = generate_quality_report(integrated_dataset, config)
  ),
  
  # Descriptive analysis
  tar_target(
    name = descriptive_stats,
    command = generate_descriptive_analysis(integrated_dataset)
  ),
  
  # Temporal analysis
  tar_target(
    name = temporal_trends,
    command = analyze_temporal_trends(integrated_dataset)
  ),
  
  # Outbreak detection
  tar_target(
    name = outbreak_detection,
    command = detect_outbreaks(integrated_dataset, config$outbreak_threshold)
  ),
  
  # Geographic analysis
  tar_target(
    name = geographic_analysis,
    command = analyze_geographic_patterns(integrated_dataset)
  ),
  
  # Alert system
  tar_target(
    name = alert_status,
    command = check_alert_conditions(outbreak_detection, temporal_trends),
    cue = tar_cue(mode = "always")
  ),
  
  # Dashboard data preparation
  tar_target(
    name = dashboard_data,
    command = prepare_dashboard_data(
      descriptive_stats, 
      temporal_trends, 
      geographic_analysis,
      outbreak_detection
    )
  ),
  
  # Report generation
  tar_render(
    name = surveillance_report,
    path = "reports/surveillance_report.Rmd",
    params = list(
      data = integrated_dataset,
      analysis = descriptive_stats,
      alerts = alert_status,
      date = Sys.Date()
    ),
    output_dir = "output/reports"
  ),
  
  # Export results
  tar_target(
    name = export_results,
    command = export_pipeline_results(
      dashboard_data,
      surveillance_report,
      quality_report,
      config
    )
  )
)
```

### Core Pipeline Functions

#### Data Import Functions
```r
# R/data_import.R

# Import surveillance data
import_surveillance_data <- function(config) {
  tryCatch({
    # Connect to data source
    if (config$data_source$type == "database") {
      data <- import_from_database(config$data_source$database)
    } else if (config$data_source$type == "files") {
      data <- import_from_files(config$data_source$path)
    } else if (config$data_source$type == "api") {
      data <- import_from_api(config$data_source$endpoint)
    }
    
    # Basic validation
    if (nrow(data) == 0) {
      warning("No surveillance data imported")
    }
    
    return(data)
  }, error = function(e) {
    stop("Failed to import surveillance data: ", e$message)
  })
}

# Import from database
import_from_database <- function(db_config) {
  library(DBI)
  
  # Create connection
  con <- dbConnect(
    RPostgreSQL::PostgreSQL(),
    host = db_config$host,
    port = db_config$port,
    dbname = db_config$database,
    user = Sys.getenv("DB_USER"),
    password = Sys.getenv("DB_PASSWORD")
  )
  
  # Query recent data
  query <- "
    SELECT *
    FROM surveillance_cases 
    WHERE report_date >= CURRENT_DATE - INTERVAL '30 days'
    ORDER BY report_date DESC
  "
  
  data <- dbGetQuery(con, query)
  dbDisconnect(con)
  
  return(data)
}

# Import from files
import_from_files <- function(file_path) {
  library(readr)
  
  # Get list of recent files
  files <- list.files(file_path, pattern = "*.csv", full.names = TRUE)
  recent_files <- files[file.mtime(files) >= Sys.Date() - 7]
  
  # Read and combine files
  data <- map_dfr(recent_files, function(file) {
    read_csv(file, col_types = cols(
      case_id = col_character(),
      report_date = col_date(),
      age = col_integer(),
      sex = col_character()
    ))
  })
  
  return(data)
}
```

#### Data Cleaning Functions
```r
# R/data_cleaning.R

clean_surveillance_data <- function(raw_data, config) {
  cleaned_data <- raw_data %>%
    # Remove duplicates
    distinct(case_id, .keep_all = TRUE) %>%
    
    # Date validation
    filter(
      !is.na(report_date),
      report_date >= as.Date(config$study_period$start),
      report_date <= Sys.Date()
    ) %>%
    
    # Standardize variables
    mutate(
      # Age groups
      age_group = case_when(
        age < 18 ~ "0-17",
        age < 65 ~ "18-64",
        TRUE ~ "65+"
      ),
      age_group = factor(age_group, levels = c("0-17", "18-64", "65+")),
      
      # Sex standardization
      sex = case_when(
        str_to_upper(sex) %in% c("M", "MALE") ~ "Male",
        str_to_upper(sex) %in% c("F", "FEMALE") ~ "Female",
        TRUE ~ "Unknown"
      ),
      
      # Temporal variables
      week = floor_date(report_date, "week"),
      month = floor_date(report_date, "month"),
      
      # Clinical variables
      hospitalized = !is.na(hospitalization_date) & hospitalization_date <= report_date,
      died = !is.na(death_date) & death_date >= report_date
    ) %>%
    
    # Additional validation
    filter(
      age >= 0 & age <= 120,
      sex %in% c("Male", "Female", "Unknown")
    )
  
  # Log cleaning results
  cat("Data cleaning completed:\n")
  cat("  Original records:", nrow(raw_data), "\n")
  cat("  Cleaned records:", nrow(cleaned_data), "\n")
  cat("  Records removed:", nrow(raw_data) - nrow(cleaned_data), "\n")
  
  return(cleaned_data)
}
```

#### Analysis Functions
```r
# R/analysis_functions.R

# Generate descriptive statistics
generate_descriptive_analysis <- function(data) {
  # Overall summary
  overall <- data %>%
    summarise(
      total_cases = n(),
      start_date = min(report_date),
      end_date = max(report_date),
      median_age = median(age, na.rm = TRUE),
      hospitalization_rate = mean(hospitalized, na.rm = TRUE),
      case_fatality_rate = mean(died, na.rm = TRUE)
    )
  
  # Demographics
  age_dist <- data %>%
    count(age_group, name = "cases") %>%
    mutate(percentage = round(cases / sum(cases) * 100, 1))
  
  sex_dist <- data %>%
    count(sex, name = "cases") %>%
    mutate(percentage = round(cases / sum(cases) * 100, 1))
  
  # Temporal patterns
  weekly_counts <- data %>%
    count(week, name = "cases") %>%
    arrange(week) %>%
    mutate(
      cases_7day_avg = slider::slide_dbl(cases, mean, .before = 6, .complete = FALSE)
    )
  
  return(list(
    overall = overall,
    demographics = list(age = age_dist, sex = sex_dist),
    temporal = weekly_counts
  ))
}

# Outbreak detection using EARS algorithm
detect_outbreaks <- function(data, threshold = 2.5) {
  # Daily case counts
  daily_counts <- data %>%
    count(report_date, name = "cases") %>%
    complete(
      report_date = seq.Date(min(report_date), max(report_date), by = "day"),
      fill = list(cases = 0)
    ) %>%
    arrange(report_date)
  
  # EARS C3 algorithm
  daily_counts <- daily_counts %>%
    mutate(
      # Baseline (8-week moving average, excluding recent 2 days)
      baseline = slider::slide_dbl(
        cases,
        ~ mean(.x, na.rm = TRUE),
        .before = 56 + 2,  # 8 weeks + 2 days
        .after = -3,
        .complete = FALSE
      ),
      
      # Standard deviation
      baseline_sd = slider::slide_dbl(
        cases,
        ~ sd(.x, na.rm = TRUE),
        .before = 56 + 2,
        .after = -3,
        .complete = FALSE
      ),
      
      # EARS statistic
      ears_statistic = (cases - baseline) / baseline_sd,
      
      # Alert indicator
      alert = ears_statistic > threshold & !is.na(ears_statistic)
    )
  
  # Identify alert periods
  alerts <- daily_counts %>%
    filter(alert) %>%
    select(report_date, cases, baseline, ears_statistic)
  
  # Current alert status
  current_alerts <- any(daily_counts$alert[daily_counts$report_date >= Sys.Date() - 3])
  
  return(list(
    daily_data = daily_counts,
    alerts = alerts,
    current_alert = current_alerts,
    threshold = threshold
  ))
}
```

## GCP Services for Epidemiology

### Cloud Architecture Overview

#### Core GCP Services
```
Data Sources → Cloud Storage → BigQuery → Compute Engine/Cloud Run → Results
     ↓              ↓            ↓              ↓                    ↓
Surveillance    Raw Data     Data Warehouse    R Analysis        Reports
Lab Results     Staging      SQL Analytics     Pipelines         Dashboards
External APIs   Archival     ML Features       APIs              Alerts
```

### Setting Up GCP Environment

#### Project Initialization
```bash
# Create project
gcloud projects create epidemiology-analysis

# Set default project
gcloud config set project epidemiology-analysis

# Enable APIs
gcloud services enable compute.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable bigquery.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable cloudfunctions.googleapis.com
gcloud services enable cloudscheduler.googleapis.com

# Create service account
gcloud iam service-accounts create epi-pipeline \
    --description="Epidemiology pipeline service account" \
    --display-name="Epi Pipeline"

# Grant permissions
gcloud projects add-iam-policy-binding epidemiology-analysis \
    --member="serviceAccount:epi-pipeline@epidemiology-analysis.iam.gserviceaccount.com" \
    --role="roles/bigquery.admin"

gcloud projects add-iam-policy-binding epidemiology-analysis \
    --member="serviceAccount:epi-pipeline@epidemiology-analysis.iam.gserviceaccount.com" \
    --role="roles/storage.admin"
```

### R Docker Container for GCP

#### Dockerfile
```dockerfile
# Dockerfile for R epidemiology environment
FROM rocker/r-ver:4.3.2

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libgdal-dev \
    libproj-dev \
    pandoc \
    && rm -rf /var/lib/apt/lists/*

# Install R packages
RUN install2.r --error \
    dplyr \
    ggplot2 \
    rmarkdown \
    targets \
    googleCloudStorageR \
    bigrquery \
    plumber \
    logger \
    config

# Set working directory
WORKDIR /app

# Copy application
COPY . /app/

# Install project dependencies
RUN R -e "renv::restore()"

# Set environment
ENV PORT=8080
EXPOSE 8080

# Run application
CMD ["R", "-e", "source('app.R')"]
```

## Real-time Surveillance Systems

### Automated Pipeline Architecture

#### Cloud Run API Service
```r
# app.R - Plumber API for surveillance automation
library(plumber)
library(targets)

#* @apiTitle Epidemiology Surveillance API
#* @apiDescription Automated surveillance analysis and reporting

#* Run surveillance pipeline
#* @post /pipeline/run
function(req, res) {
  tryCatch({
    # Run targets pipeline
    tar_make()
    
    # Get results
    results <- list(
      cases_processed = tar_read(integrated_dataset) %>% nrow(),
      alerts_detected = tar_read(alert_status),
      quality_score = tar_read(quality_report)$overall_score
    )
    
    res$status <- 200
    return(list(success = TRUE, results = results, timestamp = Sys.time()))
    
  }, error = function(e) {
    res$status <- 500
    return(list(success = FALSE, error = e$message, timestamp = Sys.time()))
  })
}

#* Get current surveillance summary
#* @get /surveillance/summary
function() {
  data <- tar_read(integrated_dataset)
  
  summary <- data %>%
    filter(report_date >= Sys.Date() - 7) %>%
    summarise(
      weekly_cases = n(),
      hospitalizations = sum(hospitalized, na.rm = TRUE),
      deaths = sum(died, na.rm = TRUE)
    )
  
  return(summary)
}

#* Check alert status
#* @get /alerts/status
function() {
  alert_status <- tar_read(alert_status)
  outbreak_data <- tar_read(outbreak_detection)
  
  return(list(
    current_alert = alert_status,
    recent_alerts = outbreak_data$alerts %>% 
      filter(report_date >= Sys.Date() - 7)
  ))
}

# Start API
pr() %>% pr_run(host = "0.0.0.0", port = as.numeric(Sys.getenv("PORT", 8080)))
```

#### Cloud Scheduler Configuration
```yaml
# cloud-scheduler.yml
schedulers:
  daily_pipeline:
    name: "daily-surveillance-pipeline"
    schedule: "0 6 * * *"  # 6 AM daily
    timezone: "America/New_York"
    target:
      type: "http"
      uri: "https://epidemiology-api-SERVICE_URL/pipeline/run"
      method: "POST"
      
  hourly_alerts:
    name: "hourly-alert-check"
    schedule: "0 * * * *"  # Every hour
    timezone: "America/New_York"
    target:
      type: "http"
      uri: "https://epidemiology-api-SERVICE_URL/alerts/check"
      method: "GET"
```

### Deployment Scripts

#### Deploy to Cloud Run
```bash
#!/bin/bash
# deploy.sh - Deploy surveillance system to GCP

# Build and push Docker image
docker build -t gcr.io/epidemiology-analysis/surveillance-api .
docker push gcr.io/epidemiology-analysis/surveillance-api

# Deploy to Cloud Run
gcloud run deploy surveillance-api \
    --image gcr.io/epidemiology-analysis/surveillance-api \
    --platform managed \
    --region us-central1 \
    --memory 4Gi \
    --cpu 2 \
    --timeout 3600 \
    --max-instances 10 \
    --service-account epi-pipeline@epidemiology-analysis.iam.gserviceaccount.com

# Create Cloud Scheduler jobs
gcloud scheduler jobs create http daily-surveillance-pipeline \
    --schedule="0 6 * * *" \
    --uri="https://surveillance-api-URL/pipeline/run" \
    --http-method=POST \
    --time-zone="America/New_York"

echo "Deployment completed successfully!"
```

This comprehensive guide provides a solid foundation for automated R pipelines and GCP deployment specifically tailored for epidemiological research. The examples show production-ready patterns for data processing, analysis automation, and cloud deployment.

Would you like me to expand on any particular aspect, such as specific outbreak detection algorithms, real-time dashboard creation, or advanced cloud architecture patterns? 
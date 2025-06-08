# R Script Runner for GCP
# Main execution script with cloud integration and comprehensive logging

# Load required libraries
suppressPackageStartupMessages({
  library(tidyverse)
  library(googleCloudStorageR)
  library(bigrquery)
  library(log4r)
  library(config)
  library(here)
  library(jsonlite)
})

# Configuration and logging setup
setup_logging <- function() {
  log_level <- Sys.getenv("LOG_LEVEL", "INFO")
  log_file <- file.path("/app/logs", paste0("r_execution_", 
                                           format(Sys.time(), "%Y%m%d_%H%M%S"), 
                                           ".log"))
  
  logger <- create.logger(logfile = log_file, level = log_level)
  return(logger)
}

# Initialize logger
logger <- setup_logging()

# Safely get environment variables
get_env_var <- function(var_name, default = NULL, required = TRUE) {
  value <- Sys.getenv(var_name, unset = NA)
  if (is.na(value)) {
    if (required) {
      error(logger, paste("Required environment variable", var_name, "not set"))
      stop(paste("Required environment variable", var_name, "not set"))
    } else {
      value <- default
    }
  }
  return(value)
}

# Load configuration
load_config <- function() {
  config <- list(
    project_id = get_env_var("GCP_PROJECT_ID"),
    region = get_env_var("GCP_REGION", "us-central1", FALSE),
    input_bucket = get_env_var("GCS_INPUT_BUCKET", "", FALSE),
    output_bucket = get_env_var("GCS_OUTPUT_BUCKET"),
    logs_bucket = get_env_var("GCS_LOGS_BUCKET", "", FALSE),
    bq_dataset = get_env_var("BQ_DATASET_ID", "", FALSE),
    bq_location = get_env_var("BQ_LOCATION", "US", FALSE),
    environment = get_env_var("ENVIRONMENT", "production", FALSE),
    debug_mode = as.logical(get_env_var("DEBUG_MODE", "FALSE", FALSE))
  )
  
  info(logger, "Configuration loaded successfully")
  return(config)
}

# GCP Authentication
authenticate_gcp <- function(config) {
  tryCatch({
    # Set authentication for BigQuery
    if (file.exists("/app/config/service-account-key.json")) {
      bq_auth(path = "/app/config/service-account-key.json")
      gcs_auth("/app/config/service-account-key.json")
      info(logger, "Authenticated with service account")
    } else {
      # Try default authentication
      bq_auth()
      gcs_auth()
      info(logger, "Using default authentication")
    }
    
    # Set project
    bq_projects()  # This will test the connection
    info(logger, paste("Connected to project:", config$project_id))
    
  }, error = function(e) {
    error(logger, paste("Authentication failed:", e$message))
    stop(paste("Authentication failed:", e$message))
  })
}

# Download input data from GCS
download_input_data <- function(config) {
  if (config$input_bucket == "") {
    info(logger, "No input bucket specified, skipping download")
    return(NULL)
  }
  
  tryCatch({
    # List objects in input bucket
    objects <- gcs_list_objects(config$input_bucket, prefix = "input/")
    
    if (nrow(objects) == 0) {
      info(logger, "No input files found")
      return(NULL)
    }
    
    # Download files to local directory
    dir.create("/app/data/input", recursive = TRUE, showWarnings = FALSE)
    
    downloaded_files <- character()
    for (i in seq_len(nrow(objects))) {
      object_name <- objects$name[i]
      local_path <- file.path("/app/data/input", basename(object_name))
      
      gcs_get_object(object_name, bucket = config$input_bucket, 
                     saveToDisk = local_path)
      downloaded_files <- c(downloaded_files, local_path)
      
      info(logger, paste("Downloaded:", object_name))
    }
    
    info(logger, paste("Downloaded", length(downloaded_files), "files"))
    return(downloaded_files)
    
  }, error = function(e) {
    error(logger, paste("Failed to download input data:", e$message))
    stop(paste("Failed to download input data:", e$message))
  })
}

# Execute the target analysis script
execute_analysis_script <- function(config, input_files) {
  target_script <- get_env_var("TARGET_SCRIPT")
  
  if (!file.exists(file.path("/app", target_script))) {
    error(logger, paste("Target script not found:", target_script))
    stop(paste("Target script not found:", target_script))
  }
  
  info(logger, paste("Executing analysis script:", target_script))
  
  tryCatch({
    # Set environment for the target script
    Sys.setenv(ANALYSIS_INPUT_FILES = paste(input_files, collapse = ","))
    Sys.setenv(ANALYSIS_OUTPUT_DIR = "/app/output")
    Sys.setenv(ANALYSIS_CONFIG = toJSON(config, auto_unbox = TRUE))
    
    # Source the target script
    source(file.path("/app", target_script), local = TRUE)
    
    info(logger, "Analysis script completed successfully")
    
  }, error = function(e) {
    error(logger, paste("Analysis script failed:", e$message))
    stop(paste("Analysis script failed:", e$message))
  })
}

# Upload results to BigQuery
upload_to_bigquery <- function(config, results_data = NULL) {
  if (config$bq_dataset == "" || is.null(results_data)) {
    info(logger, "Skipping BigQuery upload - no dataset or data specified")
    return(NULL)
  }
  
  tryCatch({
    table_name <- paste0(get_env_var("BQ_TABLE_PREFIX", "analysis_", FALSE),
                        format(Sys.time(), "%Y%m%d_%H%M%S"))
    
    # Upload data to BigQuery
    bq_table_upload(
      x = bq_table(config$project_id, config$bq_dataset, table_name),
      values = results_data,
      create_disposition = "CREATE_IF_NEEDED",
      write_disposition = "WRITE_TRUNCATE"
    )
    
    info(logger, paste("Data uploaded to BigQuery table:", table_name))
    return(table_name)
    
  }, error = function(e) {
    error(logger, paste("BigQuery upload failed:", e$message))
    warning("BigQuery upload failed, continuing with file uploads")
    return(NULL)
  })
}

# Upload output files to GCS
upload_output_files <- function(config) {
  output_dir <- "/app/output"
  
  if (!dir.exists(output_dir)) {
    info(logger, "No output directory found")
    return(NULL)
  }
  
  output_files <- list.files(output_dir, recursive = TRUE, full.names = TRUE)
  
  if (length(output_files) == 0) {
    info(logger, "No output files to upload")
    return(NULL)
  }
  
  tryCatch({
    uploaded_files <- character()
    
    for (file_path in output_files) {
      relative_path <- sub(paste0(output_dir, "/"), "", file_path)
      gcs_object_name <- file.path("output", 
                                   format(Sys.time(), "%Y/%m/%d"),
                                   relative_path)
      
      gcs_upload(file_path, 
                 bucket = config$output_bucket,
                 name = gcs_object_name)
      
      uploaded_files <- c(uploaded_files, gcs_object_name)
      info(logger, paste("Uploaded:", gcs_object_name))
    }
    
    info(logger, paste("Uploaded", length(uploaded_files), "files to GCS"))
    return(uploaded_files)
    
  }, error = function(e) {
    error(logger, paste("Failed to upload output files:", e$message))
    stop(paste("Failed to upload output files:", e$message))
  })
}

# Generate execution summary
generate_summary <- function(config, input_files, uploaded_files, bq_table) {
  summary <- list(
    execution_time = Sys.time(),
    environment = config$environment,
    project_id = config$project_id,
    input_files_count = length(input_files %||% character()),
    output_files_count = length(uploaded_files %||% character()),
    bigquery_table = bq_table,
    status = "completed"
  )
  
  # Save summary as JSON
  summary_file <- "/app/output/execution_summary.json"
  write_json(summary, summary_file, pretty = TRUE)
  
  info(logger, "Execution summary generated")
  return(summary)
}

# Main execution function
main <- function() {
  start_time <- Sys.time()
  info(logger, "Starting R script execution")
  
  tryCatch({
    # Load configuration
    config <- load_config()
    
    # Authenticate with GCP
    authenticate_gcp(config)
    
    # Download input data
    input_files <- download_input_data(config)
    
    # Execute analysis
    execute_analysis_script(config, input_files)
    
    # Upload results (this depends on your specific analysis)
    # For now, we'll assume the analysis script saves results to /app/output
    
    # Upload to BigQuery (if results data is available)
    # This would be customized based on your specific use case
    bq_table <- upload_to_bigquery(config, results_data = NULL)
    
    # Upload output files to GCS
    uploaded_files <- upload_output_files(config)
    
    # Generate execution summary
    summary <- generate_summary(config, input_files, uploaded_files, bq_table)
    
    end_time <- Sys.time()
    execution_duration <- as.numeric(difftime(end_time, start_time, units = "secs"))
    
    info(logger, paste("Execution completed successfully in", 
                      round(execution_duration, 2), "seconds"))
    
  }, error = function(e) {
    error(logger, paste("Execution failed:", e$message))
    stop(paste("Execution failed:", e$message))
  })
}

# Execute main function if this script is run directly
if (!interactive()) {
  main()
} 
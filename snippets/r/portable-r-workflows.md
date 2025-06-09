# Portable R Code Workflows

Comprehensive guide for creating reproducible, portable R code that runs seamlessly across GitHub Actions, GCP, AWS, and other cloud platforms.

## Table of Contents

- [Environment Management](#environment-management)
- [Containerization](#containerization)
- [GitHub Actions](#github-actions)
- [Google Cloud Platform](#google-cloud-platform)
- [Package Development](#package-development)
- [API Development](#api-development)
- [Deployment Patterns](#deployment-patterns)

## Environment Management

### 1. renv for Reproducible Environments

```r
# renv setup and configuration
# .Rprofile
source("renv/activate.R")

# Initialize renv in your project
renv::init()

# Install packages and snapshot
install.packages(c("dplyr", "ggplot2", "tidymodels", "plumber"))
renv::snapshot()

# Create renv.lock programmatically
create_renv_lock <- function() {
  # Define required packages
  packages <- c(
    "dplyr", "ggplot2", "tidyr", "readr",
    "tidymodels", "recipes", "parsnip",
    "plumber", "httr", "jsonlite",
    "DBI", "RPostgres", "pool",
    "testthat", "devtools", "usethis"
  )
  
  # Install if not present
  for (pkg in packages) {
    if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
      install.packages(pkg)
    }
  }
  
  # Snapshot environment
  renv::snapshot()
  
  cat("renv.lock created with", length(packages), "packages\n")
}

# Restore environment on new system
restore_environment <- function() {
  if (file.exists("renv.lock")) {
    renv::restore()
    cat("Environment restored from renv.lock\n")
  } else {
    warning("No renv.lock file found")
  }
}
```

### 2. Portable Configuration Management

```r
# config.R - Environment-aware configuration
library(config)

# config.yml
default:
  database:
    host: "localhost"
    port: 5432
    name: "mydb"
  api:
    base_url: "http://localhost:8000"
  
development:
  inherits: default
  database:
    host: "dev-db.example.com"
  
production:
  inherits: default
  database:
    host: !expr Sys.getenv("DB_HOST")
    port: !expr as.integer(Sys.getenv("DB_PORT", "5432"))
    name: !expr Sys.getenv("DB_NAME")
  api:
    base_url: !expr Sys.getenv("API_BASE_URL")

# R configuration helper
setup_config <- function(env = Sys.getenv("R_ENV", "development")) {
  config <- config::get(config = env)
  
  # Validate required environment variables in production
  if (env == "production") {
    required_vars <- c("DB_HOST", "DB_NAME", "API_BASE_URL")
    missing_vars <- required_vars[!nzchar(Sys.getenv(required_vars))]
    
    if (length(missing_vars) > 0) {
      stop("Missing required environment variables: ", 
           paste(missing_vars, collapse = ", "))
    }
  }
  
  return(config)
}

# Database connection with config
create_db_connection <- function(config) {
  pool::dbPool(
    drv = RPostgres::Postgres(),
    host = config$database$host,
    port = config$database$port,
    dbname = config$database$name,
    user = Sys.getenv("DB_USER"),
    password = Sys.getenv("DB_PASSWORD")
  )
}
```

## Containerization

### 1. Multi-stage R Dockerfile

```dockerfile
# Dockerfile
# Multi-stage build for optimized R containers
FROM rocker/r-ver:4.3.2 AS base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libpq-dev \
    libgit2-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Install renv
RUN R -e "install.packages('renv', repos = c(CRAN = 'https://cloud.r-project.org'))"

# Development stage
FROM base AS development

# Install additional dev tools
RUN apt-get update && apt-get install -y \
    vim \
    htop \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install development R packages
RUN R -e "install.packages(c('devtools', 'usethis', 'testthat', 'roxygen2'), repos = c(CRAN = 'https://cloud.r-project.org'))"

WORKDIR /workspace
COPY renv.lock renv.lock
COPY renv/activate.R renv/activate.R
COPY renv/settings.dcf renv/settings.dcf

# Restore packages
RUN R -e "renv::restore()"

# Production stage
FROM base AS production

WORKDIR /app

# Copy renv files
COPY renv.lock renv.lock
COPY renv/activate.R renv/activate.R
COPY renv/settings.dcf renv/settings.dcf

# Restore packages (production only)
RUN R -e "renv::restore()"

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash ruser
RUN chown -R ruser:ruser /app
USER ruser

# Default command
CMD ["R", "-e", "source('main.R')"]

# API stage
FROM production AS api

EXPOSE 8000

CMD ["R", "-e", "plumber::plumb('api.R')$run(host='0.0.0.0', port=8000)"]
```

### 2. Docker Compose for Development

```yaml
# docker-compose.yml
version: '3.8'

services:
  r-dev:
    build:
      context: .
      target: development
    container_name: r-development
    ports:
      - "8787:8787"  # RStudio Server
      - "8000:8000"  # Plumber API
    volumes:
      - .:/workspace
      - r-packages:/usr/local/lib/R/site-library
    environment:
      - DISABLE_AUTH=true
      - ROOT=true
      - R_ENV=development
    command: >
      bash -c "
        R -e \"install.packages('rstudioapi', repos='https://cloud.r-project.org')\" &&
        /init
      "

  r-api:
    build:
      context: .
      target: api
    container_name: r-api
    ports:
      - "8000:8000"
    environment:
      - R_ENV=production
      - DB_HOST=postgres
      - DB_NAME=rdata
      - DB_USER=ruser
      - DB_PASSWORD=rpassword
    depends_on:
      - postgres

  postgres:
    image: postgres:15
    container_name: r-postgres
    environment:
      POSTGRES_DB: rdata
      POSTGRES_USER: ruser
      POSTGRES_PASSWORD: rpassword
    ports:
      - "5432:5432"
    volumes:
      - postgres-data:/var/lib/postgresql/data

  rstudio:
    image: rocker/rstudio:4.3.2
    container_name: r-rstudio
    ports:
      - "8787:8787"
    volumes:
      - .:/home/rstudio/workspace
      - r-packages:/usr/local/lib/R/site-library
    environment:
      - DISABLE_AUTH=true
      - ROOT=true

volumes:
  r-packages:
  postgres-data:
```

## GitHub Actions

### 1. Comprehensive R CI/CD Pipeline

```yaml
# .github/workflows/r-ci-cd.yml
name: R CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  release:
    types: [ published ]

env:
  R_VERSION: '4.3.2'

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macOS-latest]
        r-version: ['4.2.3', '4.3.2']
      fail-fast: false

    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Setup R
      uses: r-lib/actions/setup-r@v2
      with:
        r-version: ${{ matrix.r-version }}
        use-public-rspm: true

    - name: Setup Pandoc
      uses: r-lib/actions/setup-pandoc@v2

    - name: Query dependencies
      run: |
        install.packages('remotes')
        saveRDS(remotes::dev_package_deps(dependencies = TRUE), ".github/depends.Rds", version = 2)
        writeLines(sprintf("R-%i.%i", getRversion()$major, getRversion()$minor), ".github/R-version")
      shell: Rscript {0}

    - name: Cache R packages
      uses: actions/cache@v3
      with:
        path: ${{ env.R_LIBS_USER }}
        key: ${{ runner.os }}-${{ hashFiles('.github/R-version') }}-1-${{ hashFiles('.github/depends.Rds') }}
        restore-keys: ${{ runner.os }}-${{ hashFiles('.github/R-version') }}-1-

    - name: Install system dependencies (Linux)
      if: runner.os == 'Linux'
      run: |
        sudo apt-get update
        sudo apt-get install -y libcurl4-openssl-dev libxml2-dev libssl-dev libpq-dev

    - name: Install dependencies
      run: |
        remotes::install_deps(dependencies = TRUE)
        remotes::install_cran("rcmdcheck")
      shell: Rscript {0}

    - name: Check package
      env:
        _R_CHECK_CRAN_INCOMING_REMOTE_: false
      run: |
        rcmdcheck::rcmdcheck(args = c("--no-manual", "--as-cran"), error_on = "warning", check_dir = "check")
      shell: Rscript {0}

    - name: Run tests
      run: |
        if (file.exists("tests/testthat.R")) {
          testthat::test_local()
        } else {
          cat("No tests found\n")
        }
      shell: Rscript {0}

    - name: Test coverage
      if: matrix.os == 'ubuntu-latest' && matrix.r-version == '4.3.2'
      run: |
        install.packages("covr")
        covr::codecov()
      shell: Rscript {0}

  lint:
    runs-on: ubuntu-latest
    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Setup R
      uses: r-lib/actions/setup-r@v2
      with:
        r-version: ${{ env.R_VERSION }}

    - name: Install lintr
      run: install.packages("lintr")
      shell: Rscript {0}

    - name: Lint code
      run: |
        lints <- lintr::lint_package()
        print(lints)
        if (length(lints) > 0) {
          quit(status = 1)
        }
      shell: Rscript {0}

  build-and-deploy:
    needs: [test, lint]
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'

    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Setup R
      uses: r-lib/actions/setup-r@v2
      with:
        r-version: ${{ env.R_VERSION }}

    - name: Build Docker image
      run: |
        docker build -t ${{ github.repository }}:latest .
        docker tag ${{ github.repository }}:latest ${{ github.repository }}:${{ github.sha }}

    - name: Login to Docker Hub
      uses: docker/login-action@v2
      with:
        username: ${{ secrets.DOCKER_USERNAME }}
        password: ${{ secrets.DOCKER_PASSWORD }}

    - name: Push Docker image
      run: |
        docker push ${{ github.repository }}:latest
        docker push ${{ github.repository }}:${{ github.sha }}

  deploy-gcp:
    needs: build-and-deploy
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'

    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Setup Google Cloud CLI
      uses: google-github-actions/setup-gcloud@v1
      with:
        service_account_key: ${{ secrets.GCP_SA_KEY }}
        project_id: ${{ secrets.GCP_PROJECT_ID }}

    - name: Deploy to Cloud Run
      run: |
        gcloud run deploy r-api \
          --image ${{ github.repository }}:${{ github.sha }} \
          --platform managed \
          --region us-central1 \
          --allow-unauthenticated \
          --set-env-vars R_ENV=production \
          --memory 2Gi \
          --cpu 2
```

### 2. R Package Testing Workflow

```yaml
# .github/workflows/r-package.yml
name: R Package Check

on: [push, pull_request]

jobs:
  R-CMD-check:
    runs-on: ${{ matrix.config.os }}
    name: ${{ matrix.config.os }} (${{ matrix.config.r }})

    strategy:
      fail-fast: false
      matrix:
        config:
          - {os: windows-latest, r: 'release'}
          - {os: macOS-latest, r: 'release'}
          - {os: ubuntu-latest, r: 'devel', http-user-agent: 'release'}
          - {os: ubuntu-latest, r: 'release'}
          - {os: ubuntu-latest, r: 'oldrel-1'}

    env:
      GITHUB_PAT: ${{ secrets.GITHUB_TOKEN }}
      R_KEEP_PKG_SOURCE: yes

    steps:
      - uses: actions/checkout@v4

      - uses: r-lib/actions/setup-pandoc@v2

      - uses: r-lib/actions/setup-r@v2
        with:
          r-version: ${{ matrix.config.r }}
          http-user-agent: ${{ matrix.config.http-user-agent }}
          use-public-rspm: true

      - uses: r-lib/actions/setup-r-dependencies@v2
        with:
          extra-packages: any::rcmdcheck
          needs: check

      - uses: r-lib/actions/check-r-package@v2
        with:
          upload-snapshots: true
```

## Google Cloud Platform

### 1. Cloud Run Deployment

```r
# deployment/gcp/cloud-run-setup.R
library(googleCloudRunner)
library(plumber)

# Setup Cloud Run deployment
setup_cloud_run <- function(project_id, region = "us-central1") {
  # Authenticate
  cr_setup(project_id = project_id, region = region)
  
  # Build configuration
  build_config <- cr_build_yaml(
    steps = list(
      cr_buildstep_docker("r-api", tag = "gcr.io/$PROJECT_ID/r-api"),
      cr_buildstep_run(
        name = "r-api",
        image = "gcr.io/$PROJECT_ID/r-api",
        env = c("R_ENV=production"),
        memory = "2Gi",
        cpu = 2,
        port = 8000,
        allowUnauthenticated = TRUE
      )
    )
  )
  
  # Submit build
  build <- cr_build_make(build_config)
  cr_build(build)
}

# Deploy function with monitoring
deploy_r_api <- function(image_url, service_name = "r-api") {
  # Cloud Run service configuration
  service <- cr_run_yaml(
    image = image_url,
    name = service_name,
    env = list(
      R_ENV = "production",
      DB_HOST = cr_run_secret("db-host"),
      DB_PASSWORD = cr_run_secret("db-password")
    ),
    resources = list(
      memory = "2Gi",
      cpu = 2
    ),
    concurrency = 100,
    port = 8000
  )
  
  # Deploy
  deployment <- cr_deploy(service)
  
  # Setup monitoring
  setup_monitoring(service_name)
  
  return(deployment)
}

# Monitoring setup
setup_monitoring <- function(service_name) {
  # Create uptime check
  cr_monitoring_uptime(
    display_name = paste0(service_name, "-uptime"),
    http_check_path = "/health",
    period = "60s"
  )
  
  # Create alerting policy
  cr_monitoring_alert(
    display_name = paste0(service_name, "-errors"),
    condition = "resource.type=\"cloud_run_revision\" AND severity>=ERROR"
  )
}
```

### 2. Dataflow R Pipeline

```r
# dataflow/r-pipeline.R
library(googleCloudStorageR)
library(bigrquery)

# Dataflow-compatible R processing function
process_data_batch <- function(input_path, output_path, config) {
  
  # Read from Cloud Storage
  input_data <- gcs_get_object(input_path, 
                               bucket = config$input_bucket,
                               parseFunction = read.csv)
  
  # Data processing pipeline
  processed_data <- input_data %>%
    dplyr::filter(!is.na(value)) %>%
    dplyr::mutate(
      processed_date = Sys.Date(),
      value_normalized = scale(value)[,1]
    ) %>%
    dplyr::group_by(category) %>%
    dplyr::summarise(
      mean_value = mean(value_normalized),
      count = n(),
      .groups = "drop"
    )
  
  # Write to BigQuery
  bq_table_upload(
    x = bq_table(config$project_id, config$dataset, config$table),
    values = processed_data,
    write_disposition = "WRITE_APPEND"
  )
  
  # Write to Cloud Storage for backup
  gcs_upload(processed_data,
             bucket = config$output_bucket,
             name = output_path,
             object_function = function(x, file) write.csv(x, file, row.names = FALSE))
  
  return(nrow(processed_data))
}

# Dataflow job submission
submit_dataflow_job <- function(template_path, parameters) {
  system2("gcloud", args = c(
    "dataflow", "jobs", "run", parameters$job_name,
    "--gcs-location", template_path,
    "--region", parameters$region,
    "--parameters", paste0(names(parameters), "=", parameters, collapse = ",")
  ))
}
```

### 3. Vertex AI Integration

```r
# vertex-ai/model-training.R
library(googleCloudVertexAIR)
library(tidymodels)

# Vertex AI training job
train_model_vertex <- function(training_data_uri, model_output_uri) {
  
  # Define training script
  training_script <- '
    library(tidymodels)
    library(googleCloudStorageR)
    
    # Load data from GCS
    data <- gcs_get_object(training_data_uri, parseFunction = readr::read_csv)
    
    # Prepare data
    data_split <- initial_split(data, prop = 0.8)
    train_data <- training(data_split)
    test_data <- testing(data_split)
    
    # Define recipe
    recipe <- recipe(target ~ ., data = train_data) %>%
      step_normalize(all_numeric_predictors()) %>%
      step_dummy(all_nominal_predictors())
    
    # Define model
    model <- rand_forest(trees = 1000) %>%
      set_engine("ranger") %>%
      set_mode("regression")
    
    # Create workflow
    workflow <- workflow() %>%
      add_recipe(recipe) %>%
      add_model(model)
    
    # Fit model
    fitted_model <- fit(workflow, train_data)
    
    # Evaluate
    predictions <- predict(fitted_model, test_data)
    metrics <- bind_cols(test_data, predictions) %>%
      metrics(truth = target, estimate = .pred)
    
    # Save model
    saveRDS(fitted_model, "model.rds")
    gcs_upload("model.rds", bucket = "model-artifacts", name = model_output_uri)
    
    # Log metrics
    cat("Model performance:\n")
    print(metrics)
  '
  
  # Submit training job
  job <- vertex_ai_training_job(
    display_name = "r-model-training",
    worker_pool_specs = list(
      vertex_ai_worker_pool_spec(
        machine_type = "n1-standard-4",
        replica_count = 1,
        container_spec = vertex_ai_container_spec(
          image_uri = "gcr.io/PROJECT_ID/r-training:latest",
          command = c("Rscript"),
          args = c("-e", training_script)
        )
      )
    )
  )
  
  return(job)
}

# Model serving with Vertex AI
deploy_model_vertex <- function(model_artifact_uri, endpoint_name) {
  
  # Create model
  model <- vertex_ai_model(
    display_name = "r-prediction-model",
    artifact_uri = model_artifact_uri,
    container_spec = vertex_ai_container_spec(
      image_uri = "gcr.io/PROJECT_ID/r-serving:latest",
      predict_route = "/predict",
      health_route = "/health"
    )
  )
  
  # Create endpoint
  endpoint <- vertex_ai_endpoint(display_name = endpoint_name)
  
  # Deploy model to endpoint
  deployment <- vertex_ai_deploy_model(
    endpoint = endpoint,
    model = model,
    machine_type = "n1-standard-2",
    min_replica_count = 1,
    max_replica_count = 10
  )
  
  return(deployment)
}
```

## Package Development

### 1. Modern R Package Structure

```r
# R package template with modern practices
create_modern_r_package <- function(package_name, path = ".") {
  
  # Create package structure
  usethis::create_package(file.path(path, package_name))
  usethis::use_git()
  usethis::use_github()
  
  # Setup testing
  usethis::use_testthat()
  usethis::use_test("main")
  
  # Setup documentation
  usethis::use_roxygen_md()
  usethis::use_pkgdown()
  
  # Setup CI/CD
  usethis::use_github_action("check-standard")
  usethis::use_github_action("test-coverage")
  usethis::use_github_action("pkgdown")
  
  # Setup code quality
  usethis::use_code_of_conduct()
  usethis::use_mit_license()
  
  # Package dependencies
  usethis::use_package("dplyr")
  usethis::use_package("ggplot2")
  usethis::use_pipe()
  
  # Development dependencies
  usethis::use_dev_package("testthat")
  usethis::use_dev_package("covr")
  
  cat("Modern R package created:", package_name, "\n")
}

# Package configuration
setup_package_config <- function() {
  # .Rbuildignore
  usethis::use_build_ignore(c(
    "^.*\\.Rproj$",
    "^\\.Rproj\\.user$",
    "^README\\.Rmd$",
    "^LICENSE\\.md$",
    "^cran-comments\\.md$",
    "^\\.github$",
    "^_pkgdown\\.yml$",
    "^docs$",
    "^pkgdown$"
  ))
  
  # GitHub Actions
  usethis::use_github_action_check_standard()
  usethis::use_coverage()
  
  # pkgdown configuration
  pkgdown_config <- list(
    url = "https://username.github.io/package-name",
    template = list(
      bootstrap = 5
    ),
    reference = list(
      list(
        title = "Main Functions",
        contents = c("main_function", "helper_function")
      )
    )
  )
  
  yaml::write_yaml(pkgdown_config, "_pkgdown.yml")
}
```

### 2. Package Testing Framework

```r
# tests/testthat/test-main.R
library(testthat)
library(mockery)

test_that("data processing works correctly", {
  # Test data
  test_data <- data.frame(
    x = 1:10,
    y = 11:20,
    group = rep(c("A", "B"), 5)
  )
  
  # Test function
  result <- process_data(test_data)
  
  # Assertions
  expect_s3_class(result, "data.frame")
  expect_equal(nrow(result), 2)  # Two groups
  expect_true(all(c("group", "mean_x", "mean_y") %in% names(result)))
})

test_that("API functions handle errors gracefully", {
  # Mock external API
  mock_api_call <- mock(stop("API Error"))
  
  with_mock(
    "package::api_call" = mock_api_call,
    {
      expect_error(fetch_external_data("invalid_endpoint"))
    }
  )
})

test_that("database connections work", {
  skip_if_not(db_available(), "Database not available")
  
  conn <- connect_to_db()
  expect_s4_class(conn, "PqConnection")
  
  # Test query
  result <- DBI::dbGetQuery(conn, "SELECT 1 as test")
  expect_equal(result$test, 1)
  
  DBI::dbDisconnect(conn)
})

# Integration tests
test_that("full pipeline works end-to-end", {
  skip_on_cran()
  skip_if_offline()
  
  # Setup test environment
  temp_dir <- tempdir()
  test_config <- list(
    input_path = file.path(temp_dir, "input.csv"),
    output_path = file.path(temp_dir, "output.csv")
  )
  
  # Create test data
  test_data <- data.frame(x = 1:100, y = rnorm(100))
  write.csv(test_data, test_config$input_path, row.names = FALSE)
  
  # Run pipeline
  result <- run_pipeline(test_config)
  
  # Verify output
  expect_true(file.exists(test_config$output_path))
  output_data <- read.csv(test_config$output_path)
  expect_gt(nrow(output_data), 0)
  
  # Cleanup
  unlink(temp_dir, recursive = TRUE)
})
```

## API Development

### 1. Plumber API with Best Practices

```r
# api.R
library(plumber)
library(dplyr)
library(jsonlite)
library(DBI)
library(pool)

# Global connection pool
pool <- NULL

#* @apiTitle ML Model API
#* @apiDescription REST API for machine learning predictions
#* @apiVersion 1.0.0

# Initialize API
#* @filter cors
function(req, res) {
  res$setHeader("Access-Control-Allow-Origin", "*")
  res$setHeader("Access-Control-Allow-Methods", "GET,POST,PUT,DELETE,OPTIONS")
  res$setHeader("Access-Control-Allow-Headers", "Content-Type,Authorization")
  
  if (req$REQUEST_METHOD == "OPTIONS") {
    res$status <- 200
    return(list())
  } else {
    plumber::forward()
  }
}

# Authentication filter
#* @filter auth
function(req, res) {
  if (!is.null(req$HTTP_AUTHORIZATION)) {
    token <- gsub("Bearer ", "", req$HTTP_AUTHORIZATION)
    if (validate_token(token)) {
      plumber::forward()
    } else {
      res$status <- 401
      return(list(error = "Invalid token"))
    }
  } else {
    res$status <- 401
    return(list(error = "Authorization required"))
  }
}

# Health check endpoint
#* @get /health
#* @serializer unboxedJSON
function() {
  list(
    status = "healthy",
    timestamp = Sys.time(),
    version = packageVersion("mypackage")
  )
}

# Prediction endpoint
#* @post /predict
#* @param features:list Input features for prediction
#* @serializer unboxedJSON
#* @preempt auth
function(features) {
  tryCatch({
    # Validate input
    if (!validate_features(features)) {
      stop("Invalid features provided")
    }
    
    # Load model
    model <- load_model()
    
    # Make prediction
    prediction <- predict(model, features)
    
    # Log prediction
    log_prediction(features, prediction)
    
    list(
      prediction = prediction,
      confidence = calculate_confidence(prediction),
      timestamp = Sys.time()
    )
  }, error = function(e) {
    res$status <- 400
    list(error = e$message)
  })
}

# Batch prediction endpoint
#* @post /predict/batch
#* @param data:data.frame Batch of features for prediction
#* @serializer unboxedJSON
#* @preempt auth
function(data) {
  tryCatch({
    # Validate batch size
    if (nrow(data) > 1000) {
      stop("Batch size too large (max 1000)")
    }
    
    # Process batch
    model <- load_model()
    predictions <- predict(model, data)
    
    list(
      predictions = predictions,
      count = length(predictions),
      timestamp = Sys.time()
    )
  }, error = function(e) {
    res$status <- 400
    list(error = e$message)
  })
}

# Model metrics endpoint
#* @get /metrics
#* @serializer unboxedJSON
#* @preempt auth
function() {
  conn <- pool::poolCheckout(pool)
  on.exit(pool::poolReturn(conn))
  
  metrics <- DBI::dbGetQuery(conn, "
    SELECT 
      DATE(created_at) as date,
      COUNT(*) as predictions,
      AVG(confidence) as avg_confidence
    FROM predictions 
    WHERE created_at >= NOW() - INTERVAL '7 days'
    GROUP BY DATE(created_at)
    ORDER BY date DESC
  ")
  
  list(
    metrics = metrics,
    summary = list(
      total_predictions = sum(metrics$predictions),
      avg_confidence = mean(metrics$avg_confidence)
    )
  )
}

# Helper functions
validate_features <- function(features) {
  required_fields <- c("feature1", "feature2", "feature3")
  all(required_fields %in% names(features))
}

validate_token <- function(token) {
  # Implement token validation logic
  return(token == Sys.getenv("API_TOKEN"))
}

load_model <- function() {
  # Load trained model
  if (!exists("cached_model", envir = .GlobalEnv)) {
    .GlobalEnv$cached_model <- readRDS("model.rds")
  }
  return(.GlobalEnv$cached_model)
}

calculate_confidence <- function(prediction) {
  # Calculate prediction confidence
  return(runif(1, 0.7, 0.99))
}

log_prediction <- function(features, prediction) {
  # Log to database
  conn <- pool::poolCheckout(pool)
  on.exit(pool::poolReturn(conn))
  
  DBI::dbExecute(conn, "
    INSERT INTO predictions (features, prediction, created_at)
    VALUES (?, ?, NOW())
  ", list(jsonlite::toJSON(features), prediction))
}

# Initialize database connection
init_db <- function() {
  pool <<- pool::dbPool(
    drv = RPostgres::Postgres(),
    host = Sys.getenv("DB_HOST"),
    port = as.integer(Sys.getenv("DB_PORT", "5432")),
    dbname = Sys.getenv("DB_NAME"),
    user = Sys.getenv("DB_USER"),
    password = Sys.getenv("DB_PASSWORD"),
    minSize = 1,
    maxSize = 10
  )
}

# Cleanup on exit
.onUnload <- function(libpath) {
  if (!is.null(pool)) {
    pool::poolClose(pool)
  }
}

# Initialize
init_db()
```

### 2. API Testing Suite

```r
# tests/test-api.R
library(testthat)
library(httr)
library(jsonlite)

# API testing helper
test_api <- function(endpoint, method = "GET", body = NULL, token = NULL) {
  base_url <- "http://localhost:8000"
  url <- paste0(base_url, endpoint)
  
  headers <- list("Content-Type" = "application/json")
  if (!is.null(token)) {
    headers$Authorization <- paste("Bearer", token)
  }
  
  if (method == "GET") {
    response <- GET(url, add_headers(.headers = headers))
  } else if (method == "POST") {
    response <- POST(url, 
                    body = toJSON(body, auto_unbox = TRUE),
                    add_headers(.headers = headers))
  }
  
  return(response)
}

# Test health endpoint
test_that("health endpoint returns success", {
  response <- test_api("/health")
  
  expect_equal(status_code(response), 200)
  
  content <- content(response, "parsed")
  expect_equal(content$status, "healthy")
  expect_true("timestamp" %in% names(content))
})

# Test authentication
test_that("authentication works correctly", {
  # Test without token
  response <- test_api("/predict", "POST", list(features = list(x = 1, y = 2)))
  expect_equal(status_code(response), 401)
  
  # Test with invalid token
  response <- test_api("/predict", "POST", 
                      list(features = list(x = 1, y = 2)),
                      token = "invalid")
  expect_equal(status_code(response), 401)
  
  # Test with valid token
  response <- test_api("/predict", "POST",
                      list(features = list(feature1 = 1, feature2 = 2, feature3 = 3)),
                      token = Sys.getenv("API_TOKEN"))
  expect_equal(status_code(response), 200)
})

# Test prediction endpoint
test_that("prediction endpoint works", {
  features <- list(
    feature1 = 1.5,
    feature2 = 2.3,
    feature3 = 0.8
  )
  
  response <- test_api("/predict", "POST", 
                      list(features = features),
                      token = Sys.getenv("API_TOKEN"))
  
  expect_equal(status_code(response), 200)
  
  content <- content(response, "parsed")
  expect_true("prediction" %in% names(content))
  expect_true("confidence" %in% names(content))
  expect_is(content$prediction, "numeric")
})

# Load testing
test_that("API handles concurrent requests", {
  skip_on_cran()
  
  library(future)
  plan(multisession, workers = 4)
  
  # Create multiple concurrent requests
  futures <- lapply(1:10, function(i) {
    future({
      test_api("/predict", "POST",
              list(features = list(feature1 = i, feature2 = i*2, feature3 = i*3)),
              token = Sys.getenv("API_TOKEN"))
    })
  })
  
  # Collect results
  responses <- value(futures)
  
  # All should succeed
  status_codes <- sapply(responses, status_code)
  expect_true(all(status_codes == 200))
})
```

## Deployment Patterns

### 1. Kubernetes Deployment

```yaml
# k8s/r-api-deployment.yml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: r-api
  labels:
    app: r-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: r-api
  template:
    metadata:
      labels:
        app: r-api
    spec:
      containers:
      - name: r-api
        image: gcr.io/PROJECT_ID/r-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: R_ENV
          value: "production"
        - name: DB_HOST
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: host
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: password
        - name: API_TOKEN
          valueFrom:
            secretKeyRef:
              name: api-secret
              key: token
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: r-api-service
spec:
  selector:
    app: r-api
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer

---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: r-api-ingress
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - api.example.com
    secretName: r-api-tls
  rules:
  - host: api.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: r-api-service
            port:
              number: 80
```

### 2. Terraform Infrastructure

```hcl
# terraform/main.tf
provider "google" {
  project = var.project_id
  region  = var.region
}

# Cloud Run service
resource "google_cloud_run_service" "r_api" {
  name     = "r-api"
  location = var.region

  template {
    spec {
      containers {
        image = "gcr.io/${var.project_id}/r-api:latest"
        
        ports {
          container_port = 8000
        }

        env {
          name  = "R_ENV"
          value = "production"
        }

        env {
          name = "DB_HOST"
          value_from {
            secret_key_ref {
              name = google_secret_manager_secret_version.db_host.secret
              key  = "latest"
            }
          }
        }

        resources {
          limits = {
            cpu    = "2"
            memory = "2Gi"
          }
        }
      }

      container_concurrency = 100
      timeout_seconds      = 300
    }

    metadata {
      annotations = {
        "autoscaling.knative.dev/minScale" = "1"
        "autoscaling.knative.dev/maxScale" = "10"
        "run.googleapis.com/cpu-throttling" = "false"
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }
}

# IAM for Cloud Run
resource "google_cloud_run_service_iam_member" "public" {
  service  = google_cloud_run_service.r_api.name
  location = google_cloud_run_service.r_api.location
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# Secret Manager
resource "google_secret_manager_secret" "db_host" {
  secret_id = "db-host"
}

resource "google_secret_manager_secret_version" "db_host" {
  secret      = google_secret_manager_secret.db_host.id
  secret_data = var.db_host
}

# Cloud SQL instance
resource "google_sql_database_instance" "main" {
  name             = "r-database"
  database_version = "POSTGRES_14"
  region           = var.region

  settings {
    tier = "db-f1-micro"

    database_flags {
      name  = "log_statement"
      value = "all"
    }
  }

  deletion_protection = false
}

# Monitoring
resource "google_monitoring_uptime_check_config" "r_api" {
  display_name = "R API Uptime Check"
  timeout      = "60s"
  period       = "300s"

  http_check {
    path         = "/health"
    port         = "443"
    use_ssl      = true
    validate_ssl = true
  }

  monitored_resource {
    type = "uptime_url"
    labels = {
      project_id = var.project_id
      host       = google_cloud_run_service.r_api.status[0].url
    }
  }
}

# Variables
variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region"
  type        = string
  default     = "us-central1"
}

variable "db_host" {
  description = "Database host"
  type        = string
  sensitive   = true
}

# Outputs
output "service_url" {
  value = google_cloud_run_service.r_api.status[0].url
}
```

## Related Resources

- [Docker Containerization](../../deployment/docker/)
- [Kubernetes Deployment](../../deployment/kubernetes/)
- [CI/CD Pipelines](../../devops/)
- [Cloud Infrastructure](../../deployment/cloud/)
- [API Development](../../snippets/python/)

## Contributing

When adding R deployment patterns:
1. Test across different R versions
2. Include comprehensive error handling
3. Document resource requirements
4. Provide monitoring setup
5. Add security considerations 
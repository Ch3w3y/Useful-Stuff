# R Environment Configuration for Cloud Deployments

Complete guide for setting up reproducible R environments across development, testing, and production environments.

## Quick Start Templates

### 1. Project Structure
```
my-r-project/
├── .github/
│   └── workflows/
│       └── deploy.yml
├── R/
│   ├── app.R
│   ├── api.R
│   └── utils.R
├── tests/
│   └── testthat/
├── renv/
├── .Rprofile
├── renv.lock
├── Dockerfile
├── docker-compose.yml
├── config.yml
└── README.md
```

### 2. renv Configuration

```r
# .Rprofile
# Activate renv for reproducible package management
source("renv/activate.R")

# Set development options
if (interactive()) {
  options(
    repos = c(CRAN = "https://cloud.r-project.org/"),
    browserNLdisabled = TRUE,
    deparse.max.lines = 2,
    warnPartialMatchArgs = TRUE,
    warnPartialMatchAttr = TRUE,
    warnPartialMatchDollar = TRUE
  )
}

# Production optimizations
if (Sys.getenv("R_ENV") == "production") {
  options(
    repos = c(CRAN = "https://packagemanager.rstudio.com/cran/__linux__/focal/latest"),
    Ncpus = parallel::detectCores(),
    timeout = 300,
    warn = 1
  )
}
```

### 3. Environment Configuration

```yaml
# config.yml
default:
  database:
    host: "localhost"
    port: 5432
    name: "myapp_dev"
  api:
    port: 8000
    host: "0.0.0.0"
  logging:
    level: "INFO"
  cache:
    ttl: 3600

development:
  inherits: default
  database:
    host: "localhost"
  logging:
    level: "DEBUG"

testing:
  inherits: default
  database:
    host: "localhost"
    name: "myapp_test"

production:
  inherits: default
  database:
    host: !expr Sys.getenv("DB_HOST")
    port: !expr as.integer(Sys.getenv("DB_PORT", "5432"))
    name: !expr Sys.getenv("DB_NAME")
    user: !expr Sys.getenv("DB_USER")
    password: !expr Sys.getenv("DB_PASSWORD")
  api:
    port: !expr as.integer(Sys.getenv("PORT", "8000"))
  logging:
    level: !expr Sys.getenv("LOG_LEVEL", "INFO")
```

### 4. Docker Configuration

```dockerfile
# Dockerfile
ARG R_VERSION=4.3.2
FROM rocker/r-ver:${R_VERSION}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libpq-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install renv
ENV RENV_VERSION 1.0.3
RUN R -e "install.packages('remotes', repos = c(CRAN = 'https://cloud.r-project.org'))"
RUN R -e "remotes::install_github('rstudio/renv@${RENV_VERSION}')"

# Copy renv files
COPY renv.lock renv.lock
COPY renv/activate.R renv/activate.R
COPY renv/settings.dcf renv/settings.dcf

# Restore packages
RUN R -e "renv::restore()"

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser
RUN chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Default command
EXPOSE 8000
CMD ["R", "-e", "source('app.R')"]
```

### 5. Docker Compose for Development

```yaml
# docker-compose.yml
version: '3.8'

services:
  r-app:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - R_ENV=development
      - DB_HOST=postgres
      - DB_NAME=myapp_dev
      - DB_USER=postgres
      - DB_PASSWORD=password
    volumes:
      - .:/app
    depends_on:
      - postgres
      - redis
    command: R -e "source('app.R')"

  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: myapp_dev
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data

  rstudio:
    image: rocker/rstudio:4.3.2
    ports:
      - "8787:8787"
    environment:
      - DISABLE_AUTH=true
      - ROOT=true
    volumes:
      - .:/home/rstudio/project
      - rstudio_home:/home/rstudio

volumes:
  postgres_data:
  redis_data:
  rstudio_home:
```

## Environment Management Scripts

### 1. Setup Script

```r
# setup.R
setup_environment <- function(env = "development") {
  cat("Setting up R environment for:", env, "\n")
  
  # Initialize renv if not already done
  if (!file.exists("renv.lock")) {
    renv::init()
  }
  
  # Install core packages
  core_packages <- c(
    "config",
    "plumber", 
    "DBI",
    "RPostgres",
    "pool",
    "redis",
    "httr",
    "jsonlite",
    "dplyr",
    "ggplot2",
    "logging"
  )
  
  # Development packages
  if (env == "development") {
    dev_packages <- c(
      "testthat",
      "devtools",
      "usethis",
      "lintr",
      "covr",
      "profvis"
    )
    core_packages <- c(core_packages, dev_packages)
  }
  
  # Install packages
  for (pkg in core_packages) {
    if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
      install.packages(pkg)
    }
  }
  
  # Snapshot environment
  renv::snapshot()
  
  cat("Environment setup complete!\n")
}

# Load configuration
load_config <- function(env = Sys.getenv("R_ENV", "development")) {
  if (!require("config", quietly = TRUE)) {
    install.packages("config")
    library(config)
  }
  
  config <- config::get(config = env)
  
  # Validate required environment variables in production
  if (env == "production") {
    required_vars <- c("DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD")
    missing_vars <- required_vars[!nzchar(Sys.getenv(required_vars))]
    
    if (length(missing_vars) > 0) {
      stop("Missing required environment variables: ", 
           paste(missing_vars, collapse = ", "))
    }
  }
  
  return(config)
}

# Database connection
setup_database <- function(config) {
  pool::dbPool(
    drv = RPostgres::Postgres(),
    host = config$database$host,
    port = config$database$port,
    dbname = config$database$name,
    user = config$database$user %||% Sys.getenv("DB_USER"),
    password = config$database$password %||% Sys.getenv("DB_PASSWORD"),
    minSize = 1,
    maxSize = 10
  )
}

# Logging setup
setup_logging <- function(config) {
  library(logging)
  
  # Set log level
  level <- config$logging$level
  basicConfig(level = level)
  
  # JSON formatter for production
  if (Sys.getenv("R_ENV") == "production") {
    json_formatter <- function(record) {
      log_entry <- list(
        timestamp = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
        level = record$levelname,
        message = record$msg,
        module = record$module %||% "main"
      )
      jsonlite::toJSON(log_entry, auto_unbox = TRUE)
    }
    
    addHandler(writeToConsole, formatter = json_formatter)
  }
}

# Run setup
if (!interactive()) {
  env <- Sys.getenv("R_ENV", "development")
  setup_environment(env)
}
```

### 2. Health Check Implementation

```r
# health.R
library(plumber)
library(DBI)

#* @get /health
#* @serializer unboxedJSON
health_check <- function() {
  status <- "healthy"
  checks <- list()
  
  # Database check
  tryCatch({
    if (exists("db_pool", envir = .GlobalEnv)) {
      conn <- pool::poolCheckout(.GlobalEnv$db_pool)
      result <- DBI::dbGetQuery(conn, "SELECT 1 as test")
      pool::poolReturn(conn)
      checks$database <- "healthy"
    } else {
      checks$database <- "not_configured"
    }
  }, error = function(e) {
    checks$database <<- "unhealthy"
    status <<- "unhealthy"
  })
  
  # Memory check
  memory_info <- gc(verbose = FALSE)
  memory_used <- sum(memory_info[, "used"])
  memory_limit <- as.numeric(Sys.getenv("MEMORY_LIMIT", "2000"))
  
  if (memory_used > memory_limit * 0.9) {
    checks$memory <- "warning"
    if (status == "healthy") status <- "degraded"
  } else {
    checks$memory <- "healthy"
  }
  
  # Disk space check (if applicable)
  tryCatch({
    disk_usage <- system("df / | tail -1 | awk '{print $5}'", intern = TRUE)
    disk_percent <- as.numeric(gsub("%", "", disk_usage))
    
    if (disk_percent > 90) {
      checks$disk <- "warning"
      if (status == "healthy") status <- "degraded"
    } else {
      checks$disk <- "healthy"
    }
  }, error = function(e) {
    checks$disk <- "unknown"
  })
  
  list(
    status = status,
    timestamp = Sys.time(),
    version = packageVersion("base"),
    checks = checks,
    uptime = Sys.time() - .GlobalEnv$start_time
  )
}

#* @get /ready
#* @serializer unboxedJSON
readiness_check <- function() {
  ready <- TRUE
  
  # Check if database pool is available
  if (!exists("db_pool", envir = .GlobalEnv)) {
    ready <- FALSE
  }
  
  # Check if configuration is loaded
  if (!exists("app_config", envir = .GlobalEnv)) {
    ready <- FALSE
  }
  
  list(
    ready = ready,
    timestamp = Sys.time()
  )
}

#* @get /metrics
#* @serializer text
metrics_endpoint <- function() {
  if (exists("prometheus_metrics", envir = .GlobalEnv)) {
    return(.GlobalEnv$prometheus_metrics$render())
  } else {
    return("# No metrics configured")
  }
}
```

### 3. Application Bootstrap

```r
# app.R
# Application entry point
library(plumber)
library(config)
library(logging)

# Record start time
.GlobalEnv$start_time <- Sys.time()

# Load configuration
cat("Loading configuration...\n")
source("setup.R")
.GlobalEnv$app_config <- load_config()

# Setup logging
setup_logging(.GlobalEnv$app_config)
loginfo("Starting application in %s mode", Sys.getenv("R_ENV", "development"))

# Setup database connection
if (!is.null(.GlobalEnv$app_config$database)) {
  loginfo("Connecting to database...")
  .GlobalEnv$db_pool <- setup_database(.GlobalEnv$app_config)
  
  # Cleanup on exit
  on.exit({
    if (exists("db_pool", envir = .GlobalEnv)) {
      pool::poolClose(.GlobalEnv$db_pool)
      loginfo("Database connection closed")
    }
  }, add = TRUE)
}

# Load API routes
source("api.R")

# Create plumber API
api <- plumber$new()

# Add health check routes
api$mount("/", pr("health.R"))

# Add main API routes  
api$mount("/api", pr("api.R"))

# Add middleware
api$filter("logger", function(req) {
  loginfo("%s %s", req$REQUEST_METHOD, req$PATH_INFO)
  plumber::forward()
})

api$filter("cors", function(req, res) {
  res$setHeader("Access-Control-Allow-Origin", "*")
  res$setHeader("Access-Control-Allow-Methods", "GET,POST,PUT,DELETE,OPTIONS")
  res$setHeader("Access-Control-Allow-Headers", "Content-Type,Authorization")
  
  if (req$REQUEST_METHOD == "OPTIONS") {
    res$status <- 200
    return(list())
  } else {
    plumber::forward()
  }
})

# Error handling
api$setErrorHandler(function(req, res, err) {
  logerror("Error in %s %s: %s", req$REQUEST_METHOD, req$PATH_INFO, err$message)
  res$status <- 500
  list(error = "Internal server error", timestamp = Sys.time())
})

# Start server
port <- .GlobalEnv$app_config$api$port
host <- .GlobalEnv$app_config$api$host

loginfo("Starting server on %s:%s", host, port)
api$run(host = host, port = port)
```

## GCP Deployment Scripts

### 1. Cloud Run Deployment

```bash
#!/bin/bash
# deploy-gcp.sh

set -e

PROJECT_ID=${1:-"your-project-id"}
REGION=${2:-"us-central1"}
SERVICE_NAME=${3:-"r-api"}

echo "Deploying to GCP Cloud Run..."
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "Service: $SERVICE_NAME"

# Build and submit to Cloud Build
gcloud builds submit --tag gcr.io/$PROJECT_ID/$SERVICE_NAME

# Deploy to Cloud Run
gcloud run deploy $SERVICE_NAME \
  --image gcr.io/$PROJECT_ID/$SERVICE_NAME \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --max-instances 20 \
  --set-env-vars R_ENV=production \
  --set-env-vars PORT=8080

# Get service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format='value(status.url)')
echo "Service deployed at: $SERVICE_URL"

# Test deployment
echo "Testing deployment..."
curl -f $SERVICE_URL/health || echo "Health check failed"
```

### 2. Environment Variables Setup

```r
# Set environment variables for GCP deployment
set_gcp_env_vars <- function(project_id, service_name, region = "us-central1") {
  
  # Database connection string
  db_vars <- list(
    "DB_HOST" = "/cloudsql/your-project:region:instance",
    "DB_NAME" = "production_db",
    "DB_USER" = "app_user"
  )
  
  # Application configuration
  app_vars <- list(
    "R_ENV" = "production",
    "LOG_LEVEL" = "INFO",
    "PORT" = "8080"
  )
  
  # Combine all variables
  env_vars <- c(db_vars, app_vars)
  
  # Format for gcloud command
  env_string <- paste(
    names(env_vars), env_vars, 
    sep = "=", 
    collapse = ","
  )
  
  # Update Cloud Run service
  system(sprintf(
    "gcloud run services update %s --region=%s --set-env-vars %s",
    service_name, region, env_string
  ))
  
  cat("Environment variables updated for", service_name, "\n")
}
```

## Testing Framework

### 1. Unit Tests

```r
# tests/testthat/test-api.R
library(testthat)
library(httr)

test_that("health endpoint returns success", {
  response <- GET("http://localhost:8000/health")
  
  expect_equal(status_code(response), 200)
  
  content <- content(response, "parsed")
  expect_equal(content$status, "healthy")
  expect_true("timestamp" %in% names(content))
})

test_that("API handles authentication", {
  # Test without authentication
  response <- POST("http://localhost:8000/api/predict")
  expect_equal(status_code(response), 401)
  
  # Test with valid token
  response <- POST(
    "http://localhost:8000/api/predict",
    add_headers(Authorization = paste("Bearer", Sys.getenv("TEST_TOKEN"))),
    body = list(data = list(x = 1, y = 2)),
    encode = "json"
  )
  expect_equal(status_code(response), 200)
})
```

### 2. Integration Tests

```r
# tests/testthat/test-integration.R
test_that("end-to-end workflow works", {
  skip_on_cran()
  skip_if_offline()
  
  # Start test server
  test_server <- callr::r_bg(function() {
    source("app.R")
  })
  
  # Wait for server to start
  Sys.sleep(5)
  
  tryCatch({
    # Test API endpoints
    health_response <- GET("http://localhost:8000/health")
    expect_equal(status_code(health_response), 200)
    
    # Test main functionality
    predict_response <- POST(
      "http://localhost:8000/api/predict",
      body = list(features = list(x = 1.5, y = 2.3)),
      encode = "json"
    )
    expect_equal(status_code(predict_response), 200)
    
  }, finally = {
    # Stop test server
    test_server$kill()
  })
})
```

## Performance Optimization

### 1. Memory Management

```r
# performance.R
optimize_r_performance <- function() {
  # Set memory limits
  if (nzchar(Sys.getenv("MEMORY_LIMIT"))) {
    memory_limit <- as.numeric(Sys.getenv("MEMORY_LIMIT"))
    options(memory.limit = memory_limit)
  }
  
  # Optimize garbage collection
  options(
    expressions = 500000,
    keep.source = FALSE,
    keep.source.pkgs = FALSE
  )
  
  # Parallel processing
  options(mc.cores = parallel::detectCores())
  
  # Package loading optimization
  options(
    defaultPackages = c("datasets", "utils", "grDevices", "graphics", "stats", "methods")
  )
}

# Monitor performance
monitor_performance <- function() {
  gc_info <- gc(verbose = FALSE)
  
  list(
    memory_used = sum(gc_info[, "used"]),
    memory_max = sum(gc_info[, "max used"]),
    r_version = R.version.string,
    system_memory = system("free -m | grep '^Mem' | awk '{print $3}'", intern = TRUE),
    cpu_usage = system("top -bn1 | grep 'Cpu(s)' | awk '{print $2}'", intern = TRUE)
  )
}
```

## Security Configuration

### 1. API Security

```r
# security.R
library(jose)

# JWT validation
validate_jwt <- function(token) {
  tryCatch({
    secret <- Sys.getenv("JWT_SECRET")
    if (nzchar(secret)) {
      payload <- jose::jwt_decode_sig(token, secret)
      return(list(valid = TRUE, payload = payload))
    }
    return(list(valid = FALSE, error = "No JWT secret configured"))
  }, error = function(e) {
    return(list(valid = FALSE, error = e$message))
  })
}

# Input sanitization
sanitize_input <- function(input) {
  if (is.character(input)) {
    # Remove potential SQL injection characters
    input <- gsub("[';\"\\\\]", "", input)
    # Limit length
    input <- substr(input, 1, 1000)
  }
  return(input)
}

# Rate limiting
implement_rate_limit <- function(identifier, max_requests = 100, window = 3600) {
  # Implementation would depend on Redis or similar store
  # This is a simplified version
  return(TRUE)
}
```

## Quick Commands

```bash
# Development
docker-compose up -d
R -e "source('setup.R')"

# Testing
R -e "testthat::test_local()"

# Build for production
docker build -t my-r-app .

# Deploy to GCP
./deploy-gcp.sh your-project-id

# Monitor logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=r-api" --limit 50
```

This comprehensive setup provides everything needed for portable R applications that can run seamlessly across GitHub Actions, GCP, AWS, and other cloud platforms! 
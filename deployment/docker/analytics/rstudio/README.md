# RStudio Server Docker Deployment for Analytics

Comprehensive Docker deployment configurations for RStudio Server optimized for data science and analytics workflows. This setup provides production-ready RStudio environments with pre-configured R packages, database connectivity, and integration with modern data infrastructure.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Docker Configurations](#docker-configurations)
- [Custom R Environment](#custom-r-environment)
- [Database Integration](#database-integration)
- [Package Management](#package-management)
- [Security & Authentication](#security--authentication)
- [Performance Optimization](#performance-optimization)
- [Monitoring & Logging](#monitoring--logging)
- [Troubleshooting](#troubleshooting)

## Overview

This deployment provides production-ready RStudio Server environments with:

- **Pre-configured R Environment**: Essential packages for data science and analytics
- **Database Connectivity**: PostgreSQL, MySQL, MongoDB, and cloud databases
- **Cloud Integration**: AWS, GCP, Azure storage and services
- **Security Features**: Authentication, HTTPS, user management
- **Performance Optimization**: Resource management and caching
- **Development Tools**: Git integration, package development tools

### Key Features
- Docker Compose orchestration
- Multiple R environments (base, tidyverse, geospatial, ML)
- Pre-installed analytics packages
- Database and cloud connectivity
- Shiny Server integration
- Version control with Git
- Package development environment

## Quick Start

### Prerequisites
```bash
# Install Docker and Docker Compose
sudo apt-get update
sudo apt-get install docker.io docker-compose

# Create project directory
mkdir rstudio-analytics && cd rstudio-analytics
```

### Basic Setup
```bash
# Create directory structure
mkdir -p data scripts packages config

# Create basic docker-compose.yml
cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  rstudio:
    image: rocker/rstudio:latest
    ports:
      - "8787:8787"
    volumes:
      - ./data:/home/rstudio/data
      - ./scripts:/home/rstudio/scripts
    environment:
      - PASSWORD=rstudio123
      - ROOT=TRUE
EOF

# Start RStudio Server
docker-compose up -d

# Access at http://localhost:8787 (rstudio/rstudio123)
```

## Docker Configurations

### Complete Production Setup
```yaml
# docker-compose.yml
version: '3.8'

services:
  rstudio:
    build:
      context: .
      dockerfile: Dockerfile.rstudio
    ports:
      - "8787:8787"
    volumes:
      - ./data:/home/rstudio/data
      - ./scripts:/home/rstudio/scripts
      - ./packages:/home/rstudio/packages
      - ./config:/home/rstudio/.config
      - rstudio-home:/home/rstudio
    environment:
      - PASSWORD=${RSTUDIO_PASSWORD:-analytics123}
      - ROOT=TRUE
      - USERID=1000
      - GROUPID=1000
      # Database connections
      - POSTGRES_HOST=postgres
      - POSTGRES_DB=analytics
      - POSTGRES_USER=rstudio
      - POSTGRES_PASSWORD=rstudio
      # Cloud storage
      - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
      - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
      - AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION:-us-east-1}
    depends_on:
      - postgres
      - redis
    networks:
      - rstudio-network
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 8G
        reservations:
          cpus: '2.0'
          memory: 4G

  postgres:
    image: postgres:13
    environment:
      - POSTGRES_DB=analytics
      - POSTGRES_USER=rstudio
      - POSTGRES_PASSWORD=rstudio
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./sql/init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    networks:
      - rstudio-network
    restart: unless-stopped

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    networks:
      - rstudio-network
    restart: unless-stopped

  shiny:
    build:
      context: .
      dockerfile: Dockerfile.shiny
    ports:
      - "3838:3838"
    volumes:
      - ./shiny-apps:/srv/shiny-server
      - ./data:/srv/data:ro
    depends_on:
      - postgres
    networks:
      - rstudio-network
    restart: unless-stopped

volumes:
  rstudio-home:
  postgres-data:
  redis-data:

networks:
  rstudio-network:
    driver: bridge
```

### Custom RStudio Dockerfile
```dockerfile
# Dockerfile.rstudio
FROM rocker/tidyverse:latest

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libpq-dev \
    libssl-dev \
    libcurl4-openssl-dev \
    libxml2-dev \
    libgdal-dev \
    libproj-dev \
    libudunits2-dev \
    libv8-dev \
    libprotobuf-dev \
    protobuf-compiler \
    libjq-dev \
    libnode-dev \
    cargo \
    && rm -rf /var/lib/apt/lists/*

# Install R packages for analytics
COPY install_packages.R /tmp/install_packages.R
RUN Rscript /tmp/install_packages.R

# Install additional tools
RUN apt-get update && apt-get install -y \
    git \
    vim \
    htop \
    tree \
    curl \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Configure Git
RUN git config --system user.name "RStudio User" && \
    git config --system user.email "rstudio@analytics.local"

# Create directories
RUN mkdir -p /home/rstudio/data \
             /home/rstudio/scripts \
             /home/rstudio/packages \
             /home/rstudio/projects

# Set up R environment
COPY .Rprofile /home/rstudio/.Rprofile
COPY .Renviron /home/rstudio/.Renviron

# Set ownership
RUN chown -R rstudio:rstudio /home/rstudio

# Expose port
EXPOSE 8787

CMD ["/init"]
```

### R Package Installation Script
```r
# install_packages.R
# Data manipulation and analysis
install.packages(c(
  "tidyverse", "data.table", "dtplyr", "dplyr", "tidyr",
  "readr", "readxl", "haven", "jsonlite", "xml2"
))

# Visualization
install.packages(c(
  "ggplot2", "plotly", "ggvis", "lattice", "grid",
  "RColorBrewer", "viridis", "scales", "ggthemes"
))

# Statistical modeling
install.packages(c(
  "caret", "randomForest", "e1071", "glmnet", "xgboost",
  "nnet", "rpart", "party", "survival", "mgcv"
))

# Time series
install.packages(c(
  "forecast", "prophet", "zoo", "xts", "lubridate",
  "tseries", "quantmod", "TTR"
))

# Database connectivity
install.packages(c(
  "DBI", "RPostgreSQL", "RMySQL", "RSQLite", "odbc",
  "mongolite", "redis", "pool"
))

# Cloud and web services
install.packages(c(
  "aws.s3", "googleCloudStorageR", "AzureStor",
  "httr", "rvest", "curl", "RCurl"
))

# Reporting and documentation
install.packages(c(
  "rmarkdown", "knitr", "bookdown", "blogdown",
  "flexdashboard", "DT", "formattable"
))

# Shiny and web applications
install.packages(c(
  "shiny", "shinydashboard", "shinyWidgets", "DT",
  "leaflet", "plotly", "htmlwidgets"
))

# Development tools
install.packages(c(
  "devtools", "roxygen2", "testthat", "usethis",
  "pkgdown", "covr", "lintr"
))

# Geospatial analysis
install.packages(c(
  "sf", "sp", "rgdal", "raster", "leaflet",
  "mapview", "tmap", "ggmap"
))

# Text mining and NLP
install.packages(c(
  "tm", "quanteda", "tidytext", "textdata",
  "wordcloud", "RColorBrewer"
))

# Parallel computing
install.packages(c(
  "parallel", "foreach", "doParallel", "future",
  "furrr", "multidplyr"
))

# Additional utilities
install.packages(c(
  "here", "fs", "glue", "stringr", "forcats",
  "purrr", "magrittr", "rlang"
))

# Install from GitHub (development versions)
if (!require(remotes)) install.packages("remotes")
remotes::install_github("rstudio/reticulate")
remotes::install_github("r-lib/conflicted")
```

### R Configuration Files
```r
# .Rprofile
# Set CRAN mirror
options(repos = c(CRAN = "https://cran.rstudio.com/"))

# Set default packages
options(defaultPackages = c(getOption("defaultPackages"), "tidyverse"))

# Configure parallel processing
options(mc.cores = parallel::detectCores())

# Set up database connections
if (file.exists(".Renviron")) {
  readRenviron(".Renviron")
}

# Helper functions
source_if_exists <- function(file) {
  if (file.exists(file)) source(file)
}

# Load custom functions
source_if_exists("scripts/utils.R")

# Welcome message
cat("Welcome to RStudio Analytics Environment!\n")
cat("Available cores:", parallel::detectCores(), "\n")
cat("R version:", R.version.string, "\n")
```

```bash
# .Renviron
# Database connections
POSTGRES_HOST=postgres
POSTGRES_DB=analytics
POSTGRES_USER=rstudio
POSTGRES_PASSWORD=rstudio

# Cloud storage
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=us-east-1

# API keys
GOOGLE_MAPS_API_KEY=your_google_maps_key
TWITTER_API_KEY=your_twitter_key

# Custom paths
DATA_PATH=/home/rstudio/data
SCRIPTS_PATH=/home/rstudio/scripts
```

## Database Integration

### Database Connection Functions
```r
# scripts/database_connections.R
library(DBI)
library(RPostgreSQL)
library(pool)

# PostgreSQL connection
get_postgres_connection <- function() {
  pool::dbPool(
    drv = RPostgreSQL::PostgreSQL(),
    host = Sys.getenv("POSTGRES_HOST"),
    dbname = Sys.getenv("POSTGRES_DB"),
    user = Sys.getenv("POSTGRES_USER"),
    password = Sys.getenv("POSTGRES_PASSWORD"),
    port = 5432
  )
}

# Query data from PostgreSQL
query_postgres <- function(query, params = NULL) {
  pool <- get_postgres_connection()
  on.exit(pool::poolClose(pool))
  
  if (is.null(params)) {
    DBI::dbGetQuery(pool, query)
  } else {
    DBI::dbGetQuery(pool, query, params = params)
  }
}

# Write data to PostgreSQL
write_postgres <- function(data, table_name, append = TRUE) {
  pool <- get_postgres_connection()
  on.exit(pool::poolClose(pool))
  
  DBI::dbWriteTable(
    pool, 
    table_name, 
    data, 
    append = append, 
    row.names = FALSE
  )
}

# MongoDB connection
library(mongolite)

get_mongo_connection <- function(collection, database = "analytics") {
  mongolite::mongo(
    collection = collection,
    db = database,
    url = "mongodb://mongo:27017"
  )
}

# Redis connection
library(redux)

get_redis_connection <- function() {
  redux::hiredis(host = "redis", port = 6379)
}
```

### Data Loading Examples
```r
# scripts/data_loading.R
library(tidyverse)
source("scripts/database_connections.R")

# Load data from PostgreSQL
load_sales_data <- function(start_date = NULL, end_date = NULL) {
  query <- "
    SELECT * FROM sales_data 
    WHERE date_column >= $1 AND date_column <= $2
  "
  
  if (is.null(start_date)) start_date <- Sys.Date() - 30
  if (is.null(end_date)) end_date <- Sys.Date()
  
  query_postgres(query, list(start_date, end_date))
}

# Load data from CSV with caching
load_csv_cached <- function(file_path, cache_key = NULL) {
  if (is.null(cache_key)) {
    cache_key <- paste0("csv_", tools::md5sum(file_path))
  }
  
  redis_conn <- get_redis_connection()
  
  # Check cache
  cached_data <- redis_conn$GET(cache_key)
  if (!is.null(cached_data)) {
    return(unserialize(cached_data))
  }
  
  # Load and cache data
  data <- readr::read_csv(file_path)
  redis_conn$SET(cache_key, serialize(data, NULL))
  redis_conn$EXPIRE(cache_key, 3600)  # 1 hour cache
  
  return(data)
}

# Load data from cloud storage
load_from_s3 <- function(bucket, key) {
  library(aws.s3)
  
  temp_file <- tempfile(fileext = ".csv")
  aws.s3::save_object(
    object = key,
    bucket = bucket,
    file = temp_file
  )
  
  data <- readr::read_csv(temp_file)
  unlink(temp_file)
  
  return(data)
}
```

## Package Management

### Package Development Environment
```dockerfile
# Dockerfile.dev
FROM rocker/tidyverse:latest

# Install development tools
RUN apt-get update && apt-get install -y \
    libgit2-dev \
    libssh2-1-dev \
    && rm -rf /var/lib/apt/lists/*

# Install development packages
RUN install2.r --error \
    devtools \
    roxygen2 \
    testthat \
    usethis \
    pkgdown \
    covr \
    lintr \
    styler \
    goodpractice

# Set up package development environment
RUN mkdir -p /home/rstudio/packages
WORKDIR /home/rstudio/packages

# Configure Git for package development
RUN git config --global user.name "Package Developer" && \
    git config --global user.email "dev@analytics.local"
```

### Package Installation Script
```r
# scripts/install_custom_packages.R
# Function to install packages with error handling
safe_install <- function(packages, repos = getOption("repos")) {
  for (pkg in packages) {
    tryCatch({
      if (!require(pkg, character.only = TRUE)) {
        install.packages(pkg, repos = repos)
        library(pkg, character.only = TRUE)
      }
      cat("✓ Successfully installed/loaded:", pkg, "\n")
    }, error = function(e) {
      cat("✗ Failed to install:", pkg, "-", e$message, "\n")
    })
  }
}

# Install packages by category
analytics_packages <- c(
  "tidyverse", "data.table", "dtplyr",
  "caret", "randomForest", "xgboost",
  "forecast", "prophet", "lubridate"
)

visualization_packages <- c(
  "ggplot2", "plotly", "ggvis",
  "leaflet", "DT", "formattable"
)

database_packages <- c(
  "DBI", "RPostgreSQL", "mongolite",
  "pool", "odbc"
)

# Install all packages
safe_install(analytics_packages)
safe_install(visualization_packages)
safe_install(database_packages)

# Create package summary
installed_packages <- installed.packages()[, c("Package", "Version")]
write.csv(installed_packages, "package_inventory.csv", row.names = FALSE)
```

## Security & Authentication

### HTTPS Configuration
```yaml
# docker-compose.secure.yml
version: '3.8'

services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - rstudio
    networks:
      - rstudio-network

  rstudio:
    # ... existing configuration
    ports: []  # Remove direct port exposure
    environment:
      - PASSWORD=${RSTUDIO_PASSWORD}
      - DISABLE_AUTH=false
    networks:
      - rstudio-network
```

### Authentication Configuration
```bash
# scripts/setup_auth.sh
#!/bin/bash

# Create user accounts
docker exec rstudio-container useradd -m -s /bin/bash analyst1
docker exec rstudio-container useradd -m -s /bin/bash analyst2

# Set passwords
echo "analyst1:password123" | docker exec -i rstudio-container chpasswd
echo "analyst2:password456" | docker exec -i rstudio-container chpasswd

# Add users to rstudio group
docker exec rstudio-container usermod -a -G rstudio analyst1
docker exec rstudio-container usermod -a -G rstudio analyst2

# Create user directories
docker exec rstudio-container mkdir -p /home/analyst1/projects
docker exec rstudio-container mkdir -p /home/analyst2/projects

# Set permissions
docker exec rstudio-container chown -R analyst1:analyst1 /home/analyst1
docker exec rstudio-container chown -R analyst2:analyst2 /home/analyst2
```

## Performance Optimization

### Resource Monitoring
```r
# scripts/performance_monitor.R
library(httr)
library(jsonlite)

# Monitor R session performance
monitor_r_session <- function() {
  # Memory usage
  memory_info <- list(
    used_mb = round(sum(gc()[, 2]) * 8 / 1024, 2),
    available_mb = round(memory.limit() - sum(gc()[, 2]) * 8 / 1024, 2),
    objects_count = length(ls(envir = .GlobalEnv))
  )
  
  # CPU usage (approximate)
  start_time <- Sys.time()
  system.time(for(i in 1:1000) sqrt(i))
  cpu_benchmark <- as.numeric(Sys.time() - start_time)
  
  # Session info
  session_info <- list(
    r_version = R.version.string,
    platform = R.version$platform,
    packages_loaded = length(.packages(TRUE)),
    working_directory = getwd()
  )
  
  # Combine all metrics
  metrics <- list(
    timestamp = Sys.time(),
    memory = memory_info,
    cpu_benchmark = cpu_benchmark,
    session = session_info
  )
  
  return(metrics)
}

# Log performance metrics
log_performance <- function(log_file = "performance.json") {
  metrics <- monitor_r_session()
  
  # Append to log file
  if (file.exists(log_file)) {
    existing_logs <- jsonlite::fromJSON(log_file)
    all_logs <- rbind(existing_logs, metrics)
  } else {
    all_logs <- metrics
  }
  
  jsonlite::write_json(all_logs, log_file, pretty = TRUE)
}

# Schedule performance monitoring
schedule_monitoring <- function(interval_minutes = 15) {
  while (TRUE) {
    log_performance()
    Sys.sleep(interval_minutes * 60)
  }
}
```

### Caching Strategies
```r
# scripts/caching_utils.R
library(memoise)
library(redux)

# Redis-based caching
redis_cache <- function(key, expr, ttl = 3600) {
  redis_conn <- get_redis_connection()
  
  # Check cache
  cached_result <- redis_conn$GET(key)
  if (!is.null(cached_result)) {
    return(unserialize(cached_result))
  }
  
  # Compute and cache result
  result <- force(expr)
  redis_conn$SET(key, serialize(result, NULL))
  redis_conn$EXPIRE(key, ttl)
  
  return(result)
}

# Memoization for expensive computations
expensive_computation <- function(data, params) {
  # Simulate expensive computation
  Sys.sleep(2)
  return(summary(data))
}

# Create memoised version
memoised_computation <- memoise::memoise(expensive_computation)

# File-based caching
cache_to_file <- function(expr, cache_file, max_age_hours = 24) {
  if (file.exists(cache_file)) {
    file_age <- difftime(Sys.time(), file.mtime(cache_file), units = "hours")
    if (file_age < max_age_hours) {
      return(readRDS(cache_file))
    }
  }
  
  result <- force(expr)
  saveRDS(result, cache_file)
  return(result)
}
```

## Monitoring & Logging

### Application Monitoring
```r
# scripts/monitoring.R
library(httr)
library(jsonlite)

# Health check function
health_check <- function() {
  checks <- list()
  
  # Database connectivity
  checks$database <- tryCatch({
    conn <- get_postgres_connection()
    DBI::dbGetQuery(conn, "SELECT 1")
    pool::poolClose(conn)
    "healthy"
  }, error = function(e) paste("unhealthy:", e$message))
  
  # Redis connectivity
  checks$redis <- tryCatch({
    redis_conn <- get_redis_connection()
    redis_conn$PING()
    "healthy"
  }, error = function(e) paste("unhealthy:", e$message))
  
  # Memory usage
  memory_usage <- sum(gc()[, 2]) * 8 / 1024  # MB
  checks$memory <- if (memory_usage < 1000) "healthy" else "warning"
  
  # Disk space
  disk_info <- system("df -h /home/rstudio", intern = TRUE)
  checks$disk <- "healthy"  # Simplified check
  
  return(checks)
}

# Log application events
log_event <- function(level, message, details = NULL) {
  log_entry <- list(
    timestamp = Sys.time(),
    level = level,
    message = message,
    details = details,
    session_id = Sys.getpid()
  )
  
  # Write to log file
  log_file <- "application.log"
  cat(jsonlite::toJSON(log_entry), "\n", file = log_file, append = TRUE)
  
  # Also print to console for development
  cat(sprintf("[%s] %s: %s\n", Sys.time(), level, message))
}

# Error handling wrapper
with_error_logging <- function(expr, context = "unknown") {
  tryCatch({
    result <- force(expr)
    log_event("INFO", paste("Successfully executed:", context))
    return(result)
  }, error = function(e) {
    log_event("ERROR", paste("Error in:", context), list(error = e$message))
    stop(e)
  })
}
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Package Installation Issues
```r
# Diagnose package installation problems
diagnose_packages <- function() {
  # Check R version
  cat("R Version:", R.version.string, "\n")
  
  # Check repositories
  cat("Repositories:\n")
  print(getOption("repos"))
  
  # Check library paths
  cat("Library paths:\n")
  print(.libPaths())
  
  # Check system dependencies
  system_deps <- c("libcurl4-openssl-dev", "libssl-dev", "libxml2-dev")
  for (dep in system_deps) {
    status <- system(paste("dpkg -l", dep), ignore.stdout = TRUE)
    cat(dep, ":", if (status == 0) "installed" else "missing", "\n")
  }
}

# Fix common package issues
fix_package_issues <- function() {
  # Update package database
  update.packages(ask = FALSE)
  
  # Reinstall problematic packages
  problematic_packages <- c("curl", "httr", "xml2")
  for (pkg in problematic_packages) {
    remove.packages(pkg)
    install.packages(pkg)
  }
}
```

#### 2. Database Connection Issues
```r
# Test database connections
test_database_connections <- function() {
  # PostgreSQL
  tryCatch({
    conn <- get_postgres_connection()
    result <- DBI::dbGetQuery(conn, "SELECT version()")
    pool::poolClose(conn)
    cat("PostgreSQL: Connected successfully\n")
    cat("Version:", result$version, "\n")
  }, error = function(e) {
    cat("PostgreSQL: Connection failed -", e$message, "\n")
  })
  
  # Redis
  tryCatch({
    redis_conn <- get_redis_connection()
    redis_conn$PING()
    cat("Redis: Connected successfully\n")
  }, error = function(e) {
    cat("Redis: Connection failed -", e$message, "\n")
  })
}
```

#### 3. Performance Issues
```r
# Performance diagnostics
performance_diagnostics <- function() {
  # Memory usage
  gc_info <- gc()
  cat("Memory usage:\n")
  print(gc_info)
  
  # Object sizes in global environment
  obj_sizes <- sapply(ls(envir = .GlobalEnv), function(x) {
    object.size(get(x, envir = .GlobalEnv))
  })
  
  cat("\nLargest objects in global environment:\n")
  print(sort(obj_sizes, decreasing = TRUE)[1:10])
  
  # Session info
  cat("\nSession info:\n")
  print(sessionInfo())
}

# Clean up environment
cleanup_environment <- function() {
  # Remove large objects
  obj_sizes <- sapply(ls(envir = .GlobalEnv), function(x) {
    object.size(get(x, envir = .GlobalEnv))
  })
  
  large_objects <- names(obj_sizes[obj_sizes > 100 * 1024^2])  # > 100MB
  
  if (length(large_objects) > 0) {
    cat("Removing large objects:", paste(large_objects, collapse = ", "), "\n")
    rm(list = large_objects, envir = .GlobalEnv)
  }
  
  # Force garbage collection
  gc()
}
```

### Health Check Scripts
```bash
#!/bin/bash
# scripts/health_check.sh

echo "=== RStudio Server Health Check ==="

# Check container status
echo "Container status:"
docker-compose ps

# Check RStudio accessibility
echo "Checking RStudio accessibility..."
curl -f http://localhost:8787 > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✓ RStudio Server is accessible"
else
    echo "✗ RStudio Server is not accessible"
fi

# Check database connectivity
echo "Checking database connectivity..."
docker-compose exec postgres pg_isready -U rstudio > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✓ PostgreSQL is ready"
else
    echo "✗ PostgreSQL is not ready"
fi

# Check Redis connectivity
echo "Checking Redis connectivity..."
docker-compose exec redis redis-cli ping > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✓ Redis is responding"
else
    echo "✗ Redis is not responding"
fi

# Check disk space
echo "Disk space usage:"
df -h

# Check memory usage
echo "Memory usage:"
free -h

# Check for errors in logs
echo "Recent errors in logs:"
docker-compose logs --tail=20 rstudio | grep -i error || echo "No recent errors found"
```

## Best Practices

### Development Workflow
1. **Use version control** with Git for all projects
2. **Organize projects** with consistent directory structure
3. **Document code** with roxygen2 comments
4. **Test functions** with testthat package
5. **Use renv** for package management in projects

### Performance Best Practices
1. **Monitor memory usage** regularly
2. **Use data.table** for large datasets
3. **Implement caching** for expensive computations
4. **Profile code** to identify bottlenecks
5. **Use parallel processing** when appropriate

### Security Best Practices
1. **Use strong passwords** and change defaults
2. **Enable HTTPS** for production deployments
3. **Limit user permissions** appropriately
4. **Regular security updates** for base images
5. **Secure database connections** with proper credentials

---

*This RStudio Server Docker deployment provides a comprehensive R analytics environment suitable for data science teams and individual analysts. The configuration supports scalability, security, and integration with modern data infrastructure.* 
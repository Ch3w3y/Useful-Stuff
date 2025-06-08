# R Project Management and Data Serialization for Epidemiology

## Table of Contents

1. [Data Serialization with .Rds Files](#data-serialization-with-rds-files)
2. [Project Portability and Reproducibility](#project-portability-and-reproducibility)
3. [Git Workflows for R Projects](#git-workflows-for-r-projects)
4. [Example Git Repositories](#example-git-repositories)
5. [Advanced Data Management](#advanced-data-management)
6. [Project Templates and Scaffolding](#project-templates-and-scaffolding)
7. [Dependency Management](#dependency-management)
8. [Testing and Validation](#testing-and-validation)
9. [Documentation Standards](#documentation-standards)
10. [Collaboration Best Practices](#collaboration-best-practices)
11. [Data Security and Privacy](#data-security-and-privacy)
12. [Performance Optimization](#performance-optimization)

## Data Serialization with .Rds Files

### Understanding .Rds Files

**.Rds files** are R's native binary format for storing single R objects efficiently and preserving all R-specific attributes.

#### Advantages of .Rds Format
1. **Efficiency**: Compressed binary format, faster read/write
2. **Preservation**: Maintains data types, attributes, factors levels
3. **Portability**: Works across R versions and platforms
4. **Compression**: Automatic compression reduces file size
5. **Single Objects**: Clean, predictable data structure

#### .Rds vs Other Formats

| Format | .Rds | .RData/.rda | CSV | Parquet | HDF5 |
|--------|------|-------------|-----|---------|------|
| **R Objects** | Single | Multiple | Tabular only | Tabular only | Multiple |
| **Compression** | Built-in | Built-in | None | Built-in | Built-in |
| **Speed** | Fast | Fast | Slow | Very Fast | Fast |
| **Attributes** | Preserved | Preserved | Lost | Limited | Limited |
| **Cross-platform** | Yes | Yes | Yes | Yes | Yes |
| **Size** | Small | Small | Large | Small | Medium |

### Practical .Rds Usage in Epidemiology

#### Saving Different Data Types
```r
# Load packages
library(dplyr)
library(lubridate)

# Save cleaned surveillance data
surveillance_clean <- raw_surveillance %>%
  filter(!is.na(case_id)) %>%
  mutate(
    report_date = ymd(report_date),
    age_group = factor(age_group, 
                      levels = c("0-17", "18-64", "65+"),
                      ordered = TRUE),
    severity = factor(severity,
                     levels = c("Mild", "Moderate", "Severe", "Critical"))
  )

# Save main dataset
saveRDS(surveillance_clean, "data/processed/surveillance_clean.rds")

# Save analysis results
case_control_results <- list(
  model = glm(outcome ~ exposure + age + sex, family = binomial, data = study_data),
  odds_ratios = or_table,
  confounders = confounding_analysis,
  diagnostics = model_diagnostics
)
saveRDS(case_control_results, "output/models/case_control_results.rds")

# Save summary statistics
weekly_summary <- surveillance_clean %>%
  group_by(week = floor_date(report_date, "week")) %>%
  summarise(
    total_cases = n(),
    hospitalizations = sum(hospitalized, na.rm = TRUE),
    deaths = sum(died, na.rm = TRUE),
    attack_rate = total_cases / population * 1000,
    .groups = "drop"
  )
saveRDS(weekly_summary, "data/processed/weekly_summary.rds")

# Save geographic data with spatial attributes
county_cases <- surveillance_clean %>%
  group_by(county_fips) %>%
  summarise(cases = n(), .groups = "drop") %>%
  left_join(county_shapefile, by = "county_fips")
saveRDS(county_cases, "data/processed/county_cases_spatial.rds")
```

#### Loading and Verifying .Rds Files
```r
# Safe loading with verification
load_and_verify_rds <- function(filepath, expected_class = NULL) {
  if (!file.exists(filepath)) {
    stop(paste("File not found:", filepath))
  }
  
  # Check file age
  file_age <- difftime(Sys.time(), file.mtime(filepath), units = "days")
  if (file_age > 7) {
    warning(paste("File is", round(file_age, 1), "days old"))
  }
  
  # Load data
  data <- readRDS(filepath)
  
  # Verify expected class
  if (!is.null(expected_class) && !inherits(data, expected_class)) {
    warning(paste("Expected class", expected_class, "but got", class(data)[1]))
  }
  
  # Basic validation
  cat("Loaded:", filepath, "\n")
  cat("Class:", class(data)[1], "\n")
  if (is.data.frame(data)) {
    cat("Dimensions:", nrow(data), "rows,", ncol(data), "columns\n")
    cat("Memory usage:", format(object.size(data), units = "MB"), "\n")
  }
  
  return(data)
}

# Usage examples
surveillance_data <- load_and_verify_rds(
  "data/processed/surveillance_clean.rds", 
  expected_class = "data.frame"
)

model_results <- load_and_verify_rds(
  "output/models/case_control_results.rds",
  expected_class = "list"
)
```

### Advanced .Rds Strategies

#### Version Control for Data
```r
# Versioned data saving
save_versioned_rds <- function(object, base_path, version = NULL) {
  if (is.null(version)) {
    version <- format(Sys.time(), "%Y%m%d_%H%M%S")
  }
  
  # Create versioned filename
  file_ext <- tools::file_ext(base_path)
  file_base <- tools::file_path_sans_ext(base_path)
  versioned_path <- paste0(file_base, "_v", version, ".rds")
  
  # Save with metadata
  metadata <- list(
    data = object,
    created = Sys.time(),
    r_version = R.version.string,
    packages = sessionInfo()$otherPkgs,
    user = Sys.getenv("USER"),
    hash = digest::digest(object)
  )
  
  saveRDS(metadata, versioned_path)
  
  # Create/update symlink to latest
  if (file.exists(base_path)) {
    file.remove(base_path)
  }
  file.symlink(basename(versioned_path), base_path)
  
  cat("Saved version:", version, "\n")
  cat("Path:", versioned_path, "\n")
  
  return(versioned_path)
}

# Usage
save_versioned_rds(
  surveillance_clean, 
  "data/processed/surveillance_clean.rds",
  version = "2024_03_15"
)
```

#### Compressed and Encrypted Storage
```r
# High compression for large datasets
save_compressed_rds <- function(object, filepath, compress_level = 9) {
  # Use maximum compression
  saveRDS(object, filepath, compress = "gzip", compression_level = compress_level)
  
  # Report compression ratio
  temp_uncompressed <- tempfile()
  saveRDS(object, temp_uncompressed, compress = FALSE)
  
  original_size <- file.size(temp_uncompressed)
  compressed_size <- file.size(filepath)
  ratio <- round((1 - compressed_size/original_size) * 100, 1)
  
  cat("Compression ratio:", ratio, "%\n")
  cat("Original size:", format(original_size, units = "MB"), "\n")
  cat("Compressed size:", format(compressed_size, units = "MB"), "\n")
  
  file.remove(temp_uncompressed)
}

# Encrypted storage for sensitive data
library(cyphr)

# Create encryption key (do this once, store securely)
key <- cyphr::key_sodium(sodium::keygen())

# Save encrypted data
save_encrypted_rds <- function(object, filepath, key) {
  encrypted_data <- cyphr::encrypt_object(object, key)
  saveRDS(encrypted_data, filepath)
  cat("Saved encrypted data to:", filepath, "\n")
}

# Load encrypted data
load_encrypted_rds <- function(filepath, key) {
  encrypted_data <- readRDS(filepath)
  decrypted_object <- cyphr::decrypt_object(encrypted_data, key)
  return(decrypted_object)
}

# Usage for PHI data
save_encrypted_rds(patient_data, "data/secure/patient_data_encrypted.rds", key)
```

### Data Pipeline with .Rds
```r
# epidemiology_data_pipeline.R
library(here)
library(digest)

# Pipeline configuration
pipeline_config <- list(
  raw_data_path = here("data", "raw"),
  processed_data_path = here("data", "processed"),
  cache_path = here("cache"),
  force_refresh = FALSE
)

# Smart caching system
cached_operation <- function(operation_func, cache_key, force_refresh = FALSE) {
  cache_file <- file.path(pipeline_config$cache_path, paste0(cache_key, ".rds"))
  
  # Check if cache exists and is valid
  if (!force_refresh && file.exists(cache_file)) {
    cache_age <- difftime(Sys.time(), file.mtime(cache_file), units = "hours")
    if (cache_age < 24) {  # Cache valid for 24 hours
      cat("Loading from cache:", cache_key, "\n")
      return(readRDS(cache_file))
    }
  }
  
  # Execute operation and cache result
  cat("Computing:", cache_key, "\n")
  result <- operation_func()
  
  # Ensure cache directory exists
  if (!dir.exists(pipeline_config$cache_path)) {
    dir.create(pipeline_config$cache_path, recursive = TRUE)
  }
  
  saveRDS(result, cache_file)
  return(result)
}

# Data processing functions
import_surveillance_data <- function() {
  # Import and basic validation
  raw_files <- list.files(pipeline_config$raw_data_path, 
                         pattern = "surveillance.*\\.csv$", 
                         full.names = TRUE)
  
  surveillance_data <- map_dfr(raw_files, read_csv) %>%
    filter(!is.na(case_id)) %>%
    arrange(report_date)
  
  return(surveillance_data)
}

clean_surveillance_data <- function() {
  raw_data <- cached_operation(
    import_surveillance_data, 
    "raw_surveillance_data",
    force_refresh = pipeline_config$force_refresh
  )
  
  cleaned_data <- raw_data %>%
    mutate(
      report_date = lubridate::ymd(report_date),
      age_group = case_when(
        age < 18 ~ "0-17",
        age < 65 ~ "18-64",
        TRUE ~ "65+"
      ),
      age_group = factor(age_group, levels = c("0-17", "18-64", "65+"))
    ) %>%
    filter(
      !is.na(report_date),
      report_date >= as.Date("2020-01-01"),
      report_date <= Sys.Date()
    )
  
  return(cleaned_data)
}

# Execute pipeline
main_pipeline <- function() {
  # Clean data
  surveillance_clean <- cached_operation(
    clean_surveillance_data,
    "surveillance_clean"
  )
  
  # Generate summary statistics
  summary_stats <- cached_operation(
    function() {
      surveillance_clean %>%
        group_by(week = lubridate::floor_date(report_date, "week")) %>%
        summarise(
          cases = n(),
          hospitalizations = sum(hospitalized, na.rm = TRUE),
          deaths = sum(died, na.rm = TRUE),
          .groups = "drop"
        )
    },
    "weekly_summary_stats"
  )
  
  # Save final processed data
  saveRDS(surveillance_clean, 
          file.path(pipeline_config$processed_data_path, "surveillance_clean.rds"))
  saveRDS(summary_stats,
          file.path(pipeline_config$processed_data_path, "weekly_summary.rds"))
  
  cat("Pipeline completed successfully\n")
  cat("Processed", nrow(surveillance_clean), "records\n")
}

# Run pipeline
main_pipeline()
```

## Project Portability and Reproducibility

### Creating Portable R Projects

#### Essential Files for Portability
```
epidemiology-project/
├── README.md                    # Project description and setup
├── epidemiology-project.Rproj  # RStudio project file
├── renv.lock                   # Package versions (renv)
├── .Renviron                   # Environment variables (template)
├── .gitignore                  # Git ignore patterns
├── .Rbuildignore              # Package build ignore patterns
├── DESCRIPTION                # Package metadata (optional)
├── setup.R                    # Project setup script
└── docker/                    # Docker configuration
    ├── Dockerfile
    └── docker-compose.yml
```

#### Setup Script for Reproducibility
```r
# setup.R - Project initialization script
cat("Setting up Epidemiology Analysis Project\n")
cat("========================================\n\n")

# Check R version
required_r_version <- "4.2.0"
current_r_version <- paste(R.version$major, R.version$minor, sep = ".")

if (compareVersion(current_r_version, required_r_version) < 0) {
  stop(paste("R version", required_r_version, "or higher required. Current:", current_r_version))
}

# Create directory structure
dirs_to_create <- c(
  "data/raw",
  "data/processed", 
  "data/external",
  "R/functions",
  "output/figures",
  "output/tables", 
  "output/models",
  "reports/weekly",
  "reports/monthly",
  "docs",
  "tests",
  "logs",
  "cache"
)

for (dir in dirs_to_create) {
  if (!dir.exists(dir)) {
    dir.create(dir, recursive = TRUE)
    cat("Created directory:", dir, "\n")
  }
}

# Initialize renv if not present
if (!file.exists("renv.lock")) {
  cat("Initializing renv...\n")
  if (!requireNamespace("renv", quietly = TRUE)) {
    install.packages("renv")
  }
  renv::init()
}

# Install required packages
required_packages <- c(
  # Data manipulation
  "dplyr", "tidyr", "lubridate", "stringr",
  
  # Analysis
  "broom", "survival", "tableone", "MatchIt", "epiR",
  
  # Visualization  
  "ggplot2", "plotly", "leaflet", "DT",
  
  # Reporting
  "rmarkdown", "knitr", "flexdashboard",
  
  # Project management
  "here", "usethis", "devtools",
  
  # Data I/O
  "readr", "readxl", "haven", "jsonlite",
  
  # Spatial analysis
  "sf", "tmap"
)

# Check and install missing packages
missing_packages <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]

if (length(missing_packages) > 0) {
  cat("Installing missing packages:", paste(missing_packages, collapse = ", "), "\n")
  install.packages(missing_packages)
}

# Load key packages to verify installation
suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(here)
})

# Create .Renviron template if it doesn't exist
if (!file.exists(".Renviron")) {
  renviron_content <- "# Environment variables for epidemiology project
# Copy this file to .Renviron and fill in your values
# Database connection
DB_HOST=localhost
DB_PORT=5432
DB_NAME=epidemiology
DB_USER=your_username
DB_PASSWORD=your_password

# API keys (if needed)
GEOCODING_API_KEY=your_geocoding_key
WEATHER_API_KEY=your_weather_key

# Project settings
PROJECT_NAME=epidemiology-analysis
CONTACT_EMAIL=your.email@institution.edu
"
  writeLines(renviron_content, ".Renviron.template")
  cat("Created .Renviron.template - copy to .Renviron and customize\n")
}

# Create gitignore if it doesn't exist
if (!file.exists(".gitignore")) {
  gitignore_content <- "# R gitignore patterns
.Rhistory
.RData
.Ruserdata
.Renviron
*.Rproj.user/

# Data files (add specific patterns as needed)
data/raw/*
!data/raw/.gitkeep
data/secure/
*.rds
*.csv
*.xlsx

# Output files
output/
reports/*.html
reports/*.pdf
reports/*.docx

# Logs and temporary files
logs/
cache/
temp/
*.log
*.tmp

# OS specific
.DS_Store
Thumbs.db

# IDE specific
.vscode/
.idea/

# Sensitive files
*credentials*
*secrets*
*key*
"
  writeLines(gitignore_content, ".gitignore")
  cat("Created .gitignore\n")
}

# Create README template
if (!file.exists("README.md")) {
  readme_content <- paste0("# ", basename(getwd()), "

## Epidemiological Analysis Project

### Overview
Brief description of the study and analysis objectives.

### Setup
1. Clone this repository
2. Open `", basename(getwd()), ".Rproj` in RStudio
3. Run `source('setup.R')` to initialize the project
4. Copy `.Renviron.template` to `.Renviron` and configure

### Project Structure
- `data/` - Data files (raw, processed, external)
- `R/` - R scripts and functions
- `reports/` - RMarkdown reports
- `output/` - Analysis outputs (figures, tables, models)
- `docs/` - Documentation

### Data Sources
- List your data sources here
- Include access instructions and restrictions

### Analysis Plan
- Describe your analysis approach
- Link to detailed statistical analysis plan

### Contact
- Principal Investigator: [Name] ([email])
- Analyst: [Name] ([email])

### Last Updated
", Sys.Date(), "
")
  writeLines(readme_content, "README.md")
  cat("Created README.md template\n")
}

# Create sample configuration file
config_content <- "# config.R - Project configuration
# Data paths
DATA_RAW <- here::here('data', 'raw')
DATA_PROCESSED <- here::here('data', 'processed')  
DATA_EXTERNAL <- here::here('data', 'external')

# Output paths
OUTPUT_FIGURES <- here::here('output', 'figures')
OUTPUT_TABLES <- here::here('output', 'tables')
OUTPUT_MODELS <- here::here('output', 'models')

# Analysis parameters
ALPHA_LEVEL <- 0.05
CONFIDENCE_LEVEL <- 0.95
MIN_CELL_SIZE <- 5

# Study parameters
STUDY_START_DATE <- as.Date('2020-01-01')
STUDY_END_DATE <- as.Date('2023-12-31')

# Color palettes for plots
EPI_COLORS <- c(
  'primary' = '#2E86AB',
  'secondary' = '#A23B72', 
  'accent' = '#F18F01',
  'warning' = '#C73E1D',
  'success' = '#7CB342'
)
"

if (!file.exists("R/config.R")) {
  writeLines(config_content, "R/config.R")
  cat("Created R/config.R\n")
}

cat("\nProject setup completed successfully!\n")
cat("Next steps:\n")
cat("1. Customize .Renviron with your settings\n")
cat("2. Add your data to data/raw/\n") 
cat("3. Begin analysis in R/01_data_import.R\n")
```

### Docker Configuration for Reproducibility

#### Dockerfile for R Epidemiology Environment
```dockerfile
# docker/Dockerfile
FROM rocker/r-ver:4.3.2

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libgdal-dev \
    libproj-dev \
    libgeos-dev \
    libudunits2-dev \
    pandoc \
    pandoc-citeproc \
    && rm -rf /var/lib/apt/lists/*

# Install R packages
RUN install2.r --error \
    dplyr \
    ggplot2 \
    rmarkdown \
    tidyr \
    lubridate \
    stringr \
    broom \
    survival \
    tableone \
    epiR \
    epitools \
    sf \
    leaflet \
    plotly \
    DT \
    knitr \
    flexdashboard \
    here \
    renv

# Create app directory
WORKDIR /app

# Copy project files
COPY . /app/

# Install project-specific packages
RUN R -e "renv::restore()"

# Expose port for Shiny apps
EXPOSE 3838

# Default command
CMD ["R"]
```

#### Docker Compose for Development
```yaml
# docker/docker-compose.yml
version: '3.8'

services:
  rstudio:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    ports:
      - "8787:8787"
    environment:
      - PASSWORD=epidemiology
      - ROOT=TRUE
    volumes:
      - ..:/home/rstudio/project
      - ../data:/home/rstudio/project/data
      - ../output:/home/rstudio/project/output
    command: /init

  postgres:
    image: postgres:13
    environment:
      POSTGRES_DB: epidemiology
      POSTGRES_USER: epi_user
      POSTGRES_PASSWORD: epi_password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql

  redis:
    image: redis:6-alpine
    ports:
      - "6379:6379"

volumes:
  postgres_data:
```

### Environment Management

#### Using renv for Package Management
```r
# Initialize renv in project
renv::init()

# Install packages (automatically tracked)
install.packages(c("dplyr", "ggplot2", "survival"))

# Take snapshot of current state
renv::snapshot()

# Check for updates
renv::update()

# Restore environment on different machine
renv::restore()

# Check package status
renv::status()

# Create portable lockfile
renv::snapshot(type = "simple")
```

#### Custom renv Configuration
```r
# renv/settings.dcf
auto.snapshot: TRUE
snapshot.type: simple
use.cache: TRUE
package.dependency.fields: Imports, Depends, LinkingTo
```

## Git Workflows for R Projects

### Git Setup for Epidemiology Projects

#### .gitignore for R Projects
```gitignore
# R-specific gitignore
.Rhistory
.RData
.Ruserdata
.Renviron
*.Rproj.user/

# Data files (customize based on your needs)
data/raw/*
!data/raw/README.md
!data/raw/.gitkeep
data/processed/*.rds
data/secure/
*.csv
*.xlsx
*.sav
*.dta

# PHI and sensitive data
*patient*
*phi*
*hipaa*
*confidential*
*sensitive*

# Output files
output/figures/*
!output/figures/README.md
output/tables/*
!output/tables/README.md
output/models/*
!output/models/README.md

# Generated reports
reports/*.html
reports/*.pdf  
reports/*.docx
!reports/*.Rmd

# Logs and cache
logs/
cache/
temp/
*.log
*.tmp

# Package development
src/*.o
src/*.so
src/*.dll

# IDE files
.vscode/
.idea/
.sublime-*

# OS files
.DS_Store
Thumbs.db
*.swp
*.swo
*~

# Credentials and keys
.env
.env.local
*credential*
*secret*
*key*
*token*
```

#### Git Configuration for Epidemiology
```bash
# Configure Git for epidemiology work
git config --global user.name "Your Name"
git config --global user.email "your.email@institution.edu"

# Set up GPG signing for sensitive work
git config --global user.signingkey YOUR_GPG_KEY
git config --global commit.gpgsign true

# Configure merge strategy for data conflicts
git config --global merge.ours.driver true

# Set up Git LFS for large files
git lfs install
git lfs track "*.rds"
git lfs track "data/external/*"
```

### Branching Strategy for Analysis Projects

#### GitFlow for Epidemiological Studies
```
main (production)
  ↑
develop (integration)
  ↑
feature/data-cleaning
feature/descriptive-analysis  
feature/case-control-analysis
feature/survival-analysis
hotfix/critical-bug-fix
release/v1.0.0
```

#### Branch Management Commands
```bash
# Start new analysis feature
git checkout develop
git pull origin develop
git checkout -b feature/case-control-analysis

# Work on analysis
git add R/case_control_analysis.R
git commit -m "Add case-control analysis functions

- Implement conditional logistic regression
- Add matching algorithm
- Create summary tables"

# Push feature branch
git push -u origin feature/case-control-analysis

# Create pull request (via GitHub/GitLab interface)

# After review and merge, clean up
git checkout develop
git pull origin develop
git branch -d feature/case-control-analysis
```

### Commit Message Conventions

#### Structured Commit Messages for Epidemiology
```bash
# Format: type(scope): description
#
# Types:
# feat: new analysis or feature
# fix: bug fix in analysis
# data: data updates or changes
# docs: documentation changes
# style: formatting, no logic change
# refactor: code restructuring
# test: adding tests
# chore: maintenance tasks

# Examples:
git commit -m "feat(analysis): add case-control logistic regression

- Implement conditional logistic regression for matched data
- Add diagnostic plots for model assumptions
- Calculate adjusted odds ratios with 95% CI"

git commit -m "data(surveillance): update weekly surveillance data

- Add week ending 2024-03-15
- 247 new cases reported  
- 12 hospitalizations, 2 deaths"

git commit -m "fix(cleaning): correct age group categorization

- Fix boundary condition for 65+ age group
- Update factor levels to maintain ordering
- Affects 23 cases in dataset"

git commit -m "docs(methods): add statistical analysis plan

- Detail primary and secondary analyses
- Specify inclusion/exclusion criteria
- Define outcome measures and endpoints"
```

### Collaborative Git Workflows

#### Code Review Process
```bash
# 1. Create feature branch
git checkout -b feature/spatial-analysis

# 2. Make changes and commit
git add R/spatial_functions.R
git commit -m "feat(spatial): add disease clustering analysis"

# 3. Push and create pull request
git push -u origin feature/spatial-analysis

# 4. Code review checklist (PR template):
```

**Pull Request Template**:
```markdown
## Analysis Description
Brief description of the analysis or changes made.

## Type of Change
- [ ] New analysis feature
- [ ] Bug fix
- [ ] Data update
- [ ] Documentation
- [ ] Refactoring

## Testing Checklist
- [ ] Code runs without errors
- [ ] Results are reproducible
- [ ] Statistical methods are appropriate
- [ ] Assumptions are tested/validated
- [ ] Output is properly documented

## Validation
- [ ] Cross-checked calculations manually
- [ ] Compared with previous results (if applicable)
- [ ] Verified data quality checks
- [ ] Reviewed for potential biases

## Documentation
- [ ] Code is well-commented
- [ ] Analysis plan is updated
- [ ] Results are interpreted correctly
- [ ] Limitations are acknowledged

## Data Privacy
- [ ] No PHI exposed in code
- [ ] Data handling complies with IRB
- [ ] Output is de-identified
- [ ] Sensitive files are not committed

## Reviewer Notes
Any specific areas that need attention or questions for reviewers.
```

#### Handling Data Conflicts
```bash
# Set up merge driver for data files
echo "*.rds merge=ours" >> .gitattributes
echo "data/processed/* merge=ours" >> .gitattributes

# Configure the merge driver
git config merge.ours.driver true

# For conflicts in analysis results
git config merge.rds.driver 'R --vanilla -e "
  args <- commandArgs(TRUE)
  local <- readRDS(args[2])
  remote <- readRDS(args[3])  
  # Custom merge logic here
  merged <- merge_analysis_results(local, remote)
  saveRDS(merged, args[1])
  "'
```

## Example Git Repositories

### Repository Templates

#### Template 1: Outbreak Investigation
```
outbreak-investigation-template/
├── README.md
├── .gitignore  
├── .gitattributes
├── outbreak-investigation.Rproj
├── renv.lock
├── setup.R
├── data/
│   ├── raw/
│   │   ├── linelist.csv
│   │   ├── laboratory.csv
│   │   └── environmental.csv
│   ├── processed/
│   └── external/
│       ├── population_data.csv
│       └── geographic_boundaries.shp
├── R/
│   ├── 01_data_import.R
│   ├── 02_data_cleaning.R
│   ├── 03_descriptive_analysis.R
│   ├── 04_epidemic_curve.R
│   ├── 05_case_definition.R
│   ├── 06_hypothesis_generation.R
│   ├── 07_analytical_study.R
│   ├── 08_environmental_analysis.R
│   └── functions/
│       ├── outbreak_functions.R
│       ├── epi_curve_functions.R
│       └── mapping_functions.R
├── reports/
│   ├── situation_report.Rmd
│   ├── preliminary_report.Rmd
│   ├── final_report.Rmd
│   └── templates/
├── output/
│   ├── figures/
│   ├── tables/
│   └── maps/
└── docs/
    ├── investigation_protocol.md
    ├── case_definition.md
    └── data_dictionary.md
```

#### Template 2: Surveillance System
```
surveillance-system/
├── README.md
├── .github/
│   └── workflows/
│       ├── daily-update.yml
│       ├── weekly-report.yml
│       └── quality-checks.yml
├── surveillance-system.Rproj
├── renv.lock
├── config/
│   ├── database.yml
│   ├── surveillance_config.R
│   └── alert_thresholds.yml
├── data/
│   ├── raw/
│   ├── processed/
│   ├── historical/
│   └── external/
├── R/
│   ├── data_processing/
│   │   ├── import_functions.R
│   │   ├── validation_functions.R
│   │   └── cleaning_functions.R
│   ├── analysis/
│   │   ├── descriptive_analysis.R
│   │   ├── trend_analysis.R
│   │   ├── outbreak_detection.R
│   │   └── forecasting.R
│   ├── visualization/
│   │   ├── dashboard_plots.R
│   │   ├── map_functions.R
│   │   └── alert_visualizations.R
│   └── automation/
│       ├── daily_pipeline.R
│       ├── weekly_reports.R
│       └── alert_system.R
├── dashboards/
│   ├── surveillance_dashboard.Rmd
│   ├── executive_dashboard.Rmd
│   └── public_dashboard.Rmd
├── reports/
│   ├── automated/
│   │   ├── daily_summary.Rmd
│   │   ├── weekly_report.Rmd
│   │   └── monthly_report.Rmd
│   └── ad_hoc/
├── tests/
│   ├── test_data_validation.R
│   ├── test_analysis_functions.R
│   └── test_alert_system.R
└── deployment/
    ├── docker/
    ├── kubernetes/
    └── scripts/
```

#### Template 3: Cohort Study Analysis
```
cohort-study-analysis/
├── README.md
├── CITATION.cff
├── LICENSE
├── cohort-study.Rproj
├── renv.lock
├── _targets.R              # targets pipeline
├── data/
│   ├── raw/
│   │   ├── baseline/
│   │   ├── followup/
│   │   └── outcomes/
│   ├── processed/
│   │   ├── analytic_cohort.rds
│   │   ├── exposure_data.rds
│   │   └── outcome_data.rds
│   └── external/
│       ├── census_data/
│       └── validation_studies/
├── R/
│   ├── data_management/
│   │   ├── import_baseline.R
│   │   ├── import_followup.R
│   │   ├── merge_datasets.R
│   │   └── create_analytic_cohort.R
│   ├── variable_derivation/
│   │   ├── exposure_variables.R
│   │   ├── outcome_variables.R
│   │   ├── covariate_variables.R
│   │   └── time_variables.R
│   ├── analysis/
│   │   ├── descriptive/
│   │   │   ├── table_one.R
│   │   │   ├── missing_data_analysis.R
│   │   │   └── cohort_description.R
│   │   ├── primary/
│   │   │   ├── cox_regression.R
│   │   │   ├── competing_risks.R
│   │   │   └── time_varying_covariates.R
│   │   ├── secondary/
│   │   │   ├── subgroup_analyses.R
│   │   │   ├── dose_response.R
│   │   │   └── effect_modification.R
│   │   └── sensitivity/
│   │       ├── missing_data_sensitivity.R
│   │       ├── unmeasured_confounding.R
│   │       └── alternative_definitions.R
│   └── functions/
│       ├── survival_functions.R
│       ├── table_functions.R
│       └── plot_functions.R
├── analysis/
│   ├── 01_cohort_assembly.R
│   ├── 02_descriptive_analysis.R
│   ├── 03_primary_analysis.R
│   ├── 04_secondary_analyses.R
│   ├── 05_sensitivity_analyses.R
│   └── 06_results_compilation.R
├── manuscripts/
│   ├── main_paper.Rmd
│   ├── supplementary_materials.Rmd
│   └── response_to_reviewers.Rmd
├── presentations/
│   ├── conference_abstract.Rmd
│   ├── oral_presentation.Rmd
│   └── poster_presentation.Rmd
└── output/
    ├── tables/
    ├── figures/
    ├── models/
    └── supplementary/
```

This comprehensive guide covers the essential aspects of R project management, data serialization, and Git workflows specifically tailored for epidemiological research. The content provides practical, production-ready solutions for managing complex epidemiological analysis projects.

Would you like me to continue with the next section covering automated R pipelines and GCP deployment? This would include detailed coverage of production pipelines, cloud deployment strategies, and scalable epidemiological analysis systems. 
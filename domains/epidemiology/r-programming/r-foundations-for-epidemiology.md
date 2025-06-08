# R Programming Foundations for Epidemiology

## Table of Contents

1. [R Ecosystem for Epidemiology](#r-ecosystem-for-epidemiology)
2. [RStudio and Posit Workbench Setup](#rstudio-and-posit-workbench-setup)
3. [R Project Organization](#r-project-organization)
4. [Essential R Packages for Epidemiology](#essential-r-packages-for-epidemiology)
5. [Data Structures and Objects](#data-structures-and-objects)
6. [Data Import and Export](#data-import-and-export)
7. [Data Cleaning and Preparation](#data-cleaning-and-preparation)
8. [Epidemiological Analysis Workflows](#epidemiological-analysis-workflows)
9. [Reproducible Research Practices](#reproducible-research-practices)
10. [Performance and Memory Management](#performance-and-memory-management)
11. [Debugging and Error Handling](#debugging-and-error-handling)
12. [Best Practices and Style](#best-practices-and-style)

## R Ecosystem for Epidemiology

### Why R for Epidemiology?

#### Advantages of R in Public Health
1. **Open Source**: Free, transparent, community-driven
2. **Statistical Focus**: Built by statisticians for statisticians
3. **Comprehensive Packages**: Extensive epidemiological libraries
4. **Reproducible Research**: RMarkdown, version control integration
5. **Data Visualization**: Advanced plotting capabilities
6. **Community Support**: Active epidemiology and public health community
7. **Integration**: Works well with databases, web services, other tools

#### R vs. Other Statistical Software

| Feature | R | SAS | SPSS | Stata | Python |
|---------|---|-----|------|-------|--------|
| **Cost** | Free | Expensive | Expensive | Expensive | Free |
| **Flexibility** | High | Medium | Low | Medium | High |
| **Epidemiology Packages** | Excellent | Good | Limited | Good | Growing |
| **Learning Curve** | Steep | Steep | Gentle | Medium | Medium |
| **Reproducibility** | Excellent | Good | Limited | Good | Excellent |
| **Visualization** | Excellent | Limited | Limited | Good | Excellent |
| **Community** | Large | Medium | Small | Medium | Large |

### R in Epidemiological Research

#### Common Applications
1. **Descriptive Analysis**: Rates, proportions, cross-tabulations
2. **Analytical Studies**: Case-control, cohort analyses
3. **Survival Analysis**: Time-to-event outcomes
4. **Spatial Epidemiology**: Geographic analysis and mapping
5. **Outbreak Investigation**: Contact tracing, epidemic curves
6. **Surveillance**: Automated reporting and monitoring
7. **Meta-analysis**: Systematic review and pooled analysis
8. **Clinical Trials**: Randomized controlled trial analysis

#### Key R Capabilities for Epidemiology

**Data Management**:
- Complex data reshaping and merging
- Date/time handling for longitudinal data
- Missing data imputation
- Data validation and quality checks

**Statistical Analysis**:
- Logistic and linear regression
- Survival analysis (Cox regression)
- Generalized linear models
- Bayesian analysis
- Machine learning methods

**Visualization**:
- Epidemic curves
- Forest plots
- Kaplan-Meier curves
- Geographic maps
- Interactive dashboards

**Reporting**:
- Automated report generation
- Dynamic documents
- Web applications
- Publication-ready tables and figures

### R Installation and Configuration

#### Installing R

**Windows**:
```bash
# Download from CRAN: https://cran.r-project.org/
# Run installer as administrator
# Recommend installing to C:\R\ (not Program Files)
```

**macOS**:
```bash
# Option 1: Download from CRAN
# https://cran.r-project.org/bin/macosx/

# Option 2: Using Homebrew
brew install r

# Install XQuartz for graphics
brew install --cask xquartz
```

**Linux (Ubuntu/Debian)**:
```bash
# Add CRAN repository
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
sudo add-apt-repository 'deb https://cloud.r-project.org/bin/linux/ubuntu focal-cran40/'

# Update and install
sudo apt update
sudo apt install r-base r-base-dev

# Install essential system dependencies
sudo apt install libcurl4-openssl-dev libssl-dev libxml2-dev libgdal-dev libproj-dev
```

#### R Configuration

**Creating .Rprofile**:
```r
# ~/.Rprofile - Global R configuration
# Set CRAN mirror
options(repos = c(CRAN = "https://cloud.r-project.org/"))

# Set default library path
.libPaths(c("~/R/library", .libPaths()))

# Load commonly used packages
suppressMessages({
  library(dplyr)
  library(ggplot2)
  library(data.table)
})

# Set default options
options(
  stringsAsFactors = FALSE,  # Don't auto-convert strings to factors
  digits = 4,               # Default number of digits to print
  scipen = 999,            # Avoid scientific notation
  max.print = 100          # Limit console output
)

# Custom functions
source("~/R/custom_functions.R")

# Startup message
cat("R configured for epidemiological analysis\n")
cat("Loaded: dplyr, ggplot2, data.table\n")
```

**Package Management Strategy**:
```r
# Create separate libraries for different projects
.libPaths(c("~/R/epi-project/renv/library", .libPaths()))

# Use renv for project-specific package management
install.packages("renv")
renv::init()  # Initialize project-specific library
```

## RStudio and Posit Workbench Setup

### RStudio Desktop Installation

#### Download and Install
1. **Download**: https://www.rstudio.com/products/rstudio/download/
2. **Requirements**: R must be installed first
3. **Installation**: Follow platform-specific installer

#### Essential RStudio Configuration

**Global Options (Tools > Global Options)**:

**General**:
```
✓ Restore .RData into workspace at startup: UNCHECKED
✓ Save workspace to .RData on exit: Never
✓ Always save history: CHECKED
```

**Code**:
```
✓ Soft-wrap R source files: CHECKED
✓ Show syntax highlighting in console input: CHECKED
✓ Auto-indent code after paste: CHECKED
✓ Insert spaces for tab: CHECKED (2 spaces)
```

**Appearance**:
```
Editor theme: Choose comfortable theme (Textmate, Solarized Dark)
Font: Source Code Pro or Fira Code (with ligatures)
Font size: 11-14 pt
```

**Pane Layout**:
```
Top Left: Source
Top Right: Environment/History
Bottom Left: Console/Terminal
Bottom Right: Files/Plots/Packages/Help
```

### Advanced RStudio Features

#### Projects and Workflow

**Creating RStudio Projects**:
```r
# File > New Project > New Directory > New Project
# Advantages:
# - Sets working directory automatically
# - Isolates project environment
# - Integrates with version control
# - Enables relative file paths
```

**Project Structure**:
```
epi-study/
├── epi-study.Rproj          # RStudio project file
├── README.md                # Project documentation
├── renv.lock               # Package versions
├── .gitignore              # Git ignore file
├── data/                   # Raw and processed data
│   ├── raw/
│   ├── processed/
│   └── external/
├── R/                      # R scripts and functions
│   ├── 01_data_import.R
│   ├── 02_data_cleaning.R
│   ├── 03_analysis.R
│   └── functions/
├── output/                 # Analysis outputs
│   ├── figures/
│   ├── tables/
│   └── reports/
├── docs/                   # Documentation
└── tests/                  # Unit tests
```

#### Code Organization Features

**Code Sections**:
```r
# Create collapsible sections with ####
# Data Import ####
library(readr)

# Data Cleaning ####
# Use Ctrl+Shift+R to insert section

# Analysis ####
# Sections appear in document outline
```

**Code Folding**:
```r
# Functions and control structures automatically fold
if (condition) {
  # This block can be folded
  long_analysis_code()
}

# Custom folding with ----
# Long data processing ----
{
  step1 <- data %>% filter(...)
  step2 <- step1 %>% mutate(...)
  step3 <- step2 %>% summarize(...)
}
```

### RStudio Addins and Productivity

#### Essential Addins
```r
# Install useful addins
install.packages(c(
  "styler",      # Code formatting
  "remedy",      # RMarkdown helpers
  "esquisse",    # ggplot2 GUI
  "datapasta",   # Copy-paste data
  "reprex"       # Reproducible examples
))
```

#### Keyboard Shortcuts
```
Ctrl+Shift+M    # Pipe operator %>%
Ctrl+Shift+R    # Insert section
Ctrl+Shift+C    # Comment/uncomment
Ctrl+Shift+K    # Knit document
Ctrl+Shift+P    # Command palette
Ctrl+Shift+F10  # Restart R session
Ctrl+Enter      # Run current line/selection
Ctrl+Shift+Enter # Run current chunk
Alt+Shift+K     # Show all shortcuts
```

### Posit Workbench (RStudio Server Pro)

#### Overview
- **Web-based RStudio**: Access from any browser
- **Multi-user environment**: Shared resources and collaboration
- **Administrative features**: User management, resource monitoring
- **Enterprise security**: LDAP/AD integration, audit logging

#### Key Features for Epidemiology Teams

**Project Sharing**:
```r
# Shared project directories
/shared/epi-projects/covid-analysis/
├── data/          # Shared datasets
├── scripts/       # Analysis scripts
└── results/       # Collaborative results
```

**Session Management**:
```r
# Configure session settings
# Tools > Global Options > General
# Default working directory: /home/username/projects
# Memory limit: Set appropriate for large datasets
```

**Load Balancing**:
- Distribute computational load across servers
- Automatic failover for high availability
- Resource monitoring and allocation

#### Administration for Epidemiology

**User Groups**:
```bash
# Create epidemiology group
sudo groupadd epidemiologists
sudo usermod -a -G epidemiologists alice
sudo usermod -a -G epidemiologists bob

# Set group permissions for data
sudo chgrp -R epidemiologists /data/surveillance
sudo chmod -R g+rw /data/surveillance
```

**Resource Management**:
```bash
# /etc/rstudio/rsession.conf
session-timeout-minutes=480
session-memory-limit-mb=8192
session-default-working-dir=/home/{USER}/projects
```

## R Project Organization

### Project Structure Philosophy

#### Principles for Epidemiological Projects
1. **Reproducibility**: Anyone should be able to recreate analysis
2. **Portability**: Projects should work across different computers
3. **Maintainability**: Code should be easy to understand and modify
4. **Scalability**: Structure should accommodate project growth
5. **Collaboration**: Multiple researchers should be able to contribute

#### Directory Structure Standards

**Basic Structure**:
```
study-name/
├── README.md                    # Project overview and setup
├── study-name.Rproj            # RStudio project file
├── .gitignore                  # Version control exclusions
├── renv.lock                   # Package versions (if using renv)
├── _targets.R                  # Pipeline definition (if using targets)
│
├── data/                       # Never edit by hand
│   ├── raw/                    # Original, immutable data
│   ├── processed/              # Cleaned, analysis-ready data
│   └── external/               # External reference data
│
├── R/                          # All R code
│   ├── 01_setup.R             # Package loading, options
│   ├── 02_import.R            # Data import functions
│   ├── 03_clean.R             # Data cleaning functions
│   ├── 04_analyze.R           # Analysis functions
│   ├── 05_visualize.R         # Plotting functions
│   └── functions/             # Custom function library
│       ├── data_utils.R
│       ├── epi_functions.R
│       └── plot_functions.R
│
├── analysis/                   # Analysis scripts
│   ├── descriptive.R
│   ├── case_control.R
│   ├── survival.R
│   └── sensitivity.R
│
├── reports/                    # RMarkdown reports
│   ├── preliminary_report.Rmd
│   ├── final_report.Rmd
│   └── supplementary.Rmd
│
├── output/                     # Analysis outputs
│   ├── figures/               # All plots and graphs
│   ├── tables/                # Analysis tables
│   └── models/                # Saved model objects
│
├── docs/                       # Documentation
│   ├── codebook.md           # Variable definitions
│   ├── protocol.md           # Study protocol
│   └── analysis_plan.md      # Statistical analysis plan
│
└── tests/                      # Unit tests
    ├── test_data_import.R
    ├── test_cleaning.R
    └── test_analysis.R
```

**Advanced Structure for Large Studies**:
```
large-cohort-study/
├── config/                     # Configuration files
│   ├── database.yml
│   ├── variables.yml
│   └── parameters.R
│
├── data/
│   ├── raw/
│   │   ├── baseline/
│   │   ├── followup/
│   │   └── outcomes/
│   ├── processed/
│   │   ├── cleaned/
│   │   ├── derived/
│   │   └── analysis/
│   └── external/
│       ├── census/
│       ├── mortality/
│       └── geocoding/
│
├── R/
│   ├── import/
│   │   ├── database_functions.R
│   │   ├── file_readers.R
│   │   └── api_functions.R
│   ├── cleaning/
│   │   ├── validation.R
│   │   ├── standardization.R
│   │   └── quality_control.R
│   ├── analysis/
│   │   ├── descriptive.R
│   │   ├── inferential.R
│   │   └── modeling.R
│   └── visualization/
│       ├── tables.R
│       ├── plots.R
│       └── maps.R
│
├── analysis/
│   ├── aim1_prevalence/
│   ├── aim2_risk_factors/
│   ├── aim3_survival/
│   └── sensitivity_analyses/
│
├── reports/
│   ├── data_monitoring/
│   ├── interim_analyses/
│   └── final_reports/
│
└── workflows/                  # Automated pipelines
    ├── data_processing.R
    ├── weekly_reports.R
    └── quality_checks.R
```

### File Naming Conventions

#### Systematic Naming Strategy
```r
# Use consistent, descriptive names
# Format: [number]_[verb]_[noun]_[details].R

# Good examples:
01_import_baseline_data.R
02_clean_exposure_variables.R
03_analyze_case_control.R
04_plot_epidemic_curve.R

# Poor examples:
script1.R
analysis.R
final.R
new_analysis_v2_final.R
```

#### File Naming Rules
1. **No spaces**: Use underscores or hyphens
2. **Lowercase**: Avoid mixed case for consistency
3. **Descriptive**: Names should indicate content
4. **Sequential**: Use numbers for workflow order
5. **Versioning**: Use dates or semantic versions for major revisions

```r
# Date-based versioning
analysis_2024-01-15.R
results_2024-01-20.R

# Semantic versioning
model_v1-0-0.R
model_v1-1-0.R  # Minor update
model_v2-0-0.R  # Major revision
```

### Configuration Management

#### Project Configuration File
```r
# config/config.R - Central configuration
# Data paths
DATA_RAW <- here::here("data", "raw")
DATA_PROCESSED <- here::here("data", "processed")
DATA_EXTERNAL <- here::here("data", "external")

# Output paths
OUTPUT_FIGURES <- here::here("output", "figures")
OUTPUT_TABLES <- here::here("output", "tables")
OUTPUT_MODELS <- here::here("output", "models")

# Database connection
DB_HOST <- Sys.getenv("DB_HOST", "localhost")
DB_PORT <- Sys.getenv("DB_PORT", "5432")
DB_NAME <- Sys.getenv("DB_NAME", "epi_study")

# Analysis parameters
SIGNIFICANCE_LEVEL <- 0.05
BOOTSTRAP_ITERATIONS <- 1000
CONFIDENCE_LEVEL <- 0.95

# Study dates
STUDY_START_DATE <- as.Date("2020-01-01")
STUDY_END_DATE <- as.Date("2023-12-31")
FOLLOWUP_CUTOFF <- as.Date("2024-01-31")
```

#### Environment Variables
```r
# .Renviron file for sensitive information
DB_USERNAME=epi_user
DB_PASSWORD=secure_password
API_KEY=your_api_key_here
GEOCODING_KEY=google_maps_key

# Load in R scripts
library(keyring)
db_password <- keyring::key_get("database_password")
```

### Package Management

#### Using renv for Reproducibility
```r
# Initialize renv in project
renv::init()

# Install packages (automatically captured)
install.packages(c("dplyr", "ggplot2", "survival"))

# Create snapshot of current state
renv::snapshot()

# Restore exact package versions
renv::restore()

# Update packages
renv::update()
```

#### renv.lock Structure
```json
{
  "R": {
    "Version": "4.3.2",
    "Repositories": [
      {
        "Name": "CRAN",
        "URL": "https://cloud.r-project.org"
      }
    ]
  },
  "Packages": {
    "dplyr": {
      "Package": "dplyr",
      "Version": "1.1.4",
      "Source": "Repository",
      "Repository": "CRAN",
      "Hash": "fedd9d00c2944ff00a0e2696ccf048ec"
    }
  }
}
```

#### Custom Package Loading
```r
# R/setup.R - Standardized package loading
# Required packages
required_packages <- c(
  # Data manipulation
  "dplyr", "tidyr", "stringr", "lubridate",
  
  # Analysis
  "broom", "survival", "tableone", "MatchIt",
  
  # Visualization
  "ggplot2", "plotly", "leaflet", "DT",
  
  # Reporting
  "rmarkdown", "knitr", "officer", "flextable",
  
  # Epidemiology specific
  "epiR", "epitools", "EpiModel", "outbreaks"
)

# Install missing packages
new_packages <- required_packages[!(required_packages %in% 
                                   installed.packages()[,"Package"])]
if(length(new_packages)) {
  install.packages(new_packages)
}

# Load all packages
suppressPackageStartupMessages({
  lapply(required_packages, library, character.only = TRUE)
})

# Verify critical packages
critical_packages <- c("dplyr", "ggplot2", "survival")
for(pkg in critical_packages) {
  if(!requireNamespace(pkg, quietly = TRUE)) {
    stop(paste("Critical package", pkg, "not available"))
  }
}
```

## Essential R Packages for Epidemiology

### Core Data Science Packages

#### Tidyverse Ecosystem
```r
# Complete tidyverse
install.packages("tidyverse")

# Individual packages for lighter load
library(dplyr)     # Data manipulation
library(tidyr)     # Data reshaping
library(ggplot2)   # Visualization
library(stringr)   # String manipulation
library(lubridate) # Date/time handling
library(readr)     # Data import
library(purrr)     # Functional programming
library(forcats)   # Factor handling
```

**dplyr for Data Manipulation**:
```r
# Essential dplyr verbs for epidemiology
library(dplyr)

# Filter cases based on criteria
cases <- study_data %>%
  filter(
    age >= 18,
    age <= 65,
    !is.na(exposure),
    outcome_date >= study_start
  )

# Create new variables
analytic_data <- cases %>%
  mutate(
    age_group = case_when(
      age < 30 ~ "18-29",
      age < 50 ~ "30-49",
      age < 65 ~ "50-64",
      TRUE ~ "65+"
    ),
    follow_up_time = as.numeric(outcome_date - enrollment_date),
    exposed = exposure > 0
  )

# Summarize by groups
summary_stats <- analytic_data %>%
  group_by(age_group, exposed) %>%
  summarise(
    n = n(),
    mean_age = mean(age),
    incidence_rate = sum(outcome) / sum(person_time),
    .groups = "drop"
  )
```

#### Data.table for High Performance
```r
library(data.table)

# Convert to data.table
DT <- as.data.table(large_dataset)

# Fast operations on large datasets
# Filter
cases_dt <- DT[age >= 18 & age <= 65 & !is.na(exposure)]

# Group operations
summary_dt <- DT[, .(
  n = .N,
  mean_age = mean(age),
  incidence = sum(outcome) / sum(person_time)
), by = .(age_group, exposed)]

# Update by reference (memory efficient)
DT[, age_group := fcase(
  age < 30, "18-29",
  age < 50, "30-49",
  age < 65, "50-64",
  default = "65+"
)]
```

### Specialized Epidemiology Packages

#### epiR - Epidemiological Analysis
```r
install.packages("epiR")
library(epiR)

# 2x2 contingency table analysis
tab <- matrix(c(
  exposed_cases = 84,
  unexposed_cases = 16,
  exposed_controls = 315,
  unexposed_controls = 485
), nrow = 2, byrow = TRUE)

# Calculate odds ratio and confidence interval
result <- epi.2by2(tab, method = "case.control")
print(result)

# Stratified analysis
stratified_result <- epi.2by2(tab_array, method = "case.control")
```

#### epitools - Epidemiological Tools
```r
install.packages("epitools")
library(epitools)

# Odds ratio calculation
or_result <- oddsratio(exposed_cases, total_cases, 
                      exposed_controls, total_controls)

# Rate ratio calculation
rr_result <- rateratio(cases_exposed, time_exposed,
                      cases_unexposed, time_unexposed)

# Age standardization
age_adjusted <- ageadjust.direct(
  count = cases_by_age,
  pop = population_by_age,
  stdpop = standard_population
)
```

#### survival - Survival Analysis
```r
library(survival)

# Kaplan-Meier survival curves
km_fit <- survfit(Surv(time, status) ~ treatment, data = lung_cancer)
plot(km_fit, xlab = "Time (months)", ylab = "Survival probability")

# Cox proportional hazards model
cox_model <- coxph(Surv(time, status) ~ age + sex + treatment, 
                   data = lung_cancer)
summary(cox_model)

# Test proportional hazards assumption
ph_test <- cox.zph(cox_model)
plot(ph_test)
```

### Specialized Analysis Packages

#### tableone - Descriptive Statistics
```r
install.packages("tableone")
library(tableone)

# Create Table 1 (baseline characteristics)
vars <- c("age", "sex", "bmi", "smoking", "comorbidities")
cat_vars <- c("sex", "smoking", "comorbidities")

table1 <- CreateTableOne(
  vars = vars,
  strata = "exposure_group",
  data = study_data,
  factorVars = cat_vars,
  test = TRUE
)

print(table1, showAllLevels = TRUE, cramVars = "sex")
```

#### broom - Tidy Model Outputs
```r
library(broom)

# Tidy logistic regression output
log_model <- glm(outcome ~ age + sex + exposure, 
                 family = binomial, data = study_data)

# Extract coefficients as data frame
tidy_results <- tidy(log_model, conf.int = TRUE, exponentiate = TRUE)
model_stats <- glance(log_model)
fitted_values <- augment(log_model)
```

#### MatchIt - Propensity Score Matching
```r
install.packages("MatchIt")
library(MatchIt)

# Propensity score matching
match_result <- matchit(
  treatment ~ age + sex + comorbidities,
  data = study_data,
  method = "nearest",
  distance = "glm",
  caliper = 0.1
)

# Check balance
summary(match_result)
plot(match_result, type = "jitter")

# Extract matched data
matched_data <- match.data(match_result)
```

### Visualization and Reporting

#### ggplot2 Advanced Epidemiological Plots
```r
library(ggplot2)
library(scales)

# Epidemic curve
epi_curve <- ggplot(outbreak_data, aes(x = onset_date)) +
  geom_histogram(binwidth = 1, fill = "steelblue", alpha = 0.7) +
  scale_x_date(date_labels = "%m/%d", date_breaks = "1 week") +
  labs(
    title = "Epidemic Curve - Foodborne Outbreak",
    x = "Date of Onset",
    y = "Number of Cases"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Forest plot for meta-analysis
forest_plot <- ggplot(meta_data, aes(x = odds_ratio, y = study)) +
  geom_point(size = 3) +
  geom_errorbarh(aes(xmin = ci_lower, xmax = ci_upper), height = 0.2) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "red") +
  scale_x_log10() +
  labs(
    title = "Forest Plot - Odds Ratios by Study",
    x = "Odds Ratio (log scale)",
    y = "Study"
  ) +
  theme_minimal()
```

#### Spatial Analysis Packages
```r
# Install spatial packages
install.packages(c("sf", "leaflet", "tmap", "SpatialEpi"))

library(sf)        # Simple features for spatial data
library(leaflet)   # Interactive maps
library(tmap)      # Thematic maps
library(SpatialEpi) # Spatial epidemiology

# Disease mapping example
# Load shapefile
county_sf <- st_read("data/counties.shp")

# Join disease data
county_disease <- county_sf %>%
  left_join(disease_rates, by = "county_id")

# Create choropleth map
tm_shape(county_disease) +
  tm_fill("incidence_rate", 
          style = "quantile", 
          palette = "YlOrRd",
          title = "Incidence Rate\nper 100,000") +
  tm_borders() +
  tm_layout(title = "Disease Incidence by County")
```

This covers the comprehensive R foundations for epidemiology. The content provides detailed setup instructions, project organization principles, and essential packages specifically tailored for epidemiological analysis. 

Would you like me to continue with the next section covering RStudio tooling, RMarkdown workflows, and project management best practices? This would include detailed coverage of .Rproj files, .Rds files, automated pipelines, and Git workflows for R projects. 
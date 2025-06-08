# RMarkdown Workflows for Epidemiological Research

## Table of Contents

1. [Introduction to RMarkdown](#introduction-to-rmarkdown)
2. [RMarkdown vs R Scripts](#rmarkdown-vs-r-scripts)
3. [Document Types and Formats](#document-types-and-formats)
4. [RMarkdown Fundamentals](#rmarkdown-fundamentals)
5. [Advanced RMarkdown Features](#advanced-rmarkdown-features)
6. [Parameterized Reports](#parameterized-reports)
7. [Automated Report Generation](#automated-report-generation)
8. [Collaborative Workflows](#collaborative-workflows)
9. [Publishing and Deployment](#publishing-and-deployment)
10. [Best Practices](#best-practices)
11. [Troubleshooting](#troubleshooting)
12. [Integration with Other Tools](#integration-with-other-tools)

## Introduction to RMarkdown

### What is RMarkdown?

**RMarkdown** is an authoring framework that combines:
- **Markdown**: Simple markup language for formatting text
- **R Code**: Executable code chunks that generate results
- **Output Formats**: HTML, PDF, Word, presentations, dashboards

#### Core Advantages for Epidemiology
1. **Reproducible Research**: Code, analysis, and narrative in one document
2. **Dynamic Reports**: Automatically update when data changes
3. **Multiple Formats**: Same source creates various output types
4. **Version Control**: Text-based format works well with Git
5. **Collaboration**: Easy to share and review
6. **Publication Ready**: Professional-quality outputs

### RMarkdown Ecosystem

#### Core Packages
```r
# Essential RMarkdown packages
install.packages(c(
  "rmarkdown",    # Core RMarkdown functionality
  "knitr",        # Engine for dynamic report generation
  "pandoc",       # Universal document converter
  "tinytex",      # Lightweight LaTeX distribution
  "bookdown",     # Books and long-form documents
  "blogdown",     # Websites and blogs
  "pagedown",     # Paged HTML documents
  "flexdashboard", # Interactive dashboards
  "xaringan",     # HTML presentations
  "distill"       # Scientific and technical writing
))
```

#### Document Rendering Process
```
.Rmd file → knitr → .md file → pandoc → output format
     ↓        ↓        ↓         ↓           ↓
   Source   Execute   Clean    Convert    Final
   Code      Code    Markdown            Document
```

## RMarkdown vs R Scripts

### When to Use Each

#### R Scripts (.R files)
**Best for**:
- Data processing and cleaning
- Function development
- Exploratory analysis
- Code development and testing
- Automated pipelines

**Characteristics**:
```r
# R Script Example: data_cleaning.R
# Load packages
library(dplyr)
library(lubridate)

# Function to clean surveillance data
clean_surveillance_data <- function(raw_data) {
  cleaned <- raw_data %>%
    filter(!is.na(case_id)) %>%
    mutate(
      report_date = ymd(report_date),
      age_group = case_when(
        age < 18 ~ "0-17",
        age < 65 ~ "18-64",
        TRUE ~ "65+"
      )
    ) %>%
    arrange(report_date)
  
  return(cleaned)
}

# Execute cleaning
surveillance_clean <- clean_surveillance_data(surveillance_raw)

# Save processed data
saveRDS(surveillance_clean, "data/processed/surveillance_clean.rds")
```

#### RMarkdown Documents (.Rmd files)
**Best for**:
- Analysis reports
- Documentation
- Presentations
- Publications
- Dashboards
- Teaching materials

**Characteristics**:
```rmd
---
title: "COVID-19 Surveillance Report"
author: "Epidemiology Team"
date: "`r Sys.Date()`"
output: html_document
---

## Executive Summary

This report analyzes COVID-19 surveillance data for the week ending `r Sys.Date()`.

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = FALSE, warning = FALSE, message = FALSE)
library(dplyr)
library(ggplot2)
surveillance_data <- readRDS("data/processed/surveillance_clean.rds")
```

### Key Findings

- Total cases this week: `r nrow(surveillance_data)`
- Average age: `r round(mean(surveillance_data$age), 1)` years

```{r epidemic-curve}
ggplot(surveillance_data, aes(x = report_date)) +
  geom_histogram(binwidth = 1, fill = "steelblue") +
  labs(title = "Daily Case Counts", x = "Date", y = "Cases") +
  theme_minimal()
```
```

### Hybrid Workflow

#### Recommended Structure
```
analysis-project/
├── R/                          # R scripts for processing
│   ├── 01_data_import.R       # Data import and validation
│   ├── 02_data_cleaning.R     # Data cleaning functions
│   ├── 03_analysis_functions.R # Analysis functions
│   └── functions/             # Reusable functions
├── reports/                   # RMarkdown documents
│   ├── weekly_report.Rmd     # Regular surveillance report
│   ├── outbreak_report.Rmd   # Outbreak investigation
│   └── annual_summary.Rmd    # Annual surveillance summary
├── output/                    # Generated reports
│   ├── html/
│   ├── pdf/
│   └── word/
└── data/                      # Data files
    ├── raw/
    └── processed/
```

#### Workflow Integration
```r
# R/master_analysis.R - Orchestrates entire workflow
source("R/01_data_import.R")
source("R/02_data_cleaning.R")
source("R/03_analysis_functions.R")

# Process data
raw_data <- import_surveillance_data()
clean_data <- clean_surveillance_data(raw_data)
analysis_results <- analyze_surveillance_data(clean_data)

# Generate reports
rmarkdown::render("reports/weekly_report.Rmd", 
                  output_dir = "output/html")
rmarkdown::render("reports/weekly_report.Rmd", 
                  output_format = "pdf_document",
                  output_dir = "output/pdf")
```

## Document Types and Formats

### Output Formats Overview

#### HTML Documents
```yaml
---
title: "Epidemiological Analysis"
output: 
  html_document:
    toc: true
    toc_float: true
    theme: flatly
    highlight: tango
    code_folding: hide
    fig_caption: true
    fig_width: 8
    fig_height: 6
---
```

**Advantages**:
- Interactive elements (plotly, DT tables)
- Easy sharing via web
- Responsive design
- Fast rendering

**Best for**: Regular reports, dashboards, interactive analysis

#### PDF Documents
```yaml
---
title: "Annual Surveillance Report"
output: 
  pdf_document:
    toc: true
    number_sections: true
    fig_caption: true
    latex_engine: xelatex
    includes:
      in_header: header.tex
geometry: margin=1in
fontsize: 11pt
---
```

**Advantages**:
- Publication quality
- Consistent formatting across platforms
- Professional appearance
- Archival format

**Best for**: Official reports, publications, formal documentation

#### Word Documents
```yaml
---
title: "Outbreak Investigation Report"
output: 
  word_document:
    reference_docx: template.docx
    toc: true
    fig_caption: true
    keep_md: true
---
```

**Advantages**:
- Easy collaboration with non-R users
- Track changes and comments
- Institution templates
- Familiar interface

**Best for**: Collaborative drafts, institutional reports

### Specialized Formats for Epidemiology

#### Flexdashboard - Interactive Dashboards
```yaml
---
title: "Disease Surveillance Dashboard"
output: 
  flexdashboard::flex_dashboard:
    orientation: columns
    vertical_layout: fill
    source_code: embed
    theme: bootstrap
runtime: shiny
---
```

**Example Dashboard Structure**:
```rmd
Column {data-width=650}
-----------------------------------------------------------------------

### Daily Case Counts

```{r}
renderPlotly({
  p <- ggplot(surveillance_data, aes(x = date, y = cases)) +
    geom_line() +
    labs(title = "Daily COVID-19 Cases")
  ggplotly(p)
})
```

Column {data-width=350}
-----------------------------------------------------------------------

### Summary Statistics

```{r}
DT::renderDataTable({
  summary_stats
}, options = list(pageLength = 10))
```

### Case Distribution

```{r}
renderPlot({
  ggplot(surveillance_data, aes(x = age_group, fill = age_group)) +
    geom_bar() +
    theme_minimal()
})
```
```

#### Bookdown - Long-form Documents
```yaml
---
title: "Epidemiological Methods Manual"
author: "Public Health Department"
site: bookdown::bookdown_site
output: bookdown::gitbook
documentclass: book
bibliography: references.bib
biblio-style: apalike
link-citations: yes
---
```

**Multi-chapter Structure**:
```
epidemiology-manual/
├── index.Rmd              # Front matter
├── 01-introduction.Rmd    # Chapter 1
├── 02-study-designs.Rmd   # Chapter 2
├── 03-measures.Rmd        # Chapter 3
├── 04-analysis.Rmd        # Chapter 4
├── references.bib         # Bibliography
└── _bookdown.yml          # Configuration
```

#### Distill - Scientific Articles
```yaml
---
title: "Risk Factors for COVID-19 Transmission"
description: |
  A case-control study of transmission risk factors
author:
  - name: Jane Smith
    affiliation: School of Public Health
date: "`r Sys.Date()`"
output: distill::distill_article
bibliography: references.bib
---
```

## RMarkdown Fundamentals

### YAML Header Configuration

#### Essential YAML Options
```yaml
---
title: "Surveillance Report"
subtitle: "Week Ending March 15, 2024"
author: 
  - name: "John Epidemiologist"
    email: "john@health.gov"
    affiliation: "State Health Department"
date: "`r format(Sys.Date(), '%B %d, %Y')`"
output:
  html_document:
    toc: true
    toc_depth: 3
    toc_float: 
      collapsed: false
      smooth_scroll: true
    number_sections: true
    theme: flatly
    highlight: tango
    code_folding: hide
    fig_width: 10
    fig_height: 6
    fig_caption: true
    df_print: paged
    keep_md: true
    self_contained: true
params:
  data_date: "2024-03-15"
  region: "all"
  include_predictions: true
bibliography: references.bib
csl: american-journal-of-epidemiology.csl
---
```

### Code Chunk Options

#### Essential Chunk Options for Epidemiology
```r
# Setup chunk - run once at beginning
```{r setup, include=FALSE}
knitr::opts_chunk$set(
  echo = FALSE,           # Hide code by default
  warning = FALSE,        # Suppress warnings
  message = FALSE,        # Suppress messages
  fig.width = 8,         # Default figure width
  fig.height = 6,        # Default figure height
  fig.align = "center",  # Center figures
  cache = FALSE,         # Don't cache results (for data updates)
  comment = NA           # Clean output formatting
)

# Load packages quietly
suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(knitr)
  library(DT)
})
```

```r
# Analysis chunk - show results but hide code
```{r descriptive-analysis, echo=FALSE, results='asis'}
# Create summary table
summary_table <- surveillance_data %>%
  group_by(age_group) %>%
  summarise(
    cases = n(),
    rate_per_100k = round(n() / population * 100000, 1)
  )

kable(summary_table, caption = "Cases by Age Group")
```

```r
# Figure chunk with custom dimensions
```{r epidemic-curve, fig.width=12, fig.height=6, fig.cap="Epidemic curve showing daily case counts over time"}
ggplot(surveillance_data, aes(x = onset_date)) +
  geom_histogram(binwidth = 1, fill = "steelblue", alpha = 0.8) +
  scale_x_date(date_labels = "%m/%d", date_breaks = "1 week") +
  labs(
    title = "COVID-19 Epidemic Curve",
    x = "Date of Symptom Onset",
    y = "Number of Cases"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
```

### Inline Code

#### Dynamic Text with Inline R
```rmd
# Surveillance Summary

As of `r format(Sys.Date(), "%B %d, %Y")`, we have identified 
**`r nrow(surveillance_data)` confirmed cases** of COVID-19. 

The outbreak began on `r min(surveillance_data$onset_date)` and the most 
recent case had symptom onset on `r max(surveillance_data$onset_date)`.

## Key Statistics

- **Attack Rate**: `r round(attack_rate * 100, 1)`%
- **Case Fatality Rate**: `r round(cfr * 100, 1)`%
- **Median Age**: `r median(surveillance_data$age)` years 
- **Age Range**: `r min(surveillance_data$age)` to `r max(surveillance_data$age)` years

The reproduction number (R₀) is estimated to be 
`r round(reproduction_number, 2)` (95% CI: `r round(r0_ci_lower, 2)` - `r round(r0_ci_upper, 2)`).
```

### Tables and Formatting

#### Professional Table Creation
```r
```{r summary-table}
library(knitr)
library(kableExtra)

# Create publication-ready table
summary_stats %>%
  kable(
    col.names = c("Age Group", "Cases", "Rate per 100,000", "95% CI"),
    digits = 1,
    caption = "COVID-19 incidence rates by age group"
  ) %>%
  kable_styling(
    bootstrap_options = c("striped", "hover", "condensed"),
    full_width = FALSE,
    position = "center"
  ) %>%
  add_header_above(c(" " = 1, "Incidence" = 3)) %>%
  footnote(
    general = "Data as of March 15, 2024",
    symbol = "Rates calculated using 2020 Census data"
  )
```

#### Interactive Tables
```r
```{r interactive-table}
library(DT)

surveillance_data %>%
  select(case_id, age, sex, onset_date, hospitalized) %>%
  datatable(
    caption = "Case Details",
    options = list(
      pageLength = 25,
      scrollX = TRUE,
      columnDefs = list(
        list(className = 'dt-center', targets = c(1, 2, 4))
      )
    ),
    filter = 'top'
  ) %>%
  formatDate('onset_date', method = 'toLocaleDateString')
```

## Advanced RMarkdown Features

### Custom Themes and Styling

#### Custom CSS for Epidemiology Reports
```css
/* custom.css - Custom styling for epi reports */

/* Header styling */
h1, h2, h3 {
  color: #2c3e50;
  font-family: 'Roboto', sans-serif;
}

h1 {
  border-bottom: 3px solid #3498db;
  padding-bottom: 10px;
}

/* Highlight key statistics */
.key-stat {
  background-color: #ecf0f1;
  border-left: 4px solid #3498db;
  padding: 15px;
  margin: 15px 0;
  font-size: 16px;
  font-weight: bold;
}

/* Warning boxes */
.alert {
  padding: 15px;
  margin: 20px 0;
  border-radius: 4px;
}

.alert-warning {
  background-color: #fcf8e3;
  border: 1px solid #faebcc;
  color: #8a6d3b;
}

.alert-danger {
  background-color: #f2dede;
  border: 1px solid #ebccd1;
  color: #a94442;
}

/* Figure styling */
.figure {
  text-align: center;
  margin: 20px 0;
}

.figure img {
  border: 1px solid #ddd;
  border-radius: 4px;
  padding: 5px;
}
```

#### Using Custom CSS
```yaml
---
title: "Outbreak Investigation Report"
output:
  html_document:
    css: custom.css
    toc: true
    toc_float: true
---
```

### Child Documents and Modularity

#### Main Report Structure
```rmd
---
title: "Comprehensive Surveillance Report"
output: html_document
---

```{r setup, include=FALSE}
source("R/setup.R")
surveillance_data <- readRDS("data/processed/surveillance_clean.rds")
```

# Executive Summary

```{r child="sections/executive_summary.Rmd"}
```

# Descriptive Analysis

```{r child="sections/descriptive_analysis.Rmd"}
```

# Geographic Analysis

```{r child="sections/geographic_analysis.Rmd"}
```

# Temporal Analysis

```{r child="sections/temporal_analysis.Rmd"}
```
```

#### Child Document Example
```rmd
<!-- sections/descriptive_analysis.Rmd -->

## Case Demographics

```{r demographics-table}
demo_table <- surveillance_data %>%
  group_by(age_group, sex) %>%
  summarise(cases = n(), .groups = "drop") %>%
  pivot_wider(names_from = sex, values_from = cases, values_fill = 0)

kable(demo_table, caption = "Case distribution by age and sex")
```

## Clinical Characteristics

```{r clinical-summary}
clinical_summary <- surveillance_data %>%
  summarise(
    hospitalized = sum(hospitalized, na.rm = TRUE),
    icu_admitted = sum(icu, na.rm = TRUE),
    deaths = sum(died, na.rm = TRUE),
    median_los = median(length_of_stay, na.rm = TRUE)
  )

# Display as formatted text
```

Of the `r nrow(surveillance_data)` cases:
- `r clinical_summary$hospitalized` (`r round(clinical_summary$hospitalized/nrow(surveillance_data)*100, 1)`%) were hospitalized
- `r clinical_summary$icu_admitted` (`r round(clinical_summary$icu_admitted/nrow(surveillance_data)*100, 1)`%) required ICU care
- `r clinical_summary$deaths` deaths occurred (CFR: `r round(clinical_summary$deaths/nrow(surveillance_data)*100, 1)`%)
```

### Custom Output Hooks

#### Custom Formatting for Statistical Results
```r
```{r custom-hooks, include=FALSE}
# Custom hook for formatting p-values
format_p <- function(p) {
  if (p < 0.001) {
    return("p < 0.001")
  } else if (p < 0.01) {
    return(paste("p =", round(p, 3)))
  } else {
    return(paste("p =", round(p, 2)))
  }
}

# Custom hook for confidence intervals
format_ci <- function(lower, upper, digits = 2) {
  paste0("(95% CI: ", round(lower, digits), " - ", round(upper, digits), ")")
}

# Custom hook for odds ratios
format_or <- function(or, lower, upper) {
  paste0("OR = ", round(or, 2), " ", format_ci(lower, upper))
}
```

```{r logistic-regression}
# Logistic regression analysis
model <- glm(outcome ~ age + sex + exposure, 
             family = binomial, data = study_data)

# Extract results
results <- broom::tidy(model, conf.int = TRUE, exponentiate = TRUE)

# Display formatted results
exposure_result <- results[results$term == "exposureTRUE", ]
```

The odds of disease among exposed individuals was `r format_or(exposure_result$estimate, exposure_result$conf.low, exposure_result$conf.high)`, `r format_p(exposure_result$p.value)`.
```

## Parameterized Reports

### Setting Up Parameters

#### Parameter Definition
```yaml
---
title: "Regional Surveillance Report: `r params$region`"
output: html_document
params:
  region: "all"
  start_date: "2024-01-01"
  end_date: "2024-03-15"
  include_maps: true
  minimum_cases: 5
  report_type: "weekly"
---
```

#### Using Parameters in Analysis
```r
```{r parameter-setup}
# Use parameters to filter data
report_data <- surveillance_data %>%
  filter(
    report_date >= as.Date(params$start_date),
    report_date <= as.Date(params$end_date)
  )

# Regional filtering
if (params$region != "all") {
  report_data <- report_data %>%
    filter(region == params$region)
}

# Apply minimum case threshold
county_data <- report_data %>%
  group_by(county) %>%
  summarise(cases = n()) %>%
  filter(cases >= params$minimum_cases)
```

#### Conditional Content
```r
```{r maps, eval=params$include_maps}
# Only include maps if parameter is TRUE
library(leaflet)

county_map <- leaflet(county_shapefile) %>%
  addTiles() %>%
  addPolygons(
    fillColor = ~color_palette(cases),
    weight = 1,
    opacity = 1,
    fillOpacity = 0.7,
    popup = ~paste("County:", county, "<br>Cases:", cases)
  )

county_map
```

### Automated Parameter Generation

#### Generate Multiple Reports
```r
# R/generate_regional_reports.R
library(rmarkdown)
library(purrr)

# Define regions to report on
regions <- c("North", "South", "East", "West", "Central")

# Generate report for each region
walk(regions, function(region) {
  render(
    input = "reports/regional_report.Rmd",
    output_file = paste0("regional_report_", region, ".html"),
    output_dir = "output/regional",
    params = list(
      region = region,
      start_date = "2024-01-01",
      end_date = Sys.Date(),
      include_maps = TRUE
    )
  )
})
```

#### Weekly Report Automation
```r
# R/weekly_report_automation.R
generate_weekly_report <- function(week_ending = Sys.Date()) {
  week_start <- week_ending - 6
  
  render(
    input = "reports/weekly_surveillance.Rmd",
    output_file = paste0("weekly_report_", week_ending, ".html"),
    output_dir = "output/weekly",
    params = list(
      start_date = as.character(week_start),
      end_date = as.character(week_ending),
      report_type = "weekly"
    )
  )
}

# Schedule with cron or Windows Task Scheduler
# Or run manually for specific weeks
generate_weekly_report("2024-03-15")
```

## Automated Report Generation

### Scheduling and Automation

#### Cron Job Setup (Linux/Mac)
```bash
# Edit crontab
crontab -e

# Add entries for automated reports
# Weekly report every Monday at 8 AM
0 8 * * 1 /usr/bin/Rscript /path/to/project/R/weekly_report.R

# Daily dashboard update every day at 6 AM
0 6 * * * /usr/bin/Rscript /path/to/project/R/daily_dashboard.R

# Monthly summary report on first day of month at 9 AM
0 9 1 * * /usr/bin/Rscript /path/to/project/R/monthly_report.R
```

#### Windows Task Scheduler
```batch
# Create batch file: run_weekly_report.bat
@echo off
cd /d "C:\path\to\project"
"C:\Program Files\R\R-4.3.2\bin\Rscript.exe" R/weekly_report.R
```

#### R-based Scheduling with taskscheduleR
```r
library(taskscheduleR)

# Schedule weekly surveillance report
taskscheduler_create(
  taskname = "weekly_surveillance_report",
  rscript = "R/weekly_report.R",
  schedule = "WEEKLY",
  starttime = "08:00",
  days = "MON"
)

# Schedule daily data update
taskscheduler_create(
  taskname = "daily_data_update",
  rscript = "R/update_surveillance_data.R",
  schedule = "DAILY",
  starttime = "06:00"
)
```

### Email Automation

#### Automated Email Reports
```r
# R/email_report.R
library(blastula)
library(rmarkdown)

send_surveillance_report <- function(recipients, report_date = Sys.Date()) {
  # Generate report
  report_file <- render(
    input = "reports/surveillance_email.Rmd",
    output_file = paste0("surveillance_", report_date, ".html"),
    output_dir = tempdir(),
    params = list(report_date = report_date)
  )
  
  # Create email
  email <- compose_email(
    body = md(paste0(
      "# Weekly Surveillance Report\n\n",
      "Please find attached the surveillance report for the week ending ",
      format(report_date, "%B %d, %Y"), ".\n\n",
      "Key highlights:\n",
      "- Total cases: ", total_cases, "\n",
      "- New cases this week: ", new_cases, "\n",
      "- Active outbreaks: ", active_outbreaks
    )),
    footer = md("Automated report generated by Surveillance System")
  ) %>%
  add_attachment(file = report_file)
  
  # Send email
  smtp_send(
    email = email,
    to = recipients,
    from = "surveillance@health.gov",
    subject = paste("Surveillance Report -", format(report_date, "%m/%d/%Y")),
    credentials = creds_file("email_creds.json")
  )
}

# Schedule weekly email
recipients <- c("director@health.gov", "epi-team@health.gov")
send_surveillance_report(recipients)
```

### Database Integration

#### Automated Data Updates
```r
# R/update_surveillance_data.R
library(DBI)
library(RPostgreSQL)

update_surveillance_dashboard <- function() {
  # Connect to database
  con <- dbConnect(
    PostgreSQL(),
    host = Sys.getenv("DB_HOST"),
    dbname = Sys.getenv("DB_NAME"),
    user = Sys.getenv("DB_USER"),
    password = Sys.getenv("DB_PASSWORD")
  )
  
  # Extract latest data
  latest_data <- dbGetQuery(con, "
    SELECT * FROM surveillance_cases 
    WHERE report_date >= CURRENT_DATE - INTERVAL '30 days'
  ")
  
  # Disconnect
  dbDisconnect(con)
  
  # Save for dashboard
  saveRDS(latest_data, "data/dashboard_data.rds")
  
  # Update dashboard
  render(
    input = "dashboards/surveillance_dashboard.Rmd",
    output_file = "surveillance_dashboard.html",
    output_dir = "output/dashboard"
  )
  
  # Log update
  cat("Dashboard updated:", Sys.time(), "\n", 
      file = "logs/dashboard_updates.log", append = TRUE)
}

# Run update
update_surveillance_dashboard()
```

This comprehensive guide covers the essential aspects of RMarkdown workflows for epidemiological research. The content provides detailed examples and practical implementations for creating reproducible, automated reporting systems.

Would you like me to continue with the next section covering .Rds files, project portability, Git workflows for R projects, and automated R pipelines? This would include detailed coverage of data serialization, project packaging, and deployment strategies. 
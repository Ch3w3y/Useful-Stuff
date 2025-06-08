# R Package Development Template
# Complete guide for creating professional R packages
# Author: Your Name
# Date: 2024

# =============================================================================
# PACKAGE STRUCTURE SETUP
# =============================================================================

# Install required development packages
if (!require("devtools")) install.packages("devtools")
if (!require("usethis")) install.packages("usethis")
if (!require("roxygen2")) install.packages("roxygen2")
if (!require("testthat")) install.packages("testthat")
if (!require("pkgdown")) install.packages("pkgdown")
if (!require("goodpractice")) install.packages("goodpractice")
if (!require("rhub")) install.packages("rhub")
if (!require("covr")) install.packages("covr")

library(devtools)
library(usethis)
library(roxygen2)
library(testthat)

# =============================================================================
# PACKAGE INITIALIZATION
# =============================================================================

#' Initialize a New R Package
#'
#' @param package_name Character string, name of the package
#' @param path Character string, path where package should be created
#' @param description Character string, brief description of package
#' @param author_name Character string, author name
#' @param author_email Character string, author email
#' @examples
#' \dontrun{
#' create_r_package("mypackage", "~/R/packages", "My awesome package", 
#'                  "Your Name", "your.email@example.com")
#' }
create_r_package <- function(package_name, path = ".", description = "What the Package Does",
                            author_name = "Your Name", author_email = "your.email@example.com") {
  
  # Create package structure
  usethis::create_package(file.path(path, package_name))
  
  # Set up basic package information
  usethis::use_description(
    fields = list(
      Title = tools::toTitleCase(gsub("[^a-zA-Z0-9]", " ", package_name)),
      Description = description,
      `Authors@R` = paste0(
        'person("', author_name, '", email = "', author_email, '", ',
        'role = c("aut", "cre"), comment = c(ORCID = "YOUR-ORCID-HERE"))'
      ),
      License = "MIT + file LICENSE",
      Encoding = "UTF-8",
      LazyData = "true",
      RoxygenNote = as.character(packageVersion("roxygen2")),
      Depends = "R (>= 3.5.0)",
      URL = paste0("https://github.com/yourusername/", package_name),
      BugReports = paste0("https://github.com/yourusername/", package_name, "/issues")
    )
  )
  
  # Set up license
  usethis::use_mit_license(author_name)
  
  # Set up roxygen2 for documentation
  usethis::use_roxygen_md()
  
  # Set up testing
  usethis::use_testthat()
  
  # Set up version control
  usethis::use_git()
  
  # Set up R build ignore
  usethis::use_build_ignore(c("^.*\\.Rproj$", "^\\.Rproj\\.user$", "^README\\.Rmd$"))
  
  # Create README
  usethis::use_readme_rmd()
  
  # Set up continuous integration
  usethis::use_github_actions_badge("R-CMD-check")
  usethis::use_github_action("check-standard")
  usethis::use_github_action("test-coverage")
  
  # Set up package documentation website
  usethis::use_pkgdown()
  
  # Create basic vignette
  usethis::use_vignette(paste0("intro-to-", package_name))
  
  cat("Package structure created successfully!\n")
  cat("Next steps:\n")
  cat("1. Edit DESCRIPTION file\n")
  cat("2. Add functions to R/ directory\n")
  cat("3. Document functions with roxygen2\n")
  cat("4. Add tests to tests/testthat/\n")
  cat("5. Build and check package\n")
}

# =============================================================================
# FUNCTION DEVELOPMENT TEMPLATES
# =============================================================================

#' Template for Data Processing Function
#'
#' This is a template showing best practices for R function development
#' including parameter validation, error handling, and documentation.
#'
#' @param data A data.frame or tibble containing the data to process
#' @param column_name Character string, name of column to process
#' @param method Character string, processing method ("normalize", "standardize", "log")
#' @param na_action Character string, how to handle missing values ("remove", "impute", "keep")
#' @param ... Additional arguments passed to processing functions
#'
#' @return A data.frame with processed data
#' @export
#'
#' @examples
#' \dontrun{
#' # Create sample data
#' sample_data <- data.frame(
#'   x = c(1, 2, 3, NA, 5),
#'   y = c(10, 20, 30, 40, 50)
#' )
#' 
#' # Process data
#' result <- process_data(sample_data, "x", method = "normalize")
#' }
#'
#' @seealso \code{\link{validate_data}}, \code{\link{handle_missing}}
process_data <- function(data, column_name, method = "normalize", 
                        na_action = "remove", ...) {
  
  # Input validation
  if (!is.data.frame(data)) {
    stop("Input 'data' must be a data.frame or tibble", call. = FALSE)
  }
  
  if (!column_name %in% names(data)) {
    stop(paste("Column", column_name, "not found in data"), call. = FALSE)
  }
  
  if (!method %in% c("normalize", "standardize", "log")) {
    stop("Method must be one of: 'normalize', 'standardize', 'log'", call. = FALSE)
  }
  
  # Extract column
  column_data <- data[[column_name]]
  
  # Handle missing values
  if (na_action == "remove") {
    valid_indices <- !is.na(column_data)
    column_data <- column_data[valid_indices]
    data <- data[valid_indices, ]
  } else if (na_action == "impute") {
    column_data[is.na(column_data)] <- mean(column_data, na.rm = TRUE)
  }
  
  # Apply processing method
  processed_data <- switch(method,
    "normalize" = (column_data - min(column_data, na.rm = TRUE)) / 
                  (max(column_data, na.rm = TRUE) - min(column_data, na.rm = TRUE)),
    "standardize" = (column_data - mean(column_data, na.rm = TRUE)) / 
                    sd(column_data, na.rm = TRUE),
    "log" = {
      if (any(column_data <= 0, na.rm = TRUE)) {
        warning("Log transformation applied to non-positive values", call. = FALSE)
      }
      log(column_data)
    }
  )
  
  # Update data
  data[[column_name]] <- processed_data
  
  # Add attributes for traceability
  attr(data, "processing_method") <- method
  attr(data, "processed_column") <- column_name
  attr(data, "processing_timestamp") <- Sys.time()
  
  return(data)
}

#' Data Validation Function
#'
#' @param data Data.frame to validate
#' @param required_columns Character vector of required column names
#' @param numeric_columns Character vector of columns that should be numeric
#' @param min_rows Integer, minimum required number of rows
#'
#' @return Logical, TRUE if data passes validation
#' @export
validate_data <- function(data, required_columns = NULL, 
                         numeric_columns = NULL, min_rows = 1) {
  
  # Check if data is a data.frame
  if (!is.data.frame(data)) {
    stop("Input must be a data.frame", call. = FALSE)
  }
  
  # Check minimum rows
  if (nrow(data) < min_rows) {
    stop(paste("Data must have at least", min_rows, "rows"), call. = FALSE)
  }
  
  # Check required columns
  if (!is.null(required_columns)) {
    missing_cols <- setdiff(required_columns, names(data))
    if (length(missing_cols) > 0) {
      stop(paste("Missing required columns:", paste(missing_cols, collapse = ", ")), 
           call. = FALSE)
    }
  }
  
  # Check numeric columns
  if (!is.null(numeric_columns)) {
    for (col in numeric_columns) {
      if (!is.numeric(data[[col]])) {
        stop(paste("Column", col, "must be numeric"), call. = FALSE)
      }
    }
  }
  
  return(TRUE)
}

# =============================================================================
# PACKAGE TESTING TEMPLATES
# =============================================================================

#' Create Test Files Template
#'
#' @param function_name Character string, name of function to test
#' @param package_name Character string, name of package
create_test_template <- function(function_name, package_name) {
  
  test_content <- paste0('
# Test file for ', function_name, '
# Generated on ', Sys.Date(), '

library(testthat)
library(', package_name, ')

test_that("', function_name, ' works with valid input", {
  # Arrange
  test_data <- data.frame(
    x = 1:10,
    y = rnorm(10)
  )
  
  # Act
  result <- ', function_name, '(test_data, "x")
  
  # Assert
  expect_is(result, "data.frame")
  expect_equal(nrow(result), 10)
  expect_true("x" %in% names(result))
})

test_that("', function_name, ' handles missing values correctly", {
  # Arrange
  test_data <- data.frame(
    x = c(1, 2, NA, 4, 5),
    y = 1:5
  )
  
  # Act & Assert
  expect_no_error(', function_name, '(test_data, "x", na_action = "remove"))
  expect_no_error(', function_name, '(test_data, "x", na_action = "impute"))
})

test_that("', function_name, ' validates input correctly", {
  # Test invalid data type
  expect_error(', function_name, '("not_a_dataframe", "x"))
  
  # Test missing column
  test_data <- data.frame(x = 1:5)
  expect_error(', function_name, '(test_data, "missing_column"))
  
  # Test invalid method
  expect_error(', function_name, '(test_data, "x", method = "invalid_method"))
})

test_that("', function_name, ' preserves data attributes", {
  # Arrange
  test_data <- data.frame(x = 1:5, y = 1:5)
  
  # Act
  result <- ', function_name, '(test_data, "x")
  
  # Assert
  expect_equal(attr(result, "processed_column"), "x")
  expect_true(!is.null(attr(result, "processing_timestamp")))
})
')
  
  # Write test file
  test_dir <- "tests/testthat"
  if (!dir.exists(test_dir)) {
    dir.create(test_dir, recursive = TRUE)
  }
  
  writeLines(test_content, file.path(test_dir, paste0("test-", function_name, ".R")))
  cat("Test template created for", function_name, "\n")
}

# =============================================================================
# DOCUMENTATION HELPERS
# =============================================================================

#' Generate Package Documentation
#'
#' @param package_path Character string, path to package directory
generate_documentation <- function(package_path = ".") {
  
  # Generate documentation from roxygen comments
  roxygen2::roxygenise(package_path)
  
  # Generate README from README.Rmd
  if (file.exists(file.path(package_path, "README.Rmd"))) {
    rmarkdown::render(file.path(package_path, "README.Rmd"))
  }
  
  # Generate pkgdown site
  if (file.exists(file.path(package_path, "_pkgdown.yml"))) {
    pkgdown::build_site(pkg = package_path)
  }
  
  cat("Documentation generated successfully!\n")
}

# =============================================================================
# PACKAGE CHECKING AND VALIDATION
# =============================================================================

#' Comprehensive Package Check
#'
#' @param package_path Character string, path to package directory
#' @param run_tests Logical, whether to run tests
#' @param check_coverage Logical, whether to check test coverage
comprehensive_check <- function(package_path = ".", run_tests = TRUE, 
                              check_coverage = TRUE) {
  
  cat("Running comprehensive package check...\n")
  
  # 1. Check package structure
  cat("1. Checking package structure...\n")
  devtools::check(pkg = package_path, quiet = TRUE)
  
  # 2. Run tests
  if (run_tests) {
    cat("2. Running tests...\n")
    test_results <- devtools::test(pkg = package_path, quiet = TRUE)
    print(test_results)
  }
  
  # 3. Check test coverage
  if (check_coverage) {
    cat("3. Checking test coverage...\n")
    coverage <- covr::package_coverage(path = package_path)
    print(coverage)
  }
  
  # 4. Good practice check
  cat("4. Checking good practices...\n")
  gp_results <- goodpractice::gp(path = package_path)
  print(gp_results)
  
  # 5. Spell check
  cat("5. Checking spelling...\n")
  spelling_errors <- spelling::spell_check_package(pkg = package_path)
  if (length(spelling_errors) > 0) {
    cat("Spelling errors found:\n")
    print(spelling_errors)
  } else {
    cat("No spelling errors found.\n")
  }
  
  cat("Package check completed!\n")
}

# =============================================================================
# CONTINUOUS INTEGRATION SETUP
# =============================================================================

#' Setup GitHub Actions for R Package
#'
#' @param package_path Character string, path to package directory
setup_github_actions <- function(package_path = ".") {
  
  # Standard R CMD check
  usethis::use_github_action("check-standard")
  
  # Test coverage
  usethis::use_github_action("test-coverage")
  
  # Render README
  usethis::use_github_action("render-readme")
  
  # Build and deploy pkgdown site
  usethis::use_github_action("pkgdown")
  
  # Create custom workflow for comprehensive checks
  workflow_content <- '
name: Comprehensive R Package Check

on:
  push:
    branches: [ main, master ]
  pull_request:
    branches: [ main, master ]

jobs:
  R-CMD-check:
    runs-on: ${{ matrix.config.os }}
    
    name: ${{ matrix.config.os }} (${{ matrix.config.r }})
    
    strategy:
      fail-fast: false
      matrix:
        config:
          - {os: windows-latest, r: "release"}
          - {os: macOS-latest, r: "release"}
          - {os: ubuntu-20.04, r: "release", rspm: "https://packagemanager.rstudio.com/cran/__linux__/focal/latest"}
          - {os: ubuntu-20.04, r: "devel", rspm: "https://packagemanager.rstudio.com/cran/__linux__/focal/latest"}
    
    env:
      R_REMOTES_NO_ERRORS_FROM_WARNINGS: true
      RSPM: ${{ matrix.config.rspm }}
      GITHUB_PAT: ${{ secrets.GITHUB_TOKEN }}
    
    steps:
      - uses: actions/checkout@v2
      
      - uses: r-lib/actions/setup-r@v1
        with:
          r-version: ${{ matrix.config.r }}
          
      - uses: r-lib/actions/setup-pandoc@v1
      
      - name: Query dependencies
        run: |
          install.packages("remotes")
          saveRDS(remotes::dev_package_deps(dependencies = TRUE), ".github/depends.Rds", version = 2)
          writeLines(sprintf("R-%i.%i", getRversion()$major, getRversion()$minor), ".github/R-version")
        shell: Rscript {0}
        
      - name: Cache R packages
        if: runner.os != "Windows"
        uses: actions/cache@v2
        with:
          path: ${{ env.R_LIBS_USER }}
          key: ${{ runner.os }}-${{ hashFiles(".github/R-version") }}-1-${{ hashFiles(".github/depends.Rds") }}
          restore-keys: ${{ runner.os }}-${{ hashFiles(".github/R-version") }}-1-
          
      - name: Install dependencies
        run: |
          remotes::install_deps(dependencies = TRUE)
          remotes::install_cran("rcmdcheck")
          remotes::install_cran("goodpractice")
          remotes::install_cran("covr")
        shell: Rscript {0}
        
      - name: Check
        env:
          _R_CHECK_CRAN_INCOMING_REMOTE_: false
        run: rcmdcheck::rcmdcheck(args = c("--no-manual", "--as-cran"), error_on = "warning", check_dir = "check")
        shell: Rscript {0}
        
      - name: Good Practice Check
        run: |
          gp_results <- goodpractice::gp(".")
          print(gp_results)
        shell: Rscript {0}
        
      - name: Test coverage
        if: runner.os == "Linux" && matrix.config.r == "release"
        run: covr::codecov()
        shell: Rscript {0}
'
  
  # Write workflow file
  workflow_dir <- file.path(package_path, ".github", "workflows")
  if (!dir.exists(workflow_dir)) {
    dir.create(workflow_dir, recursive = TRUE)
  }
  
  writeLines(workflow_content, file.path(workflow_dir, "comprehensive-check.yml"))
  
  cat("GitHub Actions setup completed!\n")
}

# =============================================================================
# PACKAGE RELEASE HELPERS
# =============================================================================

#' Prepare Package for Release
#'
#' @param package_path Character string, path to package directory
#' @param version Character string, new version number
#' @param news_entry Character string, news entry for this version
prepare_release <- function(package_path = ".", version = NULL, news_entry = NULL) {
  
  # Update version number
  if (!is.null(version)) {
    usethis::use_version(version)
  }
  
  # Update NEWS.md
  if (!is.null(news_entry)) {
    if (!file.exists(file.path(package_path, "NEWS.md"))) {
      usethis::use_news_md()
    }
    
    # Add news entry (this would need manual editing in practice)
    cat("Please manually add the following to NEWS.md:\n")
    cat(paste0("# Package version ", version, "\n\n"))
    cat(paste0("* ", news_entry, "\n\n"))
  }
  
  # Run comprehensive check
  comprehensive_check(package_path)
  
  # Generate documentation
  generate_documentation(package_path)
  
  # Submit to CRAN (preparation steps)
  cat("\nRelease preparation completed!\n")
  cat("Next steps for CRAN submission:\n")
  cat("1. Run devtools::release() for interactive release process\n")
  cat("2. Or run devtools::submit_cran() to submit directly\n")
  cat("3. Check email for CRAN feedback\n")
}

# =============================================================================
# EXAMPLE USAGE
# =============================================================================

# Example of how to use these functions:
# 
# # Create a new package
# create_r_package("myanalysispackage", "~/R/packages", 
#                  "Advanced Data Analysis Tools", 
#                  "Your Name", "your.email@example.com")
# 
# # Create test templates
# create_test_template("process_data", "myanalysispackage")
# 
# # Setup CI/CD
# setup_github_actions()
# 
# # Check package
# comprehensive_check()
# 
# # Prepare for release
# prepare_release(version = "1.0.0", news_entry = "Initial CRAN release")

cat("R Package Development Template loaded successfully!\n")
cat("Use create_r_package() to start a new package\n")
cat("Use comprehensive_check() to validate your package\n")
cat("Use prepare_release() when ready to publish\n") 
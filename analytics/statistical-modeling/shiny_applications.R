# ===============================================================================
# SHINY APPLICATIONS FOR DATA SCIENCE - COMPREHENSIVE TEMPLATE
# ===============================================================================
# Complete guide to building interactive data science dashboards with Shiny
# Author: Data Science Toolkit
# Last Updated: 2024
# ===============================================================================

# Required Libraries
suppressMessages({
  library(shiny)
  library(shinydashboard)
  library(shinythemes)
  library(DT)
  library(plotly)
  library(leaflet)
  library(dplyr)
  library(ggplot2)
  library(readr)
  library(lubridate)
  library(shinyWidgets)
  library(shinycssloaders)
  library(shinyjs)
  library(fresh)
  library(waiter)
  library(bslib)
  library(thematic)
})

# ===============================================================================
# 1. BASIC SHINY APPLICATION TEMPLATE
# ===============================================================================

#' Create Basic Data Explorer App
#' @param data Data frame to explore
create_basic_explorer <- function(data = mtcars) {
  
  # UI
  ui <- fluidPage(
    theme = bs_theme(bootswatch = "flatly"),
    
    titlePanel("Data Explorer Dashboard"),
    
    sidebarLayout(
      sidebarPanel(
        h3("Controls"),
        
        # Data upload
        fileInput("file", "Upload CSV File",
                 accept = c(".csv")),
        
        # Variable selection
        conditionalPanel(
          condition = "output.data_loaded",
          selectInput("x_var", "X Variable:", choices = NULL),
          selectInput("y_var", "Y Variable:", choices = NULL),
          selectInput("color_var", "Color Variable:", choices = NULL),
          
          # Plot type
          radioButtons("plot_type", "Plot Type:",
                      choices = list("Scatter" = "scatter",
                                   "Histogram" = "histogram",
                                   "Boxplot" = "boxplot")),
          
          # Download data
          downloadButton("download", "Download Data", class = "btn-primary")
        )
      ),
      
      mainPanel(
        tabsetPanel(
          tabPanel("Plot", 
                  withSpinner(plotlyOutput("main_plot", height = "500px"))),
          tabPanel("Data Table", 
                  DTOutput("data_table")),
          tabPanel("Summary", 
                  verbatimTextOutput("summary"))
        )
      )
    )
  )
  
  # Server
  server <- function(input, output, session) {
    
    # Reactive data
    data_reactive <- reactive({
      if (is.null(input$file)) {
        data
      } else {
        read_csv(input$file$datapath)
      }
    })
    
    # Update variable choices
    observe({
      df <- data_reactive()
      choices <- names(df)
      
      updateSelectInput(session, "x_var", choices = choices, selected = choices[1])
      updateSelectInput(session, "y_var", choices = choices, selected = choices[2])
      updateSelectInput(session, "color_var", 
                       choices = c("None" = "", choices), selected = "")
    })
    
    # Data loaded flag
    output$data_loaded <- reactive({
      !is.null(data_reactive())
    })
    outputOptions(output, "data_loaded", suspendWhenHidden = FALSE)
    
    # Main plot
    output$main_plot <- renderPlotly({
      df <- data_reactive()
      req(input$x_var, input$y_var)
      
      if (input$plot_type == "scatter") {
        p <- ggplot(df, aes_string(x = input$x_var, y = input$y_var))
        if (input$color_var != "") {
          p <- p + aes_string(color = input$color_var)
        }
        p <- p + geom_point(alpha = 0.7) + theme_minimal()
        
      } else if (input$plot_type == "histogram") {
        p <- ggplot(df, aes_string(x = input$x_var)) +
          geom_histogram(bins = 30, alpha = 0.7, fill = "steelblue") +
          theme_minimal()
        
      } else if (input$plot_type == "boxplot") {
        p <- ggplot(df, aes_string(x = input$color_var, y = input$y_var)) +
          geom_boxplot(alpha = 0.7) + theme_minimal()
      }
      
      ggplotly(p)
    })
    
    # Data table
    output$data_table <- renderDT({
      datatable(data_reactive(), options = list(scrollX = TRUE))
    })
    
    # Summary
    output$summary <- renderText({
      capture.output(summary(data_reactive()))
    })
    
    # Download handler
    output$download <- downloadHandler(
      filename = function() {
        paste("data_", Sys.Date(), ".csv", sep = "")
      },
      content = function(file) {
        write_csv(data_reactive(), file)
      }
    )
  }
  
  return(list(ui = ui, server = server))
}

# ===============================================================================
# 2. ADVANCED DASHBOARD TEMPLATE
# ===============================================================================

#' Create Advanced Analytics Dashboard
create_analytics_dashboard <- function() {
  
  # Custom CSS
  custom_css <- "
    .content-wrapper, .right-side {
      background-color: #f4f4f4;
    }
    .box {
      border-top: 3px solid #3c8dbc;
    }
    .small-box {
      border-radius: 10px;
    }
  "
  
  # Header
  header <- dashboardHeader(
    title = "Analytics Dashboard",
    tags$li(class = "dropdown",
           tags$a(href = "https://github.com", 
                  icon("github"), "Source Code"))
  )
  
  # Sidebar
  sidebar <- dashboardSidebar(
    sidebarMenu(
      menuItem("Overview", tabName = "overview", icon = icon("tachometer-alt")),
      menuItem("Data Analysis", tabName = "analysis", icon = icon("chart-line")),
      menuItem("Geospatial", tabName = "geo", icon = icon("map")),
      menuItem("Time Series", tabName = "timeseries", icon = icon("clock")),
      menuItem("Machine Learning", tabName = "ml", icon = icon("robot")),
      menuItem("Settings", tabName = "settings", icon = icon("cog"))
    )
  )
  
  # Body
  body <- dashboardBody(
    tags$head(tags$style(HTML(custom_css))),
    
    tabItems(
      # Overview tab
      tabItem(tabName = "overview",
        fluidRow(
          # Value boxes
          valueBoxOutput("total_records"),
          valueBoxOutput("avg_value"),
          valueBoxOutput("completion_rate")
        ),
        
        fluidRow(
          box(
            title = "Key Metrics", status = "primary", solidHeader = TRUE,
            width = 8, height = 400,
            plotlyOutput("overview_plot")
          ),
          
          box(
            title = "Controls", status = "warning", solidHeader = TRUE,
            width = 4, height = 400,
            
            dateRangeInput("date_range", "Date Range:",
                          start = Sys.Date() - 30, end = Sys.Date()),
            
            selectInput("metric", "Metric:",
                       choices = c("Sales" = "sales", 
                                 "Revenue" = "revenue",
                                 "Users" = "users")),
            
            actionButton("refresh", "Refresh Data", 
                        class = "btn-primary btn-block")
          )
        )
      ),
      
      # Data Analysis tab
      tabItem(tabName = "analysis",
        fluidRow(
          box(
            title = "Correlation Analysis", status = "primary", 
            solidHeader = TRUE, width = 6,
            plotlyOutput("correlation_plot")
          ),
          
          box(
            title = "Distribution Analysis", status = "success",
            solidHeader = TRUE, width = 6,
            plotlyOutput("distribution_plot")
          )
        ),
        
        fluidRow(
          box(
            title = "Data Table", status = "info", 
            solidHeader = TRUE, width = 12,
            DTOutput("analysis_table")
          )
        )
      ),
      
      # Geospatial tab
      tabItem(tabName = "geo",
        fluidRow(
          box(
            title = "Interactive Map", status = "primary",
            solidHeader = TRUE, width = 12, height = 600,
            leafletOutput("map", height = 500)
          )
        )
      ),
      
      # Time Series tab
      tabItem(tabName = "timeseries",
        fluidRow(
          box(
            title = "Time Series Plot", status = "primary",
            solidHeader = TRUE, width = 12,
            plotlyOutput("timeseries_plot", height = 400)
          )
        ),
        
        fluidRow(
          box(
            title = "Forecast", status = "warning",
            solidHeader = TRUE, width = 6,
            plotlyOutput("forecast_plot")
          ),
          
          box(
            title = "Decomposition", status = "success",
            solidHeader = TRUE, width = 6,
            plotlyOutput("decomposition_plot")
          )
        )
      ),
      
      # Machine Learning tab
      tabItem(tabName = "ml",
        fluidRow(
          box(
            title = "Model Performance", status = "primary",
            solidHeader = TRUE, width = 8,
            plotlyOutput("model_performance")
          ),
          
          box(
            title = "Feature Importance", status = "warning",
            solidHeader = TRUE, width = 4,
            plotlyOutput("feature_importance")
          )
        )
      ),
      
      # Settings tab
      tabItem(tabName = "settings",
        fluidRow(
          box(
            title = "Application Settings", status = "primary",
            solidHeader = TRUE, width = 12,
            
            h4("Data Source"),
            radioButtons("data_source", "Select Data Source:",
                        choices = list("Sample Data" = "sample",
                                     "Upload File" = "upload",
                                     "Database" = "database")),
            
            conditionalPanel(
              condition = "input.data_source == 'upload'",
              fileInput("data_file", "Choose CSV File")
            ),
            
            h4("Display Options"),
            checkboxInput("show_animations", "Enable Animations", TRUE),
            sliderInput("plot_height", "Plot Height:", 
                       min = 300, max = 800, value = 400),
            
            actionButton("save_settings", "Save Settings", 
                        class = "btn-success")
          )
        )
      )
    )
  )
  
  # UI
  ui <- dashboardPage(header, sidebar, body)
  
  # Server
  server <- function(input, output, session) {
    
    # Sample data generation
    sample_data <- reactive({
      set.seed(123)
      dates <- seq(as.Date("2023-01-01"), as.Date("2024-12-31"), by = "day")
      data.frame(
        date = dates,
        sales = abs(rnorm(length(dates), 1000, 200)),
        revenue = abs(rnorm(length(dates), 50000, 10000)),
        users = abs(rnorm(length(dates), 500, 100)),
        lat = rnorm(length(dates), 40.7, 0.1),
        lng = rnorm(length(dates), -74.0, 0.1)
      )
    })
    
    # Value boxes
    output$total_records <- renderValueBox({
      valueBox(
        value = nrow(sample_data()),
        subtitle = "Total Records",
        icon = icon("database"),
        color = "blue"
      )
    })
    
    output$avg_value <- renderValueBox({
      valueBox(
        value = paste0("$", round(mean(sample_data()$revenue), 0)),
        subtitle = "Average Revenue",
        icon = icon("dollar-sign"),
        color = "green"
      )
    })
    
    output$completion_rate <- renderValueBox({
      valueBox(
        value = "98.5%",
        subtitle = "Data Completeness",
        icon = icon("check-circle"),
        color = "yellow"
      )
    })
    
    # Overview plot
    output$overview_plot <- renderPlotly({
      df <- sample_data()
      
      p <- df %>%
        filter(date >= input$date_range[1] & date <= input$date_range[2]) %>%
        ggplot(aes_string(x = "date", y = input$metric)) +
        geom_line(color = "steelblue", size = 1) +
        geom_smooth(method = "loess", se = FALSE, color = "red") +
        theme_minimal() +
        labs(title = paste("Trend Analysis -", input$metric),
             x = "Date", y = tools::toTitleCase(input$metric))
      
      ggplotly(p)
    })
    
    # Correlation plot
    output$correlation_plot <- renderPlotly({
      df <- sample_data()
      cor_matrix <- cor(df[c("sales", "revenue", "users")])
      
      p <- plot_ly(
        x = rownames(cor_matrix),
        y = colnames(cor_matrix),
        z = cor_matrix,
        type = "heatmap",
        colors = "RdBu"
      ) %>%
        layout(title = "Correlation Matrix")
      
      p
    })
    
    # Distribution plot
    output$distribution_plot <- renderPlotly({
      df <- sample_data()
      
      p <- df %>%
        ggplot(aes_string(x = input$metric)) +
        geom_histogram(bins = 30, fill = "lightblue", alpha = 0.7) +
        geom_density(aes(y = ..count..), color = "red", size = 1) +
        theme_minimal() +
        labs(title = paste("Distribution of", input$metric))
      
      ggplotly(p)
    })
    
    # Analysis table
    output$analysis_table <- renderDT({
      datatable(sample_data(), 
                options = list(scrollX = TRUE, pageLength = 10))
    })
    
    # Map
    output$map <- renderLeaflet({
      df <- sample_data()
      
      leaflet(df) %>%
        addTiles() %>%
        addCircleMarkers(
          lng = ~lng, lat = ~lat,
          radius = ~sqrt(sales)/10,
          popup = ~paste("Sales:", sales, "<br>Date:", date),
          color = "blue",
          fillOpacity = 0.6
        ) %>%
        setView(lng = -74.0, lat = 40.7, zoom = 10)
    })
    
    # Time series plot
    output$timeseries_plot <- renderPlotly({
      df <- sample_data()
      
      p <- plot_ly(df, x = ~date, y = ~get(input$metric), type = "scatter", mode = "lines") %>%
        layout(title = paste("Time Series -", input$metric),
               xaxis = list(title = "Date"),
               yaxis = list(title = tools::toTitleCase(input$metric)))
      
      p
    })
    
    # Forecast plot (simplified)
    output$forecast_plot <- renderPlotly({
      df <- sample_data()
      
      # Simple linear trend for demonstration
      future_dates <- seq(max(df$date) + 1, max(df$date) + 30, by = "day")
      trend_model <- lm(get(input$metric) ~ as.numeric(date), data = df)
      forecast_values <- predict(trend_model, 
                                newdata = data.frame(date = as.numeric(future_dates)))
      
      forecast_df <- data.frame(
        date = future_dates,
        value = forecast_values,
        type = "Forecast"
      )
      
      historical_df <- data.frame(
        date = df$date,
        value = df[[input$metric]],
        type = "Historical"
      )
      
      combined_df <- rbind(historical_df, forecast_df)
      
      plot_ly(combined_df, x = ~date, y = ~value, color = ~type, type = "scatter", mode = "lines") %>%
        layout(title = "Forecast vs Historical")
    })
    
    # Model performance plot
    output$model_performance <- renderPlotly({
      # Simulated model performance metrics
      metrics <- data.frame(
        Model = c("Random Forest", "Linear Regression", "XGBoost", "SVM"),
        Accuracy = c(0.92, 0.85, 0.94, 0.88),
        Precision = c(0.91, 0.83, 0.93, 0.86),
        Recall = c(0.90, 0.84, 0.92, 0.87)
      )
      
      p <- metrics %>%
        tidyr::pivot_longer(cols = c(Accuracy, Precision, Recall), 
                           names_to = "Metric", values_to = "Score") %>%
        ggplot(aes(x = Model, y = Score, fill = Metric)) +
        geom_bar(stat = "identity", position = "dodge") +
        theme_minimal() +
        labs(title = "Model Performance Comparison")
      
      ggplotly(p)
    })
    
    # Feature importance
    output$feature_importance <- renderPlotly({
      features <- data.frame(
        Feature = c("Price", "Quality", "Brand", "Reviews", "Location"),
        Importance = c(0.35, 0.28, 0.18, 0.12, 0.07)
      )
      
      p <- features %>%
        ggplot(aes(x = reorder(Feature, Importance), y = Importance)) +
        geom_col(fill = "steelblue") +
        coord_flip() +
        theme_minimal() +
        labs(title = "Feature Importance", x = "Features", y = "Importance")
      
      ggplotly(p)
    })
  }
  
  return(list(ui = ui, server = server))
}

# ===============================================================================
# 3. MACHINE LEARNING DASHBOARD
# ===============================================================================

#' Create ML Model Comparison Dashboard
create_ml_dashboard <- function() {
  
  ui <- fluidPage(
    theme = bs_theme(bootswatch = "cosmo"),
    
    titlePanel("Machine Learning Model Comparison"),
    
    sidebarLayout(
      sidebarPanel(
        width = 3,
        
        h4("Data Configuration"),
        fileInput("ml_data", "Upload Training Data (.csv)"),
        
        conditionalPanel(
          condition = "output.data_uploaded",
          
          selectInput("target_var", "Target Variable:", choices = NULL),
          selectInput("feature_vars", "Feature Variables:", 
                     choices = NULL, multiple = TRUE),
          
          h4("Model Configuration"),
          checkboxGroupInput("models", "Select Models:",
                           choices = list(
                             "Linear Regression" = "lm",
                             "Random Forest" = "rf",
                             "Gradient Boosting" = "gbm",
                             "Support Vector Machine" = "svm"
                           ),
                           selected = c("lm", "rf")),
          
          sliderInput("train_split", "Training Split:", 
                     min = 0.5, max = 0.9, value = 0.8, step = 0.05),
          
          actionButton("train_models", "Train Models", 
                      class = "btn-primary btn-block"),
          
          br(), br(),
          
          conditionalPanel(
            condition = "output.models_trained",
            downloadButton("download_results", "Download Results",
                          class = "btn-success btn-block")
          )
        )
      ),
      
      mainPanel(
        width = 9,
        
        tabsetPanel(
          tabPanel("Data Overview",
                  fluidRow(
                    column(6, plotlyOutput("data_distribution")),
                    column(6, plotlyOutput("correlation_heatmap"))
                  ),
                  br(),
                  DTOutput("data_preview")
          ),
          
          tabPanel("Model Results",
                  conditionalPanel(
                    condition = "output.models_trained",
                    
                    fluidRow(
                      column(6, plotlyOutput("model_comparison")),
                      column(6, plotlyOutput("residual_plots"))
                    ),
                    
                    br(),
                    
                    fluidRow(
                      column(12, DTOutput("model_metrics"))
                    )
                  )
          ),
          
          tabPanel("Predictions",
                  conditionalPanel(
                    condition = "output.models_trained",
                    
                    fluidRow(
                      column(4,
                        wellPanel(
                          h4("Make Prediction"),
                          uiOutput("prediction_inputs"),
                          actionButton("predict", "Predict", class = "btn-warning")
                        )
                      ),
                      column(8,
                        h4("Prediction Results"),
                        verbatimTextOutput("prediction_results")
                      )
                    )
                  )
          )
        )
      )
    )
  )
  
  server <- function(input, output, session) {
    
    # Reactive values
    values <- reactiveValues(
      data = NULL,
      models = NULL,
      results = NULL
    )
    
    # Data upload
    observeEvent(input$ml_data, {
      req(input$ml_data)
      
      ext <- tools::file_ext(input$ml_data$datapath)
      if (ext != "csv") {
        showNotification("Please upload a CSV file", type = "error")
        return()
      }
      
      values$data <- read_csv(input$ml_data$datapath)
      
      # Update variable choices
      numeric_vars <- names(select_if(values$data, is.numeric))
      all_vars <- names(values$data)
      
      updateSelectInput(session, "target_var", choices = numeric_vars)
      updateSelectInput(session, "feature_vars", choices = all_vars)
    })
    
    # Data uploaded flag
    output$data_uploaded <- reactive({
      !is.null(values$data)
    })
    outputOptions(output, "data_uploaded", suspendWhenHidden = FALSE)
    
    # Models trained flag
    output$models_trained <- reactive({
      !is.null(values$results)
    })
    outputOptions(output, "models_trained", suspendWhenHidden = FALSE)
    
    # Data visualization
    output$data_distribution <- renderPlotly({
      req(values$data, input$target_var)
      
      p <- ggplot(values$data, aes_string(x = input$target_var)) +
        geom_histogram(bins = 30, fill = "lightblue", alpha = 0.7) +
        theme_minimal() +
        labs(title = paste("Distribution of", input$target_var))
      
      ggplotly(p)
    })
    
    output$correlation_heatmap <- renderPlotly({
      req(values$data, input$feature_vars)
      
      numeric_data <- select_if(values$data, is.numeric)
      cor_matrix <- cor(numeric_data, use = "complete.obs")
      
      plot_ly(
        x = colnames(cor_matrix),
        y = rownames(cor_matrix),
        z = cor_matrix,
        type = "heatmap",
        colors = "RdBu"
      ) %>%
        layout(title = "Feature Correlation Matrix")
    })
    
    output$data_preview <- renderDT({
      req(values$data)
      datatable(values$data, options = list(scrollX = TRUE, pageLength = 5))
    })
    
    # Model training
    observeEvent(input$train_models, {
      req(values$data, input$target_var, input$feature_vars, input$models)
      
      withProgress(message = "Training models...", {
        
        # Prepare data
        df <- values$data %>%
          select(all_of(c(input$target_var, input$feature_vars))) %>%
          na.omit()
        
        # Train-test split
        set.seed(123)
        train_idx <- sample(nrow(df), nrow(df) * input$train_split)
        train_data <- df[train_idx, ]
        test_data <- df[-train_idx, ]
        
        # Train models
        models <- list()
        predictions <- list()
        metrics <- data.frame()
        
        if ("lm" %in% input$models) {
          incProgress(0.2, detail = "Training Linear Regression...")
          
          formula_str <- paste(input$target_var, "~", paste(input$feature_vars, collapse = " + "))
          models$lm <- lm(as.formula(formula_str), data = train_data)
          predictions$lm <- predict(models$lm, test_data)
          
          rmse <- sqrt(mean((test_data[[input$target_var]] - predictions$lm)^2))
          mae <- mean(abs(test_data[[input$target_var]] - predictions$lm))
          r2 <- summary(models$lm)$r.squared
          
          metrics <- rbind(metrics, data.frame(
            Model = "Linear Regression",
            RMSE = rmse,
            MAE = mae,
            R_squared = r2
          ))
        }
        
        # Additional models would be implemented similarly
        # This is a simplified example
        
        values$models <- models
        values$results <- list(
          predictions = predictions,
          metrics = metrics,
          test_data = test_data
        )
        
        showNotification("Models trained successfully!", type = "success")
      })
    })
    
    # Model comparison plot
    output$model_comparison <- renderPlotly({
      req(values$results)
      
      p <- values$results$metrics %>%
        ggplot(aes(x = Model, y = RMSE, fill = Model)) +
        geom_col() +
        theme_minimal() +
        labs(title = "Model Performance (RMSE)")
      
      ggplotly(p)
    })
    
    # Model metrics table
    output$model_metrics <- renderDT({
      req(values$results)
      
      datatable(values$results$metrics, 
                options = list(dom = 't'), 
                rownames = FALSE) %>%
        formatRound(columns = c("RMSE", "MAE", "R_squared"), digits = 4)
    })
    
    # Prediction interface
    output$prediction_inputs <- renderUI({
      req(input$feature_vars)
      
      inputs <- lapply(input$feature_vars, function(var) {
        if (is.numeric(values$data[[var]])) {
          numericInput(paste0("pred_", var), var, 
                      value = mean(values$data[[var]], na.rm = TRUE))
        } else {
          selectInput(paste0("pred_", var), var,
                     choices = unique(values$data[[var]]))
        }
      })
      
      do.call(tagList, inputs)
    })
    
    # Make predictions
    observeEvent(input$predict, {
      req(values$models, input$feature_vars)
      
      # Collect input values
      pred_data <- data.frame(
        stringsAsFactors = FALSE
      )
      
      for (var in input$feature_vars) {
        pred_data[[var]] <- input[[paste0("pred_", var)]]
      }
      
      # Make predictions with available models
      results <- character()
      
      if ("lm" %in% names(values$models)) {
        pred <- predict(values$models$lm, pred_data)
        results <- c(results, paste("Linear Regression:", round(pred, 2)))
      }
      
      output$prediction_results <- renderText({
        paste(results, collapse = "\n")
      })
    })
  }
  
  return(list(ui = ui, server = server))
}

# ===============================================================================
# 4. DEPLOYMENT HELPERS
# ===============================================================================

#' Deploy Shiny App to shinyapps.io
#' @param app_dir Directory containing the Shiny app
#' @param app_name Name for the deployed app
deploy_to_shinyapps <- function(app_dir, app_name) {
  
  if (!requireNamespace("rsconnect", quietly = TRUE)) {
    stop("Please install rsconnect package: install.packages('rsconnect')")
  }
  
  # Check if account is set up
  accounts <- rsconnect::accounts()
  if (nrow(accounts) == 0) {
    cat("Please set up your shinyapps.io account first:\n")
    cat("1. Go to https://www.shinyapps.io/\n")
    cat("2. Sign up and get your token\n")
    cat("3. Run: rsconnect::setAccountInfo(name='account', token='token', secret='secret')\n")
    return(invisible())
  }
  
  # Deploy app
  cat("Deploying app to shinyapps.io...\n")
  rsconnect::deployApp(
    appDir = app_dir,
    appName = app_name,
    account = accounts$name[1],
    launch.browser = TRUE
  )
  
  cat("App deployed successfully!\n")
  cat("URL: https://", accounts$name[1], ".shinyapps.io/", app_name, "\n", sep = "")
}

#' Create Docker Configuration for Shiny App
#' @param app_dir Directory containing the Shiny app
create_docker_config <- function(app_dir) {
  
  dockerfile_content <- '
FROM rocker/shiny:latest

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    libcurl4-gnutls-dev \\
    libssl-dev \\
    libxml2-dev \\
    libgdal-dev \\
    libproj-dev

# Install R packages
RUN R -e "install.packages(c(\\'shiny\\', \\'shinydashboard\\', \\'DT\\', \\'plotly\\', \\'dplyr\\', \\'ggplot2\\'))"

# Copy app files
COPY . /srv/shiny-server/myapp/

# Set permissions
RUN chown -R shiny:shiny /srv/shiny-server/

# Expose port
EXPOSE 3838

# Run app
CMD ["/usr/bin/shiny-server"]
'
  
  writeLines(dockerfile_content, file.path(app_dir, "Dockerfile"))
  
  docker_compose_content <- '
version: "3.8"
services:
  shiny-app:
    build: .
    ports:
      - "3838:3838"
    volumes:
      - ./logs:/var/log/shiny-server
    environment:
      - SHINY_LOG_LEVEL=INFO
'
  
  writeLines(docker_compose_content, file.path(app_dir, "docker-compose.yml"))
  
  cat("Docker configuration created!\n")
  cat("To build and run:\n")
  cat("  docker-compose up --build\n")
  cat("App will be available at: http://localhost:3838/myapp\n")
}

# ===============================================================================
# EXAMPLE USAGE AND TESTING
# ===============================================================================

# Create sample apps for testing
create_sample_apps <- function() {
  
  # Basic explorer
  basic_app <- create_basic_explorer()
  
  # Save to files
  app_dir <- "basic_explorer_app"
  dir.create(app_dir, showWarnings = FALSE)
  
  # Write app.R
  app_content <- paste0(
    "# Basic Data Explorer App\n",
    "library(shiny)\n",
    "library(DT)\n",
    "library(plotly)\n",
    "library(dplyr)\n",
    "library(ggplot2)\n",
    "library(readr)\n",
    "library(bslib)\n\n",
    "# Source the app components\n",
    "source('shiny_applications.R')\n\n",
    "# Create app\n",
    "app <- create_basic_explorer()\n\n",
    "# Run app\n",
    "shinyApp(ui = app$ui, server = app$server)\n"
  )
  
  writeLines(app_content, file.path(app_dir, "app.R"))
  
  cat("Sample Shiny apps created!\n")
  cat("To run basic explorer: shiny::runApp('", app_dir, "')\n", sep = "")
  
  return(app_dir)
}

cat("Shiny Applications Template Loaded Successfully!\n")
cat("Available Functions:\n")
cat("- create_basic_explorer(): Simple data exploration app\n")
cat("- create_analytics_dashboard(): Advanced analytics dashboard\n")
cat("- create_ml_dashboard(): Machine learning comparison dashboard\n")
cat("- deploy_to_shinyapps(): Deploy to shinyapps.io\n")
cat("- create_docker_config(): Create Docker configuration\n")
cat("- create_sample_apps(): Generate sample applications\n") 
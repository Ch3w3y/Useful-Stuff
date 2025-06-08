# Shiny Server with GCP Integration
# Production server with authentication, logging, and cloud data sources

# Load required libraries
library(shiny)
library(shinydashboard)
library(DT)
library(plotly)
library(leaflet)
library(googleCloudStorageR)
library(bigrquery)
library(pool)
library(config)
library(log4r)
library(dplyr)
library(ggplot2)

# Initialize logging
logger <- create.logger(logfile = "/srv/shiny-server/logs/shiny_app.log", 
                       level = "INFO")

# Load configuration
app_config <- config::get()

# GCP Authentication
authenticate_gcp <- function() {
  tryCatch({
    if (file.exists("/srv/shiny-server/config/service-account-key.json")) {
      gcs_auth("/srv/shiny-server/config/service-account-key.json")
      bq_auth(path = "/srv/shiny-server/config/service-account-key.json")
      info(logger, "Authenticated with service account")
    } else {
      gcs_auth()
      bq_auth()
      info(logger, "Using default authentication")
    }
    return(TRUE)
  }, error = function(e) {
    error(logger, paste("Authentication failed:", e$message))
    return(FALSE)
  })
}

# Initialize GCP connection
gcp_authenticated <- authenticate_gcp()

# Database connection pool (if using Cloud SQL)
if (!is.null(app_config$database)) {
  db_pool <- pool::dbPool(
    drv = RPostgres::Postgres(),
    host = app_config$database$host,
    port = app_config$database$port,
    dbname = app_config$database$name,
    user = app_config$database$user,
    password = app_config$database$password,
    sslmode = "require"
  )
  
  # Ensure pool is closed when app exits
  onStop(function() {
    pool::poolClose(db_pool)
  })
}

# Data loading functions
load_bigquery_data <- function(query) {
  if (!gcp_authenticated) {
    return(data.frame(error = "GCP authentication failed"))
  }
  
  tryCatch({
    result <- bq_project_query(app_config$gcp$project_id, query)
    data <- bq_table_download(result)
    info(logger, paste("BigQuery data loaded:", nrow(data), "rows"))
    return(data)
  }, error = function(e) {
    error(logger, paste("BigQuery query failed:", e$message))
    return(data.frame(error = paste("Query failed:", e$message)))
  })
}

load_gcs_data <- function(bucket, object_name) {
  if (!gcp_authenticated) {
    return(data.frame(error = "GCP authentication failed"))
  }
  
  tryCatch({
    temp_file <- tempfile(fileext = ".csv")
    gcs_get_object(object_name, bucket = bucket, saveToDisk = temp_file)
    data <- read.csv(temp_file)
    unlink(temp_file)
    info(logger, paste("GCS data loaded:", nrow(data), "rows"))
    return(data)
  }, error = function(e) {
    error(logger, paste("GCS data loading failed:", e$message))
    return(data.frame(error = paste("Data loading failed:", e$message)))
  })
}

# Define server
server <- function(input, output, session) {
  
  # Authentication check (simplified - in production use proper auth)
  observe({
    if (!gcp_authenticated) {
      showModal(modalDialog(
        title = "Authentication Error",
        "Unable to authenticate with Google Cloud Platform. Please check configuration.",
        easyClose = FALSE,
        footer = NULL
      ))
    }
  })
  
  # Reactive data loading
  dataset <- reactive({
    req(input$data_source)
    
    withProgress(message = 'Loading data...', value = 0, {
      
      if (input$data_source == "bigquery") {
        incProgress(0.3, detail = "Querying BigQuery...")
        query <- paste("SELECT * FROM", input$bq_table, "LIMIT 1000")
        data <- load_bigquery_data(query)
        
      } else if (input$data_source == "gcs") {
        incProgress(0.3, detail = "Loading from Cloud Storage...")
        data <- load_gcs_data(input$gcs_bucket, input$gcs_object)
        
      } else {
        # Demo data fallback
        data <- mtcars
      }
      
      incProgress(1, detail = "Data loaded successfully")
      return(data)
    })
  })
  
  # Data summary
  output$data_summary <- renderValueBox({
    data <- dataset()
    if ("error" %in% names(data)) {
      valueBox(
        value = "Error",
        subtitle = "Data Loading Failed",
        icon = icon("exclamation-triangle"),
        color = "red"
      )
    } else {
      valueBox(
        value = nrow(data),
        subtitle = "Records Loaded",
        icon = icon("database"),
        color = "blue"
      )
    }
  })
  
  # Data table
  output$data_table <- renderDT({
    data <- dataset()
    if ("error" %in% names(data)) {
      datatable(data, options = list(pageLength = 10))
    } else {
      datatable(data, 
                options = list(
                  pageLength = 25,
                  scrollX = TRUE,
                  searchHighlight = TRUE
                ),
                filter = 'top')
    }
  })
  
  # Interactive plot
  output$interactive_plot <- renderPlotly({
    data <- dataset()
    if ("error" %in% names(data) || nrow(data) == 0) {
      p <- ggplot() + 
        geom_text(aes(x = 1, y = 1, label = "No data available"), 
                  size = 6, color = "red") +
        theme_void()
    } else {
      # Auto-select numeric columns for plotting
      numeric_cols <- names(data)[sapply(data, is.numeric)]
      
      if (length(numeric_cols) >= 2) {
        p <- ggplot(data, aes_string(x = numeric_cols[1], y = numeric_cols[2])) +
          geom_point(alpha = 0.7, size = 2) +
          geom_smooth(method = "lm", se = TRUE, alpha = 0.3) +
          theme_minimal() +
          labs(title = paste("Relationship between", numeric_cols[1], "and", numeric_cols[2]))
      } else {
        p <- ggplot(data, aes_string(x = names(data)[1])) +
          geom_histogram(bins = 30, fill = "steelblue", alpha = 0.7) +
          theme_minimal() +
          labs(title = paste("Distribution of", names(data)[1]))
      }
    }
    
    ggplotly(p) %>%
      layout(title = list(text = paste0(p$labels$title, 
                                       '<br><sup>Interactive visualization</sup>')))
  })
  
  # Map visualization (if geographic data is available)
  output$map_viz <- renderLeaflet({
    data <- dataset()
    
    # Check for latitude/longitude columns
    lat_col <- names(data)[grepl("lat|latitude", names(data), ignore.case = TRUE)][1]
    lng_col <- names(data)[grepl("lng|longitude|long", names(data), ignore.case = TRUE)][1]
    
    if (!is.na(lat_col) && !is.na(lng_col)) {
      leaflet(data) %>%
        addTiles() %>%
        addCircleMarkers(
          lng = ~get(lng_col),
          lat = ~get(lat_col),
          radius = 5,
          popup = ~paste("Lat:", get(lat_col), "<br>Lng:", get(lng_col)),
          fillOpacity = 0.7
        )
    } else {
      # Default map
      leaflet() %>%
        addTiles() %>%
        setView(lng = -98.5795, lat = 39.8283, zoom = 4)
    }
  })
  
  # Download handler
  output$download_data <- downloadHandler(
    filename = function() {
      paste("data_export_", Sys.Date(), ".csv", sep = "")
    },
    content = function(file) {
      data <- dataset()
      if (!"error" %in% names(data)) {
        write.csv(data, file, row.names = FALSE)
        info(logger, paste("Data exported:", nrow(data), "rows"))
      }
    }
  )
  
  # Log user interactions
  observe({
    if (!is.null(input$data_source)) {
      info(logger, paste("User selected data source:", input$data_source))
    }
  })
  
  # Session info
  output$session_info <- renderText({
    paste("Session started:", format(Sys.time()),
          "| User IP:", session$clientData$url_hostname,
          "| Browser:", session$clientData$url_search)
  })
} 
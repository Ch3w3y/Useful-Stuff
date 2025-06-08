# ===============================================================================
# GEOSPATIAL DATA SCIENCE IN R - COMPREHENSIVE TEMPLATE
# ===============================================================================
# Complete guide to spatial analysis, GIS operations, mapping, and remote sensing
# Author: Data Science Toolkit
# Last Updated: 2024
# ===============================================================================

# Required Libraries
suppressMessages({
  library(sf)
  library(sp)
  library(raster)
  library(rgdal)
  library(rgeos)
  library(maptools)
  library(leaflet)
  library(tmap)
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(magrittr)
  library(RColorBrewer)
  library(viridis)
  library(spatstat)
  library(spdep)
  library(gstat)
  library(geosphere)
  library(terra)
  library(stars)
  library(mapview)
  library(osmdata)
  library(ggmap)
  library(ggspatial)
  library(rnaturalearth)
  library(rnaturalearthdata)
  library(lwgeom)
  library(units)
  library(DT)
  library(plotly)
  library(htmlwidgets)
})

# ===============================================================================
# 1. DATA LOADING AND PREPARATION
# ===============================================================================

#' Load Spatial Data from Various Sources
#' @param file_path Path to spatial data file
#' @param data_type Type of spatial data
load_spatial_data <- function(file_path, data_type = "auto") {
  cat("Loading spatial data from:", file_path, "\n")
  
  if (data_type == "auto") {
    # Auto-detect file type
    ext <- tools::file_ext(file_path)
    data_type <- switch(tolower(ext),
      "shp" = "shapefile",
      "geojson" = "geojson",
      "json" = "geojson",
      "tif" = "raster",
      "tiff" = "raster",
      "csv" = "csv",
      "gpx" = "gpx",
      "kml" = "kml",
      "shapefile"
    )
  }
  
  spatial_data <- switch(data_type,
    "shapefile" = st_read(file_path),
    "geojson" = st_read(file_path),
    "raster" = rast(file_path),
    "csv" = {
      # Assume CSV has lat/lon columns
      df <- read.csv(file_path)
      if (all(c("lon", "lat") %in% names(df)) || all(c("longitude", "latitude") %in% names(df))) {
        lon_col <- ifelse("lon" %in% names(df), "lon", "longitude")
        lat_col <- ifelse("lat" %in% names(df), "lat", "latitude")
        st_as_sf(df, coords = c(lon_col, lat_col), crs = 4326)
      } else {
        df
      }
    },
    "gpx" = st_read(file_path),
    "kml" = st_read(file_path)
  )
  
  cat("Data loaded successfully. Type:", class(spatial_data)[1], "\n")
  if (inherits(spatial_data, "sf")) {
    cat("Features:", nrow(spatial_data), "\n")
    cat("CRS:", st_crs(spatial_data)$input, "\n")
  }
  
  return(spatial_data)
}

#' Create Sample Spatial Data
#' @param n_points Number of sample points
#' @param bbox Bounding box coordinates
create_sample_spatial_data <- function(n_points = 100, bbox = c(-74.1, 40.6, -73.9, 40.8)) {
  # Create random points within bounding box
  set.seed(123)
  
  lon <- runif(n_points, bbox[1], bbox[3])
  lat <- runif(n_points, bbox[2], bbox[4])
  
  # Create some sample attributes
  data <- data.frame(
    id = 1:n_points,
    lon = lon,
    lat = lat,
    value = rnorm(n_points, 100, 20),
    category = sample(c("A", "B", "C"), n_points, replace = TRUE),
    population = rpois(n_points, 1000),
    temperature = rnorm(n_points, 20, 5)
  )
  
  # Convert to sf object
  sf_data <- st_as_sf(data, coords = c("lon", "lat"), crs = 4326)
  
  return(sf_data)
}

#' Download Administrative Boundaries
#' @param country Country code or name
#' @param scale Scale of the data
download_admin_boundaries <- function(country = "united states of america", scale = "medium") {
  cat("Downloading administrative boundaries for:", country, "\n")
  
  # Get country boundaries
  country_sf <- ne_countries(scale = scale, returnclass = "sf") %>%
    filter(tolower(name_long) == tolower(country) | tolower(admin) == tolower(country))
  
  if (nrow(country_sf) == 0) {
    cat("Country not found. Available countries:\n")
    available <- ne_countries(scale = scale, returnclass = "sf")$name_long
    print(head(available, 20))
    return(NULL)
  }
  
  # Get state/province boundaries
  states_sf <- ne_states(country = country, returnclass = "sf")
  
  return(list(
    country = country_sf,
    states = states_sf
  ))
}

# ===============================================================================
# 2. COORDINATE REFERENCE SYSTEMS (CRS) AND TRANSFORMATIONS
# ===============================================================================

#' CRS Management and Transformations
#' @param spatial_data Spatial data object
#' @param target_crs Target CRS
manage_crs <- function(spatial_data, target_crs = NULL) {
  cat("\n=== CRS INFORMATION ===\n")
  
  # Current CRS
  current_crs <- st_crs(spatial_data)
  cat("Current CRS:", current_crs$input, "\n")
  cat("Is Geographic:", current_crs$is_geographic, "\n")
  cat("Units:", current_crs$units_gdal, "\n")
  
  if (!is.null(target_crs)) {
    cat("\nTransforming to CRS:", target_crs, "\n")
    spatial_data <- st_transform(spatial_data, crs = target_crs)
    
    new_crs <- st_crs(spatial_data)
    cat("New CRS:", new_crs$input, "\n")
    cat("New Units:", new_crs$units_gdal, "\n")
  }
  
  # Common CRS suggestions
  cat("\n=== COMMON CRS OPTIONS ===\n")
  cat("4326 - WGS84 (Geographic, degrees)\n")
  cat("3857 - Web Mercator (Projected, meters)\n")
  cat("32633 - UTM Zone 33N (Projected, meters)\n")
  cat("5070 - Albers Equal Area Conic (US, meters)\n")
  
  return(spatial_data)
}

#' Calculate Area and Length
#' @param spatial_data Spatial data object
calculate_geometry_metrics <- function(spatial_data) {
  cat("\n=== GEOMETRY METRICS ===\n")
  
  # Ensure projected CRS for accurate measurements
  if (st_is_longlat(spatial_data)) {
    cat("Converting to projected CRS for accurate measurements...\n")
    spatial_data <- st_transform(spatial_data, crs = 3857)  # Web Mercator
  }
  
  geom_type <- st_geometry_type(spatial_data)[1]
  
  if (geom_type %in% c("POLYGON", "MULTIPOLYGON")) {
    spatial_data$area_km2 <- as.numeric(st_area(spatial_data)) / 1e6
    cat("Added area calculations (km²)\n")
    cat("Total area:", sum(spatial_data$area_km2, na.rm = TRUE), "km²\n")
    
  } else if (geom_type %in% c("LINESTRING", "MULTILINESTRING")) {
    spatial_data$length_km <- as.numeric(st_length(spatial_data)) / 1000
    cat("Added length calculations (km)\n")
    cat("Total length:", sum(spatial_data$length_km, na.rm = TRUE), "km\n")
    
  } else if (geom_type == "POINT") {
    cat("Point data - no area/length calculations\n")
  }
  
  return(spatial_data)
}

# ===============================================================================
# 3. SPATIAL OPERATIONS AND ANALYSIS
# ===============================================================================

#' Spatial Filtering and Selection
#' @param spatial_data Spatial data object
#' @param filter_geom Geometry to filter by
perform_spatial_operations <- function(spatial_data, filter_geom = NULL) {
  cat("\n=== SPATIAL OPERATIONS ===\n")
  
  results <- list()
  
  # Basic spatial filtering
  if (!is.null(filter_geom)) {
    # Intersect
    intersects <- st_intersection(spatial_data, filter_geom)
    results$intersects <- intersects
    cat("Features intersecting filter:", nrow(intersects), "\n")
    
    # Within
    within <- spatial_data[st_within(spatial_data, filter_geom, sparse = FALSE), ]
    results$within <- within
    cat("Features within filter:", nrow(within), "\n")
    
    # Buffer analysis
    if (st_geometry_type(filter_geom)[1] == "POINT") {
      buffer_1km <- st_buffer(filter_geom, dist = 1000)  # 1km buffer
      within_buffer <- spatial_data[st_within(spatial_data, buffer_1km, sparse = FALSE), ]
      results$within_buffer <- within_buffer
      cat("Features within 1km buffer:", nrow(within_buffer), "\n")
    }
  }
  
  # Spatial aggregation
  if (nrow(spatial_data) > 1) {
    # Union (dissolve boundaries)
    if (st_geometry_type(spatial_data)[1] %in% c("POLYGON", "MULTIPOLYGON")) {
      union_result <- st_union(spatial_data)
      results$union <- union_result
      cat("Created union of all polygons\n")
    }
    
    # Convex hull
    convex_hull <- st_convex_hull(st_union(spatial_data))
    results$convex_hull <- convex_hull
    cat("Created convex hull\n")
  }
  
  return(results)
}

#' Distance Analysis
#' @param spatial_data Spatial data object
#' @param reference_point Reference point for distance calculations
analyze_distances <- function(spatial_data, reference_point = NULL) {
  cat("\n=== DISTANCE ANALYSIS ===\n")
  
  if (is.null(reference_point)) {
    # Use centroid of data as reference
    reference_point <- st_centroid(st_union(spatial_data))
    cat("Using data centroid as reference point\n")
  }
  
  # Calculate distances
  distances <- st_distance(spatial_data, reference_point)
  spatial_data$distance_km <- as.numeric(distances) / 1000
  
  cat("Distance statistics (km):\n")
  print(summary(spatial_data$distance_km))
  
  # Nearest neighbor analysis
  if (nrow(spatial_data) > 1) {
    nn_dist <- st_distance(spatial_data)
    diag(nn_dist) <- NA  # Remove self-distances
    spatial_data$nearest_neighbor_km <- apply(nn_dist, 1, min, na.rm = TRUE) / 1000
    
    cat("\nNearest neighbor statistics (km):\n")
    print(summary(spatial_data$nearest_neighbor_km))
  }
  
  return(spatial_data)
}

# ===============================================================================
# 4. SPATIAL STATISTICS
# ===============================================================================

#' Spatial Autocorrelation Analysis
#' @param spatial_data Spatial data with numeric attribute
#' @param variable Variable name for analysis
analyze_spatial_autocorrelation <- function(spatial_data, variable) {
  cat("\n=== SPATIAL AUTOCORRELATION ANALYSIS ===\n")
  
  # Convert to sp for compatibility with spdep
  spatial_sp <- as_Spatial(spatial_data)
  
  # Create spatial weights matrix
  coords <- coordinates(spatial_sp)
  
  # K-nearest neighbors (k=4)
  knn_nb <- knn2nb(knearneigh(coords, k = 4))
  knn_weights <- nb2listw(knn_nb)
  
  # Distance-based neighbors
  dist_nb <- dnearneigh(coords, 0, 5000)  # 5km threshold
  dist_weights <- nb2listw(dist_nb, zero.policy = TRUE)
  
  # Moran's I test
  cat("\n--- MORAN'S I TEST ---\n")
  values <- spatial_data[[variable]]
  
  # KNN-based
  moran_knn <- moran.test(values, knn_weights)
  cat("Moran's I (KNN):", moran_knn$estimate[1], "\n")
  cat("p-value:", moran_knn$p.value, "\n")
  
  # Distance-based
  if (length(dist_nb) > 0) {
    moran_dist <- moran.test(values, dist_weights, zero.policy = TRUE)
    cat("Moran's I (Distance):", moran_dist$estimate[1], "\n")
    cat("p-value:", moran_dist$p.value, "\n")
  }
  
  # Local Moran's I (LISA)
  lisa <- localmoran(values, knn_weights)
  spatial_data$moran_local <- lisa[, 1]
  spatial_data$moran_pvalue <- lisa[, 5]
  
  cat("\nLocal Moran's I calculated\n")
  cat("Significant hotspots/coldspots:", sum(spatial_data$moran_pvalue < 0.05), "\n")
  
  return(list(
    data = spatial_data,
    moran_knn = moran_knn,
    moran_dist = if (exists("moran_dist")) moran_dist else NULL,
    weights_knn = knn_weights,
    weights_dist = if (exists("dist_weights")) dist_weights else NULL
  ))
}

#' Point Pattern Analysis
#' @param spatial_points Point spatial data
analyze_point_patterns <- function(spatial_points) {
  cat("\n=== POINT PATTERN ANALYSIS ===\n")
  
  # Convert to spatstat format
  coords <- st_coordinates(spatial_points)
  
  # Create observation window
  bbox <- st_bbox(spatial_points)
  window <- owin(xrange = c(bbox[1], bbox[3]), yrange = c(bbox[2], bbox[4]))
  
  ppp_obj <- ppp(coords[, 1], coords[, 2], window = window)
  
  # Basic statistics
  cat("Number of points:", ppp_obj$n, "\n")
  cat("Intensity (points per unit area):", intensity(ppp_obj), "\n")
  
  # Nearest neighbor distances
  nn_dist <- nndist(ppp_obj)
  cat("Mean nearest neighbor distance:", mean(nn_dist), "\n")
  
  # G-function (nearest neighbor distribution)
  g_func <- Gest(ppp_obj)
  
  # F-function (empty space function)
  f_func <- Fest(ppp_obj)
  
  # K-function (Ripley's K)
  k_func <- Kest(ppp_obj)
  
  # L-function (normalized K)
  l_func <- Lest(ppp_obj)
  
  # Plot functions
  par(mfrow = c(2, 2))
  plot(g_func, main = "G-function")
  plot(f_func, main = "F-function")
  plot(k_func, main = "K-function")
  plot(l_func, main = "L-function")
  par(mfrow = c(1, 1))
  
  # Clark-Evans test for clustering
  ce_test <- clarkevans.test(ppp_obj)
  cat("\nClark-Evans Test:\n")
  cat("R statistic:", ce_test$statistic, "\n")
  cat("p-value:", ce_test$p.value, "\n")
  
  interpretation <- ifelse(ce_test$statistic < 1, "Clustered", 
                          ifelse(ce_test$statistic > 1, "Dispersed", "Random"))
  cat("Pattern interpretation:", interpretation, "\n")
  
  return(list(
    ppp = ppp_obj,
    g_function = g_func,
    f_function = f_func,
    k_function = k_func,
    l_function = l_func,
    clark_evans = ce_test
  ))
}

# ===============================================================================
# 5. SPATIAL INTERPOLATION AND KRIGING
# ===============================================================================

#' Spatial Interpolation
#' @param spatial_data Point data with values
#' @param variable Variable to interpolate
#' @param method Interpolation method
perform_spatial_interpolation <- function(spatial_data, variable, method = "idw") {
  cat("\n=== SPATIAL INTERPOLATION ===\n")
  
  # Create prediction grid
  bbox <- st_bbox(spatial_data)
  grid_size <- 50
  
  x_seq <- seq(bbox[1], bbox[3], length.out = grid_size)
  y_seq <- seq(bbox[2], bbox[4], length.out = grid_size)
  
  pred_grid <- expand.grid(x = x_seq, y = y_seq)
  coordinates(pred_grid) <- ~x+y
  gridded(pred_grid) <- TRUE
  
  # Convert spatial_data to sp format
  spatial_sp <- as_Spatial(spatial_data)
  
  if (method == "idw") {
    # Inverse Distance Weighting
    cat("Performing IDW interpolation...\n")
    
    # Create formula
    formula_str <- paste(variable, "~ 1")
    idw_result <- gstat::idw(as.formula(formula_str), 
                           locations = spatial_sp, 
                           newdata = pred_grid, 
                           idp = 2)
    
    # Convert back to sf
    result_sf <- st_as_sf(idw_result)
    result_sf$method <- "IDW"
    
  } else if (method == "kriging") {
    # Ordinary Kriging
    cat("Performing Ordinary Kriging...\n")
    
    # Fit variogram
    formula_str <- paste(variable, "~ 1")
    vario <- variogram(as.formula(formula_str), spatial_sp)
    vario_fit <- fit.variogram(vario, model = vgm("Sph"))
    
    # Perform kriging
    krig_result <- krige(as.formula(formula_str), 
                        locations = spatial_sp, 
                        newdata = pred_grid, 
                        model = vario_fit)
    
    # Convert back to sf
    result_sf <- st_as_sf(krig_result)
    result_sf$method <- "Kriging"
    
    # Add variogram info
    result_sf$variogram_model <- "Spherical"
    
    # Plot variogram
    plot(vario, vario_fit, main = "Fitted Variogram")
  }
  
  cat("Interpolation completed. Grid size:", grid_size, "x", grid_size, "\n")
  
  return(list(
    interpolated = result_sf,
    original_data = spatial_data,
    method = method
  ))
}

# ===============================================================================
# 6. RASTER ANALYSIS
# ===============================================================================

#' Raster Data Analysis
#' @param raster_path Path to raster file
analyze_raster_data <- function(raster_path = NULL) {
  cat("\n=== RASTER DATA ANALYSIS ===\n")
  
  if (is.null(raster_path)) {
    # Create sample raster
    cat("Creating sample raster data...\n")
    r <- rast(nrows = 100, ncols = 100, 
              xmin = -74.1, xmax = -73.9, 
              ymin = 40.6, ymax = 40.8)
    
    # Add some realistic elevation-like data
    set.seed(123)
    values(r) <- runif(ncell(r), 0, 500) + 
                100 * sin(seq(0, 4*pi, length.out = ncell(r)))
    
    names(r) <- "elevation"
    raster_data <- r
    
  } else {
    # Load raster data
    raster_data <- rast(raster_path)
    cat("Loaded raster from:", raster_path, "\n")
  }
  
  # Basic raster information
  cat("\n--- RASTER INFORMATION ---\n")
  cat("Dimensions (rows, cols, layers):", dim(raster_data), "\n")
  cat("Resolution:", res(raster_data), "\n")
  cat("Extent:", as.vector(ext(raster_data)), "\n")
  cat("CRS:", crs(raster_data), "\n")
  
  # Raster statistics
  cat("\n--- RASTER STATISTICS ---\n")
  raster_stats <- global(raster_data, c("min", "max", "mean", "sd"), na.rm = TRUE)
  print(raster_stats)
  
  # Raster operations
  operations <- list()
  
  # Slope and aspect (if elevation data)
  if (any(grepl("elevation|dem|height", names(raster_data), ignore.case = TRUE))) {
    slope_raster <- terrain(raster_data, "slope", unit = "degrees")
    aspect_raster <- terrain(raster_data, "aspect", unit = "degrees")
    
    operations$slope <- slope_raster
    operations$aspect <- aspect_raster
    
    cat("Calculated slope and aspect\n")
  }
  
  # Focal statistics (moving window)
  focal_mean <- focal(raster_data, w = matrix(1, 3, 3), fun = mean, na.rm = TRUE)
  operations$focal_mean <- focal_mean
  
  # Reclassification
  # Create classification breaks
  breaks <- quantile(values(raster_data), probs = c(0, 0.25, 0.5, 0.75, 1), na.rm = TRUE)
  rcl_matrix <- cbind(c(-Inf, breaks[-length(breaks)]), 
                      c(breaks[-1], Inf), 
                      1:4)
  
  reclassified <- classify(raster_data, rcl_matrix)
  operations$reclassified <- reclassified
  
  cat("Applied focal filter and reclassification\n")
  
  # Plot raster data
  par(mfrow = c(2, 2))
  plot(raster_data, main = "Original Raster")
  if (!is.null(operations$slope)) plot(operations$slope, main = "Slope")
  plot(operations$focal_mean, main = "Focal Mean (3x3)")
  plot(operations$reclassified, main = "Reclassified")
  par(mfrow = c(1, 1))
  
  return(list(
    original = raster_data,
    operations = operations,
    statistics = raster_stats
  ))
}

#' Raster-Vector Operations
#' @param raster_data Raster object
#' @param vector_data Vector spatial data
raster_vector_operations <- function(raster_data, vector_data) {
  cat("\n=== RASTER-VECTOR OPERATIONS ===\n")
  
  # Ensure CRS match
  if (st_crs(vector_data) != crs(raster_data)) {
    cat("Transforming vector CRS to match raster...\n")
    vector_data <- st_transform(vector_data, crs = crs(raster_data))
  }
  
  operations <- list()
  
  # Extract raster values at point locations
  if (st_geometry_type(vector_data)[1] == "POINT") {
    extracted_values <- extract(raster_data, vect(vector_data))
    vector_data$raster_value <- extracted_values[, 2]
    
    cat("Extracted raster values at", nrow(vector_data), "point locations\n")
    cat("Mean extracted value:", mean(vector_data$raster_value, na.rm = TRUE), "\n")
    
    operations$extracted_points <- vector_data
  }
  
  # Crop raster by polygon
  if (st_geometry_type(vector_data)[1] %in% c("POLYGON", "MULTIPOLYGON")) {
    cropped_raster <- crop(raster_data, vect(vector_data))
    masked_raster <- mask(cropped_raster, vect(vector_data))
    
    operations$cropped <- cropped_raster
    operations$masked <- masked_raster
    
    cat("Cropped and masked raster by polygon\n")
    
    # Zonal statistics
    zonal_stats <- zonal(raster_data, vect(vector_data), fun = c("mean", "sum", "min", "max"))
    operations$zonal_stats <- zonal_stats
    
    cat("Calculated zonal statistics\n")
    print(zonal_stats)
  }
  
  return(operations)
}

# ===============================================================================
# 7. MAPPING AND VISUALIZATION
# ===============================================================================

#' Create Static Maps with ggplot2
#' @param spatial_data Spatial data object
#' @param variable Variable to map
#' @param title Map title
create_static_map <- function(spatial_data, variable = NULL, title = "Spatial Data Map") {
  cat("\n=== CREATING STATIC MAP ===\n")
  
  # Base map
  map_plot <- ggplot(spatial_data) +
    geom_sf(aes_string(fill = variable), color = "white", size = 0.2) +
    theme_minimal() +
    theme(
      axis.text = element_blank(),
      axis.ticks = element_blank(),
      panel.grid = element_blank()
    ) +
    labs(title = title)
  
  # Add color scale if variable specified
  if (!is.null(variable) && variable %in% names(spatial_data)) {
    if (is.numeric(spatial_data[[variable]])) {
      map_plot <- map_plot +
        scale_fill_viridis_c(name = variable)
    } else {
      map_plot <- map_plot +
        scale_fill_viridis_d(name = variable)
    }
  }
  
  # Add north arrow and scale bar
  map_plot <- map_plot +
    annotation_north_arrow(location = "tl", which_north = "true") +
    annotation_scale(location = "br")
  
  print(map_plot)
  
  return(map_plot)
}

#' Create Interactive Maps with Leaflet
#' @param spatial_data Spatial data object
#' @param variable Variable to map
#' @param popup_vars Variables to show in popup
create_interactive_map <- function(spatial_data, variable = NULL, popup_vars = NULL) {
  cat("\n=== CREATING INTERACTIVE MAP ===\n")
  
  # Transform to WGS84 for leaflet
  if (st_crs(spatial_data) != st_crs(4326)) {
    spatial_data <- st_transform(spatial_data, 4326)
  }
  
  # Create base map
  map <- leaflet(spatial_data) %>%
    addProviderTiles(providers$OpenStreetMap) %>%
    setView(lng = mean(st_bbox(spatial_data)[c(1, 3)]), 
            lat = mean(st_bbox(spatial_data)[c(2, 4)]), 
            zoom = 10)
  
  # Add spatial data based on geometry type
  geom_type <- st_geometry_type(spatial_data)[1]
  
  if (geom_type == "POINT") {
    if (!is.null(variable) && is.numeric(spatial_data[[variable]])) {
      # Color by variable
      pal <- colorNumeric(palette = "viridis", domain = spatial_data[[variable]])
      
      map <- map %>%
        addCircleMarkers(
          radius = 5,
          color = ~pal(get(variable)),
          fillOpacity = 0.7,
          stroke = TRUE,
          weight = 1
        ) %>%
        addLegend(pal = pal, values = ~get(variable), title = variable)
      
    } else {
      map <- map %>%
        addCircleMarkers(radius = 5, fillOpacity = 0.7)
    }
    
  } else if (geom_type %in% c("POLYGON", "MULTIPOLYGON")) {
    if (!is.null(variable) && is.numeric(spatial_data[[variable]])) {
      # Color by variable
      pal <- colorNumeric(palette = "viridis", domain = spatial_data[[variable]])
      
      map <- map %>%
        addPolygons(
          fillColor = ~pal(get(variable)),
          fillOpacity = 0.7,
          color = "white",
          weight = 1
        ) %>%
        addLegend(pal = pal, values = ~get(variable), title = variable)
      
    } else {
      map <- map %>%
        addPolygons(fillOpacity = 0.7, color = "blue", weight = 2)
    }
  }
  
  # Add popups
  if (!is.null(popup_vars)) {
    popup_text <- paste(popup_vars, collapse = "<br>")
    # This is simplified - in practice, you'd create proper popup content
    cat("Popup variables specified:", paste(popup_vars, collapse = ", "), "\n")
  }
  
  print(map)
  return(map)
}

#' Create Thematic Maps with tmap
#' @param spatial_data Spatial data object
#' @param variable Variable to map
create_thematic_map <- function(spatial_data, variable) {
  cat("\n=== CREATING THEMATIC MAP ===\n")
  
  # Set tmap mode
  tmap_mode("plot")  # or "view" for interactive
  
  # Create thematic map
  thematic_map <- tm_shape(spatial_data) +
    tm_fill(variable, 
            palette = "viridis",
            title = variable,
            style = "quantile") +
    tm_borders(col = "white", lwd = 0.5) +
    tm_layout(
      title = paste("Thematic Map:", variable),
      title.position = c("left", "top"),
      legend.position = c("right", "bottom")
    ) +
    tm_compass(position = c("left", "bottom")) +
    tm_scale_bar(position = c("right", "top"))
  
  print(thematic_map)
  return(thematic_map)
}

# ===============================================================================
# 8. NETWORK ANALYSIS
# ===============================================================================

#' Analyze Street Networks
#' @param location Location name or bounding box
#' @param network_type Type of network to download
analyze_street_network <- function(location, network_type = "driving") {
  cat("\n=== STREET NETWORK ANALYSIS ===\n")
  
  # Download street network using osmdata
  cat("Downloading street network for:", location, "\n")
  
  # Get bounding box
  bbox <- getbb(location)
  
  # Download highway data
  highways <- opq(bbox) %>%
    add_osm_feature(key = "highway") %>%
    osmdata_sf()
  
  street_lines <- highways$osm_lines
  
  if (is.null(street_lines) || nrow(street_lines) == 0) {
    cat("No street data found for this location\n")
    return(NULL)
  }
  
  cat("Downloaded", nrow(street_lines), "street segments\n")
  
  # Basic network statistics
  total_length <- sum(st_length(street_lines), na.rm = TRUE)
  cat("Total network length:", as.numeric(total_length) / 1000, "km\n")
  
  # Highway type distribution
  highway_types <- table(street_lines$highway)
  cat("\nHighway type distribution:\n")
  print(head(highway_types, 10))
  
  # Create simple network plot
  ggplot(street_lines) +
    geom_sf(aes(color = highway), size = 0.5) +
    theme_void() +
    labs(title = paste("Street Network -", location)) +
    theme(legend.position = "none")  # Too many categories for legend
  
  return(list(
    streets = street_lines,
    total_length_km = as.numeric(total_length) / 1000,
    highway_types = highway_types
  ))
}

# ===============================================================================
# 9. COMPREHENSIVE WORKFLOW EXAMPLE
# ===============================================================================

#' Complete Geospatial Analysis Workflow
#' @param data_source Data source or path
#' @param analysis_type Type of analysis to perform
complete_geospatial_workflow <- function(data_source = "sample", analysis_type = "all") {
  cat("===============================================================================\n")
  cat("COMPLETE GEOSPATIAL ANALYSIS WORKFLOW\n")
  cat("===============================================================================\n")
  
  # 1. Load or create data
  cat("\n1. LOADING DATA...\n")
  if (data_source == "sample") {
    spatial_data <- create_sample_spatial_data(n_points = 200)
    cat("Created sample spatial data with 200 points\n")
  } else {
    spatial_data <- load_spatial_data(data_source)
  }
  
  # 2. Data exploration and CRS management
  cat("\n2. DATA EXPLORATION...\n")
  spatial_data <- manage_crs(spatial_data, target_crs = 3857)  # Web Mercator
  spatial_data <- calculate_geometry_metrics(spatial_data)
  
  # 3. Spatial operations
  cat("\n3. SPATIAL OPERATIONS...\n")
  spatial_ops <- perform_spatial_operations(spatial_data)
  
  # 4. Distance analysis
  cat("\n4. DISTANCE ANALYSIS...\n")
  spatial_data <- analyze_distances(spatial_data)
  
  # 5. Spatial statistics (if points)
  if (st_geometry_type(spatial_data)[1] == "POINT" && "value" %in% names(spatial_data)) {
    cat("\n5. SPATIAL STATISTICS...\n")
    autocorr_results <- analyze_spatial_autocorrelation(spatial_data, "value")
    spatial_data <- autocorr_results$data
    
    # Point pattern analysis
    pattern_results <- analyze_point_patterns(spatial_data)
  }
  
  # 6. Spatial interpolation (if points with values)
  if (st_geometry_type(spatial_data)[1] == "POINT" && "value" %in% names(spatial_data)) {
    cat("\n6. SPATIAL INTERPOLATION...\n")
    interp_results <- perform_spatial_interpolation(spatial_data, "value", method = "idw")
  }
  
  # 7. Mapping and visualization
  cat("\n7. CREATING MAPS...\n")
  
  # Static map
  static_map <- create_static_map(spatial_data, "value", "Sample Spatial Data")
  
  # Interactive map
  interactive_map <- create_interactive_map(spatial_data, "value")
  
  # 8. Generate summary report
  cat("\n8. GENERATING SUMMARY...\n")
  generate_geospatial_report(spatial_data)
  
  # Return comprehensive results
  results <- list(
    data = spatial_data,
    spatial_operations = spatial_ops,
    static_map = static_map,
    interactive_map = interactive_map
  )
  
  if (exists("autocorr_results")) results$autocorrelation <- autocorr_results
  if (exists("pattern_results")) results$point_patterns <- pattern_results
  if (exists("interp_results")) results$interpolation <- interp_results
  
  return(results)
}

#' Generate Geospatial Analysis Report
#' @param spatial_data Spatial data object
generate_geospatial_report <- function(spatial_data) {
  cat("===============================================================================\n")
  cat("GEOSPATIAL ANALYSIS REPORT\n")
  cat("===============================================================================\n")
  
  # Basic information
  cat("\nDATA SUMMARY:\n")
  cat("- Features:", nrow(spatial_data), "\n")
  cat("- Geometry type:", as.character(st_geometry_type(spatial_data)[1]), "\n")
  cat("- CRS:", st_crs(spatial_data)$input, "\n")
  
  # Bounding box
  bbox <- st_bbox(spatial_data)
  cat("- Bounding box:", bbox, "\n")
  
  # Variable summary
  numeric_vars <- sapply(spatial_data, is.numeric)
  numeric_vars <- names(numeric_vars)[numeric_vars]
  
  if (length(numeric_vars) > 0) {
    cat("\nNUMERIC VARIABLES:\n")
    for (var in numeric_vars) {
      if (var != "geometry") {
        cat("-", var, ":", 
            "Mean =", round(mean(spatial_data[[var]], na.rm = TRUE), 2),
            "SD =", round(sd(spatial_data[[var]], na.rm = TRUE), 2), "\n")
      }
    }
  }
  
  # Spatial extent
  if ("distance_km" %in% names(spatial_data)) {
    cat("\nSPATIAL DISTRIBUTION:\n")
    cat("- Max distance from centroid:", round(max(spatial_data$distance_km, na.rm = TRUE), 2), "km\n")
    cat("- Mean distance from centroid:", round(mean(spatial_data$distance_km, na.rm = TRUE), 2), "km\n")
  }
  
  if ("nearest_neighbor_km" %in% names(spatial_data)) {
    cat("- Mean nearest neighbor distance:", round(mean(spatial_data$nearest_neighbor_km, na.rm = TRUE), 2), "km\n")
  }
}

# ===============================================================================
# EXAMPLE USAGE
# ===============================================================================

# Run complete analysis
# results <- complete_geospatial_workflow()
# 
# # Load your own data
# my_data <- load_spatial_data("path/to/your/data.shp")
# my_results <- complete_geospatial_workflow(my_data)
# 
# # Analyze specific location's street network
# network_analysis <- analyze_street_network("New York City, NY", "driving")

cat("Geospatial Analysis Template Loaded Successfully!\n")
cat("Key Functions Available:\n")
cat("- complete_geospatial_workflow(): Run full analysis\n")
cat("- load_spatial_data(): Load various spatial formats\n")
cat("- create_static_map(): ggplot2 mapping\n")
cat("- create_interactive_map(): Leaflet mapping\n")
cat("- analyze_spatial_autocorrelation(): Spatial statistics\n")
cat("- perform_spatial_interpolation(): IDW and Kriging\n")
cat("- analyze_street_network(): OSM network analysis\n") 
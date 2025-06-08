# ===============================================================================
# ADVANCED DATA VISUALIZATION IN R - COMPREHENSIVE TEMPLATE
# ===============================================================================
# Complete guide to advanced plotting, interactive visualizations, and modern charts
# Author: Data Science Toolkit
# Last Updated: 2024
# ===============================================================================

# Required Libraries
suppressMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(plotly)
  library(htmlwidgets)
  library(DT)
  library(leaflet)
  library(crosstalk)
  library(ggpubr)
  library(ggthemes)
  library(ggridges)
  library(ggalluvial)
  library(ggbeeswarm)
  library(ggforce)
  library(ggrepel)
  library(ggdist)
  library(ggcorrplot)
  library(corrplot)
  library(pheatmap)
  library(complexheatmaps)
  library(circlize)
  library(VennDiagram)
  library(networkD3)
  library(visNetwork)
  library(DiagrammeR)
  library(gganimate)
  library(transformr)
  library(rayshader)
  library(rgl)
  library(plot3D)
  library(scatterplot3d)
  library(lattice)
  library(latticeExtra)
  library(ggExtra)
  library(cowplot)
  library(patchwork)
  library(gridExtra)
  library(grid)
  library(scales)
  library(RColorBrewer)
  library(viridis)
  library(wesanderson)
  library(ggsci)
  library(colorspace)
  library(Cairo)
  library(svglite)
})

# ===============================================================================
# 1. DATA PREPARATION AND SAMPLE DATASETS
# ===============================================================================

#' Create Sample Datasets for Visualization
#' @param dataset_type Type of dataset to create
create_sample_data <- function(dataset_type = "all") {
  set.seed(123)
  
  datasets <- list()
  
  if (dataset_type %in% c("all", "basic")) {
    # Basic scatter plot data
    datasets$basic <- data.frame(
      x = rnorm(200, 50, 15),
      y = rnorm(200, 100, 25),
      category = sample(c("A", "B", "C"), 200, replace = TRUE),
      size = runif(200, 1, 10),
      value = runif(200, 0, 100)
    )
  }
  
  if (dataset_type %in% c("all", "time_series")) {
    # Time series data
    dates <- seq(as.Date("2020-01-01"), as.Date("2023-12-31"), by = "day")
    datasets$time_series <- data.frame(
      date = dates,
      value1 = cumsum(rnorm(length(dates), 0, 1)) + 100,
      value2 = cumsum(rnorm(length(dates), 0, 1.5)) + 150,
      value3 = cumsum(rnorm(length(dates), 0, 0.8)) + 75,
      trend = 1:length(dates) * 0.1 + rnorm(length(dates), 0, 5),
      seasonal = 10 * sin(2 * pi * 1:length(dates) / 365.25) + rnorm(length(dates), 0, 2)
    )
  }
  
  if (dataset_type %in% c("all", "hierarchical")) {
    # Hierarchical/tree data
    datasets$hierarchical <- data.frame(
      level1 = rep(c("Region A", "Region B", "Region C"), each = 20),
      level2 = rep(c("State 1", "State 2", "State 3", "State 4"), 15),
      level3 = paste("City", 1:60),
      value = abs(rnorm(60, 100, 30)),
      metric2 = runif(60, 50, 200)
    )
  }
  
  if (dataset_type %in% c("all", "network")) {
    # Network data
    n_nodes <- 20
    datasets$network <- list(
      nodes = data.frame(
        id = 1:n_nodes,
        label = paste("Node", 1:n_nodes),
        group = sample(c("Group A", "Group B", "Group C"), n_nodes, replace = TRUE),
        value = runif(n_nodes, 1, 100)
      ),
      edges = data.frame(
        from = sample(1:n_nodes, 40, replace = TRUE),
        to = sample(1:n_nodes, 40, replace = TRUE),
        weight = runif(40, 0.1, 1)
      )
    )
  }
  
  if (dataset_type %in% c("all", "correlation")) {
    # Correlation matrix data
    n <- 100
    datasets$correlation <- data.frame(
      var1 = rnorm(n),
      var2 = rnorm(n),
      var3 = rnorm(n),
      var4 = rnorm(n),
      var5 = rnorm(n)
    ) %>%
    mutate(
      var2 = var1 * 0.7 + var2 * 0.3,  # Positive correlation
      var3 = -var1 * 0.5 + var3 * 0.5,  # Negative correlation
      var4 = var2 * 0.8 + var4 * 0.2   # Indirect correlation
    )
  }
  
  if (dataset_type %in% c("all", "distributions")) {
    # Multiple distributions
    datasets$distributions <- data.frame(
      normal = rnorm(1000, 50, 15),
      exponential = rexp(1000, 0.1),
      uniform = runif(1000, 0, 100),
      gamma = rgamma(1000, shape = 2, rate = 0.1),
      beta = rbeta(1000, 2, 5) * 100,
      group = rep(c("Treatment", "Control"), each = 500)
    ) %>%
    pivot_longer(cols = c(normal, exponential, uniform, gamma, beta),
                 names_to = "distribution",
                 values_to = "value")
  }
  
  if (length(datasets) == 1) {
    return(datasets[[1]])
  } else {
    return(datasets)
  }
}

# ===============================================================================
# 2. ADVANCED GGPLOT2 VISUALIZATIONS
# ===============================================================================

#' Advanced Scatter Plots with Multiple Dimensions
#' @param data Data frame
#' @param x X variable
#' @param y Y variable
#' @param color Color variable
#' @param size Size variable
#' @param facet Faceting variable
create_advanced_scatter <- function(data, x, y, color = NULL, size = NULL, facet = NULL) {
  cat("\n=== ADVANCED SCATTER PLOT ===\n")
  
  # Base plot
  p <- ggplot(data, aes_string(x = x, y = y))
  
  # Add color mapping if specified
  if (!is.null(color)) {
    p <- p + aes_string(color = color)
  }
  
  # Add size mapping if specified
  if (!is.null(size)) {
    p <- p + aes_string(size = size)
  }
  
  # Add points
  p <- p + geom_point(alpha = 0.7) +
    geom_smooth(method = "lm", se = TRUE, alpha = 0.3) +
    theme_minimal() +
    labs(title = "Advanced Scatter Plot with Multiple Dimensions",
         subtitle = paste("Relationship between", x, "and", y))
  
  # Add faceting if specified
  if (!is.null(facet)) {
    p <- p + facet_wrap(as.formula(paste("~", facet)), scales = "free")
  }
  
  # Enhance with marginal plots
  if (is.null(facet) && requireNamespace("ggExtra", quietly = TRUE)) {
    p_with_margins <- ggExtra::ggMarginal(p, type = "density", fill = "lightblue")
    print(p_with_margins)
  } else {
    print(p)
  }
  
  return(p)
}

#' Ridge Plots for Distribution Comparison
#' @param data Data frame
#' @param x Continuous variable
#' @param y Grouping variable
create_ridge_plot <- function(data, x, y) {
  cat("\n=== RIDGE PLOT ===\n")
  
  p <- ggplot(data, aes_string(x = x, y = y, fill = y)) +
    ggridges::geom_density_ridges(alpha = 0.7, scale = 2) +
    theme_ridges() +
    scale_fill_viridis_d() +
    labs(title = "Distribution Comparison with Ridge Plot",
         x = x, y = y) +
    theme(legend.position = "none")
  
  print(p)
  return(p)
}

#' Alluvial/Sankey Diagrams
#' @param data Data frame
#' @param ... Grouping variables
create_alluvial_plot <- function(data, ...) {
  cat("\n=== ALLUVIAL PLOT ===\n")
  
  # Convert to alluvial format
  vars <- quos(...)
  
  alluvial_data <- data %>%
    count(!!!vars) %>%
    filter(n > 0)
  
  p <- ggplot(alluvial_data, aes_string(axis1 = names(alluvial_data)[1],
                                       axis2 = names(alluvial_data)[2],
                                       y = "n")) +
    geom_alluvium(aes(fill = .data[[names(alluvial_data)[1]]]), width = 1/12) +
    geom_stratum(width = 1/12, fill = "black", color = "grey") +
    geom_label(stat = "stratum", aes_string(label = "after_stat(stratum)")) +
    scale_x_discrete(limits = names(alluvial_data)[1:2], expand = c(.05, .05)) +
    scale_fill_viridis_d() +
    theme_minimal() +
    labs(title = "Alluvial Flow Diagram") +
    theme(legend.position = "none")
  
  print(p)
  return(p)
}

#' Correlation Heatmap with Clustering
#' @param data Numeric data frame
#' @param method Correlation method
create_correlation_heatmap <- function(data, method = "pearson") {
  cat("\n=== CORRELATION HEATMAP ===\n")
  
  # Calculate correlation matrix
  cor_matrix <- cor(data, method = method, use = "complete.obs")
  
  # Hierarchical clustering
  hc <- hclust(as.dist(1 - abs(cor_matrix)))
  cor_matrix_ordered <- cor_matrix[hc$order, hc$order]
  
  # Create heatmap
  p1 <- corrplot(cor_matrix_ordered, 
                method = "color",
                type = "upper",
                order = "original",
                tl.cex = 0.8,
                tl.col = "black",
                title = "Correlation Heatmap with Clustering")
  
  # Alternative with ggplot2
  cor_df <- as.data.frame(as.table(cor_matrix_ordered))
  names(cor_df) <- c("Var1", "Var2", "Correlation")
  
  p2 <- ggplot(cor_df, aes(Var1, Var2, fill = Correlation)) +
    geom_tile() +
    scale_fill_gradient2(low = "blue", mid = "white", high = "red",
                        midpoint = 0, limit = c(-1, 1)) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    labs(title = "Correlation Heatmap (ggplot2 version)")
  
  print(p2)
  
  return(list(corrplot = p1, ggplot = p2, matrix = cor_matrix_ordered))
}

#' Advanced Time Series Visualization
#' @param data Time series data frame
#' @param date_col Date column name
#' @param value_cols Value column names
create_advanced_timeseries <- function(data, date_col, value_cols) {
  cat("\n=== ADVANCED TIME SERIES PLOT ===\n")
  
  # Reshape data for plotting
  plot_data <- data %>%
    select(all_of(c(date_col, value_cols))) %>%
    pivot_longer(cols = all_of(value_cols), names_to = "series", values_to = "value")
  
  # Base time series plot
  p1 <- ggplot(plot_data, aes_string(x = date_col, y = "value", color = "series")) +
    geom_line(size = 1, alpha = 0.8) +
    geom_smooth(se = FALSE, alpha = 0.5) +
    scale_color_viridis_d() +
    theme_minimal() +
    labs(title = "Multi-Series Time Series Plot",
         x = "Date", y = "Value") +
    theme(legend.position = "bottom")
  
  print(p1)
  
  # Seasonal decomposition plot (if applicable)
  if (length(value_cols) >= 1) {
    ts_data <- ts(data[[value_cols[1]]], frequency = 12)  # Assuming monthly data
    decomp <- decompose(ts_data)
    
    decomp_df <- data.frame(
      date = rep(data[[date_col]], 4),
      component = rep(c("Observed", "Trend", "Seasonal", "Residual"), 
                     each = nrow(data)),
      value = c(as.numeric(decomp$x), 
               as.numeric(decomp$trend),
               as.numeric(decomp$seasonal),
               as.numeric(decomp$random))
    ) %>%
      filter(!is.na(value))
    
    p2 <- ggplot(decomp_df, aes(x = date, y = value)) +
      geom_line(color = "steelblue") +
      facet_wrap(~ component, scales = "free_y", ncol = 1) +
      theme_minimal() +
      labs(title = "Time Series Decomposition",
           x = "Date", y = "Value")
    
    print(p2)
    
    return(list(timeseries = p1, decomposition = p2))
  }
  
  return(p1)
}

# ===============================================================================
# 3. INTERACTIVE VISUALIZATIONS
# ===============================================================================

#' Interactive Scatter Plot with Plotly
#' @param data Data frame
#' @param x X variable
#' @param y Y variable
#' @param color Color variable
#' @param size Size variable
create_interactive_scatter <- function(data, x, y, color = NULL, size = NULL) {
  cat("\n=== INTERACTIVE SCATTER PLOT ===\n")
  
  # Create base plot
  p <- plot_ly(data, x = as.formula(paste("~", x)), y = as.formula(paste("~", y)),
               type = "scatter", mode = "markers")
  
  # Add color if specified
  if (!is.null(color)) {
    p <- p %>% add_trace(color = as.formula(paste("~", color)))
  }
  
  # Add size if specified
  if (!is.null(size)) {
    p <- p %>% add_trace(size = as.formula(paste("~", size)))
  }
  
  # Enhance layout
  p <- p %>%
    layout(title = "Interactive Scatter Plot",
           xaxis = list(title = x),
           yaxis = list(title = y),
           hovermode = "closest")
  
  print(p)
  return(p)
}

#' Interactive Time Series Dashboard
#' @param data Time series data
#' @param date_col Date column
#' @param value_cols Value columns
create_interactive_timeseries <- function(data, date_col, value_cols) {
  cat("\n=== INTERACTIVE TIME SERIES DASHBOARD ===\n")
  
  # Reshape data
  plot_data <- data %>%
    select(all_of(c(date_col, value_cols))) %>%
    pivot_longer(cols = all_of(value_cols), names_to = "series", values_to = "value")
  
  # Create interactive plot
  p <- plot_ly(plot_data, x = as.formula(paste("~", date_col)), y = ~value,
               color = ~series, type = "scatter", mode = "lines") %>%
    layout(title = "Interactive Time Series Dashboard",
           xaxis = list(title = "Date", rangeslider = list(type = "date")),
           yaxis = list(title = "Value"),
           hovermode = "x unified")
  
  print(p)
  return(p)
}

#' Interactive Correlation Network
#' @param data Numeric data frame
#' @param threshold Correlation threshold for edges
create_correlation_network <- function(data, threshold = 0.5) {
  cat("\n=== INTERACTIVE CORRELATION NETWORK ===\n")
  
  # Calculate correlations
  cor_matrix <- cor(data, use = "complete.obs")
  
  # Create network data
  nodes <- data.frame(
    id = colnames(cor_matrix),
    label = colnames(cor_matrix),
    value = diag(cor_matrix)
  )
  
  # Create edges from correlation matrix
  edges <- data.frame()
  for (i in 1:(ncol(cor_matrix)-1)) {
    for (j in (i+1):ncol(cor_matrix)) {
      if (abs(cor_matrix[i, j]) > threshold) {
        edges <- rbind(edges, data.frame(
          from = colnames(cor_matrix)[i],
          to = colnames(cor_matrix)[j],
          value = abs(cor_matrix[i, j]),
          color = ifelse(cor_matrix[i, j] > 0, "red", "blue")
        ))
      }
    }
  }
  
  # Create network visualization
  if (nrow(edges) > 0) {
    network <- visNetwork(nodes, edges) %>%
      visOptions(highlightNearest = TRUE, nodesIdSelection = TRUE) %>%
      visLayout(randomSeed = 123) %>%
      visNodes(borderWidth = 2) %>%
      visEdges(smooth = FALSE)
    
    print(network)
    return(network)
  } else {
    cat("No correlations above threshold found.\n")
    return(NULL)
  }
}

#' Interactive 3D Scatter Plot
#' @param data Data frame
#' @param x X variable
#' @param y Y variable  
#' @param z Z variable
#' @param color Color variable
create_3d_scatter <- function(data, x, y, z, color = NULL) {
  cat("\n=== INTERACTIVE 3D SCATTER PLOT ===\n")
  
  p <- plot_ly(data, x = as.formula(paste("~", x)), 
               y = as.formula(paste("~", y)), 
               z = as.formula(paste("~", z)),
               type = "scatter3d", mode = "markers")
  
  if (!is.null(color)) {
    p <- p %>% add_trace(color = as.formula(paste("~", color)))
  }
  
  p <- p %>%
    layout(title = "Interactive 3D Scatter Plot",
           scene = list(
             xaxis = list(title = x),
             yaxis = list(title = y),
             zaxis = list(title = z)
           ))
  
  print(p)
  return(p)
}

# ===============================================================================
# 4. SPECIALIZED CHART TYPES
# ===============================================================================

#' Sunburst Chart for Hierarchical Data
#' @param data Hierarchical data frame
#' @param levels Column names representing hierarchy levels
#' @param values Value column
create_sunburst_chart <- function(data, levels, values) {
  cat("\n=== SUNBURST CHART ===\n")
  
  # Prepare data for sunburst
  sunburst_data <- data %>%
    group_by(across(all_of(levels))) %>%
    summarise(value = sum(.data[[values]], na.rm = TRUE), .groups = "drop")
  
  # Create labels and parents for plotly sunburst
  labels <- c()
  parents <- c()
  values_vec <- c()
  
  # Root level
  root_summary <- sunburst_data %>%
    group_by(.data[[levels[1]]]) %>%
    summarise(value = sum(value), .groups = "drop")
  
  labels <- c(labels, as.character(root_summary[[levels[1]]]))
  parents <- c(parents, rep("", nrow(root_summary)))
  values_vec <- c(values_vec, root_summary$value)
  
  # Second level
  if (length(levels) > 1) {
    second_summary <- sunburst_data %>%
      group_by(.data[[levels[1]]], .data[[levels[2]]]) %>%
      summarise(value = sum(value), .groups = "drop")
    
    labels <- c(labels, paste(second_summary[[levels[1]]], 
                             second_summary[[levels[2]]], sep = " - "))
    parents <- c(parents, as.character(second_summary[[levels[1]]]))
    values_vec <- c(values_vec, second_summary$value)
  }
  
  # Create sunburst plot
  p <- plot_ly(
    type = "sunburst",
    labels = labels,
    parents = parents,
    values = values_vec,
    branchvalues = "total"
  ) %>%
    layout(title = "Hierarchical Sunburst Chart")
  
  print(p)
  return(p)
}

#' Radar Chart
#' @param data Data frame
#' @param group_col Grouping column
#' @param value_cols Value columns for radar axes
create_radar_chart <- function(data, group_col, value_cols) {
  cat("\n=== RADAR CHART ===\n")
  
  # Normalize values to 0-1 scale
  normalized_data <- data %>%
    mutate(across(all_of(value_cols), ~ scale(.x, center = FALSE, scale = TRUE)))
  
  # Create radar chart data
  radar_data <- normalized_data %>%
    select(all_of(c(group_col, value_cols))) %>%
    pivot_longer(cols = all_of(value_cols), names_to = "metric", values_to = "value")
  
  # Plot with plotly
  groups <- unique(radar_data[[group_col]])
  
  p <- plot_ly(type = "scatterpolar", mode = "lines+markers")
  
  for (group in groups) {
    group_data <- radar_data %>% filter(.data[[group_col]] == group)
    
    p <- p %>%
      add_trace(
        r = group_data$value,
        theta = group_data$metric,
        name = group,
        type = "scatterpolar",
        mode = "lines+markers"
      )
  }
  
  p <- p %>%
    layout(
      title = "Radar Chart Comparison",
      polar = list(
        radialaxis = list(visible = TRUE, range = c(0, 1))
      )
    )
  
  print(p)
  return(p)
}

#' Treemap Visualization
#' @param data Hierarchical data
#' @param levels Hierarchy levels
#' @param values Value column
create_treemap <- function(data, levels, values) {
  cat("\n=== TREEMAP VISUALIZATION ===\n")
  
  # Aggregate data
  treemap_data <- data %>%
    group_by(across(all_of(levels))) %>%
    summarise(value = sum(.data[[values]], na.rm = TRUE), .groups = "drop") %>%
    filter(value > 0)
  
  # Create treemap
  p <- plot_ly(
    data = treemap_data,
    type = "treemap",
    labels = apply(treemap_data[levels], 1, paste, collapse = " - "),
    values = treemap_data[[values]],
    parents = if(length(levels) > 1) treemap_data[[levels[1]]] else rep("", nrow(treemap_data))
  ) %>%
    layout(title = "Treemap Visualization")
  
  print(p)
  return(p)
}

# ===============================================================================
# 5. ANIMATION AND TIME-BASED VISUALIZATIONS
# ===============================================================================

#' Animated Scatter Plot
#' @param data Data frame with time variable
#' @param x X variable
#' @param y Y variable
#' @param time_col Time column
#' @param color Color variable
create_animated_scatter <- function(data, x, y, time_col, color = NULL) {
  cat("\n=== ANIMATED SCATTER PLOT ===\n")
  
  # Create base plot
  p <- ggplot(data, aes_string(x = x, y = y)) +
    geom_point(aes_string(color = color), size = 3, alpha = 0.7) +
    theme_minimal() +
    labs(title = "Animated Scatter Plot",
         subtitle = paste("Time:", "{closest_state}")) +
    transition_states(as.name(time_col), transition_length = 1, state_length = 1) +
    ease_aes('linear')
  
  # Render animation
  anim <- animate(p, width = 800, height = 600, duration = 8)
  print(anim)
  
  return(anim)
}

#' Racing Bar Chart
#' @param data Data frame with time and category data
#' @param time_col Time column
#' @param category_col Category column
#' @param value_col Value column
create_racing_bars <- function(data, time_col, category_col, value_col) {
  cat("\n=== RACING BAR CHART ===\n")
  
  # Prepare data for racing bars
  racing_data <- data %>%
    group_by(.data[[time_col]]) %>%
    arrange(desc(.data[[value_col]])) %>%
    mutate(rank = row_number()) %>%
    filter(rank <= 10)  # Top 10 categories
  
  # Create animated bar chart
  p <- ggplot(racing_data, aes_string(x = "rank", y = value_col, fill = category_col)) +
    geom_col() +
    geom_text(aes_string(label = category_col), hjust = 1.1) +
    coord_flip(clip = "off", expand = FALSE) +
    scale_x_reverse() +
    theme_minimal() +
    theme(
      axis.text.y = element_blank(),
      axis.ticks.y = element_blank(),
      axis.line.y = element_blank(),
      legend.position = "none"
    ) +
    labs(title = "Racing Bar Chart",
         subtitle = paste("Time: {closest_state}"),
         x = "", y = value_col) +
    transition_states(as.name(time_col), transition_length = 1, state_length = 1) +
    ease_aes('cubic-in-out')
  
  # Render animation
  anim <- animate(p, width = 800, height = 600, duration = 10)
  print(anim)
  
  return(anim)
}

# ===============================================================================
# 6. PUBLICATION-READY PLOTS
# ===============================================================================

#' Create Publication-Ready Multi-Panel Figure
#' @param plots List of ggplot objects
#' @param layout Layout specification
create_publication_figure <- function(plots, layout = "2x2") {
  cat("\n=== PUBLICATION-READY FIGURE ===\n")
  
  # Apply consistent theme
  theme_pub <- theme_minimal() +
    theme(
      text = element_text(size = 12, family = "Arial"),
      plot.title = element_text(size = 14, face = "bold"),
      axis.title = element_text(size = 11),
      axis.text = element_text(size = 10),
      legend.title = element_text(size = 11),
      legend.text = element_text(size = 10),
      strip.text = element_text(size = 11, face = "bold"),
      panel.grid.minor = element_blank()
    )
  
  # Apply theme to all plots
  plots_themed <- lapply(plots, function(p) p + theme_pub)
  
  # Create multi-panel figure
  if (layout == "2x2" && length(plots_themed) == 4) {
    final_plot <- (plots_themed[[1]] | plots_themed[[2]]) / 
                  (plots_themed[[3]] | plots_themed[[4]])
  } else if (layout == "1x2" && length(plots_themed) == 2) {
    final_plot <- plots_themed[[1]] | plots_themed[[2]]
  } else if (layout == "2x1" && length(plots_themed) == 2) {
    final_plot <- plots_themed[[1]] / plots_themed[[2]]
  } else {
    # Default arrangement
    final_plot <- wrap_plots(plots_themed)
  }
  
  # Add labels
  final_plot <- final_plot + plot_annotation(
    title = "Multi-Panel Scientific Figure",
    caption = "Data visualization with R",
    tag_levels = "A"
  )
  
  print(final_plot)
  return(final_plot)
}

#' Save High-Quality Plots
#' @param plot Plot object
#' @param filename Filename without extension
#' @param formats Output formats
#' @param width Width in inches
#' @param height Height in inches
#' @param dpi Resolution
save_high_quality_plot <- function(plot, filename, formats = c("png", "pdf", "svg"), 
                                  width = 10, height = 8, dpi = 300) {
  cat("\n=== SAVING HIGH-QUALITY PLOTS ===\n")
  
  for (format in formats) {
    output_file <- paste0(filename, ".", format)
    
    if (format == "png") {
      ggsave(output_file, plot, device = "png", width = width, height = height, 
             dpi = dpi, type = "cairo")
    } else if (format == "pdf") {
      ggsave(output_file, plot, device = "pdf", width = width, height = height)
    } else if (format == "svg") {
      ggsave(output_file, plot, device = "svg", width = width, height = height)
    } else if (format == "eps") {
      ggsave(output_file, plot, device = "eps", width = width, height = height)
    }
    
    cat("Saved:", output_file, "\n")
  }
}

# ===============================================================================
# 7. COMPREHENSIVE WORKFLOW
# ===============================================================================

#' Complete Advanced Visualization Workflow
#' @param data_source Data source type
#' @param custom_data Custom data frame
complete_visualization_workflow <- function(data_source = "sample", custom_data = NULL) {
  cat("===============================================================================\n")
  cat("COMPLETE ADVANCED VISUALIZATION WORKFLOW\n")
  cat("===============================================================================\n")
  
  # 1. Load or create data
  cat("\n1. PREPARING DATA...\n")
  if (!is.null(custom_data)) {
    data_list <- list(custom = custom_data)
  } else {
    data_list <- create_sample_data("all")
  }
  
  # 2. Basic visualizations
  cat("\n2. CREATING BASIC VISUALIZATIONS...\n")
  plots <- list()
  
  if ("basic" %in% names(data_list)) {
    plots$scatter <- create_advanced_scatter(data_list$basic, "x", "y", "category", "size")
  }
  
  # 3. Distribution plots
  if ("distributions" %in% names(data_list)) {
    cat("\n3. CREATING DISTRIBUTION PLOTS...\n")
    plots$ridge <- create_ridge_plot(data_list$distributions, "value", "distribution")
  }
  
  # 4. Correlation analysis
  if ("correlation" %in% names(data_list)) {
    cat("\n4. CORRELATION ANALYSIS...\n")
    plots$correlation <- create_correlation_heatmap(data_list$correlation)
  }
  
  # 5. Time series visualization
  if ("time_series" %in% names(data_list)) {
    cat("\n5. TIME SERIES VISUALIZATION...\n")
    plots$timeseries <- create_advanced_timeseries(
      data_list$time_series, "date", c("value1", "value2", "trend")
    )
  }
  
  # 6. Interactive visualizations
  cat("\n6. CREATING INTERACTIVE VISUALIZATIONS...\n")
  interactive_plots <- list()
  
  if ("basic" %in% names(data_list)) {
    interactive_plots$scatter <- create_interactive_scatter(
      data_list$basic, "x", "y", "category", "size"
    )
  }
  
  if ("correlation" %in% names(data_list)) {
    interactive_plots$network <- create_correlation_network(data_list$correlation)
  }
  
  # 7. Specialized charts
  if ("hierarchical" %in% names(data_list)) {
    cat("\n7. SPECIALIZED CHARTS...\n")
    plots$sunburst <- create_sunburst_chart(
      data_list$hierarchical, c("level1", "level2"), "value"
    )
    plots$treemap <- create_treemap(
      data_list$hierarchical, c("level1", "level2"), "value"
    )
  }
  
  # 8. Create publication figure
  if (length(plots) >= 2) {
    cat("\n8. CREATING PUBLICATION FIGURE...\n")
    plot_list <- plots[!sapply(plots, is.null)]
    if (length(plot_list) >= 2) {
      plots$publication <- create_publication_figure(
        plot_list[1:min(4, length(plot_list))]
      )
    }
  }
  
  # 9. Generate report
  cat("\n9. GENERATING VISUALIZATION REPORT...\n")
  generate_visualization_report(plots, interactive_plots)
  
  return(list(
    static_plots = plots,
    interactive_plots = interactive_plots,
    data = data_list
  ))
}

#' Generate Visualization Report
#' @param plots Static plots list
#' @param interactive_plots Interactive plots list
generate_visualization_report <- function(plots, interactive_plots) {
  cat("===============================================================================\n")
  cat("ADVANCED VISUALIZATION ANALYSIS REPORT\n")
  cat("===============================================================================\n")
  
  cat("\nVISUALIZATION SUMMARY:\n")
  cat("- Static plots created:", length(plots), "\n")
  cat("- Interactive plots created:", length(interactive_plots), "\n")
  
  cat("\nSTATIC PLOT TYPES:\n")
  for (plot_name in names(plots)) {
    cat("- ", plot_name, "\n")
  }
  
  cat("\nINTERACTIVE PLOT TYPES:\n")
  for (plot_name in names(interactive_plots)) {
    cat("- ", plot_name, "\n")
  }
  
  cat("\nRECOMMENDATIONS:\n")
  cat("- Use interactive plots for exploratory analysis\n")
  cat("- Use static plots for publications and reports\n")
  cat("- Consider color accessibility in your visualizations\n")
  cat("- Save plots in multiple formats for different uses\n")
}

# ===============================================================================
# EXAMPLE USAGE
# ===============================================================================

# Run complete visualization workflow
# results <- complete_visualization_workflow("sample")
# 
# # Create specific visualizations
# sample_data <- create_sample_data("basic")
# scatter_plot <- create_advanced_scatter(sample_data, "x", "y", "category", "size")
# interactive_scatter <- create_interactive_scatter(sample_data, "x", "y", "category")
# 
# # Save high-quality plots
# save_high_quality_plot(scatter_plot, "my_scatter_plot", c("png", "pdf"))

cat("Advanced Visualization Template Loaded Successfully!\n")
cat("Key Functions Available:\n")
cat("- complete_visualization_workflow(): Run full visualization pipeline\n")
cat("- create_advanced_scatter(): Multi-dimensional scatter plots\n")
cat("- create_interactive_scatter(): Interactive plotly visualizations\n")
cat("- create_correlation_heatmap(): Advanced correlation analysis\n")
cat("- create_ridge_plot(): Distribution comparison\n")
cat("- create_3d_scatter(): 3D interactive plots\n")
cat("- create_publication_figure(): Publication-ready multi-panels\n")
cat("- save_high_quality_plot(): Export high-resolution plots\n") 
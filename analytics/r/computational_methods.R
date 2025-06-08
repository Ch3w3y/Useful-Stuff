# Advanced R Computational Methods
# Comprehensive guide to high-performance computing and advanced statistical methods in R
# Author: Data Science Team
# Date: 2024

# =============================================================================
# SETUP AND DEPENDENCIES
# =============================================================================

# Essential packages for computational methods
if (!require("pacman")) install.packages("pacman")

pacman::p_load(
  # Parallel computing
  parallel, foreach, doParallel, future, future.apply,
  
  # High-performance computing
  Rcpp, RcppArmadillo, RcppEigen, microbenchmark,
  
  # Numerical methods
  numDeriv, pracma, rootSolve, BB, nleqslv,
  
  # Optimization
  optimx, DEoptim, GenSA, PSO, GA, nloptr,
  
  # Matrix operations
  Matrix, slam, RSpectra, irlba,
  
  # Statistical computing
  mvtnorm, MASS, MCMCpack, coda, ggmcmc,
  
  # Simulation
  truncnorm, extraDistr, actuar, copula,
  
  # Machine learning backends
  torch, tensorflow, keras, xgboost, ranger,
  
  # Data manipulation
  data.table, dplyr, tidyr,
  
  # Visualization
  ggplot2, plotly, rayshader,
  
  # Profiling and debugging
  profvis, pryr, bench
)

# =============================================================================
# PARALLEL COMPUTING SETUP
# =============================================================================

#' Setup Parallel Computing Environment
#'
#' @param strategy Character, parallel strategy ("multisession", "multicore", "cluster")
#' @param workers Integer, number of workers (default: detectCores() - 1)
#' @param verbose Logical, whether to print setup information
#' @return List with parallel backend information
setup_parallel <- function(strategy = "multisession", workers = NULL, verbose = TRUE) {
  
  if (is.null(workers)) {
    workers <- max(1, parallel::detectCores() - 1)
  }
  
  # Setup for different parallel backends
  if (strategy == "multisession" || strategy == "multicore") {
    # future backend
    future::plan(strategy, workers = workers)
    
    if (verbose) {
      cat("Future parallel backend configured:\n")
      cat("Strategy:", strategy, "\n")
      cat("Workers:", workers, "\n")
    }
    
  } else if (strategy == "cluster") {
    # Traditional cluster setup
    cl <- parallel::makeCluster(workers)
    doParallel::registerDoParallel(cl)
    
    if (verbose) {
      cat("Cluster parallel backend configured:\n")
      cat("Workers:", workers, "\n")
    }
    
    return(list(cluster = cl, workers = workers))
  }
  
  return(list(strategy = strategy, workers = workers))
}

#' Cleanup Parallel Resources
#'
#' @param cluster Optional cluster object to stop
cleanup_parallel <- function(cluster = NULL) {
  if (!is.null(cluster)) {
    parallel::stopCluster(cluster)
  }
  
  # Reset future plan
  future::plan("sequential")
  
  cat("Parallel resources cleaned up\n")
}

# =============================================================================
# HIGH-PERFORMANCE MATRIX OPERATIONS
# =============================================================================

#' Efficient Matrix Multiplication with Different Backends
#'
#' @param A First matrix
#' @param B Second matrix
#' @param method Character, method to use ("base", "crossprod", "tcrossprod", "sparse")
#' @param benchmark Logical, whether to benchmark performance
#' @return Matrix product
matrix_multiply <- function(A, B, method = "base", benchmark = FALSE) {
  
  methods <- list(
    base = function(A, B) A %*% B,
    crossprod = function(A, B) crossprod(t(A), B),
    tcrossprod = function(A, B) tcrossprod(A, t(B)),
    sparse = function(A, B) {
      if (!inherits(A, "Matrix")) A <- Matrix::Matrix(A, sparse = TRUE)
      if (!inherits(B, "Matrix")) B <- Matrix::Matrix(B, sparse = TRUE)
      A %*% B
    }
  )
  
  if (benchmark) {
    results <- microbenchmark::microbenchmark(
      base = methods$base(A, B),
      crossprod = methods$crossprod(A, B),
      tcrossprod = methods$tcrossprod(A, B),
      sparse = methods$sparse(A, B),
      times = 10
    )
    print(results)
  }
  
  methods[[method]](A, B)
}

#' Parallel Matrix Operations
#'
#' @param X Matrix for operations
#' @param operation Character, operation to perform
#' @param chunk_size Integer, size of chunks for parallel processing
#' @return Result of parallel operation
parallel_matrix_ops <- function(X, operation = "rowSums", chunk_size = 1000) {
  
  n_rows <- nrow(X)
  n_chunks <- ceiling(n_rows / chunk_size)
  
  # Create row indices for chunks
  row_chunks <- split(1:n_rows, cut(1:n_rows, breaks = n_chunks, labels = FALSE))
  
  # Parallel operation
  results <- future.apply::future_lapply(row_chunks, function(rows) {
    X_chunk <- X[rows, , drop = FALSE]
    
    switch(operation,
           "rowSums" = rowSums(X_chunk),
           "rowMeans" = rowMeans(X_chunk),
           "rowVars" = apply(X_chunk, 1, var),
           "rowSds" = apply(X_chunk, 1, sd),
           "rowMedians" = apply(X_chunk, 1, median)
    )
  })
  
  # Combine results
  unlist(results)
}

#' Efficient Eigenvalue Computation
#'
#' @param X Symmetric matrix
#' @param k Integer, number of eigenvalues to compute
#' @param which Character, which eigenvalues ("LM", "SM", "LA", "SA")
#' @param method Character, method to use ("RSpectra", "irlba", "base")
#' @return List with eigenvalues and eigenvectors
fast_eigen <- function(X, k = min(10, ncol(X)), which = "LM", method = "RSpectra") {
  
  if (method == "RSpectra" && k < ncol(X)) {
    result <- RSpectra::eigs_sym(X, k = k, which = which)
    return(list(values = result$values, vectors = result$vectors))
    
  } else if (method == "irlba" && k < ncol(X)) {
    result <- irlba::irlba(X, nv = k)
    return(list(values = result$d, vectors = result$v))
    
  } else {
    result <- eigen(X, symmetric = TRUE)
    if (k < length(result$values)) {
      indices <- order(result$values, decreasing = (which %in% c("LM", "LA")))[1:k]
      return(list(values = result$values[indices], vectors = result$vectors[, indices]))
    }
    return(result)
  }
}

# =============================================================================
# NUMERICAL METHODS AND OPTIMIZATION
# =============================================================================

#' Advanced Numerical Integration
#'
#' @param f Function to integrate
#' @param lower Lower bound
#' @param upper Upper bound
#' @param method Character, integration method
#' @param ... Additional arguments passed to integration function
#' @return Integration result
numerical_integrate <- function(f, lower, upper, method = "adaptive", ...) {
  
  switch(method,
         "adaptive" = integrate(f, lower, upper, ...),
         "gauss" = pracma::gaussLegendre(f, lower, upper, ...),
         "romberg" = pracma::romberg(f, lower, upper, ...),
         "simpson" = pracma::simps(seq(lower, upper, length.out = 1001), 
                                   f(seq(lower, upper, length.out = 1001))),
         "monte_carlo" = {
           n <- list(...)$n %||% 10000
           x <- runif(n, lower, upper)
           (upper - lower) * mean(f(x))
         }
  )
}

#' Numerical Derivatives
#'
#' @param f Function
#' @param x Point at which to evaluate derivative
#' @param method Character, method for derivative computation
#' @param h Numeric, step size
#' @return Derivative value
numerical_derivative <- function(f, x, method = "central", h = 1e-8) {
  
  switch(method,
         "forward" = (f(x + h) - f(x)) / h,
         "backward" = (f(x) - f(x - h)) / h,
         "central" = (f(x + h) - f(x - h)) / (2 * h),
         "richardson" = numDeriv::grad(f, x, method = "Richardson"),
         "complex" = Im(f(x + 1i * h)) / h
  )
}

#' Multi-Objective Optimization
#'
#' @param objective_fns List of objective functions
#' @param lower Lower bounds
#' @param upper Upper bounds
#' @param method Character, optimization method
#' @param ... Additional arguments
#' @return Pareto optimal solutions
multi_objective_optimize <- function(objective_fns, lower, upper, 
                                   method = "nsga2", ...) {
  
  # Combine objectives into single function
  combined_fn <- function(x) {
    sapply(objective_fns, function(fn) fn(x))
  }
  
  switch(method,
         "nsga2" = {
           if (!require("nsga2R")) install.packages("nsga2R")
           nsga2R::nsga2(combined_fn, idim = length(lower), odim = length(objective_fns),
                         lower.bounds = lower, upper.bounds = upper, ...)
         },
         "mopso" = {
           if (!require("mopsoR")) install.packages("mopsoR")
           # Implementation would go here
         }
  )
}

#' Robust Optimization with Multiple Algorithms
#'
#' @param fn Objective function
#' @param lower Lower bounds
#' @param upper Upper bounds
#' @param methods Character vector, optimization methods to try
#' @param maximize Logical, whether to maximize
#' @return Best optimization result
robust_optimize <- function(fn, lower, upper, 
                           methods = c("L-BFGS-B", "DEoptim", "GenSA"), 
                           maximize = FALSE) {
  
  # Prepare objective function
  obj_fn <- if (maximize) function(x) -fn(x) else fn
  
  results <- list()
  
  for (method in methods) {
    result <- tryCatch({
      switch(method,
             "L-BFGS-B" = optim(runif(length(lower), lower, upper), obj_fn, 
                                 method = "L-BFGS-B", lower = lower, upper = upper),
             "DEoptim" = DEoptim::DEoptim(obj_fn, lower, upper, 
                                         DEoptim.control(trace = FALSE)),
             "GenSA" = GenSA::GenSA(par = NULL, fn = obj_fn, lower = lower, upper = upper,
                                   control = list(verbose = FALSE)),
             "PSO" = PSO::psoptim(rep(NA, length(lower)), obj_fn, lower = lower, upper = upper),
             "GA" = GA::ga(type = "real-valued", fitness = function(x) -obj_fn(x),
                          lower = lower, upper = upper, monitor = FALSE)
      )
    }, error = function(e) {
      warning(paste("Method", method, "failed:", e$message))
      return(NULL)
    })
    
    if (!is.null(result)) {
      results[[method]] <- result
    }
  }
  
  # Find best result
  if (length(results) == 0) {
    stop("All optimization methods failed")
  }
  
  # Extract best value (handling different result structures)
  best_values <- sapply(results, function(r) {
    if (inherits(r, "DEoptim")) {
      r$optim$bestval
    } else if (inherits(r, "GenSA")) {
      r$value
    } else if (inherits(r, "ga")) {
      -r@fitnessValue
    } else {
      r$value
    }
  })
  
  # Adjust for maximization
  if (maximize) best_values <- -best_values
  
  best_method <- names(which.min(if (maximize) -best_values else best_values))
  
  list(
    best_result = results[[best_method]],
    best_method = best_method,
    all_results = results,
    comparison = data.frame(
      method = names(best_values),
      value = if (maximize) -best_values else best_values
    )
  )
}

# =============================================================================
# MONTE CARLO METHODS AND SIMULATION
# =============================================================================

#' Advanced Monte Carlo Integration
#'
#' @param f Function to integrate
#' @param lower Lower bounds (vector for multivariate)
#' @param upper Upper bounds (vector for multivariate)
#' @param n Number of samples
#' @param method Character, sampling method
#' @param antithetic Logical, use antithetic variables
#' @return Integration estimate with confidence interval
monte_carlo_integrate <- function(f, lower, upper, n = 10000, 
                                method = "uniform", antithetic = FALSE) {
  
  d <- length(lower)  # Dimension
  
  # Generate samples
  if (method == "uniform") {
    if (antithetic && n %% 2 == 0) {
      # Antithetic sampling
      u1 <- matrix(runif(n/2 * d), ncol = d)
      u2 <- 1 - u1
      u <- rbind(u1, u2)
    } else {
      u <- matrix(runif(n * d), ncol = d)
    }
    
    # Transform to integration domain
    x <- sweep(sweep(u, 2, upper - lower, "*"), 2, lower, "+")
    
  } else if (method == "sobol") {
    if (!require("randtoolbox")) install.packages("randtoolbox")
    u <- randtoolbox::sobol(n, d, scrambling = 3)
    x <- sweep(sweep(u, 2, upper - lower, "*"), 2, lower, "+")
    
  } else if (method == "halton") {
    if (!require("randtoolbox")) install.packages("randtoolbox")
    u <- randtoolbox::halton(n, d)
    x <- sweep(sweep(u, 2, upper - lower, "*"), 2, lower, "+")
  }
  
  # Evaluate function
  if (d == 1) {
    f_vals <- apply(x, 1, function(xi) f(xi[1]))
  } else {
    f_vals <- apply(x, 1, f)
  }
  
  # Calculate integral estimate
  volume <- prod(upper - lower)
  integral <- volume * mean(f_vals)
  
  # Calculate confidence interval
  se <- volume * sd(f_vals) / sqrt(n)
  ci_lower <- integral - 1.96 * se
  ci_upper <- integral + 1.96 * se
  
  list(
    integral = integral,
    standard_error = se,
    confidence_interval = c(ci_lower, ci_upper),
    n_samples = n,
    method = method
  )
}

#' Markov Chain Monte Carlo Sampler
#'
#' @param log_density Log density function
#' @param init_params Initial parameter values
#' @param n_samples Number of samples
#' @param method Character, MCMC method
#' @param ... Additional arguments for specific methods
#' @return MCMC samples
mcmc_sampler <- function(log_density, init_params, n_samples = 10000, 
                        method = "metropolis", ...) {
  
  d <- length(init_params)
  samples <- matrix(NA, nrow = n_samples, ncol = d)
  samples[1, ] <- init_params
  
  if (method == "metropolis") {
    # Metropolis algorithm
    proposal_cov <- list(...)$proposal_cov %||% diag(0.1, d)
    
    current <- init_params
    current_log_dens <- log_density(current)
    n_accepted <- 0
    
    for (i in 2:n_samples) {
      # Propose new state
      proposal <- mvtnorm::rmvnorm(1, current, proposal_cov)[1, ]
      proposal_log_dens <- log_density(proposal)
      
      # Accept/reject
      log_ratio <- proposal_log_dens - current_log_dens
      if (log(runif(1)) < log_ratio) {
        current <- proposal
        current_log_dens <- proposal_log_dens
        n_accepted <- n_accepted + 1
      }
      
      samples[i, ] <- current
    }
    
    acceptance_rate <- n_accepted / (n_samples - 1)
    attr(samples, "acceptance_rate") <- acceptance_rate
    
  } else if (method == "hmc") {
    # Hamiltonian Monte Carlo (simplified implementation)
    if (!require("HMC")) install.packages("HMC")
    # Would implement HMC here
    
  } else if (method == "nuts") {
    # No-U-Turn Sampler
    if (!require("rstan")) install.packages("rstan")
    # Would implement NUTS here
  }
  
  samples
}

#' Parallel Simulation Framework
#'
#' @param sim_function Function that performs one simulation
#' @param n_sims Number of simulations
#' @param n_cores Number of cores to use
#' @param ... Additional arguments passed to sim_function
#' @return List of simulation results
parallel_simulation <- function(sim_function, n_sims, n_cores = NULL, ...) {
  
  if (is.null(n_cores)) {
    n_cores <- min(parallel::detectCores() - 1, n_sims)
  }
  
  # Setup parallel backend
  setup_info <- setup_parallel("multisession", n_cores, verbose = FALSE)
  
  # Run simulations in parallel
  results <- future.apply::future_lapply(1:n_sims, function(i) {
    set.seed(i)  # Ensure reproducibility
    sim_function(...)
  }, future.seed = TRUE)
  
  # Cleanup
  cleanup_parallel()
  
  results
}

# =============================================================================
# STATISTICAL COMPUTING ALGORITHMS
# =============================================================================

#' Expectation-Maximization Algorithm
#'
#' @param data Data matrix
#' @param k Number of components
#' @param model Character, model type ("gaussian", "poisson")
#' @param max_iter Maximum iterations
#' @param tol Convergence tolerance
#' @return EM algorithm results
em_algorithm <- function(data, k, model = "gaussian", max_iter = 100, tol = 1e-6) {
  
  n <- nrow(data)
  d <- ncol(data)
  
  # Initialize parameters
  if (model == "gaussian") {
    # Gaussian mixture model
    pi <- rep(1/k, k)  # Mixing proportions
    mu <- data[sample(n, k), , drop = FALSE]  # Means
    sigma <- replicate(k, cov(data), simplify = FALSE)  # Covariances
    
    log_likelihood <- numeric(max_iter)
    
    for (iter in 1:max_iter) {
      # E-step: Calculate responsibilities
      gamma <- matrix(0, n, k)
      
      for (j in 1:k) {
        gamma[, j] <- pi[j] * mvtnorm::dmvnorm(data, mu[j, ], sigma[[j]])
      }
      
      # Normalize responsibilities
      gamma <- gamma / rowSums(gamma)
      
      # M-step: Update parameters
      N_j <- colSums(gamma)
      pi <- N_j / n
      
      for (j in 1:k) {
        mu[j, ] <- colSums(gamma[, j] * data) / N_j[j]
        
        centered <- sweep(data, 2, mu[j, ], "-")
        sigma[[j]] <- (t(centered) %*% (gamma[, j] * centered)) / N_j[j]
      }
      
      # Calculate log-likelihood
      log_lik <- 0
      for (i in 1:n) {
        component_probs <- numeric(k)
        for (j in 1:k) {
          component_probs[j] <- pi[j] * mvtnorm::dmvnorm(data[i, ], mu[j, ], sigma[[j]])
        }
        log_lik <- log_lik + log(sum(component_probs))
      }
      log_likelihood[iter] <- log_lik
      
      # Check convergence
      if (iter > 1 && abs(log_likelihood[iter] - log_likelihood[iter-1]) < tol) {
        log_likelihood <- log_likelihood[1:iter]
        break
      }
    }
    
    return(list(
      pi = pi,
      mu = mu,
      sigma = sigma,
      gamma = gamma,
      log_likelihood = log_likelihood,
      iterations = length(log_likelihood)
    ))
  }
}

#' Bootstrap Methods
#'
#' @param data Data vector or matrix
#' @param statistic Function to compute statistic
#' @param R Number of bootstrap samples
#' @param method Character, bootstrap method
#' @param ... Additional arguments passed to statistic function
#' @return Bootstrap results
bootstrap_methods <- function(data, statistic, R = 1000, 
                            method = "nonparametric", ...) {
  
  n <- if (is.matrix(data)) nrow(data) else length(data)
  
  if (method == "nonparametric") {
    # Standard nonparametric bootstrap
    bootstrap_stats <- replicate(R, {
      if (is.matrix(data)) {
        boot_indices <- sample(n, replace = TRUE)
        boot_data <- data[boot_indices, , drop = FALSE]
      } else {
        boot_data <- sample(data, replace = TRUE)
      }
      statistic(boot_data, ...)
    })
    
  } else if (method == "parametric") {
    # Parametric bootstrap (requires model specification)
    # This would need model-specific implementation
    
  } else if (method == "wild") {
    # Wild bootstrap for regression residuals
    # Implementation would depend on regression context
    
  } else if (method == "block") {
    # Block bootstrap for time series
    block_length <- list(...)$block_length %||% floor(sqrt(n))
    
    bootstrap_stats <- replicate(R, {
      # Create overlapping blocks
      n_blocks <- ceiling(n / block_length)
      block_starts <- sample(1:(n - block_length + 1), n_blocks, replace = TRUE)
      
      boot_data <- numeric(0)
      for (start in block_starts) {
        boot_data <- c(boot_data, data[start:(start + block_length - 1)])
      }
      boot_data <- boot_data[1:n]  # Trim to original length
      
      statistic(boot_data, ...)
    })
  }
  
  # Calculate confidence intervals
  alpha <- 0.05
  ci_lower <- quantile(bootstrap_stats, alpha/2)
  ci_upper <- quantile(bootstrap_stats, 1 - alpha/2)
  
  list(
    statistic = bootstrap_stats,
    mean = mean(bootstrap_stats),
    sd = sd(bootstrap_stats),
    confidence_interval = c(ci_lower, ci_upper),
    method = method,
    R = R
  )
}

# =============================================================================
# PERFORMANCE PROFILING AND OPTIMIZATION
# =============================================================================

#' Function Performance Profiler
#'
#' @param expr Expression to profile
#' @param times Number of times to run
#' @param unit Time unit for reporting
#' @return Performance profiling results
profile_performance <- function(expr, times = 100, unit = "ms") {
  
  # Microbenchmark
  mb_result <- microbenchmark::microbenchmark(
    expr,
    times = times,
    unit = unit
  )
  
  # Memory profiling
  mem_before <- pryr::mem_used()
  result <- eval(expr)
  mem_after <- pryr::mem_used()
  
  # Detailed profiling with profvis (if interactive)
  if (interactive()) {
    prof_result <- profvis::profvis({
      for (i in 1:min(10, times)) {
        eval(expr)
      }
    })
    
    return(list(
      microbenchmark = mb_result,
      memory_used = mem_after - mem_before,
      profvis = prof_result,
      result = result
    ))
  } else {
    return(list(
      microbenchmark = mb_result,
      memory_used = mem_after - mem_before,
      result = result
    ))
  }
}

#' Memory-Efficient Data Processing
#'
#' @param data_file Path to data file
#' @param chunk_size Number of rows per chunk
#' @param process_function Function to apply to each chunk
#' @param combine_function Function to combine results
#' @return Processed results
chunked_processing <- function(data_file, chunk_size = 10000, 
                             process_function, combine_function = rbind) {
  
  # Read file in chunks
  con <- file(data_file, "r")
  
  results <- list()
  chunk_num <- 1
  
  while (TRUE) {
    # Read chunk
    if (tools::file_ext(data_file) == "csv") {
      chunk <- tryCatch({
        read.csv(con, nrows = chunk_size, header = (chunk_num == 1))
      }, error = function(e) NULL)
    } else {
      # Handle other file types
      break
    }
    
    if (is.null(chunk) || nrow(chunk) == 0) break
    
    # Process chunk
    processed_chunk <- process_function(chunk)
    results[[chunk_num]] <- processed_chunk
    
    chunk_num <- chunk_num + 1
    
    # Memory cleanup
    rm(chunk, processed_chunk)
    gc()
  }
  
  close(con)
  
  # Combine results
  if (length(results) > 0) {
    do.call(combine_function, results)
  } else {
    NULL
  }
}

# =============================================================================
# EXAMPLE USAGE AND DEMONSTRATIONS
# =============================================================================

#' Demonstrate Computational Methods
demonstrate_methods <- function() {
  
  cat("=== R Computational Methods Demonstration ===\n\n")
  
  # 1. Parallel Computing Demo
  cat("1. Parallel Computing Demo\n")
  setup_parallel("multisession", 2, verbose = TRUE)
  
  # Parallel matrix operation
  X <- matrix(rnorm(1000 * 100), 1000, 100)
  
  cat("Comparing serial vs parallel row sums:\n")
  
  # Serial
  serial_time <- system.time({
    serial_result <- rowSums(X)
  })
  
  # Parallel
  parallel_time <- system.time({
    parallel_result <- parallel_matrix_ops(X, "rowSums", chunk_size = 250)
  })
  
  cat("Serial time:", serial_time[3], "seconds\n")
  cat("Parallel time:", parallel_time[3], "seconds\n")
  cat("Results identical:", all.equal(serial_result, parallel_result), "\n\n")
  
  # 2. Optimization Demo
  cat("2. Multi-Algorithm Optimization Demo\n")
  
  # Rosenbrock function
  rosenbrock <- function(x) {
    (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
  }
  
  opt_result <- robust_optimize(
    rosenbrock, 
    lower = c(-2, -2), 
    upper = c(2, 2),
    methods = c("L-BFGS-B", "DEoptim")
  )
  
  print(opt_result$comparison)
  cat("\n")
  
  # 3. Monte Carlo Integration Demo
  cat("3. Monte Carlo Integration Demo\n")
  
  # Integrate e^(-x^2) from 0 to 1
  f <- function(x) exp(-x^2)
  
  mc_result <- monte_carlo_integrate(f, 0, 1, n = 10000, method = "uniform")
  
  cat("Monte Carlo estimate:", mc_result$integral, "\n")
  cat("95% CI:", paste(round(mc_result$confidence_interval, 4), collapse = " - "), "\n")
  
  # Analytical result for comparison
  analytical <- sqrt(pi)/2 * (pnorm(sqrt(2)) - 0.5)
  cat("Analytical result:", analytical, "\n\n")
  
  # 4. Performance Profiling Demo
  cat("4. Performance Profiling Demo\n")
  
  # Compare different matrix operations
  A <- matrix(rnorm(500 * 500), 500, 500)
  B <- matrix(rnorm(500 * 500), 500, 500)
  
  prof_result <- profile_performance({
    C <- A %*% B
  }, times = 10)
  
  print(prof_result$microbenchmark)
  cat("Memory used:", as.numeric(prof_result$memory_used), "bytes\n\n")
  
  cleanup_parallel()
  
  cat("=== Demonstration Complete ===\n")
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

#' Check System Capabilities
check_system_capabilities <- function() {
  
  cat("=== System Computational Capabilities ===\n")
  cat("R Version:", R.version.string, "\n")
  cat("Platform:", R.version$platform, "\n")
  cat("CPU Cores:", parallel::detectCores(), "\n")
  
  # Memory info
  if (.Platform$OS.type == "unix") {
    mem_info <- system("free -h", intern = TRUE)
    cat("Memory Info:\n")
    cat(paste(mem_info, collapse = "\n"), "\n")
  }
  
  # Package versions
  key_packages <- c("Rcpp", "parallel", "Matrix", "data.table")
  cat("\nKey Package Versions:\n")
  for (pkg in key_packages) {
    if (requireNamespace(pkg, quietly = TRUE)) {
      cat(pkg, ":", as.character(packageVersion(pkg)), "\n")
    } else {
      cat(pkg, ": Not installed\n")
    }
  }
  
  # BLAS/LAPACK info
  cat("\nBLAS Library:", sessionInfo()$BLAS, "\n")
  cat("LAPACK Library:", sessionInfo()$LAPACK, "\n")
  
  cat("=== End System Info ===\n")
}

# =============================================================================
# INITIALIZATION
# =============================================================================

# Check system capabilities on load
check_system_capabilities()

cat("\nAdvanced R Computational Methods loaded successfully!\n")
cat("Key functions available:\n")
cat("- setup_parallel(): Setup parallel computing\n")
cat("- matrix_multiply(): Efficient matrix operations\n")
cat("- robust_optimize(): Multi-algorithm optimization\n")
cat("- monte_carlo_integrate(): Advanced MC integration\n")
cat("- mcmc_sampler(): MCMC sampling\n")
cat("- bootstrap_methods(): Bootstrap procedures\n")
cat("- profile_performance(): Performance profiling\n")
cat("- demonstrate_methods(): Run demonstration\n")
cat("\nUse demonstrate_methods() to see examples in action!\n") 
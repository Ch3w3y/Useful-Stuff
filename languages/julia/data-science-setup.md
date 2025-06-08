# Julia for Data Science

Julia is a high-level, high-performance programming language designed for technical computing, with syntax that is familiar to users of other technical computing environments. This guide covers setting up Julia for data science workflows.

## Table of Contents

1. [Installation and Setup](#installation-and-setup)
2. [Package Management](#package-management)
3. [Julia Fundamentals for Data Science](#julia-fundamentals-for-data-science)
4. [Data Manipulation](#data-manipulation)
5. [Statistical Analysis](#statistical-analysis)
6. [Machine Learning](#machine-learning)
7. [Visualization](#visualization)
8. [High-Performance Computing](#high-performance-computing)
9. [Interoperability](#interoperability)
10. [Performance Optimization](#performance-optimization)

## Installation and Setup

### Installing Julia

```bash
#!/bin/bash
# install-julia.sh

# Method 1: Using juliaup (recommended)
curl -fsSL https://install.julialang.org | sh
source ~/.bashrc

# Install latest stable version
juliaup add release
juliaup default release

# Method 2: Manual installation
wget https://julialang-s3.julialang.org/bin/linux/x64/1.9/julia-1.9.4-linux-x86_64.tar.gz
tar -xzf julia-1.9.4-linux-x86_64.tar.gz
sudo mv julia-1.9.4 /opt/julia
sudo ln -s /opt/julia/bin/julia /usr/local/bin/julia

# Method 3: Using package manager
# Ubuntu/Debian
sudo apt update && sudo apt install julia

# Fedora
sudo dnf install julia

# Arch Linux
sudo pacman -S julia

# Verify installation
julia --version
```

### Julia Configuration

```julia
# ~/.julia/config/startup.jl
# Global configuration for Julia sessions

using Pkg

# Set number of threads (uncomment and adjust as needed)
# ENV["JULIA_NUM_THREADS"] = "auto"

# Add commonly used packages to standard environment
# Pkg.add(["DataFrames", "CSV", "Plots", "Statistics"])

# Custom functions for data science workflows
function activate_project(path=".")
    """Activate project environment and instantiate packages"""
    Pkg.activate(path)
    Pkg.instantiate()
    println("Activated project at: $(pwd())")
end

function setup_notebook()
    """Setup Jupyter notebook with Julia kernel"""
    Pkg.add("IJulia")
    using IJulia
    installkernel("Julia")
end

# Set plotting backend
ENV["GKSwstype"] = "100"  # For headless plotting

println("Julia Data Science Environment Loaded!")
println("Available custom functions: activate_project(), setup_notebook()")
```

### Development Environment Setup

```julia
# scripts/setup_dev_environment.jl
using Pkg

# Essential packages for data science
essential_packages = [
    "DataFrames",      # Data manipulation
    "CSV",             # CSV file handling
    "Statistics",      # Statistical functions
    "LinearAlgebra",   # Linear algebra operations
    "Plots",           # Plotting
    "StatsPlots",      # Statistical plots
    "PlotlyJS",        # Interactive plots
    "IJulia",          # Jupyter notebook kernel
    "Revise",          # Code reloading
    "BenchmarkTools",  # Performance benchmarking
    "ProfileView",     # Performance profiling
    "PackageCompiler", # Precompilation
]

# Machine learning packages
ml_packages = [
    "MLJ",             # Machine learning framework
    "ScikitLearn",     # Scikit-learn interface
    "Flux",            # Deep learning
    "MLBase",          # ML utilities
    "DecisionTree",    # Decision trees
    "Clustering",      # Clustering algorithms
    "GLM",             # Generalized linear models
    "MultivariateStats", # Multivariate statistics
]

# Statistical packages
stats_packages = [
    "StatsBase",       # Statistical base functions
    "Distributions",   # Probability distributions
    "HypothesisTests", # Statistical hypothesis tests
    "Bootstrap",       # Bootstrap methods
    "KernelDensity",   # Kernel density estimation
    "TimeSeries",      # Time series analysis
    "DSP",             # Digital signal processing
]

# Data processing packages
data_packages = [
    "Query",           # Data querying
    "DataFramesMeta",  # DataFrame macros
    "Chain",           # Piping operations
    "Pipe",            # Alternative piping
    "Tables",          # Table interface
    "Arrow",           # Apache Arrow format
    "JSON3",           # JSON parsing
    "XLSX",            # Excel file handling
]

# Optimization and numerical packages
numerical_packages = [
    "Optim",           # Optimization
    "JuMP",            # Mathematical programming
    "DifferentialEquations", # ODE/PDE solving
    "ForwardDiff",     # Automatic differentiation
    "Roots",           # Root finding
    "QuadGK",          # Numerical integration
    "FFTW",            # Fast Fourier transforms
]

# Install all packages
all_packages = vcat(
    essential_packages,
    ml_packages,
    stats_packages,
    data_packages,
    numerical_packages
)

println("Installing $(length(all_packages)) packages...")
Pkg.add(all_packages)

println("Precompiling packages...")
Pkg.precompile()

println("Julia data science environment setup complete!")

# Test installation
using DataFrames, CSV, Plots, Statistics, MLJ

# Create a test plot
test_plot = plot(1:10, rand(10), title="Julia Setup Test")
savefig(test_plot, "julia_setup_test.png")
println("Test plot saved as julia_setup_test.png")
```

## Package Management

### Project Environment Setup

```julia
# Project.toml template
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
GLM = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
MLJ = "add582a8-e3ab-11e8-2d5e-e98b27df1bc7"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"

[compat]
julia = "1.8"

# Manifest.toml is auto-generated, don't edit manually
```

### Package Development Workflow

```julia
# scripts/package_development.jl

using Pkg, PkgTemplates

# Template for creating new packages
template = Template(
    user="your-username",
    authors=["Your Name <your.email@example.com>"],
    julia=v"1.8",
    plugins=[
        License(name="MIT"),
        Git(manifest=true, ssh=true),
        GitHubActions(x86=true),
        Codecov(),
        Documenter{GitHubActions}(),
        Develop(),
    ],
)

# Create a new package
# template("MyDataSciencePackage")

function create_data_science_package(name::String)
    """Create a new data science package with standard structure"""
    
    # Create package
    template(name)
    
    # Navigate to package directory
    cd(name)
    
    # Add common dependencies
    Pkg.activate(".")
    Pkg.add([
        "DataFrames",
        "Statistics",
        "Test",
        "Documenter",
        "BenchmarkTools"
    ])
    
    # Create additional directories
    mkpath("data")
    mkpath("scripts")
    mkpath("notebooks")
    
    # Create example data analysis script
    open("scripts/example_analysis.jl", "w") do f
        write(f, """
        # Example data analysis script
        using $(name)
        using DataFrames, Statistics, Plots
        
        # Load and analyze data
        function run_example_analysis()
            # Create sample data
            df = DataFrame(
                x = randn(1000),
                y = randn(1000),
                group = rand(["A", "B", "C"], 1000)
            )
            
            # Basic statistics
            println("Dataset summary:")
            println(describe(df))
            
            # Group analysis
            grouped_stats = combine(groupby(df, :group), 
                :x => mean => :mean_x,
                :y => mean => :mean_y,
                nrow => :count
            )
            println("\\nGrouped statistics:")
            println(grouped_stats)
            
            # Visualization
            p = scatter(df.x, df.y, group=df.group, 
                       title="Example Analysis",
                       xlabel="X values", ylabel="Y values")
            savefig(p, "example_plot.png")
            
            return df, grouped_stats
        end
        
        if abspath(PROGRAM_FILE) == @__FILE__
            run_example_analysis()
        end
        """)
    end
    
    println("Created data science package: $name")
    println("Next steps:")
    println("1. cd $name")
    println("2. julia --project=.")
    println("3. Run: include(\"scripts/example_analysis.jl\")")
end

# Usage: create_data_science_package("MyAnalysisPackage")
```

## Julia Fundamentals for Data Science

### Basic Data Types and Operations

```julia
# fundamentals/basic_operations.jl

# Type system examples
abstract type AbstractDataPoint end

struct DataPoint{T} <: AbstractDataPoint
    x::T
    y::T
    label::String
end

# Multiple dispatch example
process_data(point::DataPoint{<:Real}) = "Processing numerical data: $(point.x), $(point.y)"
process_data(point::DataPoint{<:AbstractString}) = "Processing string data: $(point.x), $(point.y)"

# Array operations (vectorized and broadcast)
function demonstrate_array_operations()
    # Create arrays
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]
    
    # Vectorized operations
    z1 = x .+ y          # Element-wise addition
    z2 = x .* y          # Element-wise multiplication
    z3 = sqrt.(x)        # Element-wise square root
    z4 = exp.(x)         # Element-wise exponential
    
    # Matrix operations
    A = [1 2; 3 4]
    B = [5 6; 7 8]
    
    C1 = A * B           # Matrix multiplication
    C2 = A .* B          # Element-wise multiplication
    C3 = A^2             # Matrix power
    C4 = A .^ 2          # Element-wise power
    
    return Dict(
        "vector_ops" => (z1, z2, z3, z4),
        "matrix_ops" => (C1, C2, C3, C4)
    )
end

# Function composition and piping
function data_pipeline_example()
    # Generate sample data
    data = randn(1000)
    
    # Traditional approach
    filtered_data = filter(x -> abs(x) < 2, data)
    squared_data = map(x -> x^2, filtered_data)
    result1 = sum(squared_data) / length(squared_data)
    
    # Functional approach with piping
    result2 = data |>
        x -> filter(y -> abs(y) < 2, x) |>
        x -> map(y -> y^2, x) |>
        x -> sum(x) / length(x)
    
    # Using function composition
    f = filter ∘ (x -> abs(x) < 2)
    g = map ∘ (x -> x^2)
    h = x -> sum(x) / length(x)
    
    result3 = (h ∘ g ∘ f)(data)
    
    return result1, result2, result3
end

# Metaprogramming for data science
macro time_and_memory(expr)
    quote
        local stats = @timed $(esc(expr))
        println("Time: $(stats.time) seconds")
        println("Memory: $(stats.bytes) bytes")
        println("GC time: $(stats.gctime) seconds")
        stats.value
    end
end

# Custom performance macro
macro benchmark_data_operation(expr)
    quote
        using BenchmarkTools
        println("Benchmarking: $($(string(expr)))")
        result = @benchmark $(esc(expr))
        println("Median time: $(median(result.times) / 1e6) ms")
        println("Memory usage: $(result.memory) bytes")
        result
    end
end

# Type-stable programming examples
function type_unstable_function(x)
    if x > 0
        return x
    else
        return "negative"
    end
end

function type_stable_function(x::T) where T <: Real
    if x > 0
        return x
    else
        return zero(T)  # Return same type
    end
end
```

## Data Manipulation

### Advanced DataFrames Operations

```julia
# data/dataframes_advanced.jl

using DataFrames, CSV, Statistics, Dates, Random

# Create comprehensive sample dataset
function create_sample_dataset(n_rows=10000)
    Random.seed!(42)
    
    # Generate realistic sales data
    dates = Date(2020,1,1) .+ Day.(0:n_rows-1)
    
    df = DataFrame(
        date = dates[1:n_rows],
        product_id = rand(1:100, n_rows),
        category = rand(["Electronics", "Clothing", "Books", "Home", "Sports"], n_rows),
        region = rand(["North", "South", "East", "West"], n_rows),
        sales_amount = round.(rand(10:1000, n_rows) .* (1 .+ 0.1 * randn(n_rows)), 2),
        quantity = rand(1:10, n_rows),
        customer_age = rand(18:80, n_rows),
        is_weekend = [dayofweek(d) in [6, 7] for d in dates[1:n_rows]],
        season = [month(d) in [12, 1, 2] ? "Winter" :
                  month(d) in [3, 4, 5] ? "Spring" :
                  month(d) in [6, 7, 8] ? "Summer" : "Fall" for d in dates[1:n_rows]]
    )
    
    return df
end

# Advanced aggregation and transformation
function advanced_aggregations(df::DataFrame)
    results = Dict()
    
    # 1. Multiple aggregation functions
    results["multi_agg"] = combine(
        groupby(df, [:category, :region]),
        :sales_amount => [mean, std, minimum, maximum] => AsTable,
        :quantity => sum => :total_quantity,
        nrow => :count
    )
    
    # 2. Custom aggregation functions
    function coefficient_of_variation(x)
        return std(x) / mean(x)
    end
    
    results["custom_agg"] = combine(
        groupby(df, :category),
        :sales_amount => coefficient_of_variation => :cv_sales,
        :sales_amount => (x -> quantile(x, [0.25, 0.5, 0.75])) => [:q25, :median, :q75]
    )
    
    # 3. Window functions
    df_sorted = sort(df, [:product_id, :date])
    transform!(
        groupby(df_sorted, :product_id),
        :sales_amount => (x -> cumsum(x)) => :cumulative_sales,
        :sales_amount => (x -> x .- lag(x, 1)) => :sales_diff,
        :sales_amount => (x -> rolling(mean, x, 7)) => :rolling_mean_7d
    )
    results["windowed"] = df_sorted
    
    # 4. Complex joins and merges
    product_info = DataFrame(
        product_id = 1:100,
        product_name = ["Product_$i" for i in 1:100],
        unit_cost = round.(rand(5:50, 100), 2),
        launch_date = Date(2019,1,1) .+ Day.(rand(0:365, 100))
    )
    
    results["joined"] = leftjoin(df, product_info, on=:product_id)
    
    # 5. Pivoting and reshaping
    pivot_data = combine(
        groupby(df, [:category, :region]),
        :sales_amount => sum => :total_sales
    )
    results["pivoted"] = unstack(pivot_data, :region, :category, :total_sales)
    
    return results
end

# Data cleaning and quality checks
function data_quality_assessment(df::DataFrame)
    quality_report = Dict()
    
    # Missing values analysis
    missing_stats = combine(df, names(df) .=> (x -> count(ismissing, x)) .=> (names(df) .* "_missing"))
    quality_report["missing_values"] = missing_stats
    
    # Duplicate detection
    quality_report["duplicates"] = nrow(df) - nrow(unique(df))
    
    # Outlier detection using IQR method
    function detect_outliers(x)
        q1, q3 = quantile(x, [0.25, 0.75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return sum((x .< lower_bound) .| (x .> upper_bound))
    end
    
    numeric_cols = names(df, [eltype(col) <: Number for col in eachcol(df)])
    outlier_stats = combine(df, numeric_cols .=> detect_outliers .=> (numeric_cols .* "_outliers"))
    quality_report["outliers"] = outlier_stats
    
    # Data type consistency
    quality_report["column_types"] = [name => eltype(col) for (name, col) in pairs(eachcol(df))]
    
    # Value distributions for categorical variables
    categorical_cols = names(df, [eltype(col) <: Union{String, CategoricalValue} for col in eachcol(df)])
    quality_report["categorical_distributions"] = Dict(
        col => combine(groupby(df, col), nrow => :count) 
        for col in categorical_cols
    )
    
    return quality_report
end

# Performance optimization techniques
function optimized_operations(df::DataFrame)
    # 1. Categorical variables for memory efficiency
    df_opt = copy(df)
    categorical_cols = [:category, :region, :season]
    for col in categorical_cols
        df_opt[!, col] = categorical(df_opt[!, col])
    end
    
    # 2. Vectorized operations instead of loops
    # Bad: using loops
    function calculate_profit_loop(df)
        profit = Float64[]
        for i in 1:nrow(df)
            margin = df.category[i] == "Electronics" ? 0.2 :
                    df.category[i] == "Clothing" ? 0.4 :
                    df.category[i] == "Books" ? 0.1 : 0.3
            push!(profit, df.sales_amount[i] * margin)
        end
        return profit
    end
    
    # Good: using vectorized operations
    function calculate_profit_vectorized(df)
        margin_map = Dict(
            "Electronics" => 0.2,
            "Clothing" => 0.4,
            "Books" => 0.1,
            "Home" => 0.3,
            "Sports" => 0.3
        )
        margins = [margin_map[cat] for cat in df.category]
        return df.sales_amount .* margins
    end
    
    # Benchmark the difference
    using BenchmarkTools
    time_loop = @benchmark calculate_profit_loop($df)
    time_vectorized = @benchmark calculate_profit_vectorized($df)
    
    return Dict(
        "optimized_df" => df_opt,
        "loop_time" => median(time_loop.times) / 1e6,
        "vectorized_time" => median(time_vectorized.times) / 1e6,
        "speedup" => median(time_loop.times) / median(time_vectorized.times)
    )
end

# Main execution function
function run_dataframes_analysis()
    println("Creating sample dataset...")
    df = create_sample_dataset(10000)
    
    println("Running advanced aggregations...")
    agg_results = advanced_aggregations(df)
    
    println("Performing data quality assessment...")
    quality_results = data_quality_assessment(df)
    
    println("Testing performance optimizations...")
    perf_results = optimized_operations(df)
    
    println("Analysis complete!")
    println("Speedup from vectorization: $(round(perf_results["speedup"], 2))x")
    
    return Dict(
        "data" => df,
        "aggregations" => agg_results,
        "quality" => quality_results,
        "performance" => perf_results
    )
end

# Execute if run directly
if abspath(PROGRAM_FILE) == @__FILE__
    results = run_dataframes_analysis()
end
```

## Statistical Analysis

### Comprehensive Statistical Toolkit

```julia
# statistics/statistical_analysis.jl

using Statistics, StatsBase, Distributions, HypothesisTests
using GLM, MultivariateStats, Bootstrap
using LinearAlgebra, Random

# Descriptive statistics framework
struct DescriptiveStats{T}
    data::Vector{T}
    n::Int
    mean::T
    median::T
    std::T
    var::T
    skewness::T
    kurtosis::T
    quartiles::Vector{T}
    iqr::T
    mad::T  # Median absolute deviation
    range::T
    cv::T   # Coefficient of variation
end

function DescriptiveStats(data::Vector{T}) where T <: Real
    clean_data = filter(!isnan, data)
    n = length(clean_data)
    
    if n == 0
        throw(ArgumentError("No valid data points"))
    end
    
    μ = mean(clean_data)
    σ = std(clean_data)
    med = median(clean_data)
    q = quantile(clean_data, [0.25, 0.5, 0.75])
    
    DescriptiveStats(
        clean_data,
        n,
        μ,
        med,
        σ,
        var(clean_data),
        skewness(clean_data),
        kurtosis(clean_data),
        q,
        q[3] - q[1],  # IQR
        mad(clean_data, normalize=true),
        maximum(clean_data) - minimum(clean_data),
        σ / μ  # CV
    )
end

function Base.show(io::IO, stats::DescriptiveStats)
    println(io, "Descriptive Statistics:")
    println(io, "  Count: $(stats.n)")
    println(io, "  Mean: $(round(stats.mean, digits=4))")
    println(io, "  Median: $(round(stats.median, digits=4))")
    println(io, "  Std Dev: $(round(stats.std, digits=4))")
    println(io, "  Variance: $(round(stats.var, digits=4))")
    println(io, "  Skewness: $(round(stats.skewness, digits=4))")
    println(io, "  Kurtosis: $(round(stats.kurtosis, digits=4))")
    println(io, "  Q1, Q2, Q3: $(round.(stats.quartiles, digits=4))")
    println(io, "  IQR: $(round(stats.iqr, digits=4))")
    println(io, "  MAD: $(round(stats.mad, digits=4))")
    println(io, "  Range: $(round(stats.range, digits=4))")
    println(io, "  CV: $(round(stats.cv, digits=4))")
end

# Hypothesis testing framework
function comprehensive_hypothesis_tests(group1::Vector, group2::Vector)
    results = Dict{String, Any}()
    
    # Normality tests
    results["shapiro_wilk_g1"] = ShapiroWilkTest(group1)
    results["shapiro_wilk_g2"] = ShapiroWilkTest(group2)
    
    # Variance equality test
    results["variance_test"] = VarianceFTest(group1, group2)
    
    # Two-sample tests
    if pvalue(results["variance_test"]) > 0.05
        # Equal variances - use standard t-test
        results["t_test"] = EqualVarianceTTest(group1, group2)
    else
        # Unequal variances - use Welch's t-test
        results["welch_test"] = UnequalVarianceTTest(group1, group2)
    end
    
    # Non-parametric alternatives
    results["mann_whitney"] = MannWhitneyUTest(group1, group2)
    results["kolmogorov_smirnov"] = ApproximateTwoSampleKSTest(group1, group2)
    
    # Effect size calculations
    pooled_std = sqrt(((length(group1) - 1) * var(group1) + 
                      (length(group2) - 1) * var(group2)) / 
                     (length(group1) + length(group2) - 2))
    cohens_d = (mean(group1) - mean(group2)) / pooled_std
    results["cohens_d"] = cohens_d
    
    # Interpret effect size
    effect_interpretation = if abs(cohens_d) < 0.2
        "negligible"
    elseif abs(cohens_d) < 0.5
        "small"
    elseif abs(cohens_d) < 0.8
        "medium"
    else
        "large"
    end
    results["effect_size_interpretation"] = effect_interpretation
    
    return results
end

# ANOVA and post-hoc analysis
function anova_analysis(groups::Vector{Vector{T}}) where T <: Real
    # Prepare data for ANOVA
    all_data = vcat(groups...)
    group_labels = vcat([fill(i, length(group)) for (i, group) in enumerate(groups)]...)
    
    # One-way ANOVA
    anova_result = OneWayANOVATest(all_data, group_labels)
    
    results = Dict{String, Any}()
    results["anova"] = anova_result
    results["f_statistic"] = anova_result.F
    results["p_value"] = pvalue(anova_result)
    
    # Effect size (eta-squared)
    k = length(groups)  # number of groups
    n = length(all_data)  # total sample size
    ss_between = sum(length(group) * (mean(group) - mean(all_data))^2 for group in groups)
    ss_total = sum((x - mean(all_data))^2 for x in all_data)
    eta_squared = ss_between / ss_total
    results["eta_squared"] = eta_squared
    
    # Post-hoc tests (if ANOVA is significant)
    if pvalue(anova_result) < 0.05
        # Pairwise t-tests with Bonferroni correction
        n_comparisons = k * (k - 1) ÷ 2
        alpha_corrected = 0.05 / n_comparisons
        
        pairwise_results = []
        for i in 1:(k-1)
            for j in (i+1):k
                t_test = EqualVarianceTTest(groups[i], groups[j])
                p_val = pvalue(t_test)
                significant = p_val < alpha_corrected
                
                push!(pairwise_results, Dict(
                    "groups" => (i, j),
                    "t_statistic" => t_test.t,
                    "p_value" => p_val,
                    "significant_bonferroni" => significant
                ))
            end
        end
        results["pairwise_tests"] = pairwise_results
    end
    
    return results
end

# Regression analysis framework
function comprehensive_regression_analysis(X::Matrix, y::Vector; 
                                         feature_names::Vector{String}=String[])
    results = Dict{String, Any}()
    
    # Prepare data
    n, p = size(X)
    if isempty(feature_names)
        feature_names = ["X$i" for i in 1:p]
    end
    
    # Add intercept
    X_with_intercept = hcat(ones(n), X)
    feature_names_with_intercept = ["Intercept"; feature_names]
    
    # Fit linear regression
    model = lm(X_with_intercept, y)
    results["model"] = model
    
    # Coefficients and statistics
    β = coef(model)
    se_β = stderror(model)
    t_stats = β ./ se_β
    p_values = 2 .* (1 .- cdf.(TDist(n - p - 1), abs.(t_stats)))
    
    results["coefficients"] = DataFrame(
        feature = feature_names_with_intercept,
        coefficient = β,
        std_error = se_β,
        t_statistic = t_stats,
        p_value = p_values,
        significant = p_values .< 0.05
    )
    
    # Model fit statistics
    ŷ = predict(model)
    residuals = y - ŷ
    
    results["r_squared"] = r2(model)
    results["adj_r_squared"] = adjr2(model)
    results["rmse"] = sqrt(mean(residuals.^2))
    results["mae"] = mean(abs.(residuals))
    
    # F-test for overall model significance
    mse_model = sum((ŷ - mean(y)).^2) / p
    mse_residual = sum(residuals.^2) / (n - p - 1)
    f_stat = mse_model / mse_residual
    f_p_value = 1 - cdf(FDist(p, n - p - 1), f_stat)
    
    results["f_statistic"] = f_stat
    results["f_p_value"] = f_p_value
    
    # Residual analysis
    results["residuals"] = residuals
    results["fitted_values"] = ŷ
    results["standardized_residuals"] = residuals ./ std(residuals)
    
    # Diagnostic tests
    # Breusch-Pagan test for heteroscedasticity
    aux_model = lm(hcat(ones(n), X), residuals.^2)
    bp_stat = n * r2(aux_model)
    bp_p_value = 1 - cdf(Chisq(p), bp_stat)
    results["breusch_pagan_test"] = Dict("statistic" => bp_stat, "p_value" => bp_p_value)
    
    # Durbin-Watson test for autocorrelation
    dw_stat = sum(diff(residuals).^2) / sum(residuals.^2)
    results["durbin_watson"] = dw_stat
    
    return results
end

# Bootstrap confidence intervals
function bootstrap_confidence_intervals(data::Vector{T}, statistic::Function; 
                                      n_bootstrap::Int=10000, 
                                      confidence_level::Float64=0.95) where T <: Real
    
    bootstrap_stats = Float64[]
    n = length(data)
    
    for _ in 1:n_bootstrap
        bootstrap_sample = sample(data, n, replace=true)
        push!(bootstrap_stats, statistic(bootstrap_sample))
    end
    
    α = 1 - confidence_level
    lower_percentile = α / 2 * 100
    upper_percentile = (1 - α / 2) * 100
    
    ci_lower = quantile(bootstrap_stats, lower_percentile / 100)
    ci_upper = quantile(bootstrap_stats, upper_percentile / 100)
    
    return Dict(
        "statistic" => statistic(data),
        "bootstrap_estimates" => bootstrap_stats,
        "confidence_interval" => (ci_lower, ci_upper),
        "confidence_level" => confidence_level,
        "bias" => mean(bootstrap_stats) - statistic(data),
        "std_error" => std(bootstrap_stats)
    )
end

# Example usage and testing
function run_statistical_analysis_example()
    Random.seed!(42)
    
    # Generate example data
    n = 1000
    x1 = randn(n)
    x2 = randn(n) + 0.5 * x1
    x3 = randn(n)
    y = 2 + 1.5 * x1 + 0.8 * x2 + 0.3 * x3 + 0.5 * randn(n)
    
    X = hcat(x1, x2, x3)
    feature_names = ["X1", "X2", "X3"]
    
    # Descriptive statistics
    println("=== Descriptive Statistics ===")
    desc_stats = DescriptiveStats(y)
    println(desc_stats)
    
    # Hypothesis testing
    println("\n=== Hypothesis Testing ===")
    group1 = y[1:500]
    group2 = y[501:1000] .+ 0.5  # Add small difference
    hyp_results = comprehensive_hypothesis_tests(group1, group2)
    println("T-test p-value: $(round(pvalue(hyp_results["t_test"]), digits=6))")
    println("Cohen's d: $(round(hyp_results["cohens_d"], digits=4)) ($(hyp_results["effect_size_interpretation"]))")
    
    # ANOVA
    println("\n=== ANOVA Analysis ===")
    groups = [y[1:300], y[301:600], y[601:900], y[901:1000]]
    anova_results = anova_analysis(groups)
    println("F-statistic: $(round(anova_results["f_statistic"], digits=4))")
    println("p-value: $(round(anova_results["p_value"], digits=6))")
    println("η² (eta-squared): $(round(anova_results["eta_squared"], digits=4))")
    
    # Regression analysis
    println("\n=== Regression Analysis ===")
    reg_results = comprehensive_regression_analysis(X, y, feature_names=feature_names)
    println("R²: $(round(reg_results["r_squared"], digits=4))")
    println("Adjusted R²: $(round(reg_results["adj_r_squared"], digits=4))")
    println("RMSE: $(round(reg_results["rmse"], digits=4))")
    println("\nCoefficients:")
    println(reg_results["coefficients"])
    
    # Bootstrap confidence intervals
    println("\n=== Bootstrap Analysis ===")
    bootstrap_mean = bootstrap_confidence_intervals(y, mean, n_bootstrap=5000)
    ci = bootstrap_mean["confidence_interval"]
    println("Mean: $(round(bootstrap_mean["statistic"], digits=4))")
    println("95% CI: [$(round(ci[1], digits=4)), $(round(ci[2], digits=4))]")
    println("Bootstrap SE: $(round(bootstrap_mean["std_error"], digits=4))")
    
    return Dict(
        "descriptive" => desc_stats,
        "hypothesis_tests" => hyp_results,
        "anova" => anova_results,
        "regression" => reg_results,
        "bootstrap" => bootstrap_mean
    )
end

# Execute if run directly
if abspath(PROGRAM_FILE) == @__FILE__
    results = run_statistical_analysis_example()
end
```

This Julia guide provides comprehensive coverage of Julia for data science, including advanced DataFrames operations, statistical analysis, and performance optimization techniques. Julia's strong type system and multiple dispatch make it excellent for high-performance data science work. 
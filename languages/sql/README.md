# SQL for Data Science & Analytics

## Overview

This directory contains comprehensive SQL resources for data science, analytics, and database management. It covers everything from basic queries to advanced analytical functions, query optimization, and best practices for data analysis workflows.

## Table of Contents

- [SQL Fundamentals](#sql-fundamentals)
- [Data Analysis Queries](#data-analysis-queries)
- [Window Functions](#window-functions)
- [Advanced Analytics](#advanced-analytics)
- [Query Optimization](#query-optimization)
- [ETL Patterns](#etl-patterns)
- [Database-Specific Features](#database-specific-features)
- [Best Practices](#best-practices)
- [Performance Tuning](#performance-tuning)

## SQL Fundamentals

### Basic Data Exploration

```sql
-- Data profiling and exploration
SELECT 
    COUNT(*) as total_rows,
    COUNT(DISTINCT customer_id) as unique_customers,
    MIN(order_date) as earliest_order,
    MAX(order_date) as latest_order,
    AVG(order_amount) as avg_order_value,
    MEDIAN(order_amount) as median_order_value,
    STDDEV(order_amount) as order_amount_stddev
FROM orders;

-- Check for missing values
SELECT 
    COUNT(*) as total_rows,
    COUNT(customer_id) as non_null_customers,
    COUNT(*) - COUNT(customer_id) as null_customers,
    ROUND(100.0 * (COUNT(*) - COUNT(customer_id)) / COUNT(*), 2) as null_percentage
FROM orders;

-- Data distribution analysis
SELECT 
    order_amount,
    COUNT(*) as frequency,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 2) as percentage
FROM orders
GROUP BY order_amount
ORDER BY order_amount;
```

### Data Quality Checks

```sql
-- Duplicate detection
SELECT 
    customer_id, 
    order_date, 
    order_amount,
    COUNT(*) as duplicate_count
FROM orders
GROUP BY customer_id, order_date, order_amount
HAVING COUNT(*) > 1;

-- Outlier detection using IQR method
WITH quartiles AS (
    SELECT 
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY order_amount) as q1,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY order_amount) as q3
    FROM orders
),
outlier_bounds AS (
    SELECT 
        q1,
        q3,
        q1 - 1.5 * (q3 - q1) as lower_bound,
        q3 + 1.5 * (q3 - q1) as upper_bound
    FROM quartiles
)
SELECT 
    o.*,
    CASE 
        WHEN o.order_amount < ob.lower_bound OR o.order_amount > ob.upper_bound 
        THEN 'Outlier' 
        ELSE 'Normal' 
    END as outlier_status
FROM orders o
CROSS JOIN outlier_bounds ob
WHERE o.order_amount < ob.lower_bound OR o.order_amount > ob.upper_bound;

-- Data consistency checks
SELECT 
    'Negative amounts' as check_type,
    COUNT(*) as issue_count
FROM orders 
WHERE order_amount < 0

UNION ALL

SELECT 
    'Future dates' as check_type,
    COUNT(*) as issue_count
FROM orders 
WHERE order_date > CURRENT_DATE

UNION ALL

SELECT 
    'Invalid email formats' as check_type,
    COUNT(*) as issue_count
FROM customers 
WHERE email NOT LIKE '%@%.%';
```

## Data Analysis Queries

### Customer Analytics

```sql
-- Customer lifetime value analysis
WITH customer_metrics AS (
    SELECT 
        c.customer_id,
        c.registration_date,
        COUNT(o.order_id) as total_orders,
        SUM(o.order_amount) as total_spent,
        AVG(o.order_amount) as avg_order_value,
        MIN(o.order_date) as first_order_date,
        MAX(o.order_date) as last_order_date,
        MAX(o.order_date) - MIN(o.order_date) as customer_lifespan_days
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id
    GROUP BY c.customer_id, c.registration_date
)
SELECT 
    customer_id,
    total_orders,
    total_spent,
    avg_order_value,
    customer_lifespan_days,
    CASE 
        WHEN customer_lifespan_days > 0 
        THEN total_spent / (customer_lifespan_days::float / 365)
        ELSE total_spent 
    END as annual_value,
    CASE 
        WHEN total_spent >= 1000 THEN 'High Value'
        WHEN total_spent >= 500 THEN 'Medium Value'
        ELSE 'Low Value'
    END as customer_segment
FROM customer_metrics
ORDER BY total_spent DESC;

-- Customer cohort analysis
WITH cohort_data AS (
    SELECT 
        customer_id,
        DATE_TRUNC('month', MIN(order_date)) as cohort_month,
        DATE_TRUNC('month', order_date) as order_month
    FROM orders
    GROUP BY customer_id, DATE_TRUNC('month', order_date)
),
cohort_table AS (
    SELECT 
        cohort_month,
        order_month,
        COUNT(DISTINCT customer_id) as customers,
        EXTRACT(EPOCH FROM (order_month - cohort_month)) / (30 * 24 * 60 * 60) as period_number
    FROM cohort_data
    GROUP BY cohort_month, order_month
),
cohort_sizes AS (
    SELECT 
        cohort_month,
        COUNT(DISTINCT customer_id) as cohort_size
    FROM cohort_data
    WHERE cohort_month = order_month
    GROUP BY cohort_month
)
SELECT 
    ct.cohort_month,
    cs.cohort_size,
    ct.period_number,
    ct.customers,
    ROUND(100.0 * ct.customers / cs.cohort_size, 2) as retention_rate
FROM cohort_table ct
JOIN cohort_sizes cs ON ct.cohort_month = cs.cohort_month
ORDER BY ct.cohort_month, ct.period_number;
```

### Sales Analytics

```sql
-- Revenue analysis with growth rates
WITH monthly_revenue AS (
    SELECT 
        DATE_TRUNC('month', order_date) as month,
        SUM(order_amount) as revenue,
        COUNT(DISTINCT order_id) as order_count,
        COUNT(DISTINCT customer_id) as unique_customers,
        AVG(order_amount) as avg_order_value
    FROM orders
    GROUP BY DATE_TRUNC('month', order_date)
),
revenue_with_growth AS (
    SELECT 
        month,
        revenue,
        order_count,
        unique_customers,
        avg_order_value,
        LAG(revenue) OVER (ORDER BY month) as prev_month_revenue,
        LAG(order_count) OVER (ORDER BY month) as prev_month_orders
    FROM monthly_revenue
)
SELECT 
    month,
    revenue,
    order_count,
    unique_customers,
    avg_order_value,
    ROUND(
        100.0 * (revenue - prev_month_revenue) / prev_month_revenue, 2
    ) as revenue_growth_pct,
    ROUND(
        100.0 * (order_count - prev_month_orders) / prev_month_orders, 2
    ) as order_growth_pct
FROM revenue_with_growth
ORDER BY month;

-- Product performance analysis
SELECT 
    p.product_name,
    p.category,
    COUNT(oi.order_item_id) as times_ordered,
    SUM(oi.quantity) as total_quantity_sold,
    SUM(oi.quantity * oi.unit_price) as total_revenue,
    AVG(oi.unit_price) as avg_price,
    COUNT(DISTINCT oi.order_id) as unique_orders,
    COUNT(DISTINCT o.customer_id) as unique_customers,
    ROUND(
        SUM(oi.quantity * oi.unit_price) / SUM(SUM(oi.quantity * oi.unit_price)) OVER(), 4
    ) as revenue_share
FROM products p
JOIN order_items oi ON p.product_id = oi.product_id
JOIN orders o ON oi.order_id = o.order_id
GROUP BY p.product_id, p.product_name, p.category
ORDER BY total_revenue DESC;
```

### Time Series Analysis

```sql
-- Daily sales with moving averages
SELECT 
    order_date,
    COUNT(*) as daily_orders,
    SUM(order_amount) as daily_revenue,
    AVG(SUM(order_amount)) OVER (
        ORDER BY order_date 
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) as revenue_7day_ma,
    AVG(SUM(order_amount)) OVER (
        ORDER BY order_date 
        ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
    ) as revenue_30day_ma,
    -- Year-over-year comparison
    LAG(SUM(order_amount), 365) OVER (ORDER BY order_date) as revenue_last_year,
    ROUND(
        100.0 * (SUM(order_amount) - LAG(SUM(order_amount), 365) OVER (ORDER BY order_date)) 
        / LAG(SUM(order_amount), 365) OVER (ORDER BY order_date), 2
    ) as yoy_growth_pct
FROM orders
GROUP BY order_date
ORDER BY order_date;

-- Seasonality analysis
SELECT 
    EXTRACT(MONTH FROM order_date) as month,
    EXTRACT(QUARTER FROM order_date) as quarter,
    EXTRACT(DOW FROM order_date) as day_of_week,
    COUNT(*) as order_count,
    SUM(order_amount) as total_revenue,
    AVG(order_amount) as avg_order_value
FROM orders
GROUP BY 
    EXTRACT(MONTH FROM order_date),
    EXTRACT(QUARTER FROM order_date),
    EXTRACT(DOW FROM order_date)
ORDER BY month, day_of_week;
```

## Window Functions

### Ranking and Percentiles

```sql
-- Customer ranking by revenue
SELECT 
    customer_id,
    total_spent,
    ROW_NUMBER() OVER (ORDER BY total_spent DESC) as revenue_rank,
    RANK() OVER (ORDER BY total_spent DESC) as revenue_rank_with_ties,
    DENSE_RANK() OVER (ORDER BY total_spent DESC) as revenue_dense_rank,
    NTILE(10) OVER (ORDER BY total_spent DESC) as decile,
    PERCENT_RANK() OVER (ORDER BY total_spent) as percentile_rank,
    CUME_DIST() OVER (ORDER BY total_spent) as cumulative_distribution
FROM (
    SELECT 
        customer_id,
        SUM(order_amount) as total_spent
    FROM orders
    GROUP BY customer_id
) customer_totals
ORDER BY total_spent DESC;

-- Top N analysis per category
WITH product_sales AS (
    SELECT 
        p.category,
        p.product_name,
        SUM(oi.quantity * oi.unit_price) as total_revenue,
        ROW_NUMBER() OVER (PARTITION BY p.category ORDER BY SUM(oi.quantity * oi.unit_price) DESC) as category_rank
    FROM products p
    JOIN order_items oi ON p.product_id = oi.product_id
    GROUP BY p.category, p.product_name
)
SELECT 
    category,
    product_name,
    total_revenue,
    category_rank
FROM product_sales
WHERE category_rank <= 3
ORDER BY category, category_rank;
```

### Running Totals and Cumulative Analysis

```sql
-- Running totals and cumulative analysis
SELECT 
    order_date,
    daily_revenue,
    SUM(daily_revenue) OVER (ORDER BY order_date) as cumulative_revenue,
    AVG(daily_revenue) OVER (ORDER BY order_date) as cumulative_avg_revenue,
    daily_revenue / SUM(daily_revenue) OVER () as daily_revenue_share,
    SUM(daily_revenue) OVER (ORDER BY order_date) / SUM(daily_revenue) OVER () as cumulative_revenue_share
FROM (
    SELECT 
        order_date,
        SUM(order_amount) as daily_revenue
    FROM orders
    GROUP BY order_date
) daily_sales
ORDER BY order_date;

-- Lead and lag analysis for trend detection
WITH daily_metrics AS (
    SELECT 
        order_date,
        COUNT(*) as order_count,
        SUM(order_amount) as revenue
    FROM orders
    GROUP BY order_date
)
SELECT 
    order_date,
    order_count,
    revenue,
    LAG(revenue, 1) OVER (ORDER BY order_date) as prev_day_revenue,
    LEAD(revenue, 1) OVER (ORDER BY order_date) as next_day_revenue,
    revenue - LAG(revenue, 1) OVER (ORDER BY order_date) as day_over_day_change,
    ROUND(
        100.0 * (revenue - LAG(revenue, 1) OVER (ORDER BY order_date)) 
        / LAG(revenue, 1) OVER (ORDER BY order_date), 2
    ) as day_over_day_pct_change
FROM daily_metrics
ORDER BY order_date;
```

## Advanced Analytics

### Statistical Functions

```sql
-- Descriptive statistics
WITH order_stats AS (
    SELECT 
        customer_id,
        COUNT(*) as order_count,
        SUM(order_amount) as total_spent,
        AVG(order_amount) as avg_order_value,
        STDDEV(order_amount) as order_value_stddev,
        MIN(order_amount) as min_order_value,
        MAX(order_amount) as max_order_value,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY order_amount) as q1_order_value,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY order_amount) as median_order_value,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY order_amount) as q3_order_value
    FROM orders
    GROUP BY customer_id
)
SELECT 
    COUNT(*) as total_customers,
    AVG(order_count) as avg_orders_per_customer,
    STDDEV(order_count) as stddev_orders_per_customer,
    AVG(total_spent) as avg_total_spent,
    STDDEV(total_spent) as stddev_total_spent,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY total_spent) as median_total_spent,
    -- Coefficient of variation
    STDDEV(total_spent) / AVG(total_spent) as cv_total_spent
FROM order_stats;

-- Correlation analysis (simplified Pearson correlation)
WITH customer_metrics AS (
    SELECT 
        customer_id,
        COUNT(*) as order_frequency,
        AVG(order_amount) as avg_order_value,
        SUM(order_amount) as total_spent
    FROM orders
    GROUP BY customer_id
),
correlation_prep AS (
    SELECT 
        order_frequency,
        avg_order_value,
        order_frequency - AVG(order_frequency) OVER () as freq_deviation,
        avg_order_value - AVG(avg_order_value) OVER () as value_deviation
    FROM customer_metrics
)
SELECT 
    COUNT(*) as n,
    AVG(freq_deviation * value_deviation) as covariance,
    SQRT(AVG(freq_deviation * freq_deviation)) as freq_stddev,
    SQRT(AVG(value_deviation * value_deviation)) as value_stddev,
    AVG(freq_deviation * value_deviation) / 
    (SQRT(AVG(freq_deviation * freq_deviation)) * SQRT(AVG(value_deviation * value_deviation))) as correlation
FROM correlation_prep;
```

### Advanced Aggregations

```sql
-- ROLLUP and CUBE for multidimensional analysis
SELECT 
    COALESCE(CAST(EXTRACT(YEAR FROM order_date) AS VARCHAR), 'Total') as year,
    COALESCE(CAST(EXTRACT(MONTH FROM order_date) AS VARCHAR), 'Total') as month,
    COALESCE(p.category, 'Total') as category,
    COUNT(*) as order_count,
    SUM(oi.quantity * oi.unit_price) as total_revenue
FROM orders o
JOIN order_items oi ON o.order_id = oi.order_id
JOIN products p ON oi.product_id = p.product_id
GROUP BY ROLLUP(
    EXTRACT(YEAR FROM order_date),
    EXTRACT(MONTH FROM order_date),
    p.category
)
ORDER BY year, month, category;

-- GROUPING SETS for custom aggregation levels
SELECT 
    CASE 
        WHEN GROUPING(p.category) = 0 THEN p.category 
        ELSE 'All Categories' 
    END as category,
    CASE 
        WHEN GROUPING(EXTRACT(QUARTER FROM o.order_date)) = 0 
        THEN CAST(EXTRACT(QUARTER FROM o.order_date) AS VARCHAR)
        ELSE 'All Quarters' 
    END as quarter,
    COUNT(*) as order_count,
    SUM(oi.quantity * oi.unit_price) as revenue
FROM orders o
JOIN order_items oi ON o.order_id = oi.order_id
JOIN products p ON oi.product_id = p.product_id
GROUP BY GROUPING SETS (
    (p.category),
    (EXTRACT(QUARTER FROM o.order_date)),
    (p.category, EXTRACT(QUARTER FROM o.order_date)),
    ()
)
ORDER BY category, quarter;
```

### Recursive CTEs and Hierarchical Data

```sql
-- Recursive CTE for organizational hierarchy
WITH RECURSIVE employee_hierarchy AS (
    -- Base case: top-level managers
    SELECT 
        employee_id,
        name,
        manager_id,
        title,
        salary,
        1 as level,
        CAST(name AS VARCHAR(1000)) as hierarchy_path
    FROM employees
    WHERE manager_id IS NULL
    
    UNION ALL
    
    -- Recursive case: employees with managers
    SELECT 
        e.employee_id,
        e.name,
        e.manager_id,
        e.title,
        e.salary,
        eh.level + 1,
        CAST(eh.hierarchy_path || ' -> ' || e.name AS VARCHAR(1000))
    FROM employees e
    JOIN employee_hierarchy eh ON e.manager_id = eh.employee_id
)
SELECT 
    employee_id,
    name,
    title,
    salary,
    level,
    hierarchy_path,
    COUNT(*) OVER (PARTITION BY level) as employees_at_level
FROM employee_hierarchy
ORDER BY level, name;

-- Generate date series for time-based analysis
WITH RECURSIVE date_series AS (
    SELECT DATE '2023-01-01' as date_value
    UNION ALL
    SELECT date_value + INTERVAL '1 day'
    FROM date_series
    WHERE date_value < DATE '2023-12-31'
)
SELECT 
    ds.date_value,
    COALESCE(daily_orders.order_count, 0) as order_count,
    COALESCE(daily_orders.revenue, 0) as revenue
FROM date_series ds
LEFT JOIN (
    SELECT 
        order_date,
        COUNT(*) as order_count,
        SUM(order_amount) as revenue
    FROM orders
    WHERE order_date BETWEEN '2023-01-01' AND '2023-12-31'
    GROUP BY order_date
) daily_orders ON ds.date_value = daily_orders.order_date
ORDER BY ds.date_value;
```

## Query Optimization

### Index Strategy

```sql
-- Create indexes for common query patterns
CREATE INDEX idx_orders_customer_date ON orders(customer_id, order_date);
CREATE INDEX idx_orders_date_amount ON orders(order_date, order_amount);
CREATE INDEX idx_order_items_product ON order_items(product_id);
CREATE INDEX idx_customers_email ON customers(email);

-- Partial indexes for filtered queries
CREATE INDEX idx_high_value_orders ON orders(order_date) 
WHERE order_amount > 1000;

-- Composite indexes for multi-column queries
CREATE INDEX idx_orders_composite ON orders(customer_id, order_date, order_amount);

-- Function-based indexes
CREATE INDEX idx_orders_month ON orders(DATE_TRUNC('month', order_date));
```

### Query Rewriting for Performance

```sql
-- Original slow query
SELECT *
FROM orders o
WHERE EXISTS (
    SELECT 1 
    FROM order_items oi 
    WHERE oi.order_id = o.order_id 
    AND oi.product_id IN (SELECT product_id FROM products WHERE category = 'Electronics')
);

-- Optimized version using JOIN
SELECT DISTINCT o.*
FROM orders o
JOIN order_items oi ON o.order_id = oi.order_id
JOIN products p ON oi.product_id = p.product_id
WHERE p.category = 'Electronics';

-- Use LIMIT for large result sets
SELECT 
    customer_id,
    order_date,
    order_amount
FROM orders
ORDER BY order_date DESC
LIMIT 1000;

-- Avoid SELECT * in production queries
SELECT 
    customer_id,
    order_date,
    order_amount,
    status
FROM orders
WHERE order_date >= CURRENT_DATE - INTERVAL '30 days';
```

### Execution Plan Analysis

```sql
-- Analyze query execution plans
EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)
SELECT 
    c.customer_id,
    c.name,
    COUNT(o.order_id) as order_count,
    SUM(o.order_amount) as total_spent
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
WHERE c.registration_date >= '2023-01-01'
GROUP BY c.customer_id, c.name
HAVING COUNT(o.order_id) > 5
ORDER BY total_spent DESC;

-- Query performance monitoring
SELECT 
    query,
    calls,
    total_time,
    mean_time,
    rows,
    100.0 * shared_blks_hit / nullif(shared_blks_hit + shared_blks_read, 0) AS hit_percent
FROM pg_stat_statements
ORDER BY total_time DESC
LIMIT 10;
```

## ETL Patterns

### Data Transformation

```sql
-- Data cleaning and standardization
WITH cleaned_customers AS (
    SELECT 
        customer_id,
        TRIM(UPPER(name)) as name,
        LOWER(TRIM(email)) as email,
        REGEXP_REPLACE(phone, '[^0-9]', '', 'g') as phone,
        CASE 
            WHEN age < 0 OR age > 120 THEN NULL
            ELSE age 
        END as age,
        COALESCE(city, 'Unknown') as city,
        registration_date
    FROM raw_customers
    WHERE email LIKE '%@%.%'
)
SELECT * FROM cleaned_customers;

-- Handling duplicates and data quality issues
WITH deduplicated_orders AS (
    SELECT DISTINCT ON (customer_id, order_date, order_amount)
        order_id,
        customer_id,
        order_date,
        order_amount,
        status
    FROM raw_orders
    WHERE order_amount > 0
    AND order_date <= CURRENT_DATE
    ORDER BY customer_id, order_date, order_amount, created_at DESC
)
SELECT * FROM deduplicated_orders;
```

### Incremental Loading

```sql
-- Incremental data loading pattern
WITH latest_orders AS (
    SELECT *
    FROM raw_orders
    WHERE created_at > (
        SELECT COALESCE(MAX(last_updated), '1900-01-01'::timestamp)
        FROM etl_log
        WHERE table_name = 'orders'
    )
),
processed_orders AS (
    SELECT 
        order_id,
        customer_id,
        order_date,
        order_amount,
        status,
        CURRENT_TIMESTAMP as processed_at
    FROM latest_orders
    WHERE order_amount > 0
)
INSERT INTO orders_fact
SELECT * FROM processed_orders;

-- Update ETL log
INSERT INTO etl_log (table_name, last_updated, records_processed)
SELECT 
    'orders',
    CURRENT_TIMESTAMP,
    COUNT(*)
FROM processed_orders;
```

### Change Data Capture (CDC)

```sql
-- Track changes using temporal tables
CREATE TABLE orders_history (
    order_id INT,
    customer_id INT,
    order_amount DECIMAL(10,2),
    status VARCHAR(50),
    valid_from TIMESTAMP,
    valid_to TIMESTAMP,
    operation_type VARCHAR(10)
);

-- Trigger for change tracking
CREATE OR REPLACE FUNCTION track_order_changes()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        INSERT INTO orders_history 
        VALUES (NEW.order_id, NEW.customer_id, NEW.order_amount, NEW.status, 
                CURRENT_TIMESTAMP, NULL, 'INSERT');
        RETURN NEW;
    ELSIF TG_OP = 'UPDATE' THEN
        UPDATE orders_history 
        SET valid_to = CURRENT_TIMESTAMP 
        WHERE order_id = OLD.order_id AND valid_to IS NULL;
        
        INSERT INTO orders_history 
        VALUES (NEW.order_id, NEW.customer_id, NEW.order_amount, NEW.status, 
                CURRENT_TIMESTAMP, NULL, 'UPDATE');
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        UPDATE orders_history 
        SET valid_to = CURRENT_TIMESTAMP 
        WHERE order_id = OLD.order_id AND valid_to IS NULL;
        
        INSERT INTO orders_history 
        VALUES (OLD.order_id, OLD.customer_id, OLD.order_amount, OLD.status, 
                CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 'DELETE');
        RETURN OLD;
    END IF;
END;
$$ LANGUAGE plpgsql;
```

## Database-Specific Features

### PostgreSQL Advanced Features

```sql
-- JSON operations
SELECT 
    customer_id,
    preferences->>'newsletter' as newsletter_pref,
    preferences->'categories' as preferred_categories,
    jsonb_array_length(preferences->'categories') as category_count
FROM customers
WHERE preferences IS NOT NULL;

-- Array operations
SELECT 
    product_id,
    product_name,
    tags,
    CARDINALITY(tags) as tag_count,
    'electronics' = ANY(tags) as is_electronics
FROM products
WHERE tags IS NOT NULL;

-- Full-text search
SELECT 
    product_id,
    product_name,
    description,
    ts_rank(to_tsvector('english', description), plainto_tsquery('wireless bluetooth')) as rank
FROM products
WHERE to_tsvector('english', description) @@ plainto_tsquery('wireless bluetooth')
ORDER BY rank DESC;

-- Window frame functions
SELECT 
    order_date,
    order_amount,
    -- Median over sliding window
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY order_amount) OVER (
        ORDER BY order_date 
        ROWS BETWEEN 10 PRECEDING AND CURRENT ROW
    ) as median_amount_11day_window
FROM orders;
```

### BigQuery Specific Features

```sql
-- APPROX functions for large datasets
SELECT 
    customer_segment,
    APPROX_COUNT_DISTINCT(customer_id) as approx_unique_customers,
    APPROX_QUANTILES(order_amount, 100) as amount_percentiles,
    APPROX_TOP_COUNT(product_category, 10) as top_categories
FROM orders_fact
GROUP BY customer_segment;

-- ARRAY_AGG and STRUCT for complex aggregations
SELECT 
    customer_id,
    ARRAY_AGG(
        STRUCT(
            order_date,
            order_amount,
            products
        ) 
        ORDER BY order_date DESC 
        LIMIT 5
    ) as recent_orders
FROM (
    SELECT 
        o.customer_id,
        o.order_date,
        o.order_amount,
        ARRAY_AGG(p.product_name) as products
    FROM orders o
    JOIN order_items oi ON o.order_id = oi.order_id
    JOIN products p ON oi.product_id = p.product_id
    GROUP BY o.customer_id, o.order_date, o.order_amount
) customer_orders
GROUP BY customer_id;

-- Machine Learning integration
CREATE OR REPLACE MODEL `project.dataset.customer_ltv_model`
OPTIONS(
    model_type='linear_reg',
    input_label_cols=['lifetime_value']
) AS
SELECT 
    age,
    registration_months,
    avg_order_value,
    order_frequency,
    preferred_category,
    lifetime_value
FROM customer_features
WHERE customer_id IS NOT NULL;
```

### Snowflake Specific Features

```sql
-- Time travel queries
SELECT *
FROM orders
AT(TIMESTAMP => '2023-01-01 00:00:00'::timestamp);

-- Semi-structured data handling
SELECT 
    customer_id,
    profile:name::string as customer_name,
    profile:age::number as age,
    profile:preferences[0]::string as first_preference
FROM customer_profiles;

-- Clustering and optimization
ALTER TABLE large_orders_table 
CLUSTER BY (order_date, customer_id);

-- Zero-copy cloning
CREATE TABLE orders_staging 
CLONE orders_production;
```

## Best Practices

### Code Organization

```sql
-- Use CTEs for complex queries
WITH monthly_metrics AS (
    SELECT 
        DATE_TRUNC('month', order_date) as month,
        COUNT(*) as order_count,
        SUM(order_amount) as revenue
    FROM orders
    GROUP BY DATE_TRUNC('month', order_date)
),
growth_metrics AS (
    SELECT 
        month,
        order_count,
        revenue,
        LAG(revenue) OVER (ORDER BY month) as prev_month_revenue
    FROM monthly_metrics
)
SELECT 
    month,
    order_count,
    revenue,
    ROUND(100.0 * (revenue - prev_month_revenue) / prev_month_revenue, 2) as growth_rate
FROM growth_metrics
WHERE prev_month_revenue IS NOT NULL
ORDER BY month;

-- Proper commenting and documentation
/*
Purpose: Calculate customer retention rates by cohort
Author: Data Team
Created: 2024-01-15
Modified: 2024-01-20
Dependencies: orders, customers tables
Notes: Considers first purchase month as cohort assignment
*/

-- Comments for complex logic
SELECT 
    customer_id,
    -- Calculate recency in days from last order
    CURRENT_DATE - MAX(order_date) as recency_days,
    -- Frequency as orders per month since first order
    COUNT(*) * 30.0 / GREATEST(1, MAX(order_date) - MIN(order_date)) as frequency_monthly,
    -- Monetary value as average order amount
    AVG(order_amount) as monetary_avg
FROM orders
GROUP BY customer_id;
```

### Data Governance

```sql
-- Data lineage documentation
CREATE VIEW customer_360 AS
SELECT 
    c.customer_id,
    c.name,
    c.email,
    c.registration_date,
    om.total_orders,
    om.total_spent,
    om.avg_order_value,
    om.last_order_date,
    CASE 
        WHEN om.last_order_date >= CURRENT_DATE - INTERVAL '30 days' THEN 'Active'
        WHEN om.last_order_date >= CURRENT_DATE - INTERVAL '90 days' THEN 'At Risk'
        ELSE 'Inactive'
    END as customer_status
FROM customers c
LEFT JOIN (
    SELECT 
        customer_id,
        COUNT(*) as total_orders,
        SUM(order_amount) as total_spent,
        AVG(order_amount) as avg_order_value,
        MAX(order_date) as last_order_date
    FROM orders
    GROUP BY customer_id
) om ON c.customer_id = om.customer_id;

-- Column-level documentation
COMMENT ON COLUMN customer_360.customer_status IS 
'Customer activity status: Active (ordered within 30 days), At Risk (30-90 days), Inactive (>90 days)';

-- Data quality constraints
ALTER TABLE orders 
ADD CONSTRAINT positive_amount CHECK (order_amount > 0);

ALTER TABLE customers 
ADD CONSTRAINT valid_email CHECK (email LIKE '%@%.%');
```

### Security Best Practices

```sql
-- Row-level security
CREATE POLICY customer_data_policy ON orders
FOR ALL TO application_role
USING (customer_id IN (
    SELECT customer_id 
    FROM customer_access 
    WHERE user_id = current_user_id()
));

-- Column masking for sensitive data
CREATE VIEW orders_masked AS
SELECT 
    order_id,
    customer_id,
    order_date,
    order_amount,
    -- Mask credit card numbers
    CASE 
        WHEN current_user_role() = 'analyst' 
        THEN 'XXXX-XXXX-XXXX-' || RIGHT(credit_card_number, 4)
        ELSE credit_card_number
    END as credit_card_number
FROM orders;

-- Audit logging
CREATE TABLE query_audit (
    user_name VARCHAR(100),
    query_text TEXT,
    execution_time TIMESTAMP,
    rows_affected INT
);
```

## Performance Tuning

### Query Optimization Checklist

1. **Use appropriate indexes**
2. **Limit result sets with WHERE clauses**
3. **Use LIMIT for large queries**
4. **Avoid SELECT \* in production**
5. **Use EXISTS instead of IN for subqueries**
6. **Consider partitioning for large tables**
7. **Use appropriate join types**
8. **Analyze query execution plans**

### Common Anti-patterns to Avoid

```sql
-- AVOID: Using functions in WHERE clauses
-- BAD
SELECT * FROM orders 
WHERE EXTRACT(YEAR FROM order_date) = 2023;

-- GOOD
SELECT * FROM orders 
WHERE order_date >= '2023-01-01' 
AND order_date < '2024-01-01';

-- AVOID: N+1 query problems
-- BAD: Multiple queries in application code
SELECT * FROM customers;
-- Then for each customer: SELECT * FROM orders WHERE customer_id = ?

-- GOOD: Single query with JOIN
SELECT 
    c.*,
    COUNT(o.order_id) as order_count
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id;

-- AVOID: Large UNION operations
-- Consider using UNION ALL when duplicates don't matter
SELECT customer_id FROM customers_2023
UNION ALL
SELECT customer_id FROM customers_2024;
```

---

*This comprehensive SQL guide covers essential techniques for data science and analytics. Practice these patterns and adapt them to your specific database platform and use cases.* 
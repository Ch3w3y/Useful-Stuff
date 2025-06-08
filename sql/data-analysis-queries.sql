-- Data Analysis SQL Templates
-- ============================
-- 
-- Comprehensive collection of SQL queries for data analysis including:
-- - Exploratory Data Analysis (EDA)
-- - Window functions and analytics
-- - Time series analysis
-- - Customer analytics
-- - Statistical functions
-- - Data quality assessment
-- - Advanced aggregations
--
-- Compatible with: PostgreSQL, SQL Server, MySQL, BigQuery
-- Author: Data Science Team
-- Date: 2024

-- =================================================
-- 1. EXPLORATORY DATA ANALYSIS (EDA)
-- =================================================

-- Basic dataset overview
SELECT 
    COUNT(*) as total_rows,
    COUNT(DISTINCT customer_id) as unique_customers,
    MIN(order_date) as earliest_date,
    MAX(order_date) as latest_date,
    AVG(order_amount) as avg_order_amount,
    STDDEV(order_amount) as stddev_order_amount
FROM orders;

-- Data profiling template
WITH data_profile AS (
    SELECT 
        column_name,
        data_type,
        COUNT(*) as total_count,
        COUNT(column_name) as non_null_count,
        COUNT(*) - COUNT(column_name) as null_count,
        ROUND(
            (COUNT(*) - COUNT(column_name)) * 100.0 / COUNT(*), 2
        ) as null_percentage,
        COUNT(DISTINCT column_name) as unique_values
    FROM (
        SELECT customer_id as column_name, 'integer' as data_type FROM orders
        UNION ALL
        SELECT CAST(order_amount AS VARCHAR) as column_name, 'numeric' as data_type FROM orders
        -- Add more columns as needed
    ) unpivoted
    GROUP BY column_name, data_type
)
SELECT * FROM data_profile
ORDER BY null_percentage DESC;

-- Distribution analysis
SELECT 
    CASE 
        WHEN order_amount < 50 THEN '0-50'
        WHEN order_amount < 100 THEN '50-100'
        WHEN order_amount < 200 THEN '100-200'
        WHEN order_amount < 500 THEN '200-500'
        ELSE '500+' 
    END as amount_bucket,
    COUNT(*) as frequency,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage
FROM orders
GROUP BY 
    CASE 
        WHEN order_amount < 50 THEN '0-50'
        WHEN order_amount < 100 THEN '50-100'
        WHEN order_amount < 200 THEN '100-200'
        WHEN order_amount < 500 THEN '200-500'
        ELSE '500+' 
    END
ORDER BY MIN(order_amount);

-- Correlation analysis (for numerical columns)
WITH stats AS (
    SELECT 
        AVG(order_amount) as avg_amount,
        AVG(quantity) as avg_quantity,
        STDDEV(order_amount) as std_amount,
        STDDEV(quantity) as std_quantity
    FROM orders
)
SELECT 
    ROUND(
        SUM((order_amount - avg_amount) * (quantity - avg_quantity)) / 
        (COUNT(*) * std_amount * std_quantity), 4
    ) as correlation_coefficient
FROM orders, stats;

-- =================================================
-- 2. WINDOW FUNCTIONS AND ANALYTICS
-- =================================================

-- Running totals and moving averages
SELECT 
    order_date,
    customer_id,
    order_amount,
    -- Running total
    SUM(order_amount) OVER (
        ORDER BY order_date 
        ROWS UNBOUNDED PRECEDING
    ) as running_total,
    -- Moving average (7-day)
    AVG(order_amount) OVER (
        ORDER BY order_date 
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) as moving_avg_7day,
    -- Rank within each customer
    ROW_NUMBER() OVER (
        PARTITION BY customer_id 
        ORDER BY order_date
    ) as order_sequence,
    -- Percentile ranking
    PERCENT_RANK() OVER (
        ORDER BY order_amount
    ) as amount_percentile
FROM orders
ORDER BY order_date, customer_id;

-- Lead/lag analysis
SELECT 
    customer_id,
    order_date,
    order_amount,
    -- Previous order amount
    LAG(order_amount) OVER (
        PARTITION BY customer_id 
        ORDER BY order_date
    ) as prev_order_amount,
    -- Next order amount
    LEAD(order_amount) OVER (
        PARTITION BY customer_id 
        ORDER BY order_date
    ) as next_order_amount,
    -- Days since last order
    order_date - LAG(order_date) OVER (
        PARTITION BY customer_id 
        ORDER BY order_date
    ) as days_since_last_order,
    -- Change from previous order
    order_amount - LAG(order_amount) OVER (
        PARTITION BY customer_id 
        ORDER BY order_date
    ) as amount_change
FROM orders
ORDER BY customer_id, order_date;

-- Cohort analysis
WITH monthly_cohorts AS (
    SELECT 
        customer_id,
        DATE_TRUNC('month', MIN(order_date)) as cohort_month,
        DATE_TRUNC('month', order_date) as order_month
    FROM orders
    GROUP BY customer_id, DATE_TRUNC('month', order_date)
),
cohort_sizes AS (
    SELECT 
        cohort_month,
        COUNT(DISTINCT customer_id) as cohort_size
    FROM monthly_cohorts
    GROUP BY cohort_month
),
cohort_table AS (
    SELECT 
        c.cohort_month,
        c.order_month,
        COUNT(DISTINCT c.customer_id) as active_customers,
        cs.cohort_size,
        EXTRACT(MONTH FROM AGE(c.order_month, c.cohort_month)) as period_number
    FROM monthly_cohorts c
    JOIN cohort_sizes cs ON c.cohort_month = cs.cohort_month
    GROUP BY c.cohort_month, c.order_month, cs.cohort_size
)
SELECT 
    cohort_month,
    period_number,
    active_customers,
    cohort_size,
    ROUND(active_customers * 100.0 / cohort_size, 2) as retention_rate
FROM cohort_table
ORDER BY cohort_month, period_number;

-- =================================================
-- 3. TIME SERIES ANALYSIS
-- =================================================

-- Daily/monthly trends
SELECT 
    DATE_TRUNC('day', order_date) as date,
    COUNT(*) as order_count,
    SUM(order_amount) as total_revenue,
    AVG(order_amount) as avg_order_value,
    COUNT(DISTINCT customer_id) as unique_customers,
    -- Year-over-year comparison
    LAG(SUM(order_amount), 365) OVER (
        ORDER BY DATE_TRUNC('day', order_date)
    ) as revenue_same_day_last_year,
    -- Growth rate
    ROUND(
        (SUM(order_amount) - LAG(SUM(order_amount), 365) OVER (
            ORDER BY DATE_TRUNC('day', order_date)
        )) * 100.0 / NULLIF(LAG(SUM(order_amount), 365) OVER (
            ORDER BY DATE_TRUNC('day', order_date)
        ), 0), 2
    ) as yoy_growth_rate
FROM orders
GROUP BY DATE_TRUNC('day', order_date)
ORDER BY date;

-- Seasonality analysis
SELECT 
    EXTRACT(MONTH FROM order_date) as month,
    EXTRACT(DOW FROM order_date) as day_of_week,
    EXTRACT(HOUR FROM order_date) as hour_of_day,
    COUNT(*) as order_count,
    AVG(order_amount) as avg_amount,
    -- Compare to overall average
    ROUND(
        AVG(order_amount) / AVG(AVG(order_amount)) OVER() * 100 - 100, 2
    ) as pct_diff_from_avg
FROM orders
GROUP BY 
    EXTRACT(MONTH FROM order_date),
    EXTRACT(DOW FROM order_date),
    EXTRACT(HOUR FROM order_date)
ORDER BY month, day_of_week, hour_of_day;

-- Time series decomposition (trend component)
WITH daily_sales AS (
    SELECT 
        DATE_TRUNC('day', order_date) as date,
        SUM(order_amount) as daily_revenue
    FROM orders
    GROUP BY DATE_TRUNC('day', order_date)
),
trend_analysis AS (
    SELECT 
        date,
        daily_revenue,
        -- 7-day moving average (removes weekly seasonality)
        AVG(daily_revenue) OVER (
            ORDER BY date 
            ROWS BETWEEN 3 PRECEDING AND 3 FOLLOWING
        ) as trend_7day,
        -- 30-day moving average (longer trend)
        AVG(daily_revenue) OVER (
            ORDER BY date 
            ROWS BETWEEN 14 PRECEDING AND 14 FOLLOWING
        ) as trend_30day
    FROM daily_sales
)
SELECT 
    date,
    daily_revenue,
    trend_7day,
    trend_30day,
    daily_revenue - trend_7day as detrended_7day,
    daily_revenue - trend_30day as detrended_30day
FROM trend_analysis
ORDER BY date;

-- =================================================
-- 4. CUSTOMER ANALYTICS
-- =================================================

-- Customer lifetime value calculation
WITH customer_metrics AS (
    SELECT 
        customer_id,
        MIN(order_date) as first_order_date,
        MAX(order_date) as last_order_date,
        COUNT(*) as total_orders,
        SUM(order_amount) as total_spent,
        AVG(order_amount) as avg_order_value,
        EXTRACT(DAYS FROM (MAX(order_date) - MIN(order_date))) + 1 as lifetime_days
    FROM orders
    GROUP BY customer_id
),
clv_calculation AS (
    SELECT 
        customer_id,
        total_orders,
        total_spent,
        avg_order_value,
        lifetime_days,
        CASE 
            WHEN lifetime_days > 0 
            THEN total_orders::DECIMAL / (lifetime_days / 365.25)
            ELSE total_orders 
        END as annual_order_frequency,
        CASE 
            WHEN lifetime_days > 0 
            THEN total_spent / (lifetime_days / 365.25)
            ELSE total_spent 
        END as annual_value
    FROM customer_metrics
)
SELECT 
    customer_id,
    total_spent as historical_clv,
    annual_value as predicted_annual_clv,
    -- Simple CLV projection (assuming 3-year lifespan)
    annual_value * 3 as projected_3yr_clv,
    CASE 
        WHEN annual_value >= 1000 THEN 'High Value'
        WHEN annual_value >= 500 THEN 'Medium Value'
        ELSE 'Low Value'
    END as customer_segment
FROM clv_calculation
ORDER BY annual_value DESC;

-- RFM Analysis (Recency, Frequency, Monetary)
WITH rfm_calc AS (
    SELECT 
        customer_id,
        -- Recency (days since last order)
        CURRENT_DATE - MAX(order_date) as recency_days,
        -- Frequency (number of orders)
        COUNT(*) as frequency,
        -- Monetary (total spent)
        SUM(order_amount) as monetary
    FROM orders
    GROUP BY customer_id
),
rfm_scores AS (
    SELECT 
        customer_id,
        recency_days,
        frequency,
        monetary,
        -- RFM scores (1-5, where 5 is best)
        NTILE(5) OVER (ORDER BY recency_days DESC) as recency_score,
        NTILE(5) OVER (ORDER BY frequency) as frequency_score,
        NTILE(5) OVER (ORDER BY monetary) as monetary_score
    FROM rfm_calc
)
SELECT 
    customer_id,
    recency_days,
    frequency,
    monetary,
    recency_score,
    frequency_score,
    monetary_score,
    CONCAT(recency_score, frequency_score, monetary_score) as rfm_score,
    CASE 
        WHEN recency_score >= 4 AND frequency_score >= 4 AND monetary_score >= 4 
        THEN 'Champions'
        WHEN recency_score >= 3 AND frequency_score >= 3 AND monetary_score >= 3 
        THEN 'Loyal Customers'
        WHEN recency_score >= 4 AND frequency_score <= 2 
        THEN 'New Customers'
        WHEN recency_score <= 2 AND frequency_score >= 3 
        THEN 'At Risk'
        WHEN recency_score <= 2 AND frequency_score <= 2 
        THEN 'Lost Customers'
        ELSE 'Others'
    END as customer_segment
FROM rfm_scores
ORDER BY monetary DESC;

-- Customer churn analysis
WITH customer_activity AS (
    SELECT 
        customer_id,
        MAX(order_date) as last_order_date,
        COUNT(*) as total_orders,
        AVG(EXTRACT(DAYS FROM (order_date - LAG(order_date) OVER (
            PARTITION BY customer_id ORDER BY order_date
        )))) as avg_days_between_orders
    FROM orders
    GROUP BY customer_id
),
churn_analysis AS (
    SELECT 
        customer_id,
        last_order_date,
        total_orders,
        avg_days_between_orders,
        CURRENT_DATE - last_order_date as days_since_last_order,
        CASE 
            WHEN avg_days_between_orders IS NOT NULL 
            THEN (CURRENT_DATE - last_order_date) / avg_days_between_orders
            ELSE NULL 
        END as churn_risk_multiplier
    FROM customer_activity
)
SELECT 
    customer_id,
    last_order_date,
    days_since_last_order,
    churn_risk_multiplier,
    CASE 
        WHEN churn_risk_multiplier IS NULL THEN 'One-time Customer'
        WHEN churn_risk_multiplier <= 1.5 THEN 'Active'
        WHEN churn_risk_multiplier <= 3 THEN 'At Risk'
        ELSE 'Likely Churned'
    END as churn_status
FROM churn_analysis
ORDER BY churn_risk_multiplier DESC NULLS LAST;

-- =================================================
-- 5. STATISTICAL FUNCTIONS
-- =================================================

-- Descriptive statistics
WITH stats AS (
    SELECT 
        COUNT(*) as n,
        AVG(order_amount) as mean,
        STDDEV(order_amount) as std_dev,
        VARIANCE(order_amount) as variance,
        MIN(order_amount) as min_val,
        MAX(order_amount) as max_val,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY order_amount) as q1,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY order_amount) as median,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY order_amount) as q3,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY order_amount) as p95,
        PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY order_amount) as p99
    FROM orders
)
SELECT 
    *,
    q3 - q1 as iqr,
    (max_val - min_val) as range_val,
    std_dev / mean as coefficient_of_variation
FROM stats;

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
        q3 - q1 as iqr,
        q1 - 1.5 * (q3 - q1) as lower_bound,
        q3 + 1.5 * (q3 - q1) as upper_bound
    FROM quartiles
)
SELECT 
    o.customer_id,
    o.order_date,
    o.order_amount,
    CASE 
        WHEN o.order_amount < ob.lower_bound OR o.order_amount > ob.upper_bound 
        THEN 'Outlier'
        ELSE 'Normal'
    END as outlier_status,
    ob.lower_bound,
    ob.upper_bound
FROM orders o
CROSS JOIN outlier_bounds ob
WHERE o.order_amount < ob.lower_bound OR o.order_amount > ob.upper_bound
ORDER BY o.order_amount DESC;

-- Hypothesis testing (t-test simulation)
WITH group_stats AS (
    SELECT 
        CASE WHEN customer_id % 2 = 0 THEN 'Group A' ELSE 'Group B' END as test_group,
        COUNT(*) as n,
        AVG(order_amount) as mean,
        STDDEV(order_amount) as std_dev,
        VARIANCE(order_amount) as variance
    FROM orders
    GROUP BY CASE WHEN customer_id % 2 = 0 THEN 'Group A' ELSE 'Group B' END
),
t_test AS (
    SELECT 
        a.mean as mean_a,
        b.mean as mean_b,
        a.mean - b.mean as mean_diff,
        a.std_dev as std_a,
        b.std_dev as std_b,
        a.n as n_a,
        b.n as n_b,
        SQRT((a.variance / a.n) + (b.variance / b.n)) as standard_error,
        (a.mean - b.mean) / SQRT((a.variance / a.n) + (b.variance / b.n)) as t_statistic
    FROM group_stats a
    CROSS JOIN group_stats b
    WHERE a.test_group = 'Group A' AND b.test_group = 'Group B'
)
SELECT 
    *,
    CASE 
        WHEN ABS(t_statistic) > 1.96 THEN 'Significant (p < 0.05)'
        WHEN ABS(t_statistic) > 1.645 THEN 'Significant (p < 0.10)'
        ELSE 'Not Significant'
    END as significance_test
FROM t_test;

-- =================================================
-- 6. DATA QUALITY ASSESSMENT
-- =================================================

-- Completeness check
SELECT 
    'customer_id' as column_name,
    COUNT(*) as total_records,
    COUNT(customer_id) as non_null_records,
    COUNT(*) - COUNT(customer_id) as null_records,
    ROUND((COUNT(customer_id) * 100.0) / COUNT(*), 2) as completeness_pct
FROM orders

UNION ALL

SELECT 
    'order_amount' as column_name,
    COUNT(*) as total_records,
    COUNT(order_amount) as non_null_records,
    COUNT(*) - COUNT(order_amount) as null_records,
    ROUND((COUNT(order_amount) * 100.0) / COUNT(*), 2) as completeness_pct
FROM orders

UNION ALL

SELECT 
    'order_date' as column_name,
    COUNT(*) as total_records,
    COUNT(order_date) as non_null_records,
    COUNT(*) - COUNT(order_date) as null_records,
    ROUND((COUNT(order_date) * 100.0) / COUNT(*), 2) as completeness_pct
FROM orders;

-- Uniqueness check
SELECT 
    'Customer ID uniqueness' as check_name,
    COUNT(*) as total_records,
    COUNT(DISTINCT customer_id) as unique_values,
    COUNT(*) - COUNT(DISTINCT customer_id) as duplicates,
    ROUND((COUNT(DISTINCT customer_id) * 100.0) / COUNT(*), 2) as uniqueness_pct
FROM orders

UNION ALL

SELECT 
    'Order ID uniqueness' as check_name,
    COUNT(*) as total_records,
    COUNT(DISTINCT order_id) as unique_values,
    COUNT(*) - COUNT(DISTINCT order_id) as duplicates,
    ROUND((COUNT(DISTINCT order_id) * 100.0) / COUNT(*), 2) as uniqueness_pct
FROM orders;

-- Validity checks
SELECT 
    'Negative amounts' as check_name,
    COUNT(*) as total_records,
    SUM(CASE WHEN order_amount < 0 THEN 1 ELSE 0 END) as invalid_records,
    ROUND((SUM(CASE WHEN order_amount < 0 THEN 1 ELSE 0 END) * 100.0) / COUNT(*), 2) as invalid_pct
FROM orders

UNION ALL

SELECT 
    'Future dates' as check_name,
    COUNT(*) as total_records,
    SUM(CASE WHEN order_date > CURRENT_DATE THEN 1 ELSE 0 END) as invalid_records,
    ROUND((SUM(CASE WHEN order_date > CURRENT_DATE THEN 1 ELSE 0 END) * 100.0) / COUNT(*), 2) as invalid_pct
FROM orders

UNION ALL

SELECT 
    'Zero amounts' as check_name,
    COUNT(*) as total_records,
    SUM(CASE WHEN order_amount = 0 THEN 1 ELSE 0 END) as invalid_records,
    ROUND((SUM(CASE WHEN order_amount = 0 THEN 1 ELSE 0 END) * 100.0) / COUNT(*), 2) as invalid_pct
FROM orders;

-- =================================================
-- 7. ADVANCED AGGREGATIONS
-- =================================================

-- Pivot table simulation
SELECT 
    DATE_TRUNC('month', order_date) as month,
    SUM(CASE WHEN EXTRACT(DOW FROM order_date) = 1 THEN order_amount ELSE 0 END) as monday_sales,
    SUM(CASE WHEN EXTRACT(DOW FROM order_date) = 2 THEN order_amount ELSE 0 END) as tuesday_sales,
    SUM(CASE WHEN EXTRACT(DOW FROM order_date) = 3 THEN order_amount ELSE 0 END) as wednesday_sales,
    SUM(CASE WHEN EXTRACT(DOW FROM order_date) = 4 THEN order_amount ELSE 0 END) as thursday_sales,
    SUM(CASE WHEN EXTRACT(DOW FROM order_date) = 5 THEN order_amount ELSE 0 END) as friday_sales,
    SUM(CASE WHEN EXTRACT(DOW FROM order_date) = 6 THEN order_amount ELSE 0 END) as saturday_sales,
    SUM(CASE WHEN EXTRACT(DOW FROM order_date) = 0 THEN order_amount ELSE 0 END) as sunday_sales
FROM orders
GROUP BY DATE_TRUNC('month', order_date)
ORDER BY month;

-- Hierarchical aggregation (ROLLUP)
SELECT 
    EXTRACT(YEAR FROM order_date) as year,
    EXTRACT(MONTH FROM order_date) as month,
    COUNT(*) as order_count,
    SUM(order_amount) as total_revenue,
    AVG(order_amount) as avg_order_value
FROM orders
GROUP BY ROLLUP(EXTRACT(YEAR FROM order_date), EXTRACT(MONTH FROM order_date))
ORDER BY year, month;

-- Complex window functions for ranking
WITH customer_ranking AS (
    SELECT 
        customer_id,
        SUM(order_amount) as total_spent,
        COUNT(*) as order_count,
        AVG(order_amount) as avg_order_value,
        -- Dense rank by total spent
        DENSE_RANK() OVER (ORDER BY SUM(order_amount) DESC) as spending_rank,
        -- Percentile by spending
        PERCENT_RANK() OVER (ORDER BY SUM(order_amount)) as spending_percentile,
        -- Ntile for quintiles
        NTILE(5) OVER (ORDER BY SUM(order_amount) DESC) as spending_quintile
    FROM orders
    GROUP BY customer_id
)
SELECT 
    customer_id,
    total_spent,
    order_count,
    avg_order_value,
    spending_rank,
    ROUND(spending_percentile * 100, 1) as spending_percentile,
    spending_quintile,
    CASE spending_quintile
        WHEN 1 THEN 'Top 20%'
        WHEN 2 THEN '20-40%'
        WHEN 3 THEN '40-60%'
        WHEN 4 THEN '60-80%'
        WHEN 5 THEN 'Bottom 20%'
    END as customer_tier
FROM customer_ranking
ORDER BY spending_rank;

-- =================================================
-- 8. PERFORMANCE OPTIMIZATION EXAMPLES
-- =================================================

-- Efficient date range queries with indexing hints
SELECT /*+ INDEX(orders, idx_order_date) */
    customer_id,
    order_date,
    order_amount
FROM orders
WHERE order_date >= '2023-01-01'
    AND order_date < '2024-01-01'
    AND order_amount > 100;

-- Materialized view for common aggregations
CREATE MATERIALIZED VIEW monthly_customer_summary AS
SELECT 
    customer_id,
    DATE_TRUNC('month', order_date) as month,
    COUNT(*) as order_count,
    SUM(order_amount) as total_spent,
    AVG(order_amount) as avg_order_value,
    MIN(order_date) as first_order_date,
    MAX(order_date) as last_order_date
FROM orders
GROUP BY customer_id, DATE_TRUNC('month', order_date);

-- Common table expressions for complex analysis
WITH RECURSIVE date_series AS (
    SELECT DATE '2023-01-01' as date
    UNION ALL
    SELECT date + INTERVAL '1 day'
    FROM date_series
    WHERE date < '2023-12-31'
),
daily_orders AS (
    SELECT 
        DATE_TRUNC('day', order_date) as order_day,
        COUNT(*) as order_count,
        SUM(order_amount) as daily_revenue
    FROM orders
    WHERE order_date >= '2023-01-01' 
        AND order_date <= '2023-12-31'
    GROUP BY DATE_TRUNC('day', order_date)
)
SELECT 
    ds.date,
    COALESCE(do.order_count, 0) as order_count,
    COALESCE(do.daily_revenue, 0) as daily_revenue,
    -- Fill gaps with moving average
    AVG(COALESCE(do.daily_revenue, 0)) OVER (
        ORDER BY ds.date 
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) as smoothed_revenue
FROM date_series ds
LEFT JOIN daily_orders do ON ds.date = do.order_day
ORDER BY ds.date; 
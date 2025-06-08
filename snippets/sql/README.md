# SQL Code Snippets Collection

## Overview

Comprehensive collection of SQL code snippets for data querying, analysis, optimization, and database administration across different SQL dialects.

## Table of Contents

- [Data Querying](#data-querying)
- [Data Analysis](#data-analysis)
- [Window Functions](#window-functions)
- [Data Manipulation](#data-manipulation)
- [Performance Optimization](#performance-optimization)
- [Database Administration](#database-administration)
- [Advanced Techniques](#advanced-techniques)
- [Platform-Specific](#platform-specific)

## Data Querying

### Basic Query Patterns

```sql
-- Flexible filtering with CASE statements
SELECT 
    customer_id,
    order_date,
    total_amount,
    CASE 
        WHEN total_amount > 1000 THEN 'High Value'
        WHEN total_amount > 500 THEN 'Medium Value'
        ELSE 'Low Value'
    END AS order_category,
    CASE 
        WHEN EXTRACT(MONTH FROM order_date) IN (12, 1, 2) THEN 'Winter'
        WHEN EXTRACT(MONTH FROM order_date) IN (3, 4, 5) THEN 'Spring'
        WHEN EXTRACT(MONTH FROM order_date) IN (6, 7, 8) THEN 'Summer'
        ELSE 'Fall'
    END AS season
FROM orders
WHERE order_date >= CURRENT_DATE - INTERVAL '1 year'
ORDER BY order_date DESC;

-- Dynamic date filtering
SELECT *
FROM sales
WHERE 
    -- Last 30 days
    created_date >= CURRENT_DATE - INTERVAL '30 days'
    -- Current month
    OR (EXTRACT(YEAR FROM created_date) = EXTRACT(YEAR FROM CURRENT_DATE)
        AND EXTRACT(MONTH FROM created_date) = EXTRACT(MONTH FROM CURRENT_DATE))
    -- Previous month
    OR (EXTRACT(YEAR FROM created_date) = EXTRACT(YEAR FROM CURRENT_DATE - INTERVAL '1 month')
        AND EXTRACT(MONTH FROM created_date) = EXTRACT(MONTH FROM CURRENT_DATE - INTERVAL '1 month'));

-- Conditional aggregation
SELECT 
    product_category,
    COUNT(*) AS total_orders,
    COUNT(CASE WHEN order_status = 'completed' THEN 1 END) AS completed_orders,
    COUNT(CASE WHEN order_status = 'cancelled' THEN 1 END) AS cancelled_orders,
    AVG(CASE WHEN order_status = 'completed' THEN total_amount END) AS avg_completed_amount,
    SUM(CASE WHEN EXTRACT(MONTH FROM order_date) = EXTRACT(MONTH FROM CURRENT_DATE) 
             THEN total_amount ELSE 0 END) AS current_month_revenue
FROM orders o
JOIN products p ON o.product_id = p.product_id
GROUP BY product_category
HAVING COUNT(*) > 10
ORDER BY total_orders DESC;

-- Text search and pattern matching
SELECT 
    customer_id,
    customer_name,
    email,
    phone
FROM customers
WHERE 
    -- Fuzzy name matching
    LOWER(customer_name) LIKE '%john%'
    -- Email domain filtering
    OR email LIKE '%@gmail.com'
    -- Phone number patterns
    OR REGEXP_REPLACE(phone, '[^0-9]', '') LIKE '555%'
    -- Multiple keyword search
    OR (LOWER(customer_name) LIKE '%smith%' AND LOWER(customer_name) LIKE '%jane%');
```

### Complex Joins and Subqueries

```sql
-- Multi-level joins with aggregations
SELECT 
    c.customer_name,
    c.customer_segment,
    COUNT(DISTINCT o.order_id) AS total_orders,
    SUM(oi.quantity * oi.unit_price) AS total_revenue,
    AVG(oi.quantity * oi.unit_price) AS avg_order_value,
    MAX(o.order_date) AS last_order_date,
    STRING_AGG(DISTINCT p.product_category, ', ') AS purchased_categories
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
LEFT JOIN order_items oi ON o.order_id = oi.order_id
LEFT JOIN products p ON oi.product_id = p.product_id
WHERE o.order_date >= CURRENT_DATE - INTERVAL '2 years'
GROUP BY c.customer_id, c.customer_name, c.customer_segment
HAVING COUNT(DISTINCT o.order_id) > 0
ORDER BY total_revenue DESC;

-- Correlated subqueries for comparative analysis
SELECT 
    product_id,
    product_name,
    current_price,
    (SELECT AVG(unit_price) 
     FROM order_items oi2 
     WHERE oi2.product_id = p.product_id 
     AND oi2.order_date >= CURRENT_DATE - INTERVAL '30 days') AS avg_selling_price_30d,
    (SELECT COUNT(DISTINCT customer_id)
     FROM orders o2
     JOIN order_items oi3 ON o2.order_id = oi3.order_id
     WHERE oi3.product_id = p.product_id
     AND o2.order_date >= CURRENT_DATE - INTERVAL '90 days') AS unique_customers_90d
FROM products p
WHERE EXISTS (
    SELECT 1 
    FROM order_items oi 
    WHERE oi.product_id = p.product_id 
    AND oi.order_date >= CURRENT_DATE - INTERVAL '1 year'
);

-- Common Table Expressions (CTEs) for complex logic
WITH monthly_sales AS (
    SELECT 
        EXTRACT(YEAR FROM order_date) AS year,
        EXTRACT(MONTH FROM order_date) AS month,
        SUM(total_amount) AS monthly_revenue,
        COUNT(DISTINCT customer_id) AS unique_customers,
        COUNT(*) AS order_count
    FROM orders
    WHERE order_date >= CURRENT_DATE - INTERVAL '2 years'
    GROUP BY EXTRACT(YEAR FROM order_date), EXTRACT(MONTH FROM order_date)
),
sales_with_growth AS (
    SELECT 
        year,
        month,
        monthly_revenue,
        unique_customers,
        order_count,
        LAG(monthly_revenue) OVER (ORDER BY year, month) AS prev_month_revenue,
        (monthly_revenue - LAG(monthly_revenue) OVER (ORDER BY year, month)) / 
        NULLIF(LAG(monthly_revenue) OVER (ORDER BY year, month), 0) * 100 AS revenue_growth_pct
    FROM monthly_sales
)
SELECT 
    year,
    month,
    monthly_revenue,
    unique_customers,
    order_count,
    COALESCE(revenue_growth_pct, 0) AS revenue_growth_pct,
    CASE 
        WHEN revenue_growth_pct > 10 THEN 'High Growth'
        WHEN revenue_growth_pct > 0 THEN 'Positive Growth'
        WHEN revenue_growth_pct > -10 THEN 'Slight Decline'
        ELSE 'Significant Decline'
    END AS growth_category
FROM sales_with_growth
ORDER BY year, month;
```

## Data Analysis

### Statistical Analysis

```sql
-- Comprehensive statistical summary
SELECT 
    product_category,
    COUNT(*) AS n,
    AVG(unit_price) AS mean_price,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY unit_price) AS median_price,
    STDDEV(unit_price) AS std_dev,
    MIN(unit_price) AS min_price,
    MAX(unit_price) AS max_price,
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY unit_price) AS q1,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY unit_price) AS q3,
    -- Coefficient of variation
    STDDEV(unit_price) / NULLIF(AVG(unit_price), 0) AS cv,
    -- Skewness approximation
    (AVG(unit_price) - PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY unit_price)) / 
    NULLIF(STDDEV(unit_price), 0) AS skewness_approx
FROM products
GROUP BY product_category
ORDER BY mean_price DESC;

-- Cohort analysis
WITH customer_cohorts AS (
    SELECT 
        customer_id,
        DATE_TRUNC('month', MIN(order_date)) AS cohort_month,
        MIN(order_date) AS first_order_date
    FROM orders
    GROUP BY customer_id
),
cohort_data AS (
    SELECT 
        cc.cohort_month,
        DATE_TRUNC('month', o.order_date) AS order_month,
        COUNT(DISTINCT o.customer_id) AS customers
    FROM customer_cohorts cc
    JOIN orders o ON cc.customer_id = o.customer_id
    GROUP BY cc.cohort_month, DATE_TRUNC('month', o.order_date)
),
cohort_sizes AS (
    SELECT 
        cohort_month,
        COUNT(DISTINCT customer_id) AS cohort_size
    FROM customer_cohorts
    GROUP BY cohort_month
)
SELECT 
    cd.cohort_month,
    cd.order_month,
    EXTRACT(MONTH FROM AGE(cd.order_month, cd.cohort_month)) AS period_number,
    cd.customers,
    cs.cohort_size,
    ROUND(cd.customers::DECIMAL / cs.cohort_size * 100, 2) AS retention_rate
FROM cohort_data cd
JOIN cohort_sizes cs ON cd.cohort_month = cs.cohort_month
ORDER BY cd.cohort_month, cd.order_month;

-- RFM Analysis (Recency, Frequency, Monetary)
WITH customer_rfm AS (
    SELECT 
        customer_id,
        CURRENT_DATE - MAX(order_date) AS recency_days,
        COUNT(*) AS frequency,
        SUM(total_amount) AS monetary_value
    FROM orders
    WHERE order_date >= CURRENT_DATE - INTERVAL '2 years'
    GROUP BY customer_id
),
rfm_scores AS (
    SELECT 
        customer_id,
        recency_days,
        frequency,
        monetary_value,
        NTILE(5) OVER (ORDER BY recency_days DESC) AS recency_score,
        NTILE(5) OVER (ORDER BY frequency) AS frequency_score,
        NTILE(5) OVER (ORDER BY monetary_value) AS monetary_score
    FROM customer_rfm
)
SELECT 
    customer_id,
    recency_days,
    frequency,
    monetary_value,
    recency_score,
    frequency_score,
    monetary_score,
    CONCAT(recency_score, frequency_score, monetary_score) AS rfm_segment,
    CASE 
        WHEN recency_score >= 4 AND frequency_score >= 4 AND monetary_score >= 4 THEN 'Champions'
        WHEN recency_score >= 3 AND frequency_score >= 3 AND monetary_score >= 3 THEN 'Loyal Customers'
        WHEN recency_score >= 4 AND frequency_score <= 2 THEN 'New Customers'
        WHEN recency_score <= 2 AND frequency_score >= 3 THEN 'At Risk'
        WHEN recency_score <= 2 AND frequency_score <= 2 THEN 'Lost Customers'
        ELSE 'Potential Loyalists'
    END AS customer_segment
FROM rfm_scores
ORDER BY monetary_value DESC;
```

## Window Functions

### Advanced Analytics with Window Functions

```sql
-- Running totals and moving averages
SELECT 
    order_date,
    daily_revenue,
    SUM(daily_revenue) OVER (ORDER BY order_date) AS running_total,
    AVG(daily_revenue) OVER (ORDER BY order_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS ma_7day,
    AVG(daily_revenue) OVER (ORDER BY order_date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) AS ma_30day,
    -- Percentage of total
    daily_revenue / SUM(daily_revenue) OVER () * 100 AS pct_of_total,
    -- Rank and percentile
    RANK() OVER (ORDER BY daily_revenue DESC) AS revenue_rank,
    PERCENT_RANK() OVER (ORDER BY daily_revenue) AS revenue_percentile
FROM (
    SELECT 
        order_date,
        SUM(total_amount) AS daily_revenue
    FROM orders
    WHERE order_date >= CURRENT_DATE - INTERVAL '1 year'
    GROUP BY order_date
) daily_sales
ORDER BY order_date;

-- Year-over-year comparison
SELECT 
    product_category,
    EXTRACT(YEAR FROM order_date) AS year,
    EXTRACT(MONTH FROM order_date) AS month,
    SUM(total_amount) AS monthly_revenue,
    LAG(SUM(total_amount), 12) OVER (
        PARTITION BY product_category 
        ORDER BY EXTRACT(YEAR FROM order_date), EXTRACT(MONTH FROM order_date)
    ) AS same_month_last_year,
    (SUM(total_amount) - LAG(SUM(total_amount), 12) OVER (
        PARTITION BY product_category 
        ORDER BY EXTRACT(YEAR FROM order_date), EXTRACT(MONTH FROM order_date)
    )) / NULLIF(LAG(SUM(total_amount), 12) OVER (
        PARTITION BY product_category 
        ORDER BY EXTRACT(YEAR FROM order_date), EXTRACT(MONTH FROM order_date)
    ), 0) * 100 AS yoy_growth_pct
FROM orders o
JOIN products p ON o.product_id = p.product_id
GROUP BY product_category, EXTRACT(YEAR FROM order_date), EXTRACT(MONTH FROM order_date)
ORDER BY product_category, year, month;

-- Top N analysis with ties
SELECT 
    customer_id,
    customer_name,
    total_orders,
    total_revenue,
    revenue_rank,
    revenue_dense_rank
FROM (
    SELECT 
        c.customer_id,
        c.customer_name,
        COUNT(o.order_id) AS total_orders,
        SUM(o.total_amount) AS total_revenue,
        ROW_NUMBER() OVER (ORDER BY SUM(o.total_amount) DESC) AS revenue_rank,
        DENSE_RANK() OVER (ORDER BY SUM(o.total_amount) DESC) AS revenue_dense_rank
    FROM customers c
    JOIN orders o ON c.customer_id = o.customer_id
    WHERE o.order_date >= CURRENT_DATE - INTERVAL '1 year'
    GROUP BY c.customer_id, c.customer_name
) ranked_customers
WHERE revenue_dense_rank <= 10
ORDER BY revenue_rank;

-- Gap and island analysis
WITH order_sequences AS (
    SELECT 
        customer_id,
        order_date,
        ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date) AS rn,
        order_date - INTERVAL '1 day' * ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date) AS grp
    FROM orders
),
consecutive_periods AS (
    SELECT 
        customer_id,
        MIN(order_date) AS period_start,
        MAX(order_date) AS period_end,
        COUNT(*) AS orders_in_period,
        MAX(order_date) - MIN(order_date) + 1 AS period_length_days
    FROM order_sequences
    GROUP BY customer_id, grp
)
SELECT 
    customer_id,
    period_start,
    period_end,
    orders_in_period,
    period_length_days,
    LEAD(period_start) OVER (PARTITION BY customer_id ORDER BY period_start) - period_end - 1 AS gap_days
FROM consecutive_periods
ORDER BY customer_id, period_start;
```

## Data Manipulation

### Insert, Update, Delete Patterns

```sql
-- Upsert (INSERT or UPDATE) pattern
INSERT INTO customer_summary (customer_id, total_orders, total_revenue, last_order_date)
SELECT 
    customer_id,
    COUNT(*) AS total_orders,
    SUM(total_amount) AS total_revenue,
    MAX(order_date) AS last_order_date
FROM orders
WHERE order_date >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY customer_id
ON CONFLICT (customer_id) 
DO UPDATE SET
    total_orders = EXCLUDED.total_orders,
    total_revenue = EXCLUDED.total_revenue,
    last_order_date = EXCLUDED.last_order_date,
    updated_at = CURRENT_TIMESTAMP;

-- Conditional updates with CASE
UPDATE products 
SET 
    price_category = CASE 
        WHEN unit_price > 100 THEN 'Premium'
        WHEN unit_price > 50 THEN 'Standard'
        ELSE 'Budget'
    END,
    discount_eligible = CASE 
        WHEN inventory_count > 100 THEN TRUE
        WHEN unit_price < 20 THEN FALSE
        ELSE discount_eligible
    END,
    updated_at = CURRENT_TIMESTAMP
WHERE last_updated < CURRENT_DATE - INTERVAL '1 week';

-- Bulk data cleanup
DELETE FROM orders 
WHERE order_id IN (
    SELECT order_id 
    FROM orders o
    LEFT JOIN customers c ON o.customer_id = c.customer_id
    WHERE c.customer_id IS NULL  -- Orphaned orders
    OR o.total_amount <= 0       -- Invalid amounts
    OR o.order_date > CURRENT_DATE  -- Future dates
);

-- Data archiving with partitioning
INSERT INTO orders_archive 
SELECT * 
FROM orders 
WHERE order_date < CURRENT_DATE - INTERVAL '2 years';

DELETE FROM orders 
WHERE order_date < CURRENT_DATE - INTERVAL '2 years';
```

## Performance Optimization

### Query Optimization Techniques

```sql
-- Index-friendly queries
-- Good: Uses index on order_date
SELECT * FROM orders 
WHERE order_date >= '2023-01-01' 
AND order_date < '2024-01-01';

-- Bad: Function on indexed column
-- SELECT * FROM orders WHERE YEAR(order_date) = 2023;

-- Efficient pagination
SELECT * FROM (
    SELECT 
        *,
        ROW_NUMBER() OVER (ORDER BY order_date DESC) AS rn
    FROM orders
    WHERE order_date >= CURRENT_DATE - INTERVAL '1 year'
) paginated
WHERE rn BETWEEN 101 AND 200;

-- EXISTS vs IN for better performance
-- Good: EXISTS with correlated subquery
SELECT c.customer_id, c.customer_name
FROM customers c
WHERE EXISTS (
    SELECT 1 FROM orders o 
    WHERE o.customer_id = c.customer_id 
    AND o.order_date >= CURRENT_DATE - INTERVAL '30 days'
);

-- Avoiding SELECT *
SELECT 
    o.order_id,
    o.order_date,
    o.total_amount,
    c.customer_name
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
WHERE o.order_date >= CURRENT_DATE - INTERVAL '1 month';

-- Query hints and optimization
-- PostgreSQL
/*+ SeqScan(orders) */
SELECT /*+ USE_INDEX(orders, idx_order_date) */ 
    order_id, total_amount 
FROM orders 
WHERE order_date >= CURRENT_DATE - INTERVAL '1 week';

-- Materialized view for complex aggregations
CREATE MATERIALIZED VIEW monthly_sales_summary AS
SELECT 
    EXTRACT(YEAR FROM order_date) AS year,
    EXTRACT(MONTH FROM order_date) AS month,
    product_category,
    COUNT(*) AS order_count,
    SUM(total_amount) AS total_revenue,
    AVG(total_amount) AS avg_order_value,
    COUNT(DISTINCT customer_id) AS unique_customers
FROM orders o
JOIN products p ON o.product_id = p.product_id
GROUP BY 
    EXTRACT(YEAR FROM order_date),
    EXTRACT(MONTH FROM order_date),
    product_category;

-- Refresh materialized view
REFRESH MATERIALIZED VIEW monthly_sales_summary;
```

---

*This comprehensive SQL snippets collection provides practical, optimized queries for data analysis, reporting, and database management across different SQL platforms.* 
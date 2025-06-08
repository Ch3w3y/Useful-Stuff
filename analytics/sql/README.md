# SQL Analytics & Data Engineering

## Overview

Comprehensive guide to SQL for analytics, data engineering, ETL processes, and advanced data analysis techniques across different database platforms.

## Table of Contents

- [Analytics Fundamentals](#analytics-fundamentals)
- [Data Exploration](#data-exploration)
- [ETL & Data Pipeline](#etl--data-pipeline)
- [Advanced Analytics](#advanced-analytics)
- [Performance Optimization](#performance-optimization)
- [Reporting & Dashboards](#reporting--dashboards)
- [Data Quality](#data-quality)
- [Platform-Specific Features](#platform-specific-features)

## Analytics Fundamentals

### Data Profiling and Discovery

```sql
-- Comprehensive data profiling template
WITH data_profile AS (
    SELECT 
        -- Basic counts
        COUNT(*) AS total_rows,
        COUNT(DISTINCT customer_id) AS unique_customers,
        
        -- Date range analysis
        MIN(order_date) AS earliest_date,
        MAX(order_date) AS latest_date,
        DATEDIFF(day, MIN(order_date), MAX(order_date)) AS date_range_days,
        
        -- Completeness analysis
        COUNT(CASE WHEN customer_id IS NOT NULL THEN 1 END) AS customer_id_complete,
        COUNT(CASE WHEN order_date IS NOT NULL THEN 1 END) AS order_date_complete,
        COUNT(CASE WHEN total_amount IS NOT NULL THEN 1 END) AS amount_complete,
        
        -- Value distribution
        AVG(total_amount) AS avg_order_value,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY total_amount) AS median_order_value,
        STDDEV(total_amount) AS stddev_order_value,
        MIN(total_amount) AS min_order_value,
        MAX(total_amount) AS max_order_value,
        
        -- Categorical analysis
        COUNT(DISTINCT order_status) AS unique_statuses,
        COUNT(DISTINCT product_category) AS unique_categories
    FROM orders o
    LEFT JOIN products p ON o.product_id = p.product_id
),
completeness_summary AS (
    SELECT 
        'customer_id' AS column_name,
        customer_id_complete AS complete_count,
        total_rows - customer_id_complete AS missing_count,
        ROUND((customer_id_complete * 100.0 / total_rows), 2) AS completeness_pct
    FROM data_profile
    
    UNION ALL
    
    SELECT 
        'order_date',
        order_date_complete,
        total_rows - order_date_complete,
        ROUND((order_date_complete * 100.0 / total_rows), 2)
    FROM data_profile
    
    UNION ALL
    
    SELECT 
        'total_amount',
        amount_complete,
        total_rows - amount_complete,
        ROUND((amount_complete * 100.0 / total_rows), 2)
    FROM data_profile
)
SELECT * FROM data_profile
UNION ALL
SELECT 'Completeness Analysis' AS metric, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL
UNION ALL
SELECT 
    column_name,
    complete_count,
    missing_count,
    completeness_pct,
    NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL
FROM completeness_summary;

-- Data quality assessment
WITH quality_checks AS (
    SELECT 
        'Duplicate Records' AS check_type,
        COUNT(*) - COUNT(DISTINCT CONCAT(customer_id, order_date, total_amount)) AS issue_count,
        CASE 
            WHEN COUNT(*) = COUNT(DISTINCT CONCAT(customer_id, order_date, total_amount)) 
            THEN 'PASS' 
            ELSE 'FAIL' 
        END AS status
    FROM orders
    
    UNION ALL
    
    SELECT 
        'Future Dates',
        COUNT(CASE WHEN order_date > CURRENT_DATE THEN 1 END),
        CASE 
            WHEN COUNT(CASE WHEN order_date > CURRENT_DATE THEN 1 END) = 0 
            THEN 'PASS' 
            ELSE 'FAIL' 
        END
    FROM orders
    
    UNION ALL
    
    SELECT 
        'Negative Amounts',
        COUNT(CASE WHEN total_amount < 0 THEN 1 END),
        CASE 
            WHEN COUNT(CASE WHEN total_amount < 0 THEN 1 END) = 0 
            THEN 'PASS' 
            ELSE 'FAIL' 
        END
    FROM orders
    
    UNION ALL
    
    SELECT 
        'Orphaned Records',
        COUNT(CASE WHEN c.customer_id IS NULL THEN 1 END),
        CASE 
            WHEN COUNT(CASE WHEN c.customer_id IS NULL THEN 1 END) = 0 
            THEN 'PASS' 
            ELSE 'FAIL' 
        END
    FROM orders o
    LEFT JOIN customers c ON o.customer_id = c.customer_id
)
SELECT * FROM quality_checks;

-- Value distribution analysis
SELECT 
    'total_amount' AS metric,
    PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY total_amount) AS p5,
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY total_amount) AS p25,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY total_amount) AS p50,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY total_amount) AS p75,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY total_amount) AS p95,
    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY total_amount) AS p99
FROM orders
WHERE total_amount IS NOT NULL;
```

### Business Metrics Framework

```sql
-- KPI calculation framework
WITH date_spine AS (
    SELECT 
        date_day,
        EXTRACT(YEAR FROM date_day) AS year,
        EXTRACT(MONTH FROM date_day) AS month,
        EXTRACT(WEEK FROM date_day) AS week,
        EXTRACT(DOW FROM date_day) AS day_of_week
    FROM generate_series(
        '2023-01-01'::date, 
        CURRENT_DATE, 
        '1 day'::interval
    ) AS date_day
),
daily_metrics AS (
    SELECT 
        ds.date_day,
        ds.year,
        ds.month,
        ds.week,
        
        -- Revenue metrics
        COALESCE(SUM(o.total_amount), 0) AS daily_revenue,
        COALESCE(COUNT(o.order_id), 0) AS daily_orders,
        COALESCE(COUNT(DISTINCT o.customer_id), 0) AS daily_active_customers,
        
        -- Average order value
        CASE 
            WHEN COUNT(o.order_id) > 0 
            THEN SUM(o.total_amount) / COUNT(o.order_id) 
            ELSE 0 
        END AS avg_order_value,
        
        -- Customer metrics
        COALESCE(COUNT(DISTINCT CASE WHEN fc.first_order_date = ds.date_day THEN o.customer_id END), 0) AS new_customers,
        COALESCE(COUNT(DISTINCT CASE WHEN fc.first_order_date < ds.date_day THEN o.customer_id END), 0) AS returning_customers
        
    FROM date_spine ds
    LEFT JOIN orders o ON ds.date_day = DATE(o.order_date)
    LEFT JOIN (
        SELECT 
            customer_id, 
            MIN(DATE(order_date)) AS first_order_date
        FROM orders 
        GROUP BY customer_id
    ) fc ON o.customer_id = fc.customer_id
    GROUP BY ds.date_day, ds.year, ds.month, ds.week
),
rolling_metrics AS (
    SELECT 
        *,
        -- 7-day rolling averages
        AVG(daily_revenue) OVER (ORDER BY date_day ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS revenue_7d_avg,
        AVG(daily_orders) OVER (ORDER BY date_day ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS orders_7d_avg,
        AVG(daily_active_customers) OVER (ORDER BY date_day ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS customers_7d_avg,
        
        -- 30-day rolling averages
        AVG(daily_revenue) OVER (ORDER BY date_day ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) AS revenue_30d_avg,
        AVG(daily_orders) OVER (ORDER BY date_day ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) AS orders_30d_avg,
        
        -- Growth calculations
        LAG(daily_revenue, 7) OVER (ORDER BY date_day) AS revenue_7d_ago,
        LAG(daily_revenue, 30) OVER (ORDER BY date_day) AS revenue_30d_ago,
        LAG(daily_revenue, 365) OVER (ORDER BY date_day) AS revenue_1y_ago
        
    FROM daily_metrics
)
SELECT 
    date_day,
    daily_revenue,
    daily_orders,
    daily_active_customers,
    avg_order_value,
    new_customers,
    returning_customers,
    
    -- Rolling averages
    ROUND(revenue_7d_avg, 2) AS revenue_7d_avg,
    ROUND(revenue_30d_avg, 2) AS revenue_30d_avg,
    
    -- Growth rates
    CASE 
        WHEN revenue_7d_ago > 0 
        THEN ROUND(((daily_revenue - revenue_7d_ago) / revenue_7d_ago * 100), 2) 
        ELSE NULL 
    END AS revenue_wow_growth,
    
    CASE 
        WHEN revenue_30d_ago > 0 
        THEN ROUND(((daily_revenue - revenue_30d_ago) / revenue_30d_ago * 100), 2) 
        ELSE NULL 
    END AS revenue_mom_growth,
    
    CASE 
        WHEN revenue_1y_ago > 0 
        THEN ROUND(((daily_revenue - revenue_1y_ago) / revenue_1y_ago * 100), 2) 
        ELSE NULL 
    END AS revenue_yoy_growth
    
FROM rolling_metrics
WHERE date_day >= CURRENT_DATE - INTERVAL '90 days'
ORDER BY date_day DESC;
```

## Data Exploration

### Exploratory Data Analysis

```sql
-- Customer segmentation analysis
WITH customer_metrics AS (
    SELECT 
        c.customer_id,
        c.customer_name,
        c.registration_date,
        c.customer_segment,
        
        -- Recency (days since last order)
        CURRENT_DATE - MAX(o.order_date) AS recency_days,
        
        -- Frequency (number of orders)
        COUNT(o.order_id) AS frequency,
        
        -- Monetary (total spent)
        SUM(o.total_amount) AS monetary_value,
        
        -- Additional metrics
        AVG(o.total_amount) AS avg_order_value,
        MIN(o.order_date) AS first_order_date,
        MAX(o.order_date) AS last_order_date,
        EXTRACT(DAYS FROM (MAX(o.order_date) - MIN(o.order_date))) AS customer_lifespan_days,
        
        -- Product diversity
        COUNT(DISTINCT p.product_category) AS categories_purchased,
        STRING_AGG(DISTINCT p.product_category, ', ') AS category_list
        
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id
    LEFT JOIN products p ON o.product_id = p.product_id
    WHERE o.order_date >= CURRENT_DATE - INTERVAL '2 years'
    GROUP BY c.customer_id, c.customer_name, c.registration_date, c.customer_segment
),
rfm_scores AS (
    SELECT 
        *,
        -- RFM scoring (1-5 scale)
        NTILE(5) OVER (ORDER BY recency_days DESC) AS recency_score,
        NTILE(5) OVER (ORDER BY frequency) AS frequency_score,
        NTILE(5) OVER (ORDER BY monetary_value) AS monetary_score
    FROM customer_metrics
    WHERE frequency > 0  -- Only customers with orders
),
customer_segments AS (
    SELECT 
        *,
        CASE 
            WHEN recency_score >= 4 AND frequency_score >= 4 AND monetary_score >= 4 THEN 'Champions'
            WHEN recency_score >= 3 AND frequency_score >= 3 AND monetary_score >= 3 THEN 'Loyal Customers'
            WHEN recency_score >= 4 AND frequency_score <= 2 THEN 'New Customers'
            WHEN recency_score >= 3 AND frequency_score >= 2 AND monetary_score >= 2 THEN 'Potential Loyalists'
            WHEN recency_score >= 3 AND frequency_score <= 2 AND monetary_score <= 2 THEN 'Promising'
            WHEN recency_score <= 2 AND frequency_score >= 3 AND monetary_score >= 3 THEN 'At Risk'
            WHEN recency_score <= 2 AND frequency_score >= 2 AND monetary_score >= 2 THEN 'Cannot Lose Them'
            WHEN recency_score <= 2 AND frequency_score <= 2 AND monetary_score >= 3 THEN 'Hibernating'
            ELSE 'Lost Customers'
        END AS rfm_segment,
        
        CONCAT(recency_score, frequency_score, monetary_score) AS rfm_code
    FROM rfm_scores
)
SELECT 
    rfm_segment,
    COUNT(*) AS customer_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS segment_percentage,
    ROUND(AVG(recency_days), 1) AS avg_recency_days,
    ROUND(AVG(frequency), 1) AS avg_frequency,
    ROUND(AVG(monetary_value), 2) AS avg_monetary_value,
    ROUND(SUM(monetary_value), 2) AS total_segment_value,
    ROUND(SUM(monetary_value) * 100.0 / SUM(SUM(monetary_value)) OVER (), 2) AS value_percentage
FROM customer_segments
GROUP BY rfm_segment
ORDER BY total_segment_value DESC;

-- Product performance analysis
WITH product_metrics AS (
    SELECT 
        p.product_id,
        p.product_name,
        p.product_category,
        p.unit_price,
        
        -- Sales metrics
        COUNT(oi.order_item_id) AS units_sold,
        SUM(oi.quantity * oi.unit_price) AS total_revenue,
        AVG(oi.unit_price) AS avg_selling_price,
        COUNT(DISTINCT o.customer_id) AS unique_customers,
        COUNT(DISTINCT o.order_id) AS unique_orders,
        
        -- Time-based metrics
        MIN(o.order_date) AS first_sale_date,
        MAX(o.order_date) AS last_sale_date,
        
        -- Performance indicators
        SUM(oi.quantity * oi.unit_price) / COUNT(DISTINCT o.order_id) AS revenue_per_order,
        COUNT(oi.order_item_id) / COUNT(DISTINCT o.order_id) AS units_per_order
        
    FROM products p
    LEFT JOIN order_items oi ON p.product_id = oi.product_id
    LEFT JOIN orders o ON oi.order_id = o.order_id
    WHERE o.order_date >= CURRENT_DATE - INTERVAL '1 year'
    GROUP BY p.product_id, p.product_name, p.product_category, p.unit_price
),
product_rankings AS (
    SELECT 
        *,
        ROW_NUMBER() OVER (ORDER BY total_revenue DESC) AS revenue_rank,
        ROW_NUMBER() OVER (ORDER BY units_sold DESC) AS volume_rank,
        ROW_NUMBER() OVER (ORDER BY unique_customers DESC) AS customer_reach_rank,
        
        -- Category rankings
        ROW_NUMBER() OVER (PARTITION BY product_category ORDER BY total_revenue DESC) AS category_revenue_rank,
        
        -- Performance classification
        CASE 
            WHEN total_revenue >= PERCENTILE_CONT(0.8) WITHIN GROUP (ORDER BY total_revenue) OVER () THEN 'Star'
            WHEN total_revenue >= PERCENTILE_CONT(0.6) WITHIN GROUP (ORDER BY total_revenue) OVER () THEN 'Performer'
            WHEN total_revenue >= PERCENTILE_CONT(0.4) WITHIN GROUP (ORDER BY total_revenue) OVER () THEN 'Average'
            WHEN total_revenue >= PERCENTILE_CONT(0.2) WITHIN GROUP (ORDER BY total_revenue) OVER () THEN 'Underperformer'
            ELSE 'Poor'
        END AS performance_tier
    FROM product_metrics
    WHERE units_sold > 0
)
SELECT 
    product_category,
    performance_tier,
    COUNT(*) AS product_count,
    SUM(total_revenue) AS category_tier_revenue,
    AVG(units_sold) AS avg_units_sold,
    AVG(unique_customers) AS avg_customer_reach
FROM product_rankings
GROUP BY product_category, performance_tier
ORDER BY product_category, 
         CASE performance_tier 
             WHEN 'Star' THEN 1 
             WHEN 'Performer' THEN 2 
             WHEN 'Average' THEN 3 
             WHEN 'Underperformer' THEN 4 
             ELSE 5 
         END;
```

## ETL & Data Pipeline

### Data Transformation Patterns

```sql
-- Slowly Changing Dimension (SCD) Type 2 implementation
CREATE OR REPLACE PROCEDURE update_customer_dimension()
LANGUAGE plpgsql
AS $$
BEGIN
    -- Insert new records and update existing ones
    INSERT INTO dim_customers (
        customer_id,
        customer_name,
        email,
        phone,
        address,
        customer_segment,
        effective_date,
        expiry_date,
        is_current,
        created_at
    )
    SELECT 
        s.customer_id,
        s.customer_name,
        s.email,
        s.phone,
        s.address,
        s.customer_segment,
        CURRENT_DATE AS effective_date,
        '9999-12-31'::date AS expiry_date,
        TRUE AS is_current,
        CURRENT_TIMESTAMP AS created_at
    FROM staging_customers s
    LEFT JOIN dim_customers d ON s.customer_id = d.customer_id AND d.is_current = TRUE
    WHERE d.customer_id IS NULL  -- New customers
    
    UNION ALL
    
    SELECT 
        s.customer_id,
        s.customer_name,
        s.email,
        s.phone,
        s.address,
        s.customer_segment,
        CURRENT_DATE AS effective_date,
        '9999-12-31'::date AS expiry_date,
        TRUE AS is_current,
        CURRENT_TIMESTAMP AS created_at
    FROM staging_customers s
    INNER JOIN dim_customers d ON s.customer_id = d.customer_id AND d.is_current = TRUE
    WHERE (
        s.customer_name != d.customer_name OR
        s.email != d.email OR
        s.phone != d.phone OR
        s.address != d.address OR
        s.customer_segment != d.customer_segment
    );  -- Changed customers
    
    -- Update expiry dates for changed records
    UPDATE dim_customers 
    SET 
        expiry_date = CURRENT_DATE - INTERVAL '1 day',
        is_current = FALSE,
        updated_at = CURRENT_TIMESTAMP
    WHERE customer_id IN (
        SELECT s.customer_id
        FROM staging_customers s
        INNER JOIN dim_customers d ON s.customer_id = d.customer_id AND d.is_current = TRUE
        WHERE (
            s.customer_name != d.customer_name OR
            s.email != d.email OR
            s.phone != d.phone OR
            s.address != d.address OR
            s.customer_segment != d.customer_segment
        )
    ) AND is_current = TRUE;
    
    -- Log the operation
    INSERT INTO etl_log (
        table_name,
        operation,
        records_processed,
        execution_time,
        status
    ) VALUES (
        'dim_customers',
        'SCD Type 2 Update',
        (SELECT COUNT(*) FROM staging_customers),
        CURRENT_TIMESTAMP,
        'SUCCESS'
    );
    
END;
$$;

-- Data validation framework
CREATE OR REPLACE FUNCTION validate_data_quality(
    table_name TEXT,
    validation_rules JSONB
) RETURNS TABLE(
    rule_name TEXT,
    rule_description TEXT,
    records_failed INTEGER,
    failure_rate DECIMAL,
    status TEXT
) 
LANGUAGE plpgsql
AS $$
DECLARE
    rule JSONB;
    rule_key TEXT;
    sql_query TEXT;
    failed_count INTEGER;
    total_count INTEGER;
BEGIN
    -- Get total record count
    EXECUTE format('SELECT COUNT(*) FROM %I', table_name) INTO total_count;
    
    -- Iterate through validation rules
    FOR rule_key IN SELECT jsonb_object_keys(validation_rules)
    LOOP
        rule := validation_rules->rule_key;
        
        -- Build and execute validation query
        sql_query := format(
            'SELECT COUNT(*) FROM %I WHERE NOT (%s)',
            table_name,
            rule->>'condition'
        );
        
        EXECUTE sql_query INTO failed_count;
        
        -- Return validation result
        RETURN QUERY SELECT 
            rule_key,
            rule->>'description',
            failed_count,
            CASE WHEN total_count > 0 THEN (failed_count::DECIMAL / total_count * 100) ELSE 0 END,
            CASE WHEN failed_count = 0 THEN 'PASS' ELSE 'FAIL' END;
    END LOOP;
END;
$$;

-- Usage example for data validation
SELECT * FROM validate_data_quality(
    'orders',
    '{
        "no_future_dates": {
            "description": "Order dates should not be in the future",
            "condition": "order_date <= CURRENT_DATE"
        },
        "positive_amounts": {
            "description": "Order amounts should be positive",
            "condition": "total_amount > 0"
        },
        "valid_customer_ids": {
            "description": "All orders should have valid customer IDs",
            "condition": "customer_id IS NOT NULL"
        }
    }'::jsonb
);
```

### Incremental Data Processing

```sql
-- Incremental data loading pattern
CREATE OR REPLACE PROCEDURE process_incremental_orders()
LANGUAGE plpgsql
AS $$
DECLARE
    last_processed_date TIMESTAMP;
    current_batch_date TIMESTAMP;
    records_processed INTEGER;
BEGIN
    -- Get last processed timestamp
    SELECT COALESCE(MAX(last_processed_timestamp), '1900-01-01'::timestamp)
    INTO last_processed_date
    FROM etl_watermark
    WHERE table_name = 'orders';
    
    -- Get current batch timestamp
    current_batch_date := CURRENT_TIMESTAMP;
    
    -- Process incremental data
    WITH incremental_orders AS (
        SELECT *
        FROM source_orders
        WHERE updated_at > last_processed_date
        AND updated_at <= current_batch_date
    ),
    enriched_orders AS (
        SELECT 
            io.*,
            c.customer_segment,
            p.product_category,
            CASE 
                WHEN io.order_date >= CURRENT_DATE - INTERVAL '30 days' THEN 'Recent'
                WHEN io.order_date >= CURRENT_DATE - INTERVAL '90 days' THEN 'Medium'
                ELSE 'Old'
            END AS order_age_category
        FROM incremental_orders io
        LEFT JOIN dim_customers c ON io.customer_id = c.customer_id AND c.is_current = TRUE
        LEFT JOIN dim_products p ON io.product_id = p.product_id
    )
    INSERT INTO fact_orders (
        order_id,
        customer_id,
        product_id,
        order_date,
        total_amount,
        customer_segment,
        product_category,
        order_age_category,
        created_at,
        updated_at
    )
    SELECT 
        order_id,
        customer_id,
        product_id,
        order_date,
        total_amount,
        customer_segment,
        product_category,
        order_age_category,
        created_at,
        updated_at
    FROM enriched_orders
    ON CONFLICT (order_id) 
    DO UPDATE SET
        total_amount = EXCLUDED.total_amount,
        customer_segment = EXCLUDED.customer_segment,
        product_category = EXCLUDED.product_category,
        order_age_category = EXCLUDED.order_age_category,
        updated_at = EXCLUDED.updated_at;
    
    GET DIAGNOSTICS records_processed = ROW_COUNT;
    
    -- Update watermark
    INSERT INTO etl_watermark (table_name, last_processed_timestamp, records_processed)
    VALUES ('orders', current_batch_date, records_processed)
    ON CONFLICT (table_name)
    DO UPDATE SET
        last_processed_timestamp = EXCLUDED.last_processed_timestamp,
        records_processed = EXCLUDED.records_processed,
        updated_at = CURRENT_TIMESTAMP;
    
    -- Log the operation
    INSERT INTO etl_log (
        table_name,
        operation,
        records_processed,
        execution_time,
        status,
        details
    ) VALUES (
        'fact_orders',
        'Incremental Load',
        records_processed,
        CURRENT_TIMESTAMP,
        'SUCCESS',
        format('Processed records from %s to %s', last_processed_date, current_batch_date)
    );
    
    RAISE NOTICE 'Processed % records for incremental orders load', records_processed;
END;
$$;
```

---

*This comprehensive SQL analytics guide provides advanced techniques for data analysis, ETL processes, and business intelligence across different database platforms.* 
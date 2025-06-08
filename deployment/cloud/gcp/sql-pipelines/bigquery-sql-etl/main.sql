-- BigQuery SQL ETL Pipeline
-- Comprehensive data transformation with quality checks and error handling

-- Create or replace transformation stored procedure
CREATE OR REPLACE PROCEDURE `${project_id}.${dataset_id}.etl_main_pipeline`()
BEGIN
  DECLARE pipeline_start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP();
  DECLARE rows_processed INT64 DEFAULT 0;
  DECLARE error_message STRING;
  
  -- Error handler
  DECLARE EXIT HANDLER FOR SQLEXCEPTION
  BEGIN
    GET DIAGNOSTICS CONDITION 1
      error_message = MESSAGE_TEXT;
    
    INSERT INTO `${project_id}.${dataset_id}.pipeline_logs` (
      pipeline_name,
      execution_time,
      status,
      error_message,
      rows_processed
    ) VALUES (
      'main_etl_pipeline',
      pipeline_start_time,
      'FAILED',
      error_message,
      rows_processed
    );
    
    RAISE USING MESSAGE = error_message;
  END;
  
  -- Log pipeline start
  INSERT INTO `${project_id}.${dataset_id}.pipeline_logs` (
    pipeline_name,
    execution_time,
    status,
    message
  ) VALUES (
    'main_etl_pipeline',
    pipeline_start_time,
    'STARTED',
    'ETL pipeline execution started'
  );
  
  -- Step 1: Data Quality Checks on Raw Data
  CALL `${project_id}.${dataset_id}.validate_raw_data`();
  
  -- Step 2: Clean and standardize data
  CREATE OR REPLACE TABLE `${project_id}.${dataset_id}.cleaned_data` AS
  WITH cleaned AS (
    SELECT
      -- Data cleaning and standardization
      TRIM(UPPER(name)) as name,
      PARSE_DATE('%Y-%m-%d', date_string) as event_date,
      SAFE_CAST(value AS FLOAT64) as numeric_value,
      
      -- Handle nulls and invalid data
      COALESCE(category, 'UNKNOWN') as category,
      
      -- Standardize geographic data
      CASE 
        WHEN UPPER(country) IN ('US', 'USA', 'UNITED STATES') THEN 'US'
        WHEN UPPER(country) IN ('UK', 'UNITED KINGDOM', 'BRITAIN') THEN 'UK'
        ELSE UPPER(country)
      END as country_code,
      
      -- Create audit fields
      CURRENT_TIMESTAMP() as processed_timestamp,
      '${pipeline_run_id}' as pipeline_run_id,
      
      -- Data lineage
      _FILE_NAME as source_file,
      ROW_NUMBER() OVER (PARTITION BY id ORDER BY _FILE_NAME DESC) as row_rank
      
    FROM `${project_id}.${dataset_id}.raw_data`
    WHERE 
      -- Basic data quality filters
      name IS NOT NULL
      AND LENGTH(name) > 0
      AND date_string IS NOT NULL
      AND SAFE.PARSE_DATE('%Y-%m-%d', date_string) IS NOT NULL
  )
  SELECT * FROM cleaned
  WHERE row_rank = 1;  -- Deduplication
  
  SET rows_processed = (SELECT COUNT(*) FROM `${project_id}.${dataset_id}.cleaned_data`);
  
  -- Step 3: Create aggregated views
  CREATE OR REPLACE TABLE `${project_id}.${dataset_id}.daily_aggregates` AS
  SELECT
    event_date,
    country_code,
    category,
    COUNT(*) as record_count,
    AVG(numeric_value) as avg_value,
    SUM(numeric_value) as total_value,
    MIN(numeric_value) as min_value,
    MAX(numeric_value) as max_value,
    STDDEV(numeric_value) as stddev_value,
    CURRENT_TIMESTAMP() as created_at
  FROM `${project_id}.${dataset_id}.cleaned_data`
  GROUP BY event_date, country_code, category;
  
  -- Step 4: Create time series data for trends
  CREATE OR REPLACE TABLE `${project_id}.${dataset_id}.trend_analysis` AS
  WITH daily_metrics AS (
    SELECT
      event_date,
      SUM(total_value) as daily_total,
      AVG(avg_value) as daily_average,
      COUNT(DISTINCT category) as category_count
    FROM `${project_id}.${dataset_id}.daily_aggregates`
    GROUP BY event_date
  ),
  windowed_metrics AS (
    SELECT
      *,
      AVG(daily_total) OVER (
        ORDER BY event_date 
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
      ) as moving_avg_7d,
      LAG(daily_total, 1) OVER (ORDER BY event_date) as prev_day_total,
      LAG(daily_total, 7) OVER (ORDER BY event_date) as prev_week_total
    FROM daily_metrics
  )
  SELECT
    *,
    SAFE_DIVIDE(daily_total - prev_day_total, prev_day_total) * 100 as day_over_day_pct,
    SAFE_DIVIDE(daily_total - prev_week_total, prev_week_total) * 100 as week_over_week_pct,
    CASE 
      WHEN daily_total > moving_avg_7d * 1.2 THEN 'HIGH'
      WHEN daily_total < moving_avg_7d * 0.8 THEN 'LOW'
      ELSE 'NORMAL'
    END as trend_indicator
  FROM windowed_metrics;
  
  -- Step 5: Data quality summary
  CREATE OR REPLACE TABLE `${project_id}.${dataset_id}.data_quality_summary` AS
  SELECT
    '${pipeline_run_id}' as pipeline_run_id,
    CURRENT_TIMESTAMP() as check_timestamp,
    
    -- Record counts
    (SELECT COUNT(*) FROM `${project_id}.${dataset_id}.raw_data`) as raw_record_count,
    (SELECT COUNT(*) FROM `${project_id}.${dataset_id}.cleaned_data`) as clean_record_count,
    
    -- Data quality metrics
    (SELECT COUNT(*) FROM `${project_id}.${dataset_id}.cleaned_data` WHERE numeric_value IS NULL) as null_value_count,
    (SELECT COUNT(DISTINCT country_code) FROM `${project_id}.${dataset_id}.cleaned_data`) as unique_countries,
    (SELECT COUNT(DISTINCT category) FROM `${project_id}.${dataset_id}.cleaned_data`) as unique_categories,
    
    -- Date range coverage
    (SELECT MIN(event_date) FROM `${project_id}.${dataset_id}.cleaned_data`) as min_date,
    (SELECT MAX(event_date) FROM `${project_id}.${dataset_id}.cleaned_data`) as max_date,
    
    -- Statistical summary
    (SELECT AVG(numeric_value) FROM `${project_id}.${dataset_id}.cleaned_data`) as avg_numeric_value,
    (SELECT STDDEV(numeric_value) FROM `${project_id}.${dataset_id}.cleaned_data`) as stddev_numeric_value;
  
  -- Step 6: Update data freshness tracking
  CREATE OR REPLACE TABLE `${project_id}.${dataset_id}.data_freshness` AS
  SELECT
    table_name,
    CURRENT_TIMESTAMP() as last_updated,
    row_count,
    '${pipeline_run_id}' as pipeline_run_id
  FROM (
    SELECT 'raw_data' as table_name, COUNT(*) as row_count FROM `${project_id}.${dataset_id}.raw_data`
    UNION ALL
    SELECT 'cleaned_data', COUNT(*) FROM `${project_id}.${dataset_id}.cleaned_data`
    UNION ALL
    SELECT 'daily_aggregates', COUNT(*) FROM `${project_id}.${dataset_id}.daily_aggregates`
    UNION ALL
    SELECT 'trend_analysis', COUNT(*) FROM `${project_id}.${dataset_id}.trend_analysis`
  );
  
  -- Log successful completion
  INSERT INTO `${project_id}.${dataset_id}.pipeline_logs` (
    pipeline_name,
    execution_time,
    status,
    message,
    rows_processed,
    duration_seconds
  ) VALUES (
    'main_etl_pipeline',
    pipeline_start_time,
    'COMPLETED',
    'ETL pipeline completed successfully',
    rows_processed,
    TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), pipeline_start_time, SECOND)
  );
  
END;

-- Data validation stored procedure
CREATE OR REPLACE PROCEDURE `${project_id}.${dataset_id}.validate_raw_data`()
BEGIN
  DECLARE validation_errors ARRAY<STRING>;
  DECLARE error_count INT64;
  
  -- Initialize error array
  SET validation_errors = [];
  
  -- Check 1: Ensure raw data table exists and has data
  SET error_count = (
    SELECT COUNT(*) 
    FROM `${project_id}.${dataset_id}.INFORMATION_SCHEMA.TABLES`
    WHERE table_name = 'raw_data'
  );
  
  IF error_count = 0 THEN
    SET validation_errors = ARRAY_CONCAT(validation_errors, ['Raw data table does not exist']);
  END IF;
  
  -- Check 2: Minimum row count
  SET error_count = (SELECT COUNT(*) FROM `${project_id}.${dataset_id}.raw_data`);
  
  IF error_count < 1 THEN
    SET validation_errors = ARRAY_CONCAT(validation_errors, ['Raw data table is empty']);
  END IF;
  
  -- Check 3: Required columns exist and have valid data
  SET error_count = (
    SELECT COUNT(*)
    FROM `${project_id}.${dataset_id}.raw_data`
    WHERE name IS NULL OR LENGTH(TRIM(name)) = 0
  );
  
  IF error_count > 0 THEN
    SET validation_errors = ARRAY_CONCAT(validation_errors, [CONCAT('Found ', CAST(error_count AS STRING), ' records with missing or empty names')]);
  END IF;
  
  -- Check 4: Date format validation
  SET error_count = (
    SELECT COUNT(*)
    FROM `${project_id}.${dataset_id}.raw_data`
    WHERE date_string IS NOT NULL AND SAFE.PARSE_DATE('%Y-%m-%d', date_string) IS NULL
  );
  
  IF error_count > 0 THEN
    SET validation_errors = ARRAY_CONCAT(validation_errors, [CONCAT('Found ', CAST(error_count AS STRING), ' records with invalid date format')]);
  END IF;
  
  -- Log validation results
  INSERT INTO `${project_id}.${dataset_id}.validation_results` (
    validation_timestamp,
    pipeline_run_id,
    validation_errors,
    error_count,
    status
  ) VALUES (
    CURRENT_TIMESTAMP(),
    '${pipeline_run_id}',
    validation_errors,
    ARRAY_LENGTH(validation_errors),
    CASE WHEN ARRAY_LENGTH(validation_errors) = 0 THEN 'PASSED' ELSE 'FAILED' END
  );
  
  -- Raise exception if validation fails
  IF ARRAY_LENGTH(validation_errors) > 0 THEN
    RAISE USING MESSAGE = CONCAT('Data validation failed: ', ARRAY_TO_STRING(validation_errors, '; '));
  END IF;
  
END;

-- Create logging and monitoring tables if they don't exist
CREATE TABLE IF NOT EXISTS `${project_id}.${dataset_id}.pipeline_logs` (
  pipeline_name STRING,
  execution_time TIMESTAMP,
  status STRING,  -- STARTED, COMPLETED, FAILED
  message STRING,
  error_message STRING,
  rows_processed INT64,
  duration_seconds INT64,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
);

CREATE TABLE IF NOT EXISTS `${project_id}.${dataset_id}.validation_results` (
  validation_timestamp TIMESTAMP,
  pipeline_run_id STRING,
  validation_errors ARRAY<STRING>,
  error_count INT64,
  status STRING,  -- PASSED, FAILED
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
);

-- Create data quality monitoring view
CREATE OR REPLACE VIEW `${project_id}.${dataset_id}.pipeline_monitoring` AS
SELECT
  DATE(execution_time) as execution_date,
  pipeline_name,
  COUNT(*) as total_runs,
  COUNTIF(status = 'COMPLETED') as successful_runs,
  COUNTIF(status = 'FAILED') as failed_runs,
  AVG(duration_seconds) as avg_duration_seconds,
  MAX(rows_processed) as max_rows_processed,
  STRING_AGG(DISTINCT error_message, '; ' ORDER BY error_message) as error_messages
FROM `${project_id}.${dataset_id}.pipeline_logs`
WHERE execution_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
GROUP BY DATE(execution_time), pipeline_name
ORDER BY execution_date DESC, pipeline_name; 
# Automated Data Quality Testing

Comprehensive framework for implementing automated data quality testing, monitoring, and alerting in modern data pipelines.

## Table of Contents

- [Overview](#overview)
- [Data Quality Dimensions](#data-quality-dimensions)
- [Testing Frameworks](#testing-frameworks)
- [Implementation Patterns](#implementation-patterns)
- [Monitoring and Alerting](#monitoring-and-alerting)
- [Tool Integration](#tool-integration)

## Overview

Data quality testing has evolved from manual checks to sophisticated automated frameworks that integrate with CI/CD pipelines and provide real-time monitoring.

### Modern Requirements

- **Automated execution** in data pipelines
- **Real-time monitoring** and alerting
- **Scalable testing** for big data workloads
- **Integration** with orchestration tools
- **Observability** and lineage tracking

## Data Quality Dimensions

### 1. Completeness
```python
# Great Expectations for completeness testing
import great_expectations as ge
from great_expectations.dataset import PandasDataset

def test_completeness(df):
    """Test data completeness across multiple dimensions."""
    
    ge_df = PandasDataset(df)
    
    # Test for null values
    completeness_tests = {
        'customer_id_complete': ge_df.expect_column_values_to_not_be_null('customer_id'),
        'email_complete': ge_df.expect_column_values_to_not_be_null('email'),
        'signup_date_complete': ge_df.expect_column_values_to_not_be_null('signup_date')
    }
    
    # Test for completeness percentage
    completeness_tests['overall_completeness'] = ge_df.expect_table_row_count_to_be_between(
        min_value=int(len(df) * 0.95),  # At least 95% complete records
        max_value=len(df)
    )
    
    return completeness_tests

def test_data_freshness(df, timestamp_col='created_at', max_age_hours=24):
    """Test data freshness."""
    
    from datetime import datetime, timedelta
    
    ge_df = PandasDataset(df)
    
    cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
    
    return ge_df.expect_column_max_to_be_between(
        column=timestamp_col,
        min_value=cutoff_time,
        max_value=datetime.now()
    )
```

### 2. Accuracy and Validity
```python
def test_data_accuracy(df):
    """Test data accuracy and validity."""
    
    ge_df = PandasDataset(df)
    
    accuracy_tests = {}
    
    # Email format validation
    accuracy_tests['email_format'] = ge_df.expect_column_values_to_match_regex(
        'email',
        r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    )
    
    # Age range validation
    accuracy_tests['age_range'] = ge_df.expect_column_values_to_be_between(
        'age',
        min_value=0,
        max_value=150
    )
    
    # Phone number format
    accuracy_tests['phone_format'] = ge_df.expect_column_values_to_match_regex(
        'phone',
        r'^\+?1?-?\.?\s?\(?(\d{3})\)?[\s\-\.]?(\d{3})[\s\-\.]?(\d{4})$'
    )
    
    # Currency amounts (positive values)
    accuracy_tests['amount_positive'] = ge_df.expect_column_values_to_be_between(
        'amount',
        min_value=0,
        max_value=1000000
    )
    
    return accuracy_tests

def test_referential_integrity(child_df, parent_df, foreign_key, primary_key):
    """Test referential integrity between datasets."""
    
    orphaned_records = child_df[
        ~child_df[foreign_key].isin(parent_df[primary_key])
    ]
    
    integrity_result = {
        'orphaned_count': len(orphaned_records),
        'total_count': len(child_df),
        'integrity_percentage': (len(child_df) - len(orphaned_records)) / len(child_df) * 100,
        'orphaned_records': orphaned_records.to_dict('records') if len(orphaned_records) < 10 else None
    }
    
    return integrity_result
```

### 3. Consistency and Uniqueness
```python
def test_data_consistency(df):
    """Test data consistency and uniqueness."""
    
    ge_df = PandasDataset(df)
    
    consistency_tests = {}
    
    # Unique primary keys
    consistency_tests['unique_ids'] = ge_df.expect_column_values_to_be_unique('customer_id')
    
    # Consistent data types
    consistency_tests['date_format'] = ge_df.expect_column_values_to_match_strftime_format(
        'signup_date',
        '%Y-%m-%d'
    )
    
    # Consistent categorical values
    expected_statuses = ['active', 'inactive', 'pending', 'suspended']
    consistency_tests['status_values'] = ge_df.expect_column_values_to_be_in_set(
        'status',
        expected_statuses
    )
    
    # Cross-column consistency
    consistency_tests['date_logic'] = ge_df.expect_column_pair_values_A_to_be_greater_than_B(
        'last_login_date',
        'signup_date'
    )
    
    return consistency_tests

def test_statistical_properties(df, baseline_stats=None):
    """Test statistical properties against baseline."""
    
    import numpy as np
    from scipy import stats
    
    current_stats = {
        'mean_age': df['age'].mean(),
        'std_age': df['age'].std(),
        'median_amount': df['amount'].median(),
        'percentile_95_amount': df['amount'].quantile(0.95)
    }
    
    if baseline_stats:
        # Statistical tests for drift detection
        stat_tests = {}
        
        # KS test for distribution changes
        ks_stat, ks_p = stats.ks_2samp(
            baseline_stats['age_distribution'],
            df['age'].values
        )
        
        stat_tests['age_distribution_drift'] = {
            'ks_statistic': ks_stat,
            'p_value': ks_p,
            'drift_detected': ks_p < 0.05
        }
        
        # Mean difference test
        stat_tests['mean_age_change'] = {
            'baseline_mean': baseline_stats['mean_age'],
            'current_mean': current_stats['mean_age'],
            'change_percentage': (current_stats['mean_age'] - baseline_stats['mean_age']) / baseline_stats['mean_age'] * 100
        }
        
        return current_stats, stat_tests
    
    return current_stats
```

## Testing Frameworks

### 1. Great Expectations Integration

```python
# Complete Great Expectations setup
import great_expectations as ge
from great_expectations.core.batch import RuntimeBatchRequest
from great_expectations.checkpoint import SimpleCheckpoint

class DataQualityFramework:
    """Comprehensive data quality testing framework."""
    
    def __init__(self, data_context_path=None):
        if data_context_path:
            self.context = ge.data_context.DataContext(data_context_path)
        else:
            self.context = ge.get_context()
        
        self.expectation_suites = {}
        self.checkpoints = {}
    
    def create_expectation_suite(self, suite_name, df_sample):
        """Create expectation suite from data profiling."""
        
        # Auto-generate expectations from sample data
        suite = self.context.create_expectation_suite(
            expectation_suite_name=suite_name,
            overwrite_existing=True
        )
        
        # Add basic expectations
        suite.expect_table_columns_to_match_ordered_list([
            col for col in df_sample.columns
        ])
        
        for column in df_sample.columns:
            # Non-null expectations for required columns
            if df_sample[column].notna().sum() / len(df_sample) > 0.95:
                suite.expect_column_values_to_not_be_null(column)
            
            # Data type expectations
            if df_sample[column].dtype == 'object':
                # String length expectations
                max_length = df_sample[column].str.len().max()
                suite.expect_column_value_lengths_to_be_between(
                    column, min_value=1, max_value=max_length * 2
                )
            elif df_sample[column].dtype in ['int64', 'float64']:
                # Numeric range expectations
                min_val = df_sample[column].min()
                max_val = df_sample[column].max()
                suite.expect_column_values_to_be_between(
                    column, 
                    min_value=min_val * 0.5,  # Allow some variance
                    max_value=max_val * 1.5
                )
        
        self.context.save_expectation_suite(suite)
        self.expectation_suites[suite_name] = suite
        
        return suite
    
    def create_checkpoint(self, checkpoint_name, suite_name, datasource_name):
        """Create checkpoint for automated validation."""
        
        checkpoint_config = {
            "name": checkpoint_name,
            "config_version": 1.0,
            "template_name": None,
            "module_name": "great_expectations.checkpoint",
            "class_name": "SimpleCheckpoint",
            "run_name_template": f"{checkpoint_name}-%Y%m%d-%H%M%S",
            "expectation_suite_name": suite_name,
            "batch_request": {},
            "action_list": [
                {
                    "name": "store_validation_result",
                    "action": {
                        "class_name": "StoreValidationResultAction",
                    },
                },
                {
                    "name": "update_data_docs",
                    "action": {
                        "class_name": "UpdateDataDocsAction",
                    },
                },
            ],
        }
        
        self.context.add_checkpoint(**checkpoint_config)
        self.checkpoints[checkpoint_name] = checkpoint_config
        
        return checkpoint_config
    
    def run_validation(self, checkpoint_name, batch_request):
        """Run data validation using checkpoint."""
        
        checkpoint = self.context.get_checkpoint(checkpoint_name)
        results = checkpoint.run(batch_request=batch_request)
        
        return results
    
    def generate_data_docs(self):
        """Generate and update data documentation."""
        
        self.context.build_data_docs()
        return self.context.get_docs_sites_urls()

# Usage example
def setup_data_quality_pipeline():
    """Set up complete data quality pipeline."""
    
    # Initialize framework
    dq_framework = DataQualityFramework()
    
    # Create expectation suite from sample data
    sample_df = pd.read_csv('sample_data.csv')
    suite = dq_framework.create_expectation_suite(
        'customer_data_suite', 
        sample_df
    )
    
    # Create checkpoint
    checkpoint = dq_framework.create_checkpoint(
        'daily_customer_validation',
        'customer_data_suite',
        'customer_datasource'
    )
    
    return dq_framework
```

### 2. dbt Testing Integration

```sql
-- dbt data quality tests
-- models/customers.sql
{{ config(materialized='table') }}

SELECT 
    customer_id,
    email,
    first_name,
    last_name,
    signup_date,
    last_login_date,
    status,
    age,
    lifetime_value
FROM {{ source('raw', 'customers') }}
WHERE signup_date >= '2020-01-01'

-- tests/test_customers.yml
version: 2

models:
  - name: customers
    description: "Customer dimension table"
    columns:
      - name: customer_id
        description: "Unique customer identifier"
        tests:
          - unique
          - not_null
      
      - name: email
        description: "Customer email address"
        tests:
          - unique
          - not_null
          - relationships:
              to: ref('email_validation')
              field: email
      
      - name: age
        description: "Customer age"
        tests:
          - not_null
          - accepted_values:
              values: [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
              quote: false
      
      - name: status
        description: "Customer status"
        tests:
          - not_null
          - accepted_values:
              values: ['active', 'inactive', 'pending', 'suspended']
      
      - name: signup_date
        description: "Date customer signed up"
        tests:
          - not_null
          - relationships:
              to: ref('dim_date')
              field: date_key
      
      - name: last_login_date
        description: "Last login date"
        tests:
          - relationships:
              to: ref('dim_date')
              field: date_key

# Custom data quality tests
# macros/data_quality_tests.sql

{% macro test_email_format(model, column_name) %}
  SELECT *
  FROM {{ model }}
  WHERE {{ column_name }} IS NOT NULL
    AND NOT REGEXP_LIKE({{ column_name }}, '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$')
{% endmacro %}

{% macro test_date_range(model, column_name, start_date, end_date) %}
  SELECT *
  FROM {{ model }}
  WHERE {{ column_name }} IS NOT NULL
    AND ({{ column_name }} < '{{ start_date }}' OR {{ column_name }} > '{{ end_date }}')
{% endmacro %}

{% macro test_referential_integrity(model, column_name, ref_model, ref_column) %}
  SELECT {{ column_name }}
  FROM {{ model }}
  WHERE {{ column_name }} IS NOT NULL
    AND {{ column_name }} NOT IN (
      SELECT {{ ref_column }}
      FROM {{ ref_model }}
      WHERE {{ ref_column }} IS NOT NULL
    )
{% endmacro %}

{% macro test_data_freshness(model, timestamp_column, max_age_hours=24) %}
  SELECT *
  FROM {{ model }}
  WHERE {{ timestamp_column }} < CURRENT_TIMESTAMP - INTERVAL {{ max_age_hours }} HOUR
{% endmacro %}

# Usage in schema.yml
version: 2

models:
  - name: customers
    tests:
      - dbt_utils.expression_is_true:
          expression: "signup_date <= last_login_date"
      - test_email_format:
          column_name: email
      - test_date_range:
          column_name: signup_date
          start_date: '2020-01-01'
          end_date: '2030-12-31'
```

### 3. Pandera Schema Validation

```python
# Pandera for advanced schema validation
import pandera as pa
from pandera import Column, DataFrameSchema, Check
from pandera.typing import DataFrame, Series
import pandas as pd

# Define schema with complex validation rules
customer_schema = DataFrameSchema({
    "customer_id": Column(
        pa.Int64,
        checks=[
            Check.greater_than(0),
            Check.unique_values_eq(),
        ],
        nullable=False
    ),
    "email": Column(
        pa.String,
        checks=[
            Check.str_matches(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
            Check.unique_values_eq(),
        ],
        nullable=False
    ),
    "age": Column(
        pa.Int64,
        checks=[
            Check.in_range(18, 100),
        ],
        nullable=True
    ),
    "signup_date": Column(
        pa.DateTime,
        checks=[
            Check.greater_than_or_equal_to(pd.Timestamp('2020-01-01')),
            Check.less_than_or_equal_to(pd.Timestamp.now()),
        ],
        nullable=False
    ),
    "status": Column(
        pa.String,
        checks=[
            Check.isin(['active', 'inactive', 'pending', 'suspended']),
        ],
        nullable=False
    ),
    "lifetime_value": Column(
        pa.Float64,
        checks=[
            Check.greater_than_or_equal_to(0),
            Check.less_than(1000000),
        ],
        nullable=True
    )
}, checks=[
    # Table-level checks
    Check(lambda df: len(df) > 0, error="DataFrame cannot be empty"),
    Check(lambda df: df['signup_date'].dt.date <= df.get('last_login_date', pd.Timestamp.now()).dt.date, 
          error="Signup date must be before last login date"),
])

class DataValidator:
    """Advanced data validation with Pandera."""
    
    def __init__(self):
        self.schemas = {}
        self.validation_results = {}
    
    def register_schema(self, name, schema):
        """Register a schema for validation."""
        self.schemas[name] = schema
    
    def validate_dataframe(self, df, schema_name, lazy=True):
        """Validate DataFrame against registered schema."""
        
        if schema_name not in self.schemas:
            raise ValueError(f"Schema '{schema_name}' not found")
        
        schema = self.schemas[schema_name]
        
        try:
            if lazy:
                # Lazy validation - collect all errors
                validated_df = schema.validate(df, lazy=True)
            else:
                # Fail fast validation
                validated_df = schema.validate(df)
            
            self.validation_results[schema_name] = {
                'status': 'success',
                'row_count': len(validated_df),
                'errors': None
            }
            
            return validated_df
            
        except pa.errors.SchemaErrors as e:
            self.validation_results[schema_name] = {
                'status': 'failed',
                'row_count': len(df),
                'errors': e.failure_cases,
                'error_summary': str(e)
            }
            
            raise e
    
    def batch_validate(self, data_dict):
        """Validate multiple DataFrames in batch."""
        
        results = {}
        
        for name, df in data_dict.items():
            if name in self.schemas:
                try:
                    validated_df = self.validate_dataframe(df, name)
                    results[name] = {
                        'status': 'success',
                        'dataframe': validated_df
                    }
                except pa.errors.SchemaErrors as e:
                    results[name] = {
                        'status': 'failed',
                        'errors': e.failure_cases
                    }
        
        return results
    
    def generate_validation_report(self, output_path):
        """Generate comprehensive validation report."""
        
        report_data = []
        
        for schema_name, result in self.validation_results.items():
            report_data.append({
                'schema': schema_name,
                'status': result['status'],
                'row_count': result['row_count'],
                'error_count': len(result.get('errors', [])) if result.get('errors') is not None else 0
            })
        
        report_df = pd.DataFrame(report_data)
        report_df.to_csv(output_path, index=False)
        
        return report_df

# Usage example
validator = DataValidator()
validator.register_schema('customers', customer_schema)

# Validate customer data
customer_df = pd.read_csv('customers.csv')
try:
    validated_df = validator.validate_dataframe(customer_df, 'customers')
    print("Validation successful!")
except pa.errors.SchemaErrors as e:
    print(f"Validation failed: {e}")
    print("Error details:", e.failure_cases)
```

## Monitoring and Alerting

### Real-time Data Quality Monitoring

```python
# Real-time monitoring with custom metrics
import time
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional
import json

@dataclass
class QualityMetric:
    """Data quality metric definition."""
    name: str
    description: str
    threshold: float
    severity: str  # 'low', 'medium', 'high', 'critical'
    metric_type: str  # 'completeness', 'accuracy', 'consistency', 'timeliness'

@dataclass
class QualityAlert:
    """Data quality alert."""
    metric_name: str
    current_value: float
    threshold: float
    severity: str
    timestamp: datetime
    message: str
    dataset: str

class DataQualityMonitor:
    """Real-time data quality monitoring system."""
    
    def __init__(self, alert_handlers=None):
        self.metrics = {}
        self.alert_handlers = alert_handlers or []
        self.metric_history = {}
        self.logger = logging.getLogger(__name__)
        
    def register_metric(self, metric: QualityMetric):
        """Register a quality metric for monitoring."""
        self.metrics[metric.name] = metric
        self.metric_history[metric.name] = []
    
    def calculate_metrics(self, df, dataset_name):
        """Calculate all registered metrics for a dataset."""
        
        results = {}
        
        for metric_name, metric in self.metrics.items():
            try:
                if metric.metric_type == 'completeness':
                    value = self._calculate_completeness(df)
                elif metric.metric_type == 'accuracy':
                    value = self._calculate_accuracy(df)
                elif metric.metric_type == 'consistency':
                    value = self._calculate_consistency(df)
                elif metric.metric_type == 'timeliness':
                    value = self._calculate_timeliness(df)
                else:
                    continue
                
                results[metric_name] = value
                
                # Store in history
                self.metric_history[metric_name].append({
                    'timestamp': datetime.now(),
                    'value': value,
                    'dataset': dataset_name
                })
                
                # Check for alerts
                if value < metric.threshold:
                    alert = QualityAlert(
                        metric_name=metric_name,
                        current_value=value,
                        threshold=metric.threshold,
                        severity=metric.severity,
                        timestamp=datetime.now(),
                        message=f"{metric.description} below threshold: {value:.2f} < {metric.threshold}",
                        dataset=dataset_name
                    )
                    self._trigger_alert(alert)
                
            except Exception as e:
                self.logger.error(f"Error calculating metric {metric_name}: {str(e)}")
        
        return results
    
    def _calculate_completeness(self, df):
        """Calculate data completeness percentage."""
        total_cells = df.size
        non_null_cells = df.count().sum()
        return (non_null_cells / total_cells) * 100
    
    def _calculate_accuracy(self, df):
        """Calculate data accuracy based on validation rules."""
        total_rows = len(df)
        valid_rows = 0
        
        # Example accuracy checks
        if 'email' in df.columns:
            valid_emails = df['email'].str.match(
                r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
                na=False
            ).sum()
            valid_rows += valid_emails
        
        if 'age' in df.columns:
            valid_ages = df['age'].between(0, 150, inclusive='both').sum()
            valid_rows += valid_ages
        
        return (valid_rows / (total_rows * 2)) * 100  # Adjust based on number of checks
    
    def _calculate_consistency(self, df):
        """Calculate data consistency percentage."""
        consistency_score = 100  # Start with perfect score
        
        # Check for duplicate primary keys
        if 'id' in df.columns:
            duplicates = df['id'].duplicated().sum()
            consistency_score -= (duplicates / len(df)) * 50
        
        # Check date logic consistency
        if 'signup_date' in df.columns and 'last_login_date' in df.columns:
            invalid_dates = (df['last_login_date'] < df['signup_date']).sum()
            consistency_score -= (invalid_dates / len(df)) * 50
        
        return max(0, consistency_score)
    
    def _calculate_timeliness(self, df):
        """Calculate data timeliness score."""
        if 'created_at' in df.columns:
            cutoff = datetime.now() - timedelta(hours=24)
            recent_data = df['created_at'] > cutoff
            return (recent_data.sum() / len(df)) * 100
        return 100  # No timestamp column, assume timely
    
    def _trigger_alert(self, alert: QualityAlert):
        """Trigger alert through registered handlers."""
        for handler in self.alert_handlers:
            try:
                handler.send_alert(alert)
            except Exception as e:
                self.logger.error(f"Error sending alert: {str(e)}")
    
    def get_metric_trends(self, metric_name, hours=24):
        """Get metric trends over time."""
        if metric_name not in self.metric_history:
            return []
        
        cutoff = datetime.now() - timedelta(hours=hours)
        recent_history = [
            record for record in self.metric_history[metric_name]
            if record['timestamp'] > cutoff
        ]
        
        return recent_history
    
    def generate_dashboard_data(self):
        """Generate data for monitoring dashboard."""
        dashboard_data = {
            'metrics_summary': {},
            'recent_alerts': [],
            'trends': {}
        }
        
        # Current metric values
        for metric_name in self.metrics:
            if metric_name in self.metric_history and self.metric_history[metric_name]:
                latest = self.metric_history[metric_name][-1]
                dashboard_data['metrics_summary'][metric_name] = {
                    'current_value': latest['value'],
                    'threshold': self.metrics[metric_name].threshold,
                    'status': 'healthy' if latest['value'] >= self.metrics[metric_name].threshold else 'alert'
                }
        
        # Recent trends
        for metric_name in self.metrics:
            dashboard_data['trends'][metric_name] = self.get_metric_trends(metric_name, hours=24)
        
        return dashboard_data

class SlackAlertHandler:
    """Slack alert handler for data quality alerts."""
    
    def __init__(self, webhook_url, channel='#data-quality'):
        self.webhook_url = webhook_url
        self.channel = channel
    
    def send_alert(self, alert: QualityAlert):
        """Send alert to Slack channel."""
        import requests
        
        color = {
            'low': '#36a64f',      # Green
            'medium': '#ff9500',   # Orange
            'high': '#ff0000',     # Red
            'critical': '#8B0000'  # Dark Red
        }.get(alert.severity, '#36a64f')
        
        message = {
            'channel': self.channel,
            'attachments': [{
                'color': color,
                'title': f'Data Quality Alert - {alert.severity.upper()}',
                'fields': [
                    {
                        'title': 'Metric',
                        'value': alert.metric_name,
                        'short': True
                    },
                    {
                        'title': 'Dataset',
                        'value': alert.dataset,
                        'short': True
                    },
                    {
                        'title': 'Current Value',
                        'value': f'{alert.current_value:.2f}',
                        'short': True
                    },
                    {
                        'title': 'Threshold',
                        'value': f'{alert.threshold:.2f}',
                        'short': True
                    },
                    {
                        'title': 'Message',
                        'value': alert.message,
                        'short': False
                    }
                ],
                'timestamp': int(alert.timestamp.timestamp())
            }]
        }
        
        response = requests.post(self.webhook_url, json=message)
        response.raise_for_status()

# Usage example
def setup_monitoring():
    """Set up comprehensive data quality monitoring."""
    
    # Initialize alert handlers
    slack_handler = SlackAlertHandler('https://hooks.slack.com/services/...')
    
    # Initialize monitor
    monitor = DataQualityMonitor(alert_handlers=[slack_handler])
    
    # Register metrics
    monitor.register_metric(QualityMetric(
        name='completeness',
        description='Data completeness percentage',
        threshold=95.0,
        severity='high',
        metric_type='completeness'
    ))
    
    monitor.register_metric(QualityMetric(
        name='accuracy',
        description='Data accuracy percentage',
        threshold=90.0,
        severity='medium',
        metric_type='accuracy'
    ))
    
    monitor.register_metric(QualityMetric(
        name='consistency',
        description='Data consistency percentage',
        threshold=98.0,
        severity='high',
        metric_type='consistency'
    ))
    
    monitor.register_metric(QualityMetric(
        name='timeliness',
        description='Data timeliness percentage',
        threshold=80.0,
        severity='medium',
        metric_type='timeliness'
    ))
    
    return monitor

# Example monitoring pipeline
def run_quality_monitoring():
    """Run data quality monitoring pipeline."""
    
    monitor = setup_monitoring()
    
    # Load data
    df = pd.read_csv('daily_data.csv')
    
    # Calculate metrics
    results = monitor.calculate_metrics(df, 'daily_customer_data')
    
    # Generate dashboard data
    dashboard_data = monitor.generate_dashboard_data()
    
    print("Quality monitoring results:", results)
    return results, dashboard_data
```

## Tool Integration

### Airflow Integration

```python
# Airflow DAG for data quality monitoring
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.email import EmailOperator
from airflow.sensors.s3_key_sensor import S3KeySensor
from datetime import datetime, timedelta

def run_data_quality_checks(**context):
    """Run comprehensive data quality checks."""
    
    import pandas as pd
    from data_quality_framework import DataQualityMonitor, setup_monitoring
    
    # Get execution date
    execution_date = context['execution_date']
    
    # Load data for the execution date
    data_path = f"s3://data-bucket/daily_data/{execution_date.strftime('%Y/%m/%d')}/data.csv"
    df = pd.read_csv(data_path)
    
    # Run quality checks
    monitor = setup_monitoring()
    results = monitor.calculate_metrics(df, f"daily_data_{execution_date.strftime('%Y%m%d')}")
    
    # Store results for downstream tasks
    context['task_instance'].xcom_push(key='quality_results', value=results)
    
    return results

def validate_quality_thresholds(**context):
    """Validate that quality metrics meet thresholds."""
    
    results = context['task_instance'].xcom_pull(key='quality_results')
    
    failed_metrics = []
    for metric, value in results.items():
        if metric == 'completeness' and value < 95:
            failed_metrics.append(f"Completeness: {value:.2f}% < 95%")
        elif metric == 'accuracy' and value < 90:
            failed_metrics.append(f"Accuracy: {value:.2f}% < 90%")
    
    if failed_metrics:
        raise ValueError(f"Quality checks failed: {', '.join(failed_metrics)}")
    
    return "All quality checks passed"

# Define DAG
default_args = {
    'owner': 'data-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'email': ['data-team@company.com']
}

dag = DAG(
    'data_quality_monitoring',
    default_args=default_args,
    description='Daily data quality monitoring pipeline',
    schedule_interval='0 6 * * *',  # Daily at 6 AM
    catchup=False,
    tags=['data-quality', 'monitoring'],
)

# Wait for data to arrive
wait_for_data = S3KeySensor(
    task_id='wait_for_daily_data',
    bucket_name='data-bucket',
    bucket_key='daily_data/{{ ds }}/data.csv',
    wildcard_match=True,
    timeout=3600,  # 1 hour timeout
    poke_interval=300,  # Check every 5 minutes
    dag=dag,
)

# Run quality checks
quality_checks = PythonOperator(
    task_id='run_quality_checks',
    python_callable=run_data_quality_checks,
    dag=dag,
)

# Validate thresholds
validate_thresholds = PythonOperator(
    task_id='validate_thresholds',
    python_callable=validate_quality_thresholds,
    dag=dag,
)

# Send success notification
success_notification = EmailOperator(
    task_id='send_success_email',
    to=['data-team@company.com'],
    subject='Data Quality Check - Success',
    html_content='''
    <h3>Data Quality Check Completed Successfully</h3>
    <p>All quality metrics have passed their thresholds for {{ ds }}.</p>
    <p>Check the <a href="{{ params.dashboard_url }}">Quality Dashboard</a> for details.</p>
    ''',
    params={'dashboard_url': 'https://company.com/data-quality-dashboard'},
    dag=dag,
)

# Define task dependencies
wait_for_data >> quality_checks >> validate_thresholds >> success_notification
```

## Related Resources

- [Data Pipeline Orchestration](../../devops/modern-data-orchestration.md)
- [Monitoring and Observability](../monitoring/)
- [Infrastructure Testing](../../deployment/infrastructure/)
- [ML Model Validation](../../machine-learning/)

## Contributing

When adding new data quality patterns:
1. Include performance benchmarks
2. Document integration requirements
3. Provide example datasets
4. Add monitoring setup
5. Include alerting configurations 
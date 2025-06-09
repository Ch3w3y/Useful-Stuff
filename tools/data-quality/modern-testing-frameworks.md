# Modern Data Quality Testing Frameworks

This guide covers the latest tools and frameworks for ensuring data quality in modern data pipelines, including automated testing, monitoring, and validation approaches.

## Table of Contents

- [Overview](#overview)
- [Framework Comparison](#framework-comparison)
- [Great Expectations](#great-expectations)
- [dbt Testing](#dbt-testing)
- [Pandera](#pandera)
- [Custom Solutions](#custom-solutions)
- [CI/CD Integration](#cicd-integration)

## Overview

Data quality testing has evolved from manual checks to sophisticated automated frameworks. Modern approaches include:

- **Schema validation** and type checking
- **Statistical monitoring** for data drift
- **Real-time quality** monitoring
- **Automated alerting** and remediation
- **Integration** with data pipelines

## Framework Comparison

| Framework | Best For | Language | Real-time | Schema Validation | Statistical Tests |
|-----------|----------|----------|-----------|-------------------|-------------------|
| Great Expectations | Enterprise pipelines | Python | ✅ | ✅ | ✅ |
| dbt Tests | Transformation layer | SQL | ❌ | ✅ | ✅ |
| Pandera | Pandas workflows | Python | ❌ | ✅ | ✅ |
| Soda | Data observability | SQL/Python | ✅ | ✅ | ✅ |
| Monte Carlo | ML monitoring | Python | ✅ | ✅ | ✅ |

## Great Expectations

### Basic Setup

```python
import great_expectations as ge
from great_expectations.dataset import PandasDataset
import pandas as pd

# Initialize context
context = ge.get_context()

# Create expectation suite
suite = context.create_expectation_suite(
    expectation_suite_name="customer_data_suite",
    overwrite_existing=True
)

# Load data and create expectations
df = pd.read_csv("customer_data.csv")
batch = context.get_batch(
    {"path": "customer_data.csv", "datasource": "pandas_datasource"},
    suite
)

# Add expectations
batch.expect_column_values_to_not_be_null("customer_id")
batch.expect_column_values_to_be_unique("customer_id")
batch.expect_column_values_to_match_regex("email", r'^[^@]+@[^@]+\.[^@]+$')
batch.expect_column_values_to_be_between("age", min_value=0, max_value=120)

# Save suite
context.save_expectation_suite(suite)
```

### Advanced Expectations

```python
# Custom expectations for complex business rules
class CustomExpectations:
    @staticmethod
    def expect_revenue_consistency(batch, tolerance=0.1):
        """Expect revenue to be consistent with historical patterns."""
        current_revenue = batch.get_column_sum("revenue")
        historical_avg = 1000000  # Would come from historical data
        
        variance = abs(current_revenue - historical_avg) / historical_avg
        
        return {
            "success": variance <= tolerance,
            "result": {
                "current_revenue": current_revenue,
                "historical_average": historical_avg,
                "variance": variance,
                "tolerance": tolerance
            }
        }
    
    @staticmethod
    def expect_geographic_distribution(batch, expected_regions):
        """Expect data to cover all expected geographic regions."""
        actual_regions = set(batch.get_column_unique_values("region"))
        missing_regions = set(expected_regions) - actual_regions
        
        return {
            "success": len(missing_regions) == 0,
            "result": {
                "expected_regions": expected_regions,
                "actual_regions": list(actual_regions),
                "missing_regions": list(missing_regions)
            }
        }

# Register custom expectations
from great_expectations.execution_engine import PandasExecutionEngine

def register_custom_expectations():
    @ge.expectations.expectation(["column"])
    def expect_column_values_to_be_valid_json(
        self,
        column: str,
        mostly: float = 1
    ):
        """Expect column values to be valid JSON."""
        import json
        
        def is_valid_json(value):
            try:
                json.loads(value)
                return True
            except (ValueError, TypeError):
                return False
        
        return self.expect_column_values_to_satisfy_json_schema(
            column=column,
            json_schema={"type": "object"},
            mostly=mostly
        )
```

## dbt Testing

### Built-in Tests

```sql
-- models/customers.sql
SELECT 
    customer_id,
    email,
    first_name,
    last_name,
    signup_date,
    status
FROM {{ source('raw', 'customers') }}

-- schema.yml
version: 2

models:
  - name: customers
    columns:
      - name: customer_id
        tests:
          - unique
          - not_null
      - name: email
        tests:
          - unique
          - not_null
      - name: status
        tests:
          - accepted_values:
              values: ['active', 'inactive', 'pending']
```

### Custom dbt Tests

```sql
-- tests/assert_email_format.sql
SELECT email
FROM {{ ref('customers') }}
WHERE email IS NOT NULL
  AND NOT REGEXP_LIKE(email, r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')

-- tests/assert_revenue_positive.sql
SELECT customer_id, total_revenue
FROM {{ ref('customer_metrics') }}
WHERE total_revenue < 0

-- macros/test_helpers.sql
{% macro test_date_range(model, column_name, start_date, end_date) %}
  SELECT *
  FROM {{ model }}
  WHERE {{ column_name }} < '{{ start_date }}'
     OR {{ column_name }} > '{{ end_date }}'
{% endmacro %}

{% macro test_referential_integrity(child_model, parent_model, foreign_key, primary_key) %}
  SELECT {{ foreign_key }}
  FROM {{ child_model }}
  WHERE {{ foreign_key }} IS NOT NULL
    AND {{ foreign_key }} NOT IN (
      SELECT {{ primary_key }}
      FROM {{ parent_model }}
      WHERE {{ primary_key }} IS NOT NULL
    )
{% endmacro %}
```

## Pandera

### Schema Definition

```python
import pandera as pa
from pandera import Column, DataFrameSchema, Check
from pandera.typing import DataFrame, Series
import pandas as pd

# Define comprehensive schema
customer_schema = DataFrameSchema({
    "customer_id": Column(
        pa.Int64,
        checks=[Check.greater_than(0), Check.unique_values_eq()],
        nullable=False
    ),
    "email": Column(
        pa.String,
        checks=[
            Check.str_matches(r'^[^@]+@[^@]+\.[^@]+$'),
            Check.unique_values_eq()
        ],
        nullable=False
    ),
    "age": Column(
        pa.Int64,
        checks=[Check.in_range(0, 120)],
        nullable=True
    ),
    "signup_date": Column(
        pa.DateTime,
        checks=[
            Check.greater_than_or_equal_to(pd.Timestamp('2020-01-01')),
            Check.less_than_or_equal_to(pd.Timestamp.now())
        ],
        nullable=False
    ),
    "status": Column(
        pa.String,
        checks=[Check.isin(['active', 'inactive', 'pending'])],
        nullable=False
    )
}, strict=True)

# Custom validation functions
@pa.check_output(customer_schema)
def validate_customer_data(df: DataFrame) -> DataFrame:
    """Validate customer data with additional business logic."""
    
    # Check for duplicate emails
    if df['email'].duplicated().any():
        raise ValueError("Duplicate emails found")
    
    # Check age distribution
    if df['age'].mean() < 18:
        raise ValueError("Average age too low")
    
    return df

# Schema inheritance for related tables
order_schema = customer_schema.add_columns({
    "order_id": Column(pa.String, checks=[Check.unique_values_eq()]),
    "order_amount": Column(pa.Float64, checks=[Check.greater_than(0)]),
    "order_date": Column(pa.DateTime)
}).remove_columns(["age", "signup_date"])
```

### Advanced Validation

```python
class DataQualityValidator:
    """Advanced data quality validation with Pandera."""
    
    def __init__(self):
        self.schemas = {}
        self.validation_history = []
    
    def register_schema(self, name: str, schema: DataFrameSchema):
        """Register a schema for validation."""
        self.schemas[name] = schema
    
    def validate_with_sampling(self, df: pd.DataFrame, schema_name: str, sample_size: int = 1000):
        """Validate large datasets using sampling."""
        if len(df) > sample_size:
            sample_df = df.sample(n=sample_size, random_state=42)
            # Validate sample first
            schema = self.schemas[schema_name]
            schema.validate(sample_df)
            
            # If sample passes, validate full dataset
            return schema.validate(df, lazy=True)
        else:
            return self.schemas[schema_name].validate(df)
    
    def batch_validate(self, data_dict: dict):
        """Validate multiple datasets in batch."""
        results = {}
        
        for name, df in data_dict.items():
            if name in self.schemas:
                try:
                    validated_df = self.schemas[name].validate(df, lazy=True)
                    results[name] = {
                        'status': 'success',
                        'row_count': len(validated_df),
                        'errors': []
                    }
                except pa.errors.SchemaErrors as e:
                    results[name] = {
                        'status': 'failed',
                        'row_count': len(df),
                        'errors': e.failure_cases.to_dict('records')
                    }
        
        self.validation_history.append({
            'timestamp': pd.Timestamp.now(),
            'results': results
        })
        
        return results
    
    def generate_data_profile(self, df: pd.DataFrame):
        """Generate data profile for schema creation."""
        profile = {}
        
        for column in df.columns:
            col_data = df[column]
            
            profile[column] = {
                'dtype': str(col_data.dtype),
                'null_count': col_data.isnull().sum(),
                'null_percentage': col_data.isnull().mean() * 100,
                'unique_count': col_data.nunique(),
                'unique_percentage': col_data.nunique() / len(col_data) * 100
            }
            
            if col_data.dtype in ['int64', 'float64']:
                profile[column].update({
                    'min': col_data.min(),
                    'max': col_data.max(),
                    'mean': col_data.mean(),
                    'std': col_data.std(),
                    'quantiles': col_data.quantile([0.25, 0.5, 0.75]).to_dict()
                })
            
            elif col_data.dtype == 'object':
                profile[column].update({
                    'min_length': col_data.str.len().min(),
                    'max_length': col_data.str.len().max(),
                    'mean_length': col_data.str.len().mean(),
                    'top_values': col_data.value_counts().head(5).to_dict()
                })
        
        return profile
```

## Custom Solutions

### Real-time Quality Monitoring

```python
import asyncio
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Callable, Any
import pandas as pd

@dataclass
class QualityMetric:
    name: str
    value: float
    threshold: float
    status: str  # 'healthy', 'warning', 'critical'
    timestamp: datetime
    metadata: Dict[str, Any]

class RealTimeQualityMonitor:
    """Real-time data quality monitoring system."""
    
    def __init__(self):
        self.metrics = {}
        self.alert_callbacks = []
        self.metric_history = {}
        self.running = False
    
    def register_metric(self, name: str, calculator: Callable, threshold: float, 
                       warning_threshold: float = None):
        """Register a quality metric."""
        self.metrics[name] = {
            'calculator': calculator,
            'threshold': threshold,
            'warning_threshold': warning_threshold or threshold * 0.9,
            'last_check': None
        }
        self.metric_history[name] = []
    
    def add_alert_callback(self, callback: Callable):
        """Add callback for quality alerts."""
        self.alert_callbacks.append(callback)
    
    async def start_monitoring(self, check_interval: int = 60):
        """Start real-time monitoring."""
        self.running = True
        
        while self.running:
            await self.run_quality_checks()
            await asyncio.sleep(check_interval)
    
    def stop_monitoring(self):
        """Stop monitoring."""
        self.running = False
    
    async def run_quality_checks(self):
        """Run all registered quality checks."""
        for metric_name, config in self.metrics.items():
            try:
                value = await self._run_metric_calculation(config['calculator'])
                
                # Determine status
                if value >= config['threshold']:
                    status = 'healthy'
                elif value >= config['warning_threshold']:
                    status = 'warning'
                else:
                    status = 'critical'
                
                metric = QualityMetric(
                    name=metric_name,
                    value=value,
                    threshold=config['threshold'],
                    status=status,
                    timestamp=datetime.now(),
                    metadata={}
                )
                
                # Store metric
                self.metric_history[metric_name].append(metric)
                
                # Trigger alerts if needed
                if status in ['warning', 'critical']:
                    await self._trigger_alerts(metric)
                
                config['last_check'] = datetime.now()
                
            except Exception as e:
                print(f"Error calculating metric {metric_name}: {e}")
    
    async def _run_metric_calculation(self, calculator: Callable) -> float:
        """Run metric calculation asynchronously."""
        if asyncio.iscoroutinefunction(calculator):
            return await calculator()
        else:
            return calculator()
    
    async def _trigger_alerts(self, metric: QualityMetric):
        """Trigger alerts for quality issues."""
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(metric)
                else:
                    callback(metric)
            except Exception as e:
                print(f"Error in alert callback: {e}")
    
    def get_metric_trends(self, metric_name: str, hours: int = 24) -> List[QualityMetric]:
        """Get metric trends over time."""
        if metric_name not in self.metric_history:
            return []
        
        cutoff = datetime.now() - timedelta(hours=hours)
        return [
            metric for metric in self.metric_history[metric_name]
            if metric.timestamp > cutoff
        ]
    
    def export_metrics(self) -> Dict:
        """Export all metrics for external systems."""
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'metrics': {}
        }
        
        for metric_name, history in self.metric_history.items():
            if history:
                latest = history[-1]
                export_data['metrics'][metric_name] = asdict(latest)
        
        return export_data

# Example metric calculators
async def calculate_completeness_metric():
    """Calculate data completeness."""
    # Would connect to actual data source
    df = pd.read_csv('latest_data.csv')
    total_cells = df.size
    non_null_cells = df.count().sum()
    return (non_null_cells / total_cells) * 100

def calculate_schema_compliance():
    """Calculate schema compliance percentage."""
    # Mock implementation
    return 98.5

async def slack_alert_callback(metric: QualityMetric):
    """Send alert to Slack."""
    import aiohttp
    
    webhook_url = "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"
    
    message = {
        "text": f"Data Quality Alert: {metric.name}",
        "attachments": [{
            "color": "danger" if metric.status == "critical" else "warning",
            "fields": [
                {"title": "Metric", "value": metric.name, "short": True},
                {"title": "Value", "value": f"{metric.value:.2f}", "short": True},
                {"title": "Threshold", "value": f"{metric.threshold:.2f}", "short": True},
                {"title": "Status", "value": metric.status.upper(), "short": True}
            ]
        }]
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(webhook_url, json=message) as response:
            if response.status != 200:
                print(f"Failed to send Slack alert: {response.status}")

# Usage example
async def setup_monitoring():
    monitor = RealTimeQualityMonitor()
    
    # Register metrics
    monitor.register_metric(
        'completeness',
        calculate_completeness_metric,
        threshold=95.0,
        warning_threshold=90.0
    )
    
    monitor.register_metric(
        'schema_compliance',
        calculate_schema_compliance,
        threshold=99.0,
        warning_threshold=95.0
    )
    
    # Add alert callbacks
    monitor.add_alert_callback(slack_alert_callback)
    
    # Start monitoring
    await monitor.start_monitoring(check_interval=30)  # Check every 30 seconds
```

## CI/CD Integration

### GitHub Actions Workflow

```yaml
# .github/workflows/data-quality.yml
name: Data Quality Checks

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 6 * * *'  # Daily at 6 AM

jobs:
  data-quality:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:13
        env:
          POSTGRES_PASSWORD: postgres
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Run Great Expectations validation
      run: |
        great_expectations checkpoint run customer_data_checkpoint
      env:
        DATABASE_URL: postgresql://postgres:postgres@localhost:5432/testdb
    
    - name: Run dbt tests
      run: |
        dbt deps
        dbt test --profiles-dir ./profiles
      env:
        DBT_PROFILES_DIR: ./profiles
    
    - name: Run custom quality checks
      run: |
        python scripts/quality_checks.py
    
    - name: Upload quality reports
      uses: actions/upload-artifact@v3
      with:
        name: quality-reports
        path: |
          great_expectations/uncommitted/data_docs/
          target/
          reports/
    
    - name: Notify on failure
      if: failure()
      uses: 8398a7/action-slack@v3
      with:
        status: failure
        channel: '#data-quality'
        webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

### Kubernetes Quality Jobs

```yaml
# k8s/quality-check-job.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: data-quality-check
  labels:
    app: data-quality
spec:
  template:
    spec:
      containers:
      - name: quality-checker
        image: data-quality:latest
        command: ["python", "quality_pipeline.py"]
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: url
        - name: SLACK_WEBHOOK
          valueFrom:
            secretKeyRef:
              name: alert-secret
              key: slack_webhook
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1"
        volumeMounts:
        - name: config-volume
          mountPath: /app/config
      volumes:
      - name: config-volume
        configMap:
          name: quality-config
      restartPolicy: Never
  backoffLimit: 3

---
apiVersion: v1
kind: ConfigMap
metadata:
  name: quality-config
data:
  expectations.json: |
    {
      "customer_completeness_threshold": 95.0,
      "email_format_compliance": 99.0,
      "data_freshness_hours": 24
    }

---
apiVersion: batch/v1
kind: CronJob
metadata:
  name: daily-quality-check
spec:
  schedule: "0 6 * * *"  # Daily at 6 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: quality-checker
            image: data-quality:latest
            command: ["python", "daily_quality_pipeline.py"]
          restartPolicy: OnFailure
```

## Related Resources

- [Data Pipeline Orchestration](../../devops/modern-data-orchestration.md)
- [Monitoring and Observability](../monitoring/)
- [ML Model Validation](../../machine-learning/)
- [Infrastructure Testing](../../deployment/infrastructure/)

## Contributing

When adding new quality testing patterns:
1. Include performance benchmarks
2. Document integration requirements
3. Provide example datasets
4. Add monitoring setup
5. Include alerting configurations 
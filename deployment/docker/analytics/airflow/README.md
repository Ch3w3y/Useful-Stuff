# Apache Airflow Docker Deployment

Comprehensive Docker deployment configurations for Apache Airflow, designed for data pipeline orchestration, workflow management, and analytics automation. This setup provides production-ready Airflow environments with scalability, monitoring, and security features.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Configuration Files](#configuration-files)
- [DAG Examples](#dag-examples)
- [Production Setup](#production-setup)
- [Monitoring & Logging](#monitoring--logging)
- [Security Configuration](#security-configuration)
- [Scaling & Performance](#scaling--performance)
- [Troubleshooting](#troubleshooting)

## Overview

Apache Airflow is a platform to programmatically author, schedule, and monitor workflows. This Docker setup provides:

- **Multi-container Architecture**: Separate containers for webserver, scheduler, worker, and database
- **Scalable Workers**: Support for CeleryExecutor with Redis/RabbitMQ
- **Production Ready**: Security, monitoring, and backup configurations
- **Analytics Focus**: Pre-configured for data science and analytics workflows

### Key Features
- Docker Compose orchestration
- PostgreSQL backend database
- Redis for message broker
- Flower for monitoring Celery workers
- Custom DAG examples for analytics workflows
- Integrated logging and monitoring
- Security best practices

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Airflow Web   │    │  Airflow        │    │   Airflow       │
│   Server        │    │  Scheduler      │    │   Worker(s)     │
│   (Port 8080)   │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
         ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
         │   PostgreSQL    │    │     Redis       │    │     Flower      │
         │   Database      │    │   Message       │    │   Monitoring    │
         │                 │    │   Broker        │    │   (Port 5555)   │
         └─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Quick Start

### Prerequisites
```bash
# Install Docker and Docker Compose
sudo apt-get update
sudo apt-get install docker.io docker-compose

# Create project directory
mkdir airflow-docker && cd airflow-docker
```

### Basic Setup
```bash
# Download configuration files
curl -LfO 'https://airflow.apache.org/docs/apache-airflow/2.7.0/docker-compose.yaml'

# Create required directories
mkdir -p ./dags ./logs ./plugins ./config

# Set Airflow UID
echo -e "AIRFLOW_UID=$(id -u)" > .env

# Initialize database
docker-compose up airflow-init

# Start services
docker-compose up -d
```

### Access Points
- **Airflow Web UI**: http://localhost:8080 (admin/admin)
- **Flower Monitoring**: http://localhost:5555
- **PostgreSQL**: localhost:5432 (airflow/airflow)
- **Redis**: localhost:6379

## Configuration Files

### Docker Compose Configuration
```yaml
# docker-compose.yml
version: '3.8'

x-airflow-common:
  &airflow-common
  image: apache/airflow:2.7.0-python3.9
  environment:
    &airflow-common-env
    AIRFLOW__CORE__EXECUTOR: CeleryExecutor
    AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@postgres/airflow
    AIRFLOW__CELERY__RESULT_BACKEND: db+postgresql://airflow:airflow@postgres/airflow
    AIRFLOW__CELERY__BROKER_URL: redis://:@redis:6379/0
    AIRFLOW__CORE__FERNET_KEY: ''
    AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION: 'true'
    AIRFLOW__CORE__LOAD_EXAMPLES: 'false'
    AIRFLOW__API__AUTH_BACKENDS: 'airflow.api.auth.backend.basic_auth,airflow.api.auth.backend.session'
    AIRFLOW__SCHEDULER__ENABLE_HEALTH_CHECK: 'true'
    # Analytics-specific configurations
    AIRFLOW__CORE__DEFAULT_TIMEZONE: UTC
    AIRFLOW__WEBSERVER__EXPOSE_CONFIG: 'true'
    AIRFLOW__WEBSERVER__RBAC: 'true'
    AIRFLOW__CORE__REMOTE_LOGGING: 'false'
    AIRFLOW__LOGGING__LOGGING_LEVEL: INFO
    # Data processing configurations
    AIRFLOW__CORE__PARALLELISM: 32
    AIRFLOW__CORE__DAG_CONCURRENCY: 16
    AIRFLOW__CORE__MAX_ACTIVE_RUNS_PER_DAG: 16
    AIRFLOW__CELERY__WORKER_CONCURRENCY: 16
  volumes:
    - ${AIRFLOW_PROJ_DIR:-.}/dags:/opt/airflow/dags
    - ${AIRFLOW_PROJ_DIR:-.}/logs:/opt/airflow/logs
    - ${AIRFLOW_PROJ_DIR:-.}/plugins:/opt/airflow/plugins
    - ${AIRFLOW_PROJ_DIR:-.}/config:/opt/airflow/config
  user: "${AIRFLOW_UID:-50000}:0"
  depends_on:
    &airflow-common-depends-on
    redis:
      condition: service_healthy
    postgres:
      condition: service_healthy

services:
  postgres:
    image: postgres:13
    environment:
      POSTGRES_USER: airflow
      POSTGRES_PASSWORD: airflow
      POSTGRES_DB: airflow
    volumes:
      - postgres-db-volume:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD", "pg_isready", "-U", "airflow"]
      interval: 10s
      retries: 5
      start_period: 5s
    restart: always
    ports:
      - "5432:5432"

  redis:
    image: redis:latest
    expose:
      - 6379
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 30s
      retries: 50
      start_period: 30s
    restart: always
    ports:
      - "6379:6379"

  airflow-webserver:
    <<: *airflow-common
    command: webserver
    ports:
      - "8080:8080"
    healthcheck:
      test: ["CMD", "curl", "--fail", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 30s
    restart: always
    depends_on:
      <<: *airflow-common-depends-on
      airflow-init:
        condition: service_completed_successfully

  airflow-scheduler:
    <<: *airflow-common
    command: scheduler
    healthcheck:
      test: ["CMD", "curl", "--fail", "http://localhost:8974/health"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 30s
    restart: always
    depends_on:
      <<: *airflow-common-depends-on
      airflow-init:
        condition: service_completed_successfully

  airflow-worker:
    <<: *airflow-common
    command: celery worker
    healthcheck:
      test:
        - "CMD-SHELL"
        - 'celery --app airflow.executors.celery_executor.app inspect ping -d "celery@$${HOSTNAME}"'
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 30s
    environment:
      <<: *airflow-common-env
      DUMB_INIT_SETSID: "0"
    restart: always
    depends_on:
      <<: *airflow-common-depends-on
      airflow-init:
        condition: service_completed_successfully

  airflow-triggerer:
    <<: *airflow-common
    command: triggerer
    healthcheck:
      test: ["CMD-SHELL", 'airflow jobs check --job-type TriggererJob --hostname "$${HOSTNAME}"']
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 30s
    restart: always
    depends_on:
      <<: *airflow-common-depends-on
      airflow-init:
        condition: service_completed_successfully

  airflow-init:
    <<: *airflow-common
    entrypoint: /bin/bash
    command:
      - -c
      - |
        function ver() {
          printf "%04d%04d%04d%04d" $${1//./ }
        }
        airflow_version=$$(AIRFLOW__LOGGING__LOGGING_LEVEL=INFO && gosu airflow airflow version)
        airflow_version_comparable=$$(ver $${airflow_version})
        min_airflow_version=2.2.0
        min_airflow_version_comparable=$$(ver $${min_airflow_version})
        if (( airflow_version_comparable < min_airflow_version_comparable )); then
          echo
          echo -e "\033[1;31mERROR!!!: Too old Airflow version $${airflow_version}!\e[0m"
          echo "The minimum Airflow version supported: $${min_airflow_version}. Only use this or higher!"
          echo
          exit 1
        fi
        if [[ -z "${AIRFLOW_UID}" ]]; then
          echo
          echo -e "\033[1;33mWARNING!!!: AIRFLOW_UID not set!\e[0m"
          echo "If you are on Linux, you SHOULD follow the instructions below to set "
          echo "AIRFLOW_UID environment variable, otherwise files will be owned by root."
          echo "For other operating systems you can get rid of the warning with manually created .env file:"
          echo "    See: https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html#setting-the-right-airflow-user"
          echo
        fi
        one_meg=1048576
        mem_available=$$(($$(getconf _PHYS_PAGES) * $$(getconf PAGE_SIZE) / one_meg))
        cpus_available=$$(grep -cE 'cpu[0-9]+' /proc/stat)
        disk_available=$$(df / | tail -1 | awk '{print $$4}')
        warning_resources="false"
        if (( mem_available < 4000 )) ; then
          echo
          echo -e "\033[1;33mWARNING!!!: Not enough memory available for Docker.\e[0m"
          echo "At least 4GB of memory required. You have $$(numfmt --to iec $$((mem_available * one_meg)))"
          echo
          warning_resources="true"
        fi
        if (( cpus_available < 2 )); then
          echo
          echo -e "\033[1;33mWARNING!!!: Not enough CPUS available for Docker.\e[0m"
          echo "At least 2 CPUs recommended. You have $${cpus_available}"
          echo
          warning_resources="true"
        fi
        if (( disk_available < one_meg * 10 )); then
          echo
          echo -e "\033[1;33mWARNING!!!: Not enough Disk space available for Docker.\e[0m"
          echo "At least 10 GBs recommended. You have $$(numfmt --to iec $$((disk_available * 1024 )))"
          echo
          warning_resources="true"
        fi
        if [[ $${warning_resources} == "true" ]]; then
          echo
          echo -e "\033[1;33mWARNING!!!: You have not enough resources to run Airflow (see above)!\e[0m"
          echo "Please follow the instructions to increase amount of resources available:"
          echo "   https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html#before-you-begin"
          echo
        fi
        mkdir -p /sources/logs /sources/dags /sources/plugins
        chown -R "${AIRFLOW_UID}:0" /sources/{logs,dags,plugins}
        exec /entrypoint airflow version
    environment:
      <<: *airflow-common-env
      _AIRFLOW_DB_UPGRADE: 'true'
      _AIRFLOW_WWW_USER_CREATE: 'true'
      _AIRFLOW_WWW_USER_USERNAME: ${_AIRFLOW_WWW_USER_USERNAME:-admin}
      _AIRFLOW_WWW_USER_PASSWORD: ${_AIRFLOW_WWW_USER_PASSWORD:-admin}
      _PIP_ADDITIONAL_REQUIREMENTS: ''
    user: "0:0"
    volumes:
      - ${AIRFLOW_PROJ_DIR:-.}:/sources

  flower:
    <<: *airflow-common
    command: celery flower
    profiles:
      - flower
    ports:
      - "5555:5555"
    healthcheck:
      test: ["CMD", "curl", "--fail", "http://localhost:5555/"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 30s
    restart: always
    depends_on:
      <<: *airflow-common-depends-on
      airflow-init:
        condition: service_completed_successfully

volumes:
  postgres-db-volume:
```

### Environment Configuration
```bash
# .env
AIRFLOW_UID=1000
AIRFLOW_PROJ_DIR=.

# Database configuration
POSTGRES_USER=airflow
POSTGRES_PASSWORD=airflow
POSTGRES_DB=airflow

# Airflow configuration
_AIRFLOW_WWW_USER_USERNAME=admin
_AIRFLOW_WWW_USER_PASSWORD=admin

# Security
AIRFLOW__CORE__FERNET_KEY=your-fernet-key-here

# Performance tuning
AIRFLOW__CORE__PARALLELISM=32
AIRFLOW__CORE__DAG_CONCURRENCY=16
AIRFLOW__CELERY__WORKER_CONCURRENCY=16
```

### Custom Airflow Configuration
```python
# config/airflow.cfg
[core]
# Analytics-optimized settings
default_timezone = UTC
executor = CeleryExecutor
parallelism = 32
dag_concurrency = 16
max_active_runs_per_dag = 16
load_examples = False
dags_are_paused_at_creation = True

[webserver]
# Security settings
expose_config = True
rbac = True
authenticate = True
auth_backend = airflow.contrib.auth.backends.password_auth

[celery]
# Worker configuration
worker_concurrency = 16
task_soft_time_limit = 600
task_time_limit = 1200

[scheduler]
# Scheduler optimization
job_heartbeat_sec = 5
scheduler_heartbeat_sec = 5
num_runs = -1
processor_poll_interval = 1
min_file_process_interval = 0
dag_dir_list_interval = 300
print_stats_interval = 30
child_process_timeout = 60
scheduler_zombie_task_threshold = 300
catchup_by_default = False
max_threads = 2

[logging]
# Logging configuration
logging_level = INFO
fab_logging_level = WARN
log_format = [%%(asctime)s] {%%(filename)s:%%(lineno)d} %%(levelname)s - %%(message)s
simple_log_format = %%(asctime)s %%(levelname)s - %%(message)s
```

## DAG Examples

### Data Pipeline DAG
```python
# dags/data_pipeline_example.py
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
import pandas as pd
import requests

default_args = {
    'owner': 'data-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'catchup': False
}

dag = DAG(
    'analytics_data_pipeline',
    default_args=default_args,
    description='Analytics data processing pipeline',
    schedule_interval='@daily',
    max_active_runs=1,
    tags=['analytics', 'etl', 'data-pipeline']
)

def extract_data(**context):
    """Extract data from API or database"""
    # Example: Extract from API
    response = requests.get('https://api.example.com/data')
    data = response.json()
    
    # Save to temporary location
    df = pd.DataFrame(data)
    df.to_csv('/tmp/raw_data.csv', index=False)
    
    return f"Extracted {len(df)} records"

def transform_data(**context):
    """Transform and clean data"""
    # Load raw data
    df = pd.read_csv('/tmp/raw_data.csv')
    
    # Data transformations
    df['created_date'] = pd.to_datetime(df['created_date'])
    df = df.dropna()
    df['processed_at'] = datetime.now()
    
    # Feature engineering
    df['month'] = df['created_date'].dt.month
    df['year'] = df['created_date'].dt.year
    
    # Save transformed data
    df.to_csv('/tmp/transformed_data.csv', index=False)
    
    return f"Transformed {len(df)} records"

def load_data(**context):
    """Load data to data warehouse"""
    df = pd.read_csv('/tmp/transformed_data.csv')
    
    # Load to PostgreSQL
    postgres_hook = PostgresHook(postgres_conn_id='postgres_default')
    df.to_sql('analytics_table', postgres_hook.get_sqlalchemy_engine(), 
              if_exists='append', index=False)
    
    return f"Loaded {len(df)} records to database"

def data_quality_check(**context):
    """Perform data quality checks"""
    postgres_hook = PostgresHook(postgres_conn_id='postgres_default')
    
    # Check for duplicates
    duplicate_count = postgres_hook.get_first(
        "SELECT COUNT(*) FROM analytics_table GROUP BY id HAVING COUNT(*) > 1"
    )[0] if postgres_hook.get_first(
        "SELECT COUNT(*) FROM analytics_table GROUP BY id HAVING COUNT(*) > 1"
    ) else 0
    
    # Check for null values
    null_count = postgres_hook.get_first(
        "SELECT COUNT(*) FROM analytics_table WHERE important_field IS NULL"
    )[0]
    
    if duplicate_count > 0:
        raise ValueError(f"Found {duplicate_count} duplicate records")
    
    if null_count > 100:  # Threshold
        raise ValueError(f"Found {null_count} null values in important_field")
    
    return "Data quality checks passed"

# Define tasks
extract_task = PythonOperator(
    task_id='extract_data',
    python_callable=extract_data,
    dag=dag
)

transform_task = PythonOperator(
    task_id='transform_data',
    python_callable=transform_data,
    dag=dag
)

load_task = PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    dag=dag
)

quality_check_task = PythonOperator(
    task_id='data_quality_check',
    python_callable=data_quality_check,
    dag=dag
)

# Create table if not exists
create_table_task = PostgresOperator(
    task_id='create_table',
    postgres_conn_id='postgres_default',
    sql="""
    CREATE TABLE IF NOT EXISTS analytics_table (
        id SERIAL PRIMARY KEY,
        created_date TIMESTAMP,
        processed_at TIMESTAMP,
        month INTEGER,
        year INTEGER,
        important_field VARCHAR(255)
    );
    """,
    dag=dag
)

# Cleanup task
cleanup_task = BashOperator(
    task_id='cleanup_temp_files',
    bash_command='rm -f /tmp/raw_data.csv /tmp/transformed_data.csv',
    dag=dag
)

# Define task dependencies
create_table_task >> extract_task >> transform_task >> load_task >> quality_check_task >> cleanup_task
```

### Machine Learning Pipeline DAG
```python
# dags/ml_pipeline_example.py
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn

default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'ml_training_pipeline',
    default_args=default_args,
    description='Machine learning model training pipeline',
    schedule_interval='@weekly',
    catchup=False,
    tags=['ml', 'training', 'model']
)

def prepare_training_data(**context):
    """Prepare data for model training"""
    postgres_hook = PostgresHook(postgres_conn_id='postgres_default')
    
    # Extract training data
    sql = """
    SELECT * FROM analytics_table 
    WHERE created_date >= CURRENT_DATE - INTERVAL '30 days'
    """
    df = postgres_hook.get_pandas_df(sql)
    
    # Feature engineering
    df['feature_1'] = df['value_1'] / df['value_2']
    df['feature_2'] = df['category'].map({'A': 1, 'B': 2, 'C': 3})
    
    # Save prepared data
    df.to_csv('/tmp/training_data.csv', index=False)
    
    return f"Prepared {len(df)} records for training"

def train_model(**context):
    """Train machine learning model"""
    # Load training data
    df = pd.read_csv('/tmp/training_data.csv')
    
    # Prepare features and target
    features = ['feature_1', 'feature_2', 'value_1', 'value_2']
    X = df[features]
    y = df['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Log with MLflow
    with mlflow.start_run():
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model")
    
    # Save model
    joblib.dump(model, '/tmp/trained_model.pkl')
    
    return f"Model trained with accuracy: {accuracy:.4f}"

def validate_model(**context):
    """Validate model performance"""
    model = joblib.load('/tmp/trained_model.pkl')
    df = pd.read_csv('/tmp/training_data.csv')
    
    # Validation logic
    features = ['feature_1', 'feature_2', 'value_1', 'value_2']
    X = df[features]
    y = df['target']
    
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    
    # Set minimum accuracy threshold
    if accuracy < 0.8:
        raise ValueError(f"Model accuracy {accuracy:.4f} below threshold 0.8")
    
    return f"Model validation passed with accuracy: {accuracy:.4f}"

def deploy_model(**context):
    """Deploy model to production"""
    # Copy model to production location
    import shutil
    shutil.copy('/tmp/trained_model.pkl', '/opt/airflow/models/production_model.pkl')
    
    # Update model registry or API
    # This would typically involve updating a model serving system
    
    return "Model deployed to production"

# Define tasks
prepare_data_task = PythonOperator(
    task_id='prepare_training_data',
    python_callable=prepare_training_data,
    dag=dag
)

train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag
)

validate_model_task = PythonOperator(
    task_id='validate_model',
    python_callable=validate_model,
    dag=dag
)

deploy_model_task = PythonOperator(
    task_id='deploy_model',
    python_callable=deploy_model,
    dag=dag
)

# Define dependencies
prepare_data_task >> train_model_task >> validate_model_task >> deploy_model_task
```

## Production Setup

### Security Configuration
```python
# config/security.py
import os
from cryptography.fernet import Fernet

# Generate Fernet key for encryption
def generate_fernet_key():
    key = Fernet.generate_key()
    return key.decode()

# Set in environment
os.environ['AIRFLOW__CORE__FERNET_KEY'] = generate_fernet_key()
```

### Resource Limits
```yaml
# docker-compose.override.yml
version: '3.8'

services:
  airflow-webserver:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G

  airflow-scheduler:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G

  airflow-worker:
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 8G
        reservations:
          cpus: '2.0'
          memory: 4G
    scale: 3  # Multiple workers
```

### Backup Configuration
```bash
#!/bin/bash
# scripts/backup.sh

# Backup PostgreSQL database
docker exec airflow-docker_postgres_1 pg_dump -U airflow airflow > backup_$(date +%Y%m%d_%H%M%S).sql

# Backup DAGs and configuration
tar -czf dags_backup_$(date +%Y%m%d_%H%M%S).tar.gz dags/ config/ plugins/

# Upload to cloud storage (example with AWS S3)
aws s3 cp backup_*.sql s3://your-backup-bucket/airflow/
aws s3 cp dags_backup_*.tar.gz s3://your-backup-bucket/airflow/
```

## Monitoring & Logging

### Prometheus Metrics
```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'airflow'
    static_configs:
      - targets: ['airflow-webserver:8080']
    metrics_path: '/admin/metrics'
```

### Custom Logging Configuration
```python
# config/log_config.py
import os
from airflow.configuration import conf

# Custom logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'airflow': {
            'format': '[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s'
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'airflow',
            'stream': 'ext://sys.stdout'
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'formatter': 'airflow',
            'filename': '/opt/airflow/logs/airflow.log',
            'maxBytes': 104857600,  # 100MB
            'backupCount': 5
        }
    },
    'loggers': {
        'airflow.processor': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False
        },
        'airflow.task': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False
        }
    },
    'root': {
        'handlers': ['console', 'file'],
        'level': 'INFO'
    }
}
```

## Scaling & Performance

### Horizontal Scaling
```bash
# Scale workers
docker-compose up --scale airflow-worker=5 -d

# Monitor worker performance
docker-compose exec flower celery inspect active
```

### Performance Tuning
```python
# config/performance.py
# Airflow performance configurations

# Scheduler optimization
AIRFLOW__SCHEDULER__JOB_HEARTBEAT_SEC = 5
AIRFLOW__SCHEDULER__SCHEDULER_HEARTBEAT_SEC = 5
AIRFLOW__SCHEDULER__NUM_RUNS = -1
AIRFLOW__SCHEDULER__PROCESSOR_POLL_INTERVAL = 1

# Database connection pooling
AIRFLOW__DATABASE__SQL_ALCHEMY_POOL_SIZE = 20
AIRFLOW__DATABASE__SQL_ALCHEMY_MAX_OVERFLOW = 30
AIRFLOW__DATABASE__SQL_ALCHEMY_POOL_RECYCLE = 3600

# Celery optimization
AIRFLOW__CELERY__WORKER_CONCURRENCY = 16
AIRFLOW__CELERY__TASK_SOFT_TIME_LIMIT = 600
AIRFLOW__CELERY__TASK_TIME_LIMIT = 1200
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Memory Issues
```bash
# Check memory usage
docker stats

# Increase memory limits in docker-compose.yml
services:
  airflow-worker:
    deploy:
      resources:
        limits:
          memory: 8G
```

#### 2. Database Connection Issues
```bash
# Check database connectivity
docker-compose exec airflow-webserver airflow db check

# Reset database
docker-compose down
docker volume rm airflow-docker_postgres-db-volume
docker-compose up airflow-init
```

#### 3. DAG Import Errors
```bash
# Check DAG syntax
docker-compose exec airflow-webserver python /opt/airflow/dags/your_dag.py

# View DAG import errors in UI
# Go to Admin -> DAG Import Errors
```

#### 4. Worker Issues
```bash
# Check worker status
docker-compose exec flower celery inspect active

# Restart workers
docker-compose restart airflow-worker

# Scale workers
docker-compose up --scale airflow-worker=3 -d
```

### Health Checks
```bash
#!/bin/bash
# scripts/health_check.sh

# Check all services
docker-compose ps

# Check Airflow components
curl -f http://localhost:8080/health || echo "Webserver unhealthy"
curl -f http://localhost:5555/ || echo "Flower unhealthy"

# Check database
docker-compose exec postgres pg_isready -U airflow || echo "Database unhealthy"

# Check Redis
docker-compose exec redis redis-cli ping || echo "Redis unhealthy"
```

### Log Analysis
```bash
# View logs
docker-compose logs airflow-webserver
docker-compose logs airflow-scheduler
docker-compose logs airflow-worker

# Follow logs in real-time
docker-compose logs -f airflow-scheduler

# Search logs for errors
docker-compose logs airflow-scheduler | grep ERROR
```

## Best Practices

### DAG Development
1. **Use meaningful DAG and task IDs**
2. **Set appropriate retry policies**
3. **Implement proper error handling**
4. **Use XComs for small data passing**
5. **Avoid heavy computations in DAG files**

### Security
1. **Use Fernet encryption for sensitive data**
2. **Implement proper authentication**
3. **Secure database connections**
4. **Regular security updates**
5. **Network isolation**

### Performance
1. **Optimize DAG parsing**
2. **Use appropriate pool configurations**
3. **Monitor resource usage**
4. **Implement proper caching**
5. **Scale workers based on workload**

---

*This Airflow Docker deployment provides a robust foundation for data pipeline orchestration and analytics workflow management. The configuration is optimized for production use with proper security, monitoring, and scaling capabilities.* 
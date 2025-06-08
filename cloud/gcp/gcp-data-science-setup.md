# Google Cloud Platform for Data Science
## Complete Setup and Best Practices Guide

### 🎯 Overview

This guide provides comprehensive instructions for setting up and using Google Cloud Platform (GCP) for data science projects, from basic setup to advanced ML workflows.

## 📋 Table of Contents

1. [Initial Setup](#initial-setup)
2. [Core Services](#core-services)
3. [Data Engineering](#data-engineering)
4. [Machine Learning](#machine-learning)
5. [Cost Optimization](#cost-optimization)
6. [Security & Compliance](#security--compliance)
7. [Monitoring & Logging](#monitoring--logging)
8. [Best Practices](#best-practices)

## 🚀 Initial Setup

### Prerequisites
```bash
# Install Google Cloud SDK
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Initialize gcloud
gcloud init

# Set default project
gcloud config set project YOUR_PROJECT_ID

# Enable required APIs
gcloud services enable compute.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable bigquery.googleapis.com
gcloud services enable ml.googleapis.com
gcloud services enable aiplatform.googleapis.com
gcloud services enable container.googleapis.com
```

### Authentication Setup
```bash
# Create service account
gcloud iam service-accounts create data-science-sa \
    --description="Data Science Service Account" \
    --display-name="Data Science SA"

# Grant necessary roles
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:data-science-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/bigquery.admin"

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:data-science-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/storage.admin"

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:data-science-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/ml.admin"

# Create and download key
gcloud iam service-accounts keys create ~/gcp-key.json \
    --iam-account=data-science-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com

# Set environment variable
export GOOGLE_APPLICATION_CREDENTIALS=~/gcp-key.json
```

## 🔧 Core Services

### 1. Google Cloud Storage (Data Lake)

#### Creating Buckets
```bash
# Create bucket for data lake
gsutil mb -p YOUR_PROJECT_ID -c STANDARD -l US-CENTRAL1 gs://your-data-lake-bucket

# Create bucket for ML models
gsutil mb -p YOUR_PROJECT_ID -c STANDARD -l US-CENTRAL1 gs://your-ml-models-bucket

# Set bucket lifecycle (optional)
gsutil lifecycle set lifecycle.json gs://your-data-lake-bucket
```

#### Lifecycle Configuration (`lifecycle.json`)
```json
{
  "lifecycle": {
    "rule": [
      {
        "action": {"type": "SetStorageClass", "storageClass": "NEARLINE"},
        "condition": {"age": 30}
      },
      {
        "action": {"type": "SetStorageClass", "storageClass": "COLDLINE"},
        "condition": {"age": 90}
      },
      {
        "action": {"type": "Delete"},
        "condition": {"age": 365}
      }
    ]
  }
}
```

#### Python Integration
```python
from google.cloud import storage
import pandas as pd

# Initialize client
client = storage.Client()

# Upload data
def upload_dataframe_to_gcs(df, bucket_name, blob_name):
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(df.to_csv(index=False), content_type='text/csv')

# Download data
def download_dataframe_from_gcs(bucket_name, blob_name):
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    content = blob.download_as_text()
    return pd.read_csv(StringIO(content))
```

### 2. BigQuery (Data Warehouse)

#### Creating Datasets and Tables
```sql
-- Create dataset
CREATE SCHEMA `your-project.analytics`
OPTIONS(
  description="Analytics dataset for data science projects",
  location="US"
);

-- Create partitioned table
CREATE TABLE `your-project.analytics.user_events`
(
  user_id STRING,
  event_type STRING,
  timestamp TIMESTAMP,
  properties JSON
)
PARTITION BY DATE(timestamp)
CLUSTER BY user_id, event_type;
```

#### Python Integration
```python
from google.cloud import bigquery
import pandas as pd

# Initialize client
client = bigquery.Client()

# Query data
def query_bigquery(query):
    return client.query(query).to_dataframe()

# Upload dataframe
def upload_dataframe_to_bigquery(df, table_id):
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",
        autodetect=True
    )
    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()  # Wait for job to complete

# Example usage
query = """
SELECT 
    user_id,
    COUNT(*) as event_count,
    MAX(timestamp) as last_event
FROM `your-project.analytics.user_events`
WHERE DATE(timestamp) >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
GROUP BY user_id
"""

df = query_bigquery(query)
```

## 🔄 Data Engineering

### Cloud Composer (Apache Airflow)

#### DAG Example
```python
from airflow import DAG
from airflow.providers.google.cloud.operators.bigquery import BigQueryCreateEmptyTableOperator
from airflow.providers.google.cloud.operators.gcs_to_bigquery import GCSToBigQueryOperator
from airflow.providers.google.cloud.operators.bigquery import BigQueryInsertJobOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'data-science-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'data_pipeline',
    default_args=default_args,
    description='Daily data processing pipeline',
    schedule_interval=timedelta(days=1),
    catchup=False,
)

# Extract task
extract_task = GCSToBigQueryOperator(
    task_id='extract_raw_data',
    bucket='your-data-lake-bucket',
    source_objects=['raw_data/*.csv'],
    destination_project_dataset_table='your-project.staging.raw_data',
    write_disposition='WRITE_TRUNCATE',
    dag=dag,
)

# Transform task
transform_query = """
    CREATE OR REPLACE TABLE `your-project.analytics.processed_data` AS
    SELECT
        user_id,
        event_type,
        DATE(timestamp) as event_date,
        COUNT(*) as event_count
    FROM `your-project.staging.raw_data`
    GROUP BY user_id, event_type, DATE(timestamp)
"""

transform_task = BigQueryInsertJobOperator(
    task_id='transform_data',
    configuration={
        "query": {
            "query": transform_query,
            "useLegacySql": False,
        }
    },
    dag=dag,
)

extract_task >> transform_task
```

### Cloud Dataflow (Apache Beam)

#### Batch Processing
```python
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.io.gcp.bigquery import WriteToBigQuery

def run_pipeline():
    pipeline_options = PipelineOptions([
        '--project=your-project-id',
        '--region=us-central1',
        '--runner=DataflowRunner',
        '--temp_location=gs://your-temp-bucket/temp',
        '--staging_location=gs://your-temp-bucket/staging'
    ])
    
    with beam.Pipeline(options=pipeline_options) as pipeline:
        (pipeline
         | 'ReadFromBigQuery' >> beam.io.ReadFromBigQuery(
             query='SELECT * FROM `your-project.raw.events`')
         | 'ProcessData' >> beam.Map(process_record)
         | 'WriteToBigQuery' >> WriteToBigQuery(
             'your-project.processed.events',
             write_disposition=beam.io.BigQueryDisposition.WRITE_TRUNCATE))

def process_record(record):
    # Your processing logic here
    return {
        'user_id': record['user_id'],
        'processed_timestamp': record['timestamp'],
        'feature_1': calculate_feature_1(record),
        'feature_2': calculate_feature_2(record)
    }
```

## 🤖 Machine Learning

### Vertex AI Setup

#### Model Training
```python
from google.cloud import aiplatform
from google.cloud.aiplatform import gapic as aip

# Initialize Vertex AI
aiplatform.init(project='your-project-id', location='us-central1')

# Custom training job
def create_training_job(display_name, script_path, container_uri):
    job = aiplatform.CustomTrainingJob(
        display_name=display_name,
        script_path=script_path,
        container_uri=container_uri,
        requirements=["scikit-learn", "pandas", "numpy"],
        model_serving_container_image_uri="gcr.io/cloud-aiplatform/prediction/sklearn-cpu.0-24:latest"
    )
    
    model = job.run(
        dataset=None,
        replica_count=1,
        machine_type="n1-standard-4",
        accelerator_type="NVIDIA_TESLA_T4",
        accelerator_count=1,
        base_output_dir="gs://your-ml-models-bucket/training_output"
    )
    
    return model

# AutoML training
def train_automl_model(dataset_id, target_column):
    dataset = aiplatform.TabularDataset(dataset_id)
    
    job = aiplatform.AutoMLTabularTrainingJob(
        display_name="automl-model",
        optimization_prediction_type="regression",
        optimization_objective="minimize-rmse"
    )
    
    model = job.run(
        dataset=dataset,
        target_column=target_column,
        training_fraction_split=0.8,
        validation_fraction_split=0.1,
        test_fraction_split=0.1
    )
    
    return model
```

#### Model Deployment
```python
# Deploy model to endpoint
def deploy_model(model, endpoint_display_name):
    endpoint = aiplatform.Endpoint.create(display_name=endpoint_display_name)
    
    model.deploy(
        endpoint=endpoint,
        deployed_model_display_name="deployed-model",
        machine_type="n1-standard-2",
        min_replica_count=1,
        max_replica_count=10,
        accelerator_type="NVIDIA_TESLA_T4",
        accelerator_count=1
    )
    
    return endpoint

# Make predictions
def predict(endpoint, instances):
    predictions = endpoint.predict(instances=instances)
    return predictions
```

### MLflow Integration
```python
import mlflow
import mlflow.sklearn
from google.cloud import storage

# Set MLflow tracking URI to GCS
mlflow.set_tracking_uri("gs://your-ml-models-bucket/mlflow")

# Log experiment
def log_experiment(model, metrics, params, artifacts_path):
    with mlflow.start_run():
        # Log parameters
        for key, value in params.items():
            mlflow.log_param(key, value)
        
        # Log metrics
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Log artifacts
        mlflow.log_artifacts(artifacts_path)
```

## 💰 Cost Optimization

### Storage Cost Optimization
```bash
# Set bucket lifecycle policy
gsutil lifecycle set - gs://your-bucket << EOF
{
  "lifecycle": {
    "rule": [
      {
        "action": {"type": "SetStorageClass", "storageClass": "NEARLINE"},
        "condition": {"age": 30}
      }
    ]
  }
}
EOF

# Use requester pays
gsutil requesterpays set on gs://your-bucket
```

### Compute Cost Optimization
```python
# Use preemptible instances
from google.cloud import compute_v1

def create_preemptible_instance():
    instance_client = compute_v1.InstancesClient()
    
    config = {
        "name": "preemptible-ml-instance",
        "machine_type": "zones/us-central1-a/machineTypes/n1-standard-8",
        "scheduling": {
            "preemptible": True
        },
        "disks": [
            {
                "boot": True,
                "auto_delete": True,
                "initialize_params": {
                    "source_image": "projects/debian-cloud/global/images/family/debian-11"
                }
            }
        ],
        "network_interfaces": [
            {
                "name": "global/networks/default",
                "access_configs": [{"type": "ONE_TO_ONE_NAT", "name": "External NAT"}]
            }
        ]
    }
    
    operation = instance_client.insert(
        project="your-project-id",
        zone="us-central1-a",
        instance_resource=config
    )
    
    return operation
```

### BigQuery Cost Optimization
```sql
-- Use clustering and partitioning
CREATE TABLE `your-project.dataset.optimized_table`
(
  date DATE,
  user_id STRING,
  event_type STRING,
  value FLOAT64
)
PARTITION BY date
CLUSTER BY user_id, event_type;

-- Use approximate aggregation functions
SELECT
  user_id,
  APPROX_COUNT_DISTINCT(session_id) as approx_sessions,
  APPROX_QUANTILES(revenue, 100)[OFFSET(50)] as median_revenue
FROM `your-project.dataset.events`
GROUP BY user_id;
```

## 🔒 Security & Compliance

### IAM Best Practices
```bash
# Create custom role with minimal permissions
gcloud iam roles create dataScientistRole \
    --project=your-project-id \
    --title="Data Scientist Role" \
    --description="Custom role for data scientists" \
    --permissions=bigquery.datasets.get,bigquery.jobs.create,storage.objects.get

# Use workload identity for GKE
gcloud container clusters update your-cluster \
    --workload-pool=your-project-id.svc.id.goog

kubectl create serviceaccount ml-workload-sa

gcloud iam service-accounts add-iam-policy-binding \
    ml-service-account@your-project-id.iam.gserviceaccount.com \
    --role roles/iam.workloadIdentityUser \
    --member "serviceAccount:your-project-id.svc.id.goog[default/ml-workload-sa]"
```

### Data Encryption
```python
from google.cloud import kms

# Encrypt sensitive data
def encrypt_data(project_id, location, key_ring, key_name, plaintext):
    client = kms.KeyManagementServiceClient()
    key_name = client.crypto_key_path(project_id, location, key_ring, key_name)
    
    response = client.encrypt(request={
        "name": key_name,
        "plaintext": plaintext.encode("utf-8")
    })
    
    return response.ciphertext

# Decrypt data
def decrypt_data(project_id, location, key_ring, key_name, ciphertext):
    client = kms.KeyManagementServiceClient()
    key_name = client.crypto_key_path(project_id, location, key_ring, key_name)
    
    response = client.decrypt(request={
        "name": key_name,
        "ciphertext": ciphertext
    })
    
    return response.plaintext.decode("utf-8")
```

## 📊 Monitoring & Logging

### Cloud Monitoring Setup
```python
from google.cloud import monitoring_v3

def create_alert_policy():
    client = monitoring_v3.AlertPolicyServiceClient()
    project_name = f"projects/{project_id}"
    
    policy = monitoring_v3.AlertPolicy(
        display_name="BigQuery Slot Usage Alert",
        conditions=[
            monitoring_v3.AlertPolicy.Condition(
                display_name="BigQuery slot usage high",
                condition_threshold=monitoring_v3.AlertPolicy.Condition.MetricThreshold(
                    filter='resource.type="bigquery_project"',
                    comparison=monitoring_v3.ComparisonType.COMPARISON_GREATER_THAN,
                    threshold_value=80.0
                )
            )
        ],
        notification_channels=[notification_channel_name],
        alert_strategy=monitoring_v3.AlertPolicy.AlertStrategy(
            auto_close=86400  # 24 hours
        )
    )
    
    policy = client.create_alert_policy(name=project_name, alert_policy=policy)
    return policy
```

### Cloud Logging
```python
from google.cloud import logging

def setup_logging():
    logging_client = logging.Client()
    logging_client.setup_logging()
    
    logger = logging_client.logger("ml-pipeline")
    
    # Log structured data
    logger.log_struct({
        "message": "Model training started",
        "severity": "INFO",
        "model_name": "customer_churn_model",
        "training_data_size": 1000000,
        "hyperparameters": {
            "learning_rate": 0.01,
            "batch_size": 32
        }
    })
```

## 🎯 Best Practices

### 1. Project Structure
```
gcp-ml-project/
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── notebooks/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── visualization/
├── configs/
├── scripts/
└── tests/
```

### 2. Environment Management
```bash
# Use different projects for different environments
gcloud config configurations create dev
gcloud config configurations create staging  
gcloud config configurations create prod

# Activate configuration
gcloud config configurations activate dev
```

### 3. Code Organization
```python
# config.py - Centralized configuration
import os
from dataclasses import dataclass

@dataclass
class Config:
    PROJECT_ID: str = os.getenv('GOOGLE_CLOUD_PROJECT')
    REGION: str = os.getenv('GOOGLE_CLOUD_REGION', 'us-central1')
    BUCKET_NAME: str = os.getenv('BUCKET_NAME')
    DATASET_ID: str = os.getenv('BIGQUERY_DATASET')
    
    # Model configuration
    MODEL_NAME: str = 'customer_churn_model'
    MODEL_VERSION: str = 'v1.0'
    
    # Training configuration
    TRAIN_SPLIT: float = 0.8
    VALIDATION_SPLIT: float = 0.1
    TEST_SPLIT: float = 0.1

config = Config()
```

### 4. CI/CD Pipeline
```yaml
# .github/workflows/deploy.yml
name: Deploy ML Pipeline

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Setup Cloud SDK
      uses: google-github-actions/setup-gcloud@v0
      with:
        service_account_key: ${{ secrets.GCP_SA_KEY }}
        project_id: ${{ secrets.GCP_PROJECT_ID }}
    
    - name: Deploy to Cloud Functions
      run: |
        gcloud functions deploy ml-prediction-api \
          --runtime python39 \
          --trigger-http \
          --allow-unauthenticated \
          --source . \
          --entry-point predict
```

## 🔗 Useful Resources

- [GCP Data Science Documentation](https://cloud.google.com/products/ai)
- [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
- [BigQuery ML Documentation](https://cloud.google.com/bigquery-ml/docs)
- [Cloud Composer Documentation](https://cloud.google.com/composer/docs)

## 📈 Next Steps

1. Set up your GCP project following this guide
2. Create your first ML pipeline using Vertex AI
3. Implement monitoring and alerting
4. Set up CI/CD for automated deployments
5. Explore advanced features like AutoML and BigQuery ML

---

**Happy Data Science on GCP! ☁️🚀** 
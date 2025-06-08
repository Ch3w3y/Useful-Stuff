# AWS Data Science Setup Guide
==============================

> **Complete guide for setting up AWS infrastructure optimized for data science and machine learning workflows**

## Table of Contents
- [Getting Started](#getting-started)
- [Core AWS Services for Data Science](#core-aws-services-for-data-science)
- [EC2 for Computing](#ec2-for-computing)
- [S3 for Data Storage](#s3-for-data-storage)
- [RDS and Database Services](#rds-and-database-services)
- [SageMaker for ML](#sagemaker-for-ml)
- [Lambda for Serverless Computing](#lambda-for-serverless-computing)
- [EMR for Big Data](#emr-for-big-data)
- [Redshift for Data Warehousing](#redshift-for-data-warehousing)
- [Athena for Analytics](#athena-for-analytics)
- [Glue for ETL](#glue-for-etl)
- [IAM Security Best Practices](#iam-security-best-practices)
- [Cost Optimization](#cost-optimization)
- [Monitoring and Logging](#monitoring-and-logging)
- [Best Practices and Tips](#best-practices-and-tips)
- [Sample Architectures](#sample-architectures)

---

## Getting Started

### 1. Account Setup and Initial Configuration

#### Create AWS Account
```bash
# Visit https://aws.amazon.com and create account
# Enable billing alerts immediately
# Set up MFA for root account
```

#### Install AWS CLI
```bash
# Install AWS CLI v2
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Verify installation
aws --version

# Configure credentials
aws configure
# AWS Access Key ID: [Your access key]
# AWS Secret Access Key: [Your secret key]
# Default region name: us-east-1
# Default output format: json
```

#### Create IAM User for Development
```bash
# Create IAM user with programmatic access
aws iam create-user --user-name data-scientist

# Attach necessary policies
aws iam attach-user-policy \
  --user-name data-scientist \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

aws iam attach-user-policy \
  --user-name data-scientist \
  --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess

aws iam attach-user-policy \
  --user-name data-scientist \
  --policy-arn arn:aws:iam::aws:policy/AmazonEC2FullAccess
```

### 2. Basic AWS CLI Configuration

```bash
# Configure multiple profiles
aws configure --profile development
aws configure --profile production

# Set environment variables
export AWS_PROFILE=development
export AWS_DEFAULT_REGION=us-east-1

# Test configuration
aws sts get-caller-identity
```

---

## Core AWS Services for Data Science

### Service Overview Matrix

| Service | Use Case | Cost Level | Learning Curve |
|---------|----------|------------|----------------|
| EC2 | Custom compute environments | Medium | Medium |
| S3 | Data storage and backup | Low | Low |
| RDS | Managed databases | Medium | Low |
| SageMaker | End-to-end ML platform | High | Medium |
| Lambda | Serverless functions | Low | Low |
| EMR | Big data processing | High | High |
| Redshift | Data warehousing | High | Medium |
| Athena | Serverless analytics | Low | Low |
| Glue | ETL and data catalog | Medium | Medium |

---

## EC2 for Computing

### 1. Instance Types for Data Science

#### General Purpose Instances
```bash
# t3.large - Development and small datasets
# m5.xlarge - Balanced compute for medium workloads
# m5.2xlarge - Production workloads

# Launch instance
aws ec2 run-instances \
  --image-id ami-0abcdef1234567890 \
  --instance-type m5.xlarge \
  --key-name my-key-pair \
  --security-group-ids sg-903004f8 \
  --subnet-id subnet-6e7f829e \
  --user-data file://user-data-script.sh
```

#### Compute Optimized Instances
```bash
# c5.xlarge - CPU-intensive ML training
# c5.4xlarge - Heavy computation workloads

# Create launch template
aws ec2 create-launch-template \
  --launch-template-name data-science-template \
  --launch-template-data '{
    "ImageId": "ami-0abcdef1234567890",
    "InstanceType": "c5.xlarge",
    "KeyName": "my-key-pair",
    "SecurityGroupIds": ["sg-903004f8"],
    "UserData": "'$(base64 -w 0 user-data-script.sh)'"
  }'
```

#### Memory Optimized Instances
```bash
# r5.large - In-memory analytics
# r5.2xlarge - Large datasets in memory
# x1e.xlarge - High memory requirements

# Launch memory-optimized instance
aws ec2 run-instances \
  --image-id ami-0abcdef1234567890 \
  --instance-type r5.2xlarge \
  --key-name my-key-pair \
  --security-group-ids sg-903004f8
```

#### GPU Instances
```bash
# p3.2xlarge - Deep learning training
# p3.8xlarge - Multi-GPU training
# g4dn.xlarge - ML inference

# Launch GPU instance
aws ec2 run-instances \
  --image-id ami-0abcdef1234567890 \
  --instance-type p3.2xlarge \
  --key-name my-key-pair \
  --security-group-ids sg-903004f8
```

### 2. EC2 Setup Scripts

#### User Data Script for Data Science Environment
```bash
#!/bin/bash
# user-data-script.sh

# Update system
yum update -y

# Install development tools
yum groupinstall -y "Development Tools"
yum install -y htop tmux git vim

# Install Python and pip
yum install -y python3 python3-pip
pip3 install --upgrade pip

# Install Anaconda
cd /tmp
wget https://repo.anaconda.com/archive/Anaconda3-2023.09-Linux-x86_64.sh
bash Anaconda3-2023.09-Linux-x86_64.sh -b -p /opt/anaconda3
echo 'export PATH="/opt/anaconda3/bin:$PATH"' >> /etc/environment

# Install R
amazon-linux-extras install -y R4
yum install -y R-devel

# Install Docker
yum install -y docker
systemctl start docker
systemctl enable docker
usermod -aG docker ec2-user

# Install AWS CLI v2
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
./aws/install

# Create directories
mkdir -p /home/ec2-user/{projects,data,notebooks}
chown -R ec2-user:ec2-user /home/ec2-user/{projects,data,notebooks}

# Install Jupyter Lab
pip3 install jupyterlab
pip3 install ipykernel

# Setup Jupyter as a service
cat > /etc/systemd/system/jupyter.service << 'EOF'
[Unit]
Description=Jupyter Lab
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/home/ec2-user/notebooks
ExecStart=/opt/anaconda3/bin/jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
Restart=always

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable jupyter
systemctl start jupyter
```

### 3. Security Groups Configuration

```bash
# Create security group for data science workstation
aws ec2 create-security-group \
  --group-name data-science-sg \
  --description "Security group for data science instances"

# Get security group ID
SG_ID=$(aws ec2 describe-security-groups \
  --group-names data-science-sg \
  --query 'SecurityGroups[0].GroupId' \
  --output text)

# Allow SSH access
aws ec2 authorize-security-group-ingress \
  --group-id $SG_ID \
  --protocol tcp \
  --port 22 \
  --cidr 0.0.0.0/0

# Allow Jupyter Lab access
aws ec2 authorize-security-group-ingress \
  --group-id $SG_ID \
  --protocol tcp \
  --port 8888 \
  --cidr 0.0.0.0/0

# Allow RStudio Server access
aws ec2 authorize-security-group-ingress \
  --group-id $SG_ID \
  --protocol tcp \
  --port 8787 \
  --cidr 0.0.0.0/0
```

---

## S3 for Data Storage

### 1. Bucket Setup and Configuration

```bash
# Create S3 bucket for data science projects
aws s3 mb s3://my-datascience-bucket-$(date +%s)

# Set bucket versioning
aws s3api put-bucket-versioning \
  --bucket my-datascience-bucket \
  --versioning-configuration Status=Enabled

# Configure lifecycle policy
cat > lifecycle-policy.json << 'EOF'
{
  "Rules": [
    {
      "ID": "DataScienceLifecycle",
      "Status": "Enabled",
      "Filter": {"Prefix": "data/"},
      "Transitions": [
        {
          "Days": 30,
          "StorageClass": "STANDARD_INFREQUENT_ACCESS"
        },
        {
          "Days": 90,
          "StorageClass": "GLACIER"
        }
      ]
    }
  ]
}
EOF

aws s3api put-bucket-lifecycle-configuration \
  --bucket my-datascience-bucket \
  --lifecycle-configuration file://lifecycle-policy.json
```

### 2. Data Organization Structure

```bash
# Create organized folder structure
aws s3api put-object \
  --bucket my-datascience-bucket \
  --key raw-data/

aws s3api put-object \
  --bucket my-datascience-bucket \
  --key processed-data/

aws s3api put-object \
  --bucket my-datascience-bucket \
  --key models/

aws s3api put-object \
  --bucket my-datascience-bucket \
  --key notebooks/

aws s3api put-object \
  --bucket my-datascience-bucket \
  --key results/

# Sync local project to S3
aws s3 sync ./local-project/ s3://my-datascience-bucket/projects/project1/
```

### 3. Data Transfer Optimization

```bash
# Configure AWS CLI for faster transfers
aws configure set default.s3.max_concurrent_requests 10
aws configure set default.s3.max_bandwidth 50MB/s
aws configure set default.s3.multipart_threshold 64MB
aws configure set default.s3.multipart_chunksize 16MB

# Use S3 Transfer Acceleration
aws s3api put-bucket-accelerate-configuration \
  --bucket my-datascience-bucket \
  --accelerate-configuration Status=Enabled

# Transfer large datasets
aws s3 cp large-dataset.csv s3://my-datascience-bucket/raw-data/ \
  --storage-class STANDARD_IA
```

---

## RDS and Database Services

### 1. PostgreSQL Setup for Analytics

```bash
# Create RDS PostgreSQL instance
aws rds create-db-instance \
  --db-instance-identifier analytics-postgres \
  --db-instance-class db.t3.medium \
  --engine postgres \
  --engine-version 15.3 \
  --master-username postgres \
  --master-user-password SecurePassword123! \
  --allocated-storage 100 \
  --storage-type gp2 \
  --vpc-security-group-ids sg-903004f8 \
  --backup-retention-period 7 \
  --multi-az false \
  --publicly-accessible true
```

### 2. DynamoDB for NoSQL Data

```bash
# Create DynamoDB table for real-time data
aws dynamodb create-table \
  --table-name UserBehavior \
  --attribute-definitions \
    AttributeName=user_id,AttributeType=S \
    AttributeName=timestamp,AttributeType=N \
  --key-schema \
    AttributeName=user_id,KeyType=HASH \
    AttributeName=timestamp,KeyType=RANGE \
  --billing-mode PAY_PER_REQUEST

# Create global secondary index
aws dynamodb update-table \
  --table-name UserBehavior \
  --attribute-definitions \
    AttributeName=session_id,AttributeType=S \
  --global-secondary-index-updates \
    '[{
      "Create": {
        "IndexName": "SessionIndex",
        "KeySchema": [
          {"AttributeName": "session_id", "KeyType": "HASH"}
        ],
        "Projection": {"ProjectionType": "ALL"},
        "BillingMode": "PAY_PER_REQUEST"
      }
    }]'
```

---

## SageMaker for ML

### 1. SageMaker Studio Setup

```bash
# Create SageMaker execution role
aws iam create-role \
  --role-name SageMakerExecutionRole \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [
      {
        "Effect": "Allow",
        "Principal": {"Service": "sagemaker.amazonaws.com"},
        "Action": "sts:AssumeRole"
      }
    ]
  }'

# Attach policies to role
aws iam attach-role-policy \
  --role-name SageMakerExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess

# Create SageMaker domain
aws sagemaker create-domain \
  --domain-name my-data-science-domain \
  --auth-mode IAM \
  --default-user-settings '{
    "ExecutionRole": "arn:aws:iam::123456789012:role/SageMakerExecutionRole"
  }' \
  --vpc-id vpc-12345678 \
  --subnet-ids subnet-12345678 subnet-87654321
```

### 2. SageMaker Training Jobs

```python
# Python script for SageMaker training
import boto3
import sagemaker
from sagemaker.sklearn.estimator import SKLearn

sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

# Define training job
sklearn_estimator = SKLearn(
    entry_point='train.py',
    framework_version='1.0-1',
    instance_type='ml.m5.large',
    role=role,
    sagemaker_session=sagemaker_session
)

# Start training
sklearn_estimator.fit({'training': 's3://my-bucket/training-data'})
```

### 3. SageMaker Endpoints

```python
# Deploy model to endpoint
predictor = sklearn_estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.t2.medium'
)

# Make predictions
result = predictor.predict(test_data)

# Clean up endpoint
predictor.delete_endpoint()
```

---

## Lambda for Serverless Computing

### 1. Data Processing Lambda

```python
# lambda_function.py
import json
import boto3
import pandas as pd
from io import StringIO

s3 = boto3.client('s3')

def lambda_handler(event, context):
    # Get the bucket and key from the event
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']
    
    # Read CSV from S3
    response = s3.get_object(Bucket=bucket, Key=key)
    csv_content = response['Body'].read().decode('utf-8')
    
    # Process data with pandas
    df = pd.read_csv(StringIO(csv_content))
    
    # Simple data cleaning
    df_cleaned = df.dropna()
    df_cleaned['processed_date'] = pd.Timestamp.now()
    
    # Save processed data back to S3
    output_key = key.replace('raw-data/', 'processed-data/')
    csv_buffer = StringIO()
    df_cleaned.to_csv(csv_buffer, index=False)
    
    s3.put_object(
        Bucket=bucket,
        Key=output_key,
        Body=csv_buffer.getvalue()
    )
    
    return {
        'statusCode': 200,
        'body': json.dumps(f'Processed {len(df_cleaned)} rows')
    }
```

### 2. Deploy Lambda Function

```bash
# Create deployment package
zip -r function.zip lambda_function.py

# Create Lambda function
aws lambda create-function \
  --function-name data-processor \
  --runtime python3.9 \
  --role arn:aws:iam::123456789012:role/lambda-execution-role \
  --handler lambda_function.lambda_handler \
  --zip-file fileb://function.zip

# Add S3 trigger
aws lambda add-permission \
  --function-name data-processor \
  --principal s3.amazonaws.com \
  --action lambda:InvokeFunction \
  --source-arn arn:aws:s3:::my-datascience-bucket \
  --statement-id s3-trigger
```

---

## EMR for Big Data

### 1. EMR Cluster Configuration

```bash
# Create EMR cluster
aws emr create-cluster \
  --name "Data Science Cluster" \
  --release-label emr-6.15.0 \
  --instance-groups '[
    {
      "Name": "Master",
      "Market": "ON_DEMAND",
      "InstanceRole": "MASTER",
      "InstanceType": "m5.xlarge",
      "InstanceCount": 1
    },
    {
      "Name": "Core",
      "Market": "ON_DEMAND", 
      "InstanceRole": "CORE",
      "InstanceType": "m5.xlarge",
      "InstanceCount": 2
    }
  ]' \
  --applications Name=Hadoop Name=Spark Name=Jupyter Name=Zeppelin \
  --ec2-attributes '{
    "KeyName": "my-key-pair",
    "InstanceProfile": "EMR_EC2_DefaultRole"
  }' \
  --service-role EMR_DefaultRole \
  --enable-debugging \
  --log-uri s3://my-emr-logs/
```

### 2. Spark Job Submission

```python
# spark_job.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

spark = SparkSession.builder.appName("DataAnalysis").getOrCreate()

# Read data from S3
df = spark.read.csv("s3://my-bucket/large-dataset.csv", header=True, inferSchema=True)

# Perform analysis
result = df.groupBy("category").agg(
    avg("value").alias("avg_value"),
    count("*").alias("count")
).orderBy("avg_value", ascending=False)

# Write results back to S3
result.write.csv("s3://my-bucket/analysis-results/", header=True, mode="overwrite")

spark.stop()
```

---

## IAM Security Best Practices

### 1. Least Privilege Policies

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject"
      ],
      "Resource": "arn:aws:s3:::my-datascience-bucket/*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:ListBucket"
      ],
      "Resource": "arn:aws:s3:::my-datascience-bucket"
    },
    {
      "Effect": "Allow",
      "Action": [
        "sagemaker:CreateTrainingJob",
        "sagemaker:DescribeTrainingJob",
        "sagemaker:StopTrainingJob"
      ],
      "Resource": "*"
    }
  ]
}
```

### 2. Role-Based Access

```bash
# Create data scientist role
aws iam create-role \
  --role-name DataScientistRole \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [
      {
        "Effect": "Allow",
        "Principal": {"AWS": "arn:aws:iam::123456789012:root"},
        "Action": "sts:AssumeRole",
        "Condition": {
          "StringEquals": {
            "sts:ExternalId": "unique-external-id"
          }
        }
      }
    ]
  }'

# Attach custom policy
aws iam attach-role-policy \
  --role-name DataScientistRole \
  --policy-arn arn:aws:iam::123456789012:policy/DataScientistPolicy
```

---

## Cost Optimization

### 1. Spot Instances for Training

```bash
# Launch Spot instance for training
aws ec2 request-spot-instances \
  --spot-price "0.05" \
  --instance-count 1 \
  --type "one-time" \
  --launch-specification '{
    "ImageId": "ami-0abcdef1234567890",
    "InstanceType": "p3.2xlarge",
    "KeyName": "my-key-pair",
    "SecurityGroupIds": ["sg-903004f8"],
    "UserData": "'$(base64 -w 0 spot-training-script.sh)'"
  }'
```

### 2. Auto Scaling for Variable Workloads

```bash
# Create Auto Scaling group
aws autoscaling create-auto-scaling-group \
  --auto-scaling-group-name data-processing-asg \
  --launch-template '{
    "LaunchTemplateName": "data-science-template",
    "Version": "$Latest"
  }' \
  --min-size 0 \
  --max-size 10 \
  --desired-capacity 0 \
  --vpc-zone-identifier "subnet-12345678,subnet-87654321"

# Create scaling policies
aws autoscaling put-scaling-policy \
  --auto-scaling-group-name data-processing-asg \
  --policy-name scale-up \
  --scaling-adjustment 2 \
  --adjustment-type ChangeInCapacity
```

### 3. Cost Monitoring

```bash
# Create billing alarm
aws cloudwatch put-metric-alarm \
  --alarm-name "Monthly-Bill-Alarm" \
  --alarm-description "Alarm when monthly bill exceeds $100" \
  --metric-name EstimatedCharges \
  --namespace AWS/Billing \
  --statistic Maximum \
  --period 86400 \
  --threshold 100 \
  --comparison-operator GreaterThanThreshold \
  --dimensions Name=Currency,Value=USD \
  --evaluation-periods 1 \
  --alarm-actions arn:aws:sns:us-east-1:123456789012:billing-alarm
```

---

## Sample Architectures

### 1. Real-time Analytics Pipeline

```yaml
# CloudFormation template
AWSTemplateFormatVersion: '2010-09-09'
Description: 'Real-time analytics pipeline for data science'

Resources:
  # Kinesis Data Stream
  DataStream:
    Type: AWS::Kinesis::Stream
    Properties:
      Name: analytics-stream
      ShardCount: 2
      
  # Lambda for processing
  ProcessingLambda:
    Type: AWS::Lambda::Function
    Properties:
      FunctionName: stream-processor
      Runtime: python3.9
      Handler: index.handler
      Code:
        ZipFile: |
          def handler(event, context):
              # Process Kinesis records
              pass
              
  # DynamoDB for storage
  AnalyticsTable:
    Type: AWS::DynamoDB::Table
    Properties:
      TableName: analytics-results
      BillingMode: PAY_PER_REQUEST
      AttributeDefinitions:
        - AttributeName: id
          AttributeType: S
      KeySchema:
        - AttributeName: id
          KeyType: HASH
```

### 2. Batch Processing Architecture

```python
# Airflow DAG for batch processing
from airflow import DAG
from airflow.providers.amazon.aws.operators.emr import EmrCreateJobFlowOperator
from airflow.providers.amazon.aws.sensors.emr import EmrJobFlowSensor
from datetime import datetime, timedelta

default_args = {
    'owner': 'data-science-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'batch_ml_pipeline',
    default_args=default_args,
    description='Batch ML pipeline',
    schedule_interval='@daily',
    catchup=False
)

# EMR cluster configuration
JOB_FLOW_OVERRIDES = {
    'Name': 'batch-ml-cluster',
    'ReleaseLabel': 'emr-6.15.0',
    'Instances': {
        'InstanceGroups': [
            {
                'Name': 'Master nodes',
                'Market': 'ON_DEMAND',
                'InstanceRole': 'MASTER',
                'InstanceType': 'm5.xlarge',
                'InstanceCount': 1,
            },
            {
                'Name': 'Worker nodes',
                'Market': 'SPOT',
                'InstanceRole': 'CORE',
                'InstanceType': 'm5.xlarge',
                'InstanceCount': 2,
            }
        ],
        'Ec2KeyName': 'my-key-pair',
        'KeepJobFlowAliveWhenNoSteps': False,
    },
    'Applications': [{'Name': 'Spark'}],
    'JobFlowRole': 'EMR_EC2_DefaultRole',
    'ServiceRole': 'EMR_DefaultRole',
}

# Create EMR cluster
create_emr_cluster = EmrCreateJobFlowOperator(
    task_id='create_emr_cluster',
    job_flow_overrides=JOB_FLOW_OVERRIDES,
    dag=dag
)

# Wait for cluster to be ready
wait_for_completion = EmrJobFlowSensor(
    task_id='wait_for_completion',
    job_flow_id="{{ task_instance.xcom_pull(task_ids='create_emr_cluster', key='return_value') }}",
    dag=dag
)

create_emr_cluster >> wait_for_completion
```

---

## Best Practices and Tips

### 1. Security Best Practices

- **Use IAM roles instead of access keys**
- **Enable MFA for all users**
- **Use VPC endpoints for S3 access**
- **Encrypt data at rest and in transit**
- **Regularly rotate credentials**
- **Use AWS CloudTrail for auditing**

### 2. Cost Optimization Tips

- **Use Spot instances for training jobs**
- **Set up billing alerts**
- **Use S3 lifecycle policies**
- **Stop unused EC2 instances**
- **Use Reserved Instances for predictable workloads**
- **Monitor with AWS Cost Explorer**

### 3. Performance Optimization

- **Use appropriate instance types**
- **Leverage placement groups for HPC**
- **Use EBS-optimized instances**
- **Implement data partitioning strategies**
- **Use compression for data storage**
- **Cache frequently accessed data**

### 4. Monitoring and Troubleshooting

```bash
# CloudWatch metrics for monitoring
aws cloudwatch put-metric-data \
  --namespace "DataScience/Pipeline" \
  --metric-data MetricName=ProcessingTime,Value=120,Unit=Seconds

# Set up log aggregation
aws logs create-log-group --log-group-name /aws/datascience/processing

# Create dashboard
aws cloudwatch put-dashboard \
  --dashboard-name DataScienceDashboard \
  --dashboard-body file://dashboard.json
```

---

**Happy AWS data science computing!** ☁️📊

> This guide is continuously updated with new AWS services and best practices. Check back regularly for updates! 
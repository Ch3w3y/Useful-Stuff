# ☁️ AWS Deployment Guide

> Comprehensive guide for deploying data science and machine learning workloads on Amazon Web Services.

## 📁 Directory Structure

```
aws/
├── compute/                 # EC2, Lambda, Batch configurations
├── storage/                 # S3, EFS, EBS setup and management
├── databases/              # RDS, DynamoDB, Redshift configurations
├── ml-services/            # SageMaker, Comprehend, Rekognition
├── analytics/              # EMR, Glue, Athena, QuickSight
├── containers/             # ECS, EKS, Fargate deployments
├── serverless/             # Lambda functions and serverless architectures
├── networking/             # VPC, Security Groups, Load Balancers
├── monitoring/             # CloudWatch, X-Ray, CloudTrail
├── security/               # IAM, KMS, Secrets Manager
├── infrastructure/         # CloudFormation, CDK, Terraform
└── templates/              # Ready-to-use deployment templates
```

## 🚀 Core AWS Services for Data Science

### Compute Services

#### Amazon EC2
```bash
# Launch EC2 instance with data science AMI
aws ec2 run-instances \
    --image-id ami-0abcdef1234567890 \
    --instance-type p3.2xlarge \
    --key-name my-key-pair \
    --security-group-ids sg-903004f8 \
    --subnet-id subnet-6e7f829e \
    --user-data file://setup-script.sh
```

#### AWS Lambda
```python
import json
import boto3
import pandas as pd

def lambda_handler(event, context):
    """Lambda function for data processing"""
    
    # Initialize AWS services
    s3 = boto3.client('s3')
    
    # Process data
    bucket = event['bucket']
    key = event['key']
    
    # Download data from S3
    obj = s3.get_object(Bucket=bucket, Key=key)
    df = pd.read_csv(obj['Body'])
    
    # Perform analysis
    result = df.describe().to_dict()
    
    return {
        'statusCode': 200,
        'body': json.dumps(result)
    }
```

#### AWS Batch
```yaml
# batch-job-definition.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: ml-training-job
spec:
  template:
    spec:
      containers:
      - name: ml-container
        image: your-account.dkr.ecr.region.amazonaws.com/ml-training:latest
        resources:
          requests:
            nvidia.com/gpu: 1
        env:
        - name: S3_BUCKET
          value: "your-data-bucket"
      restartPolicy: Never
```

### Storage Services

#### Amazon S3
```python
import boto3
from botocore.exceptions import ClientError

class S3DataManager:
    def __init__(self, bucket_name):
        self.s3_client = boto3.client('s3')
        self.bucket_name = bucket_name
    
    def upload_data(self, local_file, s3_key):
        """Upload file to S3"""
        try:
            self.s3_client.upload_file(local_file, self.bucket_name, s3_key)
            return True
        except ClientError as e:
            print(f"Error uploading file: {e}")
            return False
    
    def download_data(self, s3_key, local_file):
        """Download file from S3"""
        try:
            self.s3_client.download_file(self.bucket_name, s3_key, local_file)
            return True
        except ClientError as e:
            print(f"Error downloading file: {e}")
            return False
    
    def list_objects(self, prefix=""):
        """List objects in S3 bucket"""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            return [obj['Key'] for obj in response.get('Contents', [])]
        except ClientError as e:
            print(f"Error listing objects: {e}")
            return []

# Usage
s3_manager = S3DataManager('my-data-bucket')
s3_manager.upload_data('local_data.csv', 'data/processed/dataset.csv')
```

### Machine Learning Services

#### Amazon SageMaker
```python
import boto3
import sagemaker
from sagemaker.sklearn.estimator import SKLearn

# Initialize SageMaker session
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

# Define training script
sklearn_estimator = SKLearn(
    entry_point='train.py',
    role=role,
    instance_type='ml.m5.large',
    framework_version='0.23-1',
    py_version='py3',
    script_mode=True,
    hyperparameters={
        'n_estimators': 100,
        'max_depth': 5
    }
)

# Start training job
sklearn_estimator.fit({'training': 's3://my-bucket/training-data'})

# Deploy model
predictor = sklearn_estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.t2.medium'
)
```

#### SageMaker Training Script (train.py)
```python
import argparse
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

def model_fn(model_dir):
    """Load model for inference"""
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=5)
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--training', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    
    args = parser.parse_args()
    
    # Load data
    train_data = pd.read_csv(os.path.join(args.training, 'train.csv'))
    X = train_data.drop('target', axis=1)
    y = train_data['target']
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    val_predictions = model.predict(X_val)
    accuracy = accuracy_score(y_val, val_predictions)
    print(f"Validation Accuracy: {accuracy}")
    
    # Save model
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))

if __name__ == '__main__':
    train()
```

### Analytics Services

#### Amazon EMR
```python
import boto3

def create_emr_cluster():
    """Create EMR cluster for big data processing"""
    emr_client = boto3.client('emr')
    
    response = emr_client.run_job_flow(
        Name='DataProcessingCluster',
        ReleaseLabel='emr-6.4.0',
        Instances={
            'InstanceGroups': [
                {
                    'Name': "Master nodes",
                    'Market': 'ON_DEMAND',
                    'InstanceRole': 'MASTER',
                    'InstanceType': 'm5.xlarge',
                    'InstanceCount': 1,
                },
                {
                    'Name': "Worker nodes",
                    'Market': 'ON_DEMAND',
                    'InstanceRole': 'CORE',
                    'InstanceType': 'm5.xlarge',
                    'InstanceCount': 2,
                }
            ],
            'Ec2KeyName': 'my-key-pair',
            'KeepJobFlowAliveWhenNoSteps': True,
        },
        Applications=[
            {'Name': 'Spark'},
            {'Name': 'Hadoop'},
            {'Name': 'Jupyter'},
        ],
        ServiceRole='EMR_DefaultRole',
        JobFlowRole='EMR_EC2_DefaultRole',
        BootstrapActions=[
            {
                'Name': 'Install additional packages',
                'ScriptBootstrapAction': {
                    'Path': 's3://my-bucket/bootstrap-script.sh'
                }
            }
        ]
    )
    
    return response['JobFlowId']
```

#### AWS Glue
```python
import boto3

def create_glue_job():
    """Create AWS Glue ETL job"""
    glue_client = boto3.client('glue')
    
    response = glue_client.create_job(
        Name='data-etl-job',
        Role='arn:aws:iam::account:role/GlueServiceRole',
        Command={
            'Name': 'glueetl',
            'ScriptLocation': 's3://my-bucket/scripts/etl_script.py',
            'PythonVersion': '3'
        },
        DefaultArguments={
            '--TempDir': 's3://my-bucket/temp/',
            '--job-bookmark-option': 'job-bookmark-enable',
            '--enable-metrics': ''
        },
        MaxRetries=1,
        Timeout=60,
        GlueVersion='3.0'
    )
    
    return response

# Glue ETL Script Example
glue_script = """
import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job

args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Read data from S3
datasource = glueContext.create_dynamic_frame.from_catalog(
    database="my_database",
    table_name="raw_data"
)

# Transform data
transformed_data = ApplyMapping.apply(
    frame=datasource,
    mappings=[
        ("col1", "string", "column1", "string"),
        ("col2", "int", "column2", "int"),
        ("col3", "double", "column3", "double")
    ]
)

# Write to S3
glueContext.write_dynamic_frame.from_options(
    frame=transformed_data,
    connection_type="s3",
    connection_options={"path": "s3://my-bucket/processed/"},
    format="parquet"
)

job.commit()
"""
```

## 🐳 Container Deployments

### Amazon ECS
```yaml
# ecs-task-definition.json
{
  "family": "ml-inference-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::account:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "ml-inference",
      "image": "your-account.dkr.ecr.region.amazonaws.com/ml-inference:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "MODEL_PATH",
          "value": "s3://my-bucket/models/latest/"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/ml-inference",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

### Amazon EKS
```yaml
# kubernetes-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-model-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-model
  template:
    metadata:
      labels:
        app: ml-model
    spec:
      containers:
      - name: ml-model
        image: your-account.dkr.ecr.region.amazonaws.com/ml-model:latest
        ports:
        - containerPort: 8000
        env:
        - name: AWS_DEFAULT_REGION
          value: "us-west-2"
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
---
apiVersion: v1
kind: Service
metadata:
  name: ml-model-service
spec:
  selector:
    app: ml-model
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

## 🏗️ Infrastructure as Code

### CloudFormation Template
```yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: 'Data Science Infrastructure Stack'

Parameters:
  InstanceType:
    Type: String
    Default: t3.medium
    Description: EC2 instance type for Jupyter notebook server

Resources:
  # S3 Bucket for data storage
  DataBucket:
    Type: AWS::S3::Bucket
    Properties:
      BucketName: !Sub '${AWS::StackName}-data-bucket'
      VersioningConfiguration:
        Status: Enabled
      PublicAccessBlockConfiguration:
        BlockPublicAcls: true
        BlockPublicPolicy: true
        IgnorePublicAcls: true
        RestrictPublicBuckets: true

  # IAM Role for EC2 instances
  EC2Role:
    Type: AWS::IAM::Role
    Properties:
      AssumeRolePolicyDocument:
        Version: '2012-10-17'
        Statement:
          - Effect: Allow
            Principal:
              Service: ec2.amazonaws.com
            Action: sts:AssumeRole
      ManagedPolicyArns:
        - arn:aws:iam::aws:policy/AmazonS3FullAccess
        - arn:aws:iam::aws:policy/AmazonSageMakerFullAccess

  # Instance Profile
  EC2InstanceProfile:
    Type: AWS::IAM::InstanceProfile
    Properties:
      Roles:
        - !Ref EC2Role

  # Security Group
  JupyterSecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupDescription: Security group for Jupyter notebook server
      SecurityGroupIngress:
        - IpProtocol: tcp
          FromPort: 8888
          ToPort: 8888
          CidrIp: 0.0.0.0/0
        - IpProtocol: tcp
          FromPort: 22
          ToPort: 22
          CidrIp: 0.0.0.0/0

  # EC2 Instance
  JupyterInstance:
    Type: AWS::EC2::Instance
    Properties:
      ImageId: ami-0abcdef1234567890  # Deep Learning AMI
      InstanceType: !Ref InstanceType
      IamInstanceProfile: !Ref EC2InstanceProfile
      SecurityGroupIds:
        - !Ref JupyterSecurityGroup
      UserData:
        Fn::Base64: !Sub |
          #!/bin/bash
          yum update -y
          # Install additional packages
          pip install --upgrade pip
          pip install boto3 sagemaker pandas numpy scikit-learn

Outputs:
  DataBucketName:
    Description: Name of the S3 data bucket
    Value: !Ref DataBucket
  JupyterInstanceId:
    Description: Instance ID of the Jupyter server
    Value: !Ref JupyterInstance
```

### Terraform Configuration
```hcl
# main.tf
provider "aws" {
  region = var.aws_region
}

# S3 Bucket for data storage
resource "aws_s3_bucket" "data_bucket" {
  bucket = "${var.project_name}-data-bucket"
}

resource "aws_s3_bucket_versioning" "data_bucket_versioning" {
  bucket = aws_s3_bucket.data_bucket.id
  versioning_configuration {
    status = "Enabled"
  }
}

# SageMaker Notebook Instance
resource "aws_sagemaker_notebook_instance" "ml_notebook" {
  name          = "${var.project_name}-notebook"
  role_arn      = aws_iam_role.sagemaker_role.arn
  instance_type = var.notebook_instance_type

  tags = {
    Name = "${var.project_name}-notebook"
  }
}

# IAM Role for SageMaker
resource "aws_iam_role" "sagemaker_role" {
  name = "${var.project_name}-sagemaker-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "sagemaker_execution_role" {
  role       = aws_iam_role.sagemaker_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

# Variables
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-west-2"
}

variable "project_name" {
  description = "Name of the project"
  type        = string
}

variable "notebook_instance_type" {
  description = "SageMaker notebook instance type"
  type        = string
  default     = "ml.t3.medium"
}
```

## 📊 Monitoring and Logging

### CloudWatch Monitoring
```python
import boto3
import json

def create_custom_metric():
    """Create custom CloudWatch metric"""
    cloudwatch = boto3.client('cloudwatch')
    
    # Put custom metric
    cloudwatch.put_metric_data(
        Namespace='ML/ModelPerformance',
        MetricData=[
            {
                'MetricName': 'ModelAccuracy',
                'Value': 0.95,
                'Unit': 'Percent',
                'Dimensions': [
                    {
                        'Name': 'ModelName',
                        'Value': 'RandomForestClassifier'
                    }
                ]
            }
        ]
    )

def create_alarm():
    """Create CloudWatch alarm"""
    cloudwatch = boto3.client('cloudwatch')
    
    cloudwatch.put_metric_alarm(
        AlarmName='ModelAccuracyLow',
        ComparisonOperator='LessThanThreshold',
        EvaluationPeriods=2,
        MetricName='ModelAccuracy',
        Namespace='ML/ModelPerformance',
        Period=300,
        Statistic='Average',
        Threshold=0.8,
        ActionsEnabled=True,
        AlarmActions=[
            'arn:aws:sns:region:account:model-alerts'
        ],
        AlarmDescription='Alert when model accuracy drops below 80%'
    )
```

## 🔒 Security Best Practices

### IAM Policies
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
      "Resource": "arn:aws:s3:::my-data-bucket/*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "sagemaker:CreateTrainingJob",
        "sagemaker:DescribeTrainingJob",
        "sagemaker:CreateModel",
        "sagemaker:CreateEndpoint"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Deny",
      "Action": "*",
      "Resource": "*",
      "Condition": {
        "Bool": {
          "aws:SecureTransport": "false"
        }
      }
    }
  ]
}
```

### Secrets Management
```python
import boto3
import json

def get_secret(secret_name, region_name="us-west-2"):
    """Retrieve secret from AWS Secrets Manager"""
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )
    
    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
        secret = get_secret_value_response['SecretString']
        return json.loads(secret)
    except Exception as e:
        print(f"Error retrieving secret: {e}")
        return None

# Usage
db_credentials = get_secret("prod/database/credentials")
if db_credentials:
    username = db_credentials['username']
    password = db_credentials['password']
```

## 💰 Cost Optimization

### Spot Instances
```python
import boto3

def launch_spot_instance():
    """Launch EC2 spot instance for cost savings"""
    ec2 = boto3.client('ec2')
    
    response = ec2.request_spot_instances(
        SpotPrice='0.10',
        InstanceCount=1,
        Type='one-time',
        LaunchSpecification={
            'ImageId': 'ami-0abcdef1234567890',
            'InstanceType': 'm5.large',
            'KeyName': 'my-key-pair',
            'SecurityGroupIds': ['sg-903004f8'],
            'UserData': base64.b64encode(user_data_script.encode()).decode()
        }
    )
    
    return response['SpotInstanceRequests'][0]['SpotInstanceRequestId']
```

### Auto Scaling
```yaml
# auto-scaling-group.yaml
Resources:
  AutoScalingGroup:
    Type: AWS::AutoScaling::AutoScalingGroup
    Properties:
      VPCZoneIdentifier:
        - subnet-12345678
        - subnet-87654321
      LaunchTemplate:
        LaunchTemplateId: !Ref LaunchTemplate
        Version: !GetAtt LaunchTemplate.LatestVersionNumber
      MinSize: 1
      MaxSize: 10
      DesiredCapacity: 2
      TargetGroupARNs:
        - !Ref TargetGroup
      HealthCheckType: ELB
      HealthCheckGracePeriod: 300

  ScalingPolicy:
    Type: AWS::AutoScaling::ScalingPolicy
    Properties:
      AutoScalingGroupName: !Ref AutoScalingGroup
      PolicyType: TargetTrackingScaling
      TargetTrackingConfiguration:
        PredefinedMetricSpecification:
          PredefinedMetricType: ASGAverageCPUUtilization
        TargetValue: 70.0
```

## 🚀 Deployment Automation

### CI/CD Pipeline with CodePipeline
```yaml
# buildspec.yml
version: 0.2

phases:
  pre_build:
    commands:
      - echo Logging in to Amazon ECR...
      - aws ecr get-login-password --region $AWS_DEFAULT_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com
  build:
    commands:
      - echo Build started on `date`
      - echo Building the Docker image...
      - docker build -t $IMAGE_REPO_NAME:$IMAGE_TAG .
      - docker tag $IMAGE_REPO_NAME:$IMAGE_TAG $AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com/$IMAGE_REPO_NAME:$IMAGE_TAG
  post_build:
    commands:
      - echo Build completed on `date`
      - echo Pushing the Docker image...
      - docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com/$IMAGE_REPO_NAME:$IMAGE_TAG
      - echo Writing image definitions file...
      - printf '[{"name":"ml-container","imageUri":"%s"}]' $AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com/$IMAGE_REPO_NAME:$IMAGE_TAG > imagedefinitions.json

artifacts:
  files:
    - imagedefinitions.json
```

## 📚 Best Practices

### Data Management
1. **Use S3 lifecycle policies** for cost-effective storage
2. **Enable versioning** for critical datasets
3. **Implement data encryption** at rest and in transit
4. **Use S3 Transfer Acceleration** for large file uploads
5. **Organize data with consistent naming conventions**

### Model Development
1. **Use SageMaker for managed ML workflows**
2. **Implement model versioning and tracking**
3. **Set up automated model validation**
4. **Use spot instances for training to reduce costs**
5. **Monitor model performance in production**

### Security
1. **Follow principle of least privilege** for IAM roles
2. **Use VPC endpoints** for secure service communication
3. **Enable CloudTrail** for audit logging
4. **Encrypt sensitive data** using KMS
5. **Regularly rotate access keys and passwords**

### Cost Management
1. **Use Reserved Instances** for predictable workloads
2. **Implement auto-scaling** to match demand
3. **Monitor costs** with AWS Cost Explorer
4. **Use Spot Instances** for fault-tolerant workloads
5. **Set up billing alerts** for cost control

## 🔗 Additional Resources

- [AWS Machine Learning Documentation](https://docs.aws.amazon.com/machine-learning/)
- [SageMaker Developer Guide](https://docs.aws.amazon.com/sagemaker/)
- [AWS Well-Architected Framework](https://aws.amazon.com/architecture/well-architected/)
- [AWS Cost Optimization](https://aws.amazon.com/aws-cost-management/)
- [AWS Security Best Practices](https://aws.amazon.com/security/security-resources/) 
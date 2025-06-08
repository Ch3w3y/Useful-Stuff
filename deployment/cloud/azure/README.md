# Microsoft Azure for Data Science & ML Deployment

## Overview

This directory provides comprehensive guidance for deploying data science and machine learning solutions on Microsoft Azure. It covers Azure's ML services, data platforms, deployment patterns, and best practices for scalable AI/ML infrastructure.

## Table of Contents

- [Azure ML Platform](#azure-ml-platform)
- [Data Services](#data-services)
- [Deployment Patterns](#deployment-patterns)
- [MLOps with Azure](#mlops-with-azure)
- [Infrastructure as Code](#infrastructure-as-code)
- [Security & Governance](#security--governance)
- [Cost Optimization](#cost-optimization)
- [Monitoring & Observability](#monitoring--observability)
- [Best Practices](#best-practices)

## Azure ML Platform

### Azure Machine Learning Studio

```python
# Azure ML Python SDK v2 setup
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    Workspace, Environment, Model, CodeConfiguration,
    ManagedOnlineEndpoint, ManagedOnlineDeployment
)
from azure.identity import DefaultAzureCredential

# Initialize ML Client
ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id="your-subscription-id",
    resource_group_name="your-resource-group",
    workspace_name="your-workspace-name"
)

# Create or connect to workspace
workspace = Workspace(
    name="ml-workspace",
    location="eastus2",
    display_name="ML Development Workspace",
    description="Workspace for data science projects",
    tags={"environment": "development", "team": "data-science"}
)

# Create workspace
workspace_result = ml_client.workspaces.begin_create(workspace)
```

### Model Training Pipeline

```python
from azure.ai.ml import command, Input, Output
from azure.ai.ml.entities import Environment
from azure.ai.ml.constants import AssetTypes

# Define training environment
environment = Environment(
    name="sklearn-environment",
    conda_file="environment.yml",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest"
)

# Create training job
train_job = command(
    code="./src",
    command="python train.py --data-path ${{inputs.training_data}} --model-output ${{outputs.model_output}}",
    environment=environment,
    inputs={
        "training_data": Input(
            type=AssetTypes.URI_FOLDER,
            path="azureml://datastores/workspaceblobstore/paths/data/train"
        )
    },
    outputs={
        "model_output": Output(
            type=AssetTypes.URI_FOLDER,
            path="azureml://datastores/workspaceblobstore/paths/model"
        )
    },
    compute="cpu-cluster",
    display_name="sklearn-training-job",
    experiment_name="customer-churn-prediction"
)

# Submit job
job_result = ml_client.jobs.create_or_update(train_job)
print(f"Job submitted: {job_result.name}")
```

### Automated ML (AutoML)

```python
from azure.ai.ml.automl import classification
from azure.ai.ml.entities._inputs_outputs import Input

# AutoML classification task
classification_job = classification(
    training_data=Input(
        type=AssetTypes.MLTABLE,
        path="azureml://datastores/workspaceblobstore/paths/data/train"
    ),
    target_column_name="target",
    primary_metric="accuracy",
    experiment_name="automl-classification",
    compute="cpu-cluster",
    tags={"project": "customer-segmentation"}
)

# Configure AutoML settings
classification_job.set_limits(
    timeout_minutes=60,
    trial_timeout_minutes=20,
    max_trials=10,
    max_concurrent_trials=4
)

classification_job.set_training(
    blocked_training_algorithms=["LogisticRegression"],
    enable_onnx_compatible_models=True
)

# Submit AutoML job
automl_job = ml_client.jobs.create_or_update(classification_job)
```

### Model Registration and Versioning

```python
from azure.ai.ml.entities import Model

# Register model
model = Model(
    name="customer-churn-model",
    path="./outputs/model.pkl",
    description="Customer churn prediction model",
    type="custom_model",
    tags={"framework": "scikit-learn", "algorithm": "random-forest"}
)

registered_model = ml_client.models.create_or_update(model)
print(f"Model registered: {registered_model.name}, Version: {registered_model.version}")
```

## Data Services

### Azure Synapse Analytics

```python
# Synapse Spark session configuration
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("DataPreprocessing") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .getOrCreate()

# Read data from Azure Data Lake
df = spark.read.format("delta") \
    .option("header", "true") \
    .load("abfss://container@storageaccount.dfs.core.windows.net/data/raw/")

# Data transformation pipeline
processed_df = df \
    .filter(df.amount > 0) \
    .withColumn("date_processed", current_timestamp()) \
    .groupBy("customer_id") \
    .agg(
        sum("amount").alias("total_amount"),
        count("*").alias("transaction_count"),
        avg("amount").alias("avg_amount")
    )

# Write to processed layer
processed_df.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .save("abfss://container@storageaccount.dfs.core.windows.net/data/processed/")
```

### Azure Cosmos DB for ML Features

```python
import azure.cosmos.cosmos_client as cosmos_client
import azure.cosmos.errors as errors
from azure.cosmos.partition_key import PartitionKey

# Cosmos DB client
cosmos_client = cosmos_client.CosmosClient(
    url=cosmos_endpoint,
    credential=cosmos_key
)

# Create database and container for feature store
database = cosmos_client.create_database_if_not_exists(
    id="feature_store"
)

container = database.create_container_if_not_exists(
    id="customer_features",
    partition_key=PartitionKey(path="/customer_id"),
    offer_throughput=400
)

# Store feature vectors
def store_features(customer_id, features):
    feature_document = {
        "id": f"features_{customer_id}",
        "customer_id": customer_id,
        "features": features,
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0"
    }
    
    container.create_item(body=feature_document)

# Retrieve features for inference
def get_features(customer_id):
    query = f"SELECT * FROM c WHERE c.customer_id = '{customer_id}'"
    items = list(container.query_items(
        query=query,
        enable_cross_partition_query=True
    ))
    return items[0] if items else None
```

### Azure Data Factory ETL Pipeline

```json
{
  "name": "DataPreprocessingPipeline",
  "properties": {
    "activities": [
      {
        "name": "ExtractFromSQL",
        "type": "Copy",
        "typeProperties": {
          "source": {
            "type": "SqlSource",
            "sqlReaderQuery": "SELECT * FROM customers WHERE last_updated > '@{pipeline().parameters.last_run_date}'"
          },
          "sink": {
            "type": "DelimitedTextSink",
            "storeSettings": {
              "type": "AzureBlobFSWriteSettings",
              "copyBehavior": "FlattenHierarchy"
            }
          }
        },
        "inputs": [
          {
            "referenceName": "SqlServerDataset",
            "type": "DatasetReference"
          }
        ],
        "outputs": [
          {
            "referenceName": "DataLakeDataset",
            "type": "DatasetReference"
          }
        ]
      },
      {
        "name": "TransformData",
        "type": "ExecuteDataFlow",
        "typeProperties": {
          "dataflow": {
            "referenceName": "CustomerDataTransformation",
            "type": "DataFlowReference"
          },
          "compute": {
            "coreCount": 8,
            "computeType": "General"
          }
        }
      },
      {
        "name": "TriggerMLPipeline",
        "type": "AzureMLExecutePipeline",
        "typeProperties": {
          "mlPipelineId": "your-ml-pipeline-id"
        }
      }
    ],
    "parameters": {
      "last_run_date": {
        "type": "string",
        "defaultValue": "2024-01-01"
      }
    }
  }
}
```

## Deployment Patterns

### Real-time Inference Endpoint

```python
# Create online endpoint
endpoint = ManagedOnlineEndpoint(
    name="customer-churn-endpoint",
    description="Real-time customer churn prediction",
    auth_mode="key",
    tags={"environment": "production", "model": "churn-prediction"}
)

# Create endpoint
endpoint_result = ml_client.online_endpoints.begin_create_or_update(endpoint)

# Create deployment
deployment = ManagedOnlineDeployment(
    name="churn-deployment-v1",
    endpoint_name="customer-churn-endpoint",
    model=registered_model,
    environment=environment,
    code_configuration=CodeConfiguration(
        code="./src",
        scoring_script="score.py"
    ),
    instance_type="Standard_DS3_v2",
    instance_count=2,
    request_settings={
        "max_concurrent_requests_per_instance": 4,
        "request_timeout_ms": 60000
    },
    liveness_probe={
        "period": 10,
        "initial_delay": 30,
        "timeout": 5
    }
)

deployment_result = ml_client.online_deployments.begin_create_or_update(deployment)

# Set traffic allocation
endpoint.traffic = {"churn-deployment-v1": 100}
ml_client.online_endpoints.begin_create_or_update(endpoint)
```

### Batch Inference Pipeline

```python
from azure.ai.ml.entities import BatchEndpoint, BatchDeployment

# Create batch endpoint
batch_endpoint = BatchEndpoint(
    name="batch-scoring-endpoint",
    description="Batch scoring for customer segmentation"
)

batch_endpoint_result = ml_client.batch_endpoints.begin_create_or_update(batch_endpoint)

# Create batch deployment
batch_deployment = BatchDeployment(
    name="segmentation-batch-v1",
    endpoint_name="batch-scoring-endpoint",
    model=registered_model,
    environment=environment,
    code_configuration=CodeConfiguration(
        code="./src",
        scoring_script="batch_score.py"
    ),
    compute="cpu-cluster",
    instance_count=4,
    max_concurrency_per_instance=2,
    mini_batch_size=10,
    output_action="append_row"
)

batch_deployment_result = ml_client.batch_deployments.begin_create_or_update(batch_deployment)
```

### Azure Container Instances Deployment

```python
from azure.ai.ml.entities import OnlineRequestSettings

# Deploy to Azure Container Instances
aci_deployment = ManagedOnlineDeployment(
    name="aci-deployment",
    endpoint_name="ml-endpoint",
    model=registered_model,
    environment=environment,
    instance_type="Standard_DS2_v2",
    instance_count=1,
    request_settings=OnlineRequestSettings(
        max_concurrent_requests_per_instance=1,
        request_timeout_ms=30000
    )
)
```

### Azure Kubernetes Service (AKS) Deployment

```yaml
# aks-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-model-deployment
  labels:
    app: ml-model
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
        image: your-acr.azurecr.io/ml-model:latest
        ports:
        - containerPort: 5000
        env:
        - name: MODEL_PATH
          value: "/models/model.pkl"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 5000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: ml-model-service
spec:
  selector:
    app: ml-model
  ports:
  - protocol: TCP
    port: 80
    targetPort: 5000
  type: LoadBalancer
```

## MLOps with Azure

### Azure DevOps Pipeline for ML

```yaml
# azure-pipelines.yml
trigger:
  branches:
    include:
    - main
  paths:
    include:
    - src/*
    - data/*

variables:
  azureServiceConnectionName: 'azure-ml-service-connection'
  workspaceName: 'ml-workspace'
  resourceGroup: 'ml-resource-group'
  subscriptionId: 'your-subscription-id'

stages:
- stage: DataValidation
  displayName: 'Data Validation'
  jobs:
  - job: ValidateData
    displayName: 'Validate Training Data'
    pool:
      vmImage: 'ubuntu-latest'
    steps:
    - task: UsePythonVersion@0
      inputs:
        versionSpec: '3.8'
    - script: |
        pip install azure-ai-ml pandas great-expectations
        python scripts/validate_data.py
      displayName: 'Run Data Validation'

- stage: ModelTraining
  displayName: 'Model Training'
  dependsOn: DataValidation
  condition: succeeded()
  jobs:
  - job: TrainModel
    displayName: 'Train ML Model'
    pool:
      vmImage: 'ubuntu-latest'
    steps:
    - task: AzureCLI@2
      displayName: 'Submit Training Job'
      inputs:
        azureSubscription: $(azureServiceConnectionName)
        scriptType: 'bash'
        scriptLocation: 'inlineScript'
        inlineScript: |
          az extension add -n ml
          az ml job create --file training-job.yml \
            --workspace-name $(workspaceName) \
            --resource-group $(resourceGroup) \
            --subscription $(subscriptionId)

- stage: ModelValidation
  displayName: 'Model Validation'
  dependsOn: ModelTraining
  jobs:
  - job: ValidateModel
    displayName: 'Validate Model Performance'
    steps:
    - script: |
        python scripts/validate_model.py \
          --model-name customer-churn-model \
          --threshold 0.85
      displayName: 'Validate Model Metrics'

- stage: ModelDeployment
  displayName: 'Model Deployment'
  dependsOn: ModelValidation
  condition: and(succeeded(), eq(variables['Build.SourceBranch'], 'refs/heads/main'))
  jobs:
  - deployment: DeployToStaging
    displayName: 'Deploy to Staging'
    environment: 'staging'
    strategy:
      runOnce:
        deploy:
          steps:
          - task: AzureCLI@2
            displayName: 'Deploy to Staging Endpoint'
            inputs:
              azureSubscription: $(azureServiceConnectionName)
              scriptType: 'bash'
              scriptLocation: 'inlineScript'
              inlineScript: |
                az ml online-deployment create \
                  --file staging-deployment.yml \
                  --workspace-name $(workspaceName) \
                  --resource-group $(resourceGroup)

  - deployment: DeployToProduction
    displayName: 'Deploy to Production'
    dependsOn: DeployToStaging
    environment: 'production'
    strategy:
      runOnce:
        deploy:
          steps:
          - task: AzureCLI@2
            displayName: 'Deploy to Production Endpoint'
            inputs:
              azureSubscription: $(azureServiceConnectionName)
              scriptType: 'bash'
              scriptLocation: 'inlineScript'
              inlineScript: |
                az ml online-deployment create \
                  --file production-deployment.yml \
                  --workspace-name $(workspaceName) \
                  --resource-group $(resourceGroup)
```

### Model Monitoring and Drift Detection

```python
from azure.ai.ml.entities import AlertNotification
from azure.ai.ml.constants import MonitorTargetTasks

# Configure data drift monitoring
monitor_definition = {
    "compute": "cpu-cluster",
    "monitoring_target": {
        "ml_task": MonitorTargetTasks.CLASSIFICATION,
        "endpoint_deployment_id": "/subscriptions/.../endpoints/churn-endpoint/deployments/v1"
    },
    "monitoring_signals": {
        "data_drift": {
            "type": "data_drift",
            "target_dataset": {
                "input_data": {
                    "type": "mltable",
                    "path": "azureml://datastores/workspaceblobstore/paths/reference_data"
                }
            },
            "features": {
                "top_n_feature_importance": 10
            },
            "metric_thresholds": {
                "jensen_shannon_distance": 0.1
            }
        }
    },
    "alert_notification": {
        "emails": ["ml-team@company.com"]
    }
}
```

### A/B Testing for Model Deployment

```python
# Blue-Green deployment pattern
def deploy_with_ab_testing():
    # Deploy new model version to green slot
    green_deployment = ManagedOnlineDeployment(
        name="churn-deployment-v2",
        endpoint_name="customer-churn-endpoint",
        model=new_model_version,
        environment=environment,
        instance_type="Standard_DS3_v2",
        instance_count=1
    )
    
    ml_client.online_deployments.begin_create_or_update(green_deployment)
    
    # Gradually shift traffic
    traffic_stages = [
        {"churn-deployment-v1": 90, "churn-deployment-v2": 10},
        {"churn-deployment-v1": 70, "churn-deployment-v2": 30},
        {"churn-deployment-v1": 50, "churn-deployment-v2": 50},
        {"churn-deployment-v1": 0, "churn-deployment-v2": 100}
    ]
    
    for stage in traffic_stages:
        endpoint.traffic = stage
        ml_client.online_endpoints.begin_create_or_update(endpoint)
        
        # Monitor metrics and decide whether to continue
        time.sleep(600)  # Wait 10 minutes between stages
        metrics = get_deployment_metrics("churn-deployment-v2")
        
        if metrics["error_rate"] > 0.05:
            # Rollback if error rate too high
            endpoint.traffic = {"churn-deployment-v1": 100, "churn-deployment-v2": 0}
            ml_client.online_endpoints.begin_create_or_update(endpoint)
            break
```

## Infrastructure as Code

### ARM Template for ML Workspace

```json
{
  "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
  "contentVersion": "1.0.0.0",
  "parameters": {
    "workspaceName": {
      "type": "string",
      "metadata": {
        "description": "Name of the Azure ML workspace"
      }
    },
    "location": {
      "type": "string",
      "defaultValue": "[resourceGroup().location]",
      "metadata": {
        "description": "Location for all resources"
      }
    }
  },
  "variables": {
    "storageAccountName": "[concat('mlstorage', uniqueString(resourceGroup().id))]",
    "keyVaultName": "[concat('mlkv', uniqueString(resourceGroup().id))]",
    "applicationInsightsName": "[concat('mlai', uniqueString(resourceGroup().id))]"
  },
  "resources": [
    {
      "type": "Microsoft.Storage/storageAccounts",
      "apiVersion": "2021-04-01",
      "name": "[variables('storageAccountName')]",
      "location": "[parameters('location')]",
      "sku": {
        "name": "Standard_LRS"
      },
      "kind": "StorageV2",
      "properties": {
        "encryption": {
          "services": {
            "blob": {
              "enabled": true
            },
            "file": {
              "enabled": true
            }
          },
          "keySource": "Microsoft.Storage"
        },
        "supportsHttpsTrafficOnly": true
      }
    },
    {
      "type": "Microsoft.KeyVault/vaults",
      "apiVersion": "2021-04-01-preview",
      "name": "[variables('keyVaultName')]",
      "location": "[parameters('location')]",
      "properties": {
        "tenantId": "[subscription().tenantId]",
        "sku": {
          "name": "standard",
          "family": "A"
        },
        "accessPolicies": []
      }
    },
    {
      "type": "Microsoft.MachineLearningServices/workspaces",
      "apiVersion": "2021-07-01",
      "name": "[parameters('workspaceName')]",
      "location": "[parameters('location')]",
      "dependsOn": [
        "[resourceId('Microsoft.Storage/storageAccounts', variables('storageAccountName'))]",
        "[resourceId('Microsoft.KeyVault/vaults', variables('keyVaultName'))]",
        "[resourceId('Microsoft.Insights/components', variables('applicationInsightsName'))]"
      ],
      "properties": {
        "storageAccount": "[resourceId('Microsoft.Storage/storageAccounts', variables('storageAccountName'))]",
        "keyVault": "[resourceId('Microsoft.KeyVault/vaults', variables('keyVaultName'))]",
        "applicationInsights": "[resourceId('Microsoft.Insights/components', variables('applicationInsightsName'))]"
      }
    }
  ]
}
```

### Terraform Configuration

```hcl
# main.tf
terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~>3.0"
    }
  }
}

provider "azurerm" {
  features {}
}

resource "azurerm_resource_group" "ml_rg" {
  name     = "ml-workspace-rg"
  location = var.location
  
  tags = {
    Environment = var.environment
    Purpose     = "Machine Learning"
  }
}

resource "azurerm_storage_account" "ml_storage" {
  name                     = "mlstorage${random_string.suffix.result}"
  resource_group_name      = azurerm_resource_group.ml_rg.name
  location                = azurerm_resource_group.ml_rg.location
  account_tier            = "Standard"
  account_replication_type = "LRS"
  
  tags = azurerm_resource_group.ml_rg.tags
}

resource "azurerm_machine_learning_workspace" "ml_workspace" {
  name                = "ml-workspace"
  location           = azurerm_resource_group.ml_rg.location
  resource_group_name = azurerm_resource_group.ml_rg.name
  
  application_insights_id = azurerm_application_insights.ml_insights.id
  key_vault_id           = azurerm_key_vault.ml_keyvault.id
  storage_account_id     = azurerm_storage_account.ml_storage.id
  
  identity {
    type = "SystemAssigned"
  }
  
  tags = azurerm_resource_group.ml_rg.tags
}

resource "azurerm_machine_learning_compute_cluster" "ml_cluster" {
  name                          = "cpu-cluster"
  location                     = azurerm_resource_group.ml_rg.location
  machine_learning_workspace_id = azurerm_machine_learning_workspace.ml_workspace.id
  vm_priority                  = "Dedicated"
  vm_size                     = "Standard_DS3_v2"
  
  scale_settings {
    min_node_count                       = 0
    max_node_count                       = 4
    scale_down_nodes_after_idle_duration = "PT30S"
  }
}
```

### Bicep Template

```bicep
// ml-workspace.bicep
@description('Name of the Azure ML workspace')
param workspaceName string

@description('Location for all resources')
param location string = resourceGroup().location

@description('Environment tag')
param environment string = 'dev'

var storageAccountName = 'mlstorage${uniqueString(resourceGroup().id)}'
var keyVaultName = 'mlkv${uniqueString(resourceGroup().id)}'
var applicationInsightsName = 'mlai${uniqueString(resourceGroup().id)}'

resource storageAccount 'Microsoft.Storage/storageAccounts@2021-04-01' = {
  name: storageAccountName
  location: location
  sku: {
    name: 'Standard_LRS'
  }
  kind: 'StorageV2'
  properties: {
    encryption: {
      services: {
        blob: {
          enabled: true
        }
        file: {
          enabled: true
        }
      }
      keySource: 'Microsoft.Storage'
    }
    supportsHttpsTrafficOnly: true
  }
  tags: {
    Environment: environment
  }
}

resource keyVault 'Microsoft.KeyVault/vaults@2021-04-01-preview' = {
  name: keyVaultName
  location: location
  properties: {
    tenantId: subscription().tenantId
    sku: {
      name: 'standard'
      family: 'A'
    }
    accessPolicies: []
  }
  tags: {
    Environment: environment
  }
}

resource mlWorkspace 'Microsoft.MachineLearningServices/workspaces@2021-07-01' = {
  name: workspaceName
  location: location
  properties: {
    storageAccount: storageAccount.id
    keyVault: keyVault.id
    applicationInsights: applicationInsights.id
  }
  tags: {
    Environment: environment
  }
}

output workspaceId string = mlWorkspace.id
output workspaceName string = mlWorkspace.name
```

## Security & Governance

### Network Security Configuration

```python
from azure.ai.ml.entities import Workspace

# Create workspace with private endpoint
workspace = Workspace(
    name="secure-ml-workspace",
    location="eastus2",
    public_network_access="Disabled",
    image_build_compute="cpu-cluster",
    tags={"security": "private-endpoint"}
)

# Configure private endpoint
workspace.managed_network = {
    "isolation_mode": "AllowInternetOutbound",
    "outbound_rules": {
        "required-outbound": {
            "type": "service_tag",
            "destination": {
                "service_tag": "AzureActiveDirectory",
                "protocol": "TCP",
                "port_ranges": "443"
            }
        }
    }
}
```

### Role-Based Access Control (RBAC)

```json
{
  "properties": {
    "roleName": "ML Engineer",
    "description": "Custom role for ML engineers",
    "assignableScopes": [
      "/subscriptions/{subscription-id}/resourceGroups/{resource-group}"
    ],
    "permissions": [
      {
        "actions": [
          "Microsoft.MachineLearningServices/workspaces/read",
          "Microsoft.MachineLearningServices/workspaces/computes/read",
          "Microsoft.MachineLearningServices/workspaces/computes/write",
          "Microsoft.MachineLearningServices/workspaces/experiments/read",
          "Microsoft.MachineLearningServices/workspaces/experiments/write",
          "Microsoft.MachineLearningServices/workspaces/models/read",
          "Microsoft.MachineLearningServices/workspaces/models/write"
        ],
        "notActions": [
          "Microsoft.MachineLearningServices/workspaces/delete",
          "Microsoft.MachineLearningServices/workspaces/computes/delete"
        ],
        "dataActions": [
          "Microsoft.MachineLearningServices/workspaces/datasets/read",
          "Microsoft.MachineLearningServices/workspaces/datasets/write"
        ]
      }
    ]
  }
}
```

### Data Encryption and Key Management

```python
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential

# Configure encryption with customer-managed keys
encryption_config = {
    "keyVaultProperties": {
        "keyVaultArmId": "/subscriptions/.../vaults/ml-keyvault",
        "keyIdentifier": "https://ml-keyvault.vault.azure.net/keys/workspace-key/version",
        "identityClientId": "managed-identity-client-id"
    },
    "status": "Enabled"
}

# Access secrets securely
credential = DefaultAzureCredential()
secret_client = SecretClient(
    vault_url="https://ml-keyvault.vault.azure.net/",
    credential=credential
)

# Store sensitive model parameters
secret_client.set_secret("model-api-key", "your-secret-api-key")
api_key = secret_client.get_secret("model-api-key").value
```

## Cost Optimization

### Auto-scaling Compute Resources

```python
from azure.ai.ml.entities import AmlCompute

# Create auto-scaling compute cluster
compute_config = AmlCompute(
    name="auto-scale-cluster",
    type="amlcompute",
    size="Standard_DS3_v2",
    min_instances=0,
    max_instances=10,
    idle_time_before_scale_down=120,  # 2 minutes
    tier="Dedicated"
)

compute_cluster = ml_client.compute.begin_create_or_update(compute_config)
```

### Spot Instance Configuration

```python
# Use low-priority VMs for cost savings
spot_compute = AmlCompute(
    name="spot-cluster",
    type="amlcompute",
    size="Standard_DS3_v2",
    min_instances=0,
    max_instances=20,
    tier="LowPriority"  # Use spot instances
)
```

### Cost Monitoring and Alerts

```python
# Azure Cost Management integration
from azure.mgmt.consumption import ConsumptionManagementClient

consumption_client = ConsumptionManagementClient(
    credential=DefaultAzureCredential(),
    subscription_id="your-subscription-id"
)

# Get cost breakdown by resource group
costs = consumption_client.usage_details.list(
    scope=f"/subscriptions/{subscription_id}/resourceGroups/{resource_group}",
    filter="properties/usageStart ge 2024-01-01 and properties/usageEnd le 2024-01-31"
)

for cost in costs:
    print(f"Resource: {cost.instance_name}, Cost: ${cost.cost}")
```

## Monitoring & Observability

### Application Insights Integration

```python
from applicationinsights import TelemetryClient
from azure.ai.ml.entities import Model

# Initialize telemetry client
telemetry_client = TelemetryClient("your-instrumentation-key")

# Custom scoring script with monitoring
def init():
    global model, telemetry_client
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.pkl')
    model = joblib.load(model_path)
    telemetry_client.track_event('Model initialized')

def run(raw_data):
    start_time = time.time()
    
    try:
        data = json.loads(raw_data)['data']
        predictions = model.predict(data)
        
        # Log prediction metrics
        telemetry_client.track_metric('prediction_count', len(predictions))
        telemetry_client.track_metric('inference_duration', time.time() - start_time)
        
        return predictions.tolist()
        
    except Exception as e:
        telemetry_client.track_exception()
        raise e
```

### Custom Metrics and Logging

```python
import logging
from azure.monitor.opentelemetry.exporter import AzureMonitorLogExporter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Azure Monitor exporter
exporter = AzureMonitorLogExporter(
    connection_string="InstrumentationKey=your-key"
)

# Log custom metrics during training
def log_training_metrics(epoch, loss, accuracy):
    logger.info(f"Epoch: {epoch}, Loss: {loss}, Accuracy: {accuracy}")
    
    telemetry_client.track_metric('training_loss', loss)
    telemetry_client.track_metric('training_accuracy', accuracy)
    telemetry_client.track_metric('epoch', epoch)
```

### Health Checks and Probes

```python
# Health check endpoint for model service
from flask import Flask, jsonify
import joblib
import os

app = Flask(__name__)

@app.route('/health')
def health_check():
    try:
        # Check if model is loaded
        model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.pkl')
        if os.path.exists(model_path):
            return jsonify({'status': 'healthy', 'model': 'loaded'}), 200
        else:
            return jsonify({'status': 'unhealthy', 'error': 'model not found'}), 503
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 503

@app.route('/ready')
def readiness_check():
    try:
        # Perform a dummy prediction to ensure everything works
        dummy_input = [[1, 2, 3, 4, 5]]
        prediction = model.predict(dummy_input)
        return jsonify({'status': 'ready'}), 200
    except Exception as e:
        return jsonify({'status': 'not ready', 'error': str(e)}), 503
```

## Best Practices

### Model Versioning and Lineage

```python
from azure.ai.ml.entities import Model
import mlflow

# Track model lineage with MLflow
mlflow.set_tracking_uri(workspace.get_mlflow_tracking_uri())

with mlflow.start_run() as run:
    # Log parameters
    mlflow.log_param("algorithm", "random_forest")
    mlflow.log_param("n_estimators", 100)
    
    # Log metrics
    mlflow.log_metric("accuracy", 0.85)
    mlflow.log_metric("f1_score", 0.82)
    
    # Log model
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="customer-churn-model"
    )
    
    # Log additional artifacts
    mlflow.log_artifact("feature_importance.png")
    mlflow.log_artifact("confusion_matrix.png")
```

### Environment Management

```yaml
# environment.yml
name: ml-environment
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.8
  - pip
  - scikit-learn=1.1.0
  - pandas=1.4.0
  - numpy=1.21.0
  - matplotlib=3.5.0
  - seaborn=0.11.0
  - pip:
    - azure-ai-ml==1.5.0
    - mlflow==2.2.0
    - joblib==1.1.0
    - flask==2.0.3
```

### Data Governance and Compliance

```python
# Data lineage tracking
from azure.ai.ml.entities import Data

# Register dataset with lineage information
training_data = Data(
    name="customer-churn-dataset",
    version="1.0",
    description="Customer churn prediction training data",
    path="azureml://datastores/workspaceblobstore/paths/data/train/",
    tags={
        "source": "crm_database",
        "last_updated": "2024-01-15",
        "privacy_level": "confidential",
        "retention_policy": "7_years"
    }
)

ml_client.data.create_or_update(training_data)

# Data quality validation
def validate_data_quality(data_path):
    import great_expectations as ge
    
    df = ge.read_csv(data_path)
    
    # Define expectations
    df.expect_column_to_exist("customer_id")
    df.expect_column_values_to_not_be_null("customer_id")
    df.expect_column_values_to_be_between("age", min_value=18, max_value=100)
    
    # Validate expectations
    validation_result = df.validate()
    
    if not validation_result.success:
        raise ValueError("Data quality validation failed")
    
    return validation_result
```

### Performance Optimization

```python
# Optimize model inference
import onnx
import onnxruntime as ort
from skl2onnx import convert_sklearn

# Convert sklearn model to ONNX for faster inference
onnx_model = convert_sklearn(
    model,
    initial_types=[("input", FloatTensorType([None, n_features]))],
    target_opset=11
)

# Save ONNX model
with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

# Create inference session
ort_session = ort.InferenceSession("model.onnx")

def predict_onnx(input_data):
    input_name = ort_session.get_inputs()[0].name
    predictions = ort_session.run(None, {input_name: input_data})
    return predictions[0]
```

---

*This comprehensive guide covers Azure's machine learning and data science capabilities. Adapt these patterns and examples to your specific use cases and organizational requirements.* 
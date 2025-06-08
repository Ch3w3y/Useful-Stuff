# Azure Data Science Complete Setup Guide

A comprehensive guide for setting up Azure infrastructure for data science and machine learning projects, covering all relevant services, security, cost optimization, and best practices.

## Table of Contents

1. [Initial Setup and Prerequisites](#initial-setup-and-prerequisites)
2. [Core Azure Services for Data Science](#core-azure-services-for-data-science)
3. [Azure Machine Learning Studio](#azure-machine-learning-studio)
4. [Data Storage Solutions](#data-storage-solutions)
5. [Computing Resources](#computing-resources)
6. [Analytics and BI Services](#analytics-and-bi-services)
7. [Security and Compliance](#security-and-compliance)
8. [Cost Management and Optimization](#cost-management-and-optimization)
9. [Monitoring and Logging](#monitoring-and-logging)
10. [Sample Architectures](#sample-architectures)
11. [Best Practices](#best-practices)
12. [Troubleshooting](#troubleshooting)

## Initial Setup and Prerequisites

### Install Azure CLI

#### Ubuntu/Debian
```bash
# Method 1: Using Microsoft's repository
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Method 2: Using snap
sudo snap install azure-cli --classic

# Method 3: Using pip
pip install azure-cli
```

#### Arch Linux/CachyOS
```bash
# Using AUR
yay -S azure-cli

# Or using pip
pip install azure-cli
```

### Authentication Setup

```bash
# Login to Azure
az login

# Set default subscription
az account set --subscription "Your-Subscription-Name"

# List all subscriptions
az account list --output table

# Create service principal for programmatic access
az ad sp create-for-rbac --name "data-science-sp" --role contributor
```

### Install Additional Tools

```bash
# Azure Storage Explorer (GUI)
sudo snap install storage-explorer

# Azure Functions Core Tools
sudo npm install -g azure-functions-core-tools@4 --unsafe-perm true

# Azure Data CLI
pip install azure-data-cli

# Terraform for Infrastructure as Code
sudo snap install terraform
```

## Core Azure Services for Data Science

### Resource Group Setup

```bash
# Create resource group
az group create \
  --name rg-datascience-prod \
  --location eastus2 \
  --tags environment=production team=datascience

# List resource groups
az group list --output table
```

### Virtual Network Configuration

```bash
# Create virtual network
az network vnet create \
  --resource-group rg-datascience-prod \
  --name vnet-datascience \
  --address-prefix 10.0.0.0/16 \
  --subnet-name subnet-compute \
  --subnet-prefix 10.0.1.0/24

# Create additional subnets
az network vnet subnet create \
  --resource-group rg-datascience-prod \
  --vnet-name vnet-datascience \
  --name subnet-data \
  --address-prefix 10.0.2.0/24

az network vnet subnet create \
  --resource-group rg-datascience-prod \
  --vnet-name vnet-datascience \
  --name subnet-ml \
  --address-prefix 10.0.3.0/24
```

## Azure Machine Learning Studio

### Create ML Workspace

```bash
# Create Application Insights
az monitor app-insights component create \
  --app ml-workspace-insights \
  --location eastus2 \
  --resource-group rg-datascience-prod

# Create Key Vault
az keyvault create \
  --name kv-mlworkspace-unique123 \
  --resource-group rg-datascience-prod \
  --location eastus2

# Create Container Registry
az acr create \
  --name acrmlworkspaceunique123 \
  --resource-group rg-datascience-prod \
  --sku Basic \
  --admin-enabled true

# Create ML Workspace
az ml workspace create \
  --workspace-name ml-workspace-prod \
  --resource-group rg-datascience-prod \
  --location eastus2 \
  --application-insights /subscriptions/{subscription-id}/resourceGroups/rg-datascience-prod/providers/Microsoft.Insights/components/ml-workspace-insights \
  --keyvault /subscriptions/{subscription-id}/resourceGroups/rg-datascience-prod/providers/Microsoft.KeyVault/vaults/kv-mlworkspace-unique123 \
  --container-registry /subscriptions/{subscription-id}/resourceGroups/rg-datascience-prod/providers/Microsoft.ContainerRegistry/registries/acrmlworkspaceunique123
```

### Create Compute Instances

```bash
# Create CPU compute instance
az ml compute create \
  --name cpu-instance-01 \
  --type ComputeInstance \
  --size Standard_DS3_v2 \
  --workspace-name ml-workspace-prod \
  --resource-group rg-datascience-prod

# Create GPU compute instance
az ml compute create \
  --name gpu-instance-01 \
  --type ComputeInstance \
  --size Standard_NC6s_v3 \
  --workspace-name ml-workspace-prod \
  --resource-group rg-datascience-prod

# Create compute cluster for training
az ml compute create \
  --name training-cluster \
  --type AmlCompute \
  --min-instances 0 \
  --max-instances 4 \
  --size Standard_DS3_v2 \
  --workspace-name ml-workspace-prod \
  --resource-group rg-datascience-prod
```

### Create Datastores

```bash
# Register blob storage datastore
az ml datastore create \
  --name blob-datastore \
  --type AzureBlob \
  --account-name yourstorageaccount \
  --container-name data \
  --workspace-name ml-workspace-prod \
  --resource-group rg-datascience-prod

# Register file share datastore
az ml datastore create \
  --name fileshare-datastore \
  --type AzureFile \
  --account-name yourstorageaccount \
  --file-share-name datasets \
  --workspace-name ml-workspace-prod \
  --resource-group rg-datascience-prod
```

## Data Storage Solutions

### Azure Blob Storage

```bash
# Create storage account
az storage account create \
  --name datasciencestorageuniq \
  --resource-group rg-datascience-prod \
  --location eastus2 \
  --sku Standard_LRS \
  --kind StorageV2 \
  --access-tier Cool \
  --enable-hierarchical-namespace true

# Create containers
az storage container create \
  --name raw-data \
  --account-name datasciencestorageuniq

az storage container create \
  --name processed-data \
  --account-name datasciencestorageuniq

az storage container create \
  --name models \
  --account-name datasciencestorageuniq

az storage container create \
  --name notebooks \
  --account-name datasciencestorageuniq
```

### Azure Data Lake Storage Gen2

```bash
# Enable hierarchical namespace (Data Lake)
az storage account update \
  --name datasciencestorageuniq \
  --resource-group rg-datascience-prod \
  --enable-hierarchical-namespace true

# Create file systems
az storage fs create \
  --name bronze \
  --account-name datasciencestorageuniq

az storage fs create \
  --name silver \
  --account-name datasciencestorageuniq

az storage fs create \
  --name gold \
  --account-name datasciencestorageuniq
```

### Azure SQL Database

```bash
# Create SQL Server
az sql server create \
  --name sql-datascience-server \
  --resource-group rg-datascience-prod \
  --location eastus2 \
  --admin-user sqladmin \
  --admin-password 'YourSecurePassword123!'

# Create SQL Database
az sql db create \
  --resource-group rg-datascience-prod \
  --server sql-datascience-server \
  --name datascience-db \
  --service-objective S2

# Configure firewall
az sql server firewall-rule create \
  --resource-group rg-datascience-prod \
  --server sql-datascience-server \
  --name AllowAzureServices \
  --start-ip-address 0.0.0.0 \
  --end-ip-address 0.0.0.0
```

### Azure Cosmos DB

```bash
# Create Cosmos DB account
az cosmosdb create \
  --name cosmos-datascience-db \
  --resource-group rg-datascience-prod \
  --locations regionName=eastus2 failoverPriority=0 isZoneRedundant=False

# Create SQL API database
az cosmosdb sql database create \
  --account-name cosmos-datascience-db \
  --resource-group rg-datascience-prod \
  --name analytics-db

# Create container
az cosmosdb sql container create \
  --account-name cosmos-datascience-db \
  --database-name analytics-db \
  --resource-group rg-datascience-prod \
  --name experiments \
  --partition-key-path "/experimentId" \
  --throughput 400
```

## Computing Resources

### Virtual Machine Scale Sets

```bash
# Create VM Scale Set for distributed computing
az vmss create \
  --resource-group rg-datascience-prod \
  --name vmss-datascience \
  --image UbuntuLTS \
  --upgrade-policy-mode automatic \
  --instance-count 2 \
  --admin-username azureuser \
  --generate-ssh-keys \
  --vm-sku Standard_DS2_v2 \
  --vnet-name vnet-datascience \
  --subnet subnet-compute

# Install extensions
az vmss extension set \
  --publisher Microsoft.Azure.Extensions \
  --version 2.0 \
  --name CustomScript \
  --resource-group rg-datascience-prod \
  --vmss-name vmss-datascience \
  --settings '{"fileUris":["https://raw.githubusercontent.com/Azure/azure-quickstart-templates/master/application-workloads/datascience/datascience-vm-ubuntu/scripts/deploy.sh"],"commandToExecute":"./deploy.sh"}'
```

### Azure Batch

```bash
# Create Batch account
az batch account create \
  --name batchdatascienceuniq \
  --resource-group rg-datascience-prod \
  --location eastus2

# Create pool
az batch pool create \
  --id data-processing-pool \
  --vm-size Standard_D2s_v3 \
  --target-dedicated-nodes 2 \
  --image canonical:0001-com-ubuntu-server-focal:20_04-lts \
  --node-agent-sku-id "batch.node.ubuntu 20.04"
```

### Azure Container Instances

```bash
# Create container group for Jupyter
az container create \
  --resource-group rg-datascience-prod \
  --name jupyter-container \
  --image jupyter/datascience-notebook:latest \
  --dns-name-label jupyter-datascience-unique \
  --ports 8888 \
  --cpu 2 \
  --memory 4 \
  --environment-variables JUPYTER_ENABLE_LAB=yes
```

## Analytics and BI Services

### Azure Synapse Analytics

```bash
# Create Synapse workspace
az synapse workspace create \
  --name synapse-datascience-ws \
  --resource-group rg-datascience-prod \
  --location eastus2 \
  --storage-account datasciencestorageuniq \
  --file-system synapse \
  --sql-admin-login-user sqladmin \
  --sql-admin-login-password 'YourSecurePassword123!'

# Create SQL pool
az synapse sql pool create \
  --name DataWarehouse \
  --performance-level DW100c \
  --workspace-name synapse-datascience-ws \
  --resource-group rg-datascience-prod

# Create Spark pool
az synapse spark pool create \
  --name sparkpool01 \
  --workspace-name synapse-datascience-ws \
  --resource-group rg-datascience-prod \
  --spark-version 3.1 \
  --node-count 3 \
  --node-size Medium
```

### Azure Data Factory

```bash
# Create Data Factory
az datafactory create \
  --resource-group rg-datascience-prod \
  --factory-name adf-datascience-etl \
  --location eastus2

# Create linked service for blob storage
az datafactory linked-service create \
  --resource-group rg-datascience-prod \
  --factory-name adf-datascience-etl \
  --linked-service-name BlobStorage \
  --properties '{
    "type": "AzureBlobStorage",
    "typeProperties": {
      "connectionString": "DefaultEndpointsProtocol=https;AccountName=datasciencestorageuniq;AccountKey=YOUR_ACCOUNT_KEY"
    }
  }'
```

### Azure Stream Analytics

```bash
# Create Stream Analytics job
az stream-analytics job create \
  --resource-group rg-datascience-prod \
  --name sa-realtime-analytics \
  --location eastus2 \
  --sku Standard \
  --output-error-policy Drop \
  --events-out-of-order-policy Adjust \
  --events-out-of-order-max-delay 00:00:05
```

## Security and Compliance

### Azure Key Vault Setup

```bash
# Create Key Vault
az keyvault create \
  --name kv-datascience-secrets \
  --resource-group rg-datascience-prod \
  --location eastus2 \
  --enabled-for-disk-encryption \
  --enabled-for-deployment \
  --enabled-for-template-deployment

# Add secrets
az keyvault secret set \
  --vault-name kv-datascience-secrets \
  --name database-connection-string \
  --value "Server=sql-datascience-server.database.windows.net;Database=datascience-db;User Id=sqladmin;Password=YourSecurePassword123!;"

az keyvault secret set \
  --vault-name kv-datascience-secrets \
  --name storage-account-key \
  --value "YOUR_STORAGE_ACCOUNT_KEY"
```

### Role-Based Access Control (RBAC)

```bash
# Create custom role for data scientists
az role definition create --role-definition '{
  "Name": "Data Scientist",
  "Description": "Can access ML workspace and data resources",
  "Actions": [
    "Microsoft.MachineLearningServices/workspaces/read",
    "Microsoft.MachineLearningServices/workspaces/write",
    "Microsoft.Storage/storageAccounts/blobServices/containers/read",
    "Microsoft.Storage/storageAccounts/blobServices/containers/write",
    "Microsoft.Sql/servers/databases/read"
  ],
  "NotActions": [],
  "AssignableScopes": ["/subscriptions/{subscription-id}"]
}'

# Assign role to user
az role assignment create \
  --assignee user@example.com \
  --role "Data Scientist" \
  --scope "/subscriptions/{subscription-id}/resourceGroups/rg-datascience-prod"
```

### Network Security

```bash
# Create Network Security Group
az network nsg create \
  --resource-group rg-datascience-prod \
  --name nsg-datascience

# Allow SSH
az network nsg rule create \
  --resource-group rg-datascience-prod \
  --nsg-name nsg-datascience \
  --name AllowSSH \
  --protocol tcp \
  --priority 1001 \
  --destination-port-range 22 \
  --access allow

# Allow Jupyter
az network nsg rule create \
  --resource-group rg-datascience-prod \
  --nsg-name nsg-datascience \
  --name AllowJupyter \
  --protocol tcp \
  --priority 1002 \
  --destination-port-range 8888 \
  --access allow
```

## Cost Management and Optimization

### Set Up Budgets and Alerts

```bash
# Create budget
az consumption budget create \
  --budget-name datascience-monthly-budget \
  --amount 1000 \
  --time-grain Monthly \
  --time-period start-date=2024-01-01 \
  --resource-group rg-datascience-prod
```

### Cost Optimization Strategies

#### 1. Auto-shutdown for VMs
```bash
# Enable auto-shutdown for VM
az vm auto-shutdown \
  --resource-group rg-datascience-prod \
  --name your-vm-name \
  --time 1900 \
  --email your-email@example.com
```

#### 2. Use Spot Instances
```bash
# Create VM with spot pricing
az vm create \
  --resource-group rg-datascience-prod \
  --name spot-vm-datascience \
  --image UbuntuLTS \
  --size Standard_DS2_v2 \
  --priority Spot \
  --max-price 0.05 \
  --eviction-policy Deallocate
```

#### 3. Storage Lifecycle Management
```bash
# Create lifecycle policy
az storage account management-policy create \
  --account-name datasciencestorageuniq \
  --resource-group rg-datascience-prod \
  --policy '{
    "rules": [
      {
        "name": "MoveToIA",
        "type": "Lifecycle",
        "definition": {
          "filters": {
            "blobTypes": ["blockBlob"]
          },
          "actions": {
            "baseBlob": {
              "tierToCool": {
                "daysAfterModificationGreaterThan": 30
              },
              "tierToArchive": {
                "daysAfterModificationGreaterThan": 90
              }
            }
          }
        }
      }
    ]
  }'
```

## Monitoring and Logging

### Azure Monitor Setup

```bash
# Create Log Analytics workspace
az monitor log-analytics workspace create \
  --resource-group rg-datascience-prod \
  --workspace-name law-datascience-monitoring \
  --location eastus2

# Enable monitoring for VM
az vm extension set \
  --resource-group rg-datascience-prod \
  --vm-name your-vm-name \
  --name OmsAgentForLinux \
  --publisher Microsoft.EnterpriseCloud.Monitoring \
  --version 1.13
```

### Custom Metrics and Alerts

```bash
# Create metric alert for high CPU usage
az monitor metrics alert create \
  --name "High CPU Usage" \
  --resource-group rg-datascience-prod \
  --scopes /subscriptions/{subscription-id}/resourceGroups/rg-datascience-prod/providers/Microsoft.Compute/virtualMachines/your-vm-name \
  --condition "avg Percentage CPU > 80" \
  --window-size 5m \
  --evaluation-frequency 1m \
  --action-group your-action-group
```

## Sample Architectures

### Architecture 1: ML Pipeline with Azure ML

```bash
# Create environment file
cat > ml-pipeline-env.yml << EOF
name: ml-pipeline
dependencies:
  - python=3.8
  - pandas
  - scikit-learn
  - azure-ml-sdk
  - azureml-defaults
EOF

# Create training script
cat > train.py << EOF
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import argparse
from azureml.core import Run

# Get run context
run = Run.get_context()

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data-path', type=str, help='Path to training data')
parser.add_argument('--model-path', type=str, help='Path to save model')
args = parser.parse_args()

# Load data
data = pd.read_csv(args.data_path)
X = data.drop('target', axis=1)
y = data['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Log metrics
run.log('accuracy', accuracy)

# Save model
joblib.dump(model, args.model_path)
print(f'Model saved to {args.model_path}')
EOF
```

### Architecture 2: Real-time Analytics with Stream Analytics

```sql
-- Stream Analytics Query
SELECT 
    System.Timestamp AS EventTime,
    DeviceId,
    AVG(Temperature) AS AvgTemperature,
    COUNT(*) AS EventCount
INTO 
    [PowerBIOutput]
FROM 
    [IoTInput] TIMESTAMP BY EventTime
GROUP BY 
    DeviceId, 
    TumblingWindow(minute, 5)
HAVING 
    AVG(Temperature) > 25
```

### Architecture 3: Data Lake with Databricks

```python
# Databricks notebook cell
# Mount Azure Data Lake Storage
dbutils.fs.mount(
  source = "abfss://data@datasciencestorageuniq.dfs.core.windows.net/",
  mount_point = "/mnt/datalake",
  extra_configs = {
    "fs.azure.account.auth.type.datasciencestorageuniq.dfs.core.windows.net": "OAuth",
    "fs.azure.account.oauth.provider.type.datasciencestorageuniq.dfs.core.windows.net": "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider",
    "fs.azure.account.oauth2.client.id.datasciencestorageuniq.dfs.core.windows.net": "your-client-id",
    "fs.azure.account.oauth2.client.secret.datasciencestorageuniq.dfs.core.windows.net": "your-client-secret",
    "fs.azure.account.oauth2.client.endpoint.datasciencestorageuniq.dfs.core.windows.net": "https://login.microsoftonline.com/your-tenant-id/oauth2/token"
  }
)

# Read data from mounted storage
df = spark.read.format("delta").load("/mnt/datalake/processed/")
df.show()
```

## Best Practices

### 1. Infrastructure as Code

Create Terraform configuration:

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

resource "azurerm_resource_group" "datascience" {
  name     = "rg-datascience-terraform"
  location = "East US 2"
}

resource "azurerm_machine_learning_workspace" "main" {
  name                = "ml-workspace-terraform"
  location            = azurerm_resource_group.datascience.location
  resource_group_name = azurerm_resource_group.datascience.name
  application_insights_id = azurerm_application_insights.main.id
  key_vault_id           = azurerm_key_vault.main.id
  storage_account_id     = azurerm_storage_account.main.id

  identity {
    type = "SystemAssigned"
  }
}
```

### 2. Environment Management

```bash
# Create conda environment
conda create -n azure-ml python=3.8
conda activate azure-ml
pip install azure-ml-sdk[notebooks,automl] azureml-defaults

# Save environment
conda env export > azure-ml-environment.yml
```

### 3. Data Governance

```bash
# Apply tags for data classification
az resource tag \
  --tags DataClassification=Confidential Owner=DataScience Environment=Production \
  --ids /subscriptions/{subscription-id}/resourceGroups/rg-datascience-prod/providers/Microsoft.Storage/storageAccounts/datasciencestorageuniq
```

### 4. Backup and Disaster Recovery

```bash
# Enable backup for SQL Database
az sql db ltr-policy set \
  --resource-group rg-datascience-prod \
  --server sql-datascience-server \
  --database datascience-db \
  --weekly-retention P12W \
  --monthly-retention P12M \
  --yearly-retention P5Y \
  --week-of-year 1

# Enable geo-replication
az sql db replica create \
  --resource-group rg-datascience-prod \
  --server sql-datascience-server \
  --name datascience-db \
  --partner-server sql-datascience-server-secondary \
  --partner-resource-group rg-datascience-prod
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Authentication Issues
```bash
# Clear Azure CLI cache
az account clear
az login --tenant your-tenant-id

# Check current context
az account show
```

#### 2. Network Connectivity
```bash
# Test connectivity to storage account
nslookup datasciencestorageuniq.blob.core.windows.net

# Check network security group rules
az network nsg rule list \
  --resource-group rg-datascience-prod \
  --nsg-name nsg-datascience \
  --output table
```

#### 3. Resource Quotas
```bash
# Check compute quotas
az vm list-usage --location eastus2 --output table

# Request quota increase
az support tickets create \
  --ticket-name "Increase VM Quota" \
  --severity minimal \
  --issue-type quota
```

#### 4. Performance Issues
```bash
# Monitor resource utilization
az monitor metrics list \
  --resource /subscriptions/{subscription-id}/resourceGroups/rg-datascience-prod/providers/Microsoft.Compute/virtualMachines/your-vm-name \
  --metric "Percentage CPU" \
  --start-time 2024-01-01T00:00:00Z \
  --end-time 2024-01-02T00:00:00Z
```

### Diagnostic Scripts

```bash
#!/bin/bash
# azure-health-check.sh

echo "=== Azure Data Science Environment Health Check ==="

# Check resource group
echo "Checking resource group..."
az group show --name rg-datascience-prod || echo "ERROR: Resource group not found"

# Check ML workspace
echo "Checking ML workspace..."
az ml workspace show --name ml-workspace-prod --resource-group rg-datascience-prod || echo "ERROR: ML workspace not accessible"

# Check storage account
echo "Checking storage account..."
az storage account show --name datasciencestorageuniq --resource-group rg-datascience-prod || echo "ERROR: Storage account not accessible"

# Check compute instances
echo "Checking compute instances..."
az ml compute list --workspace-name ml-workspace-prod --resource-group rg-datascience-prod --output table

# Check network connectivity
echo "Checking network connectivity..."
ping -c 3 google.com

echo "=== Health check completed ==="
```

### Performance Monitoring Script

```python
#!/usr/bin/env python3
# azure-monitor.py

import subprocess
import json
import time
from datetime import datetime

def run_az_command(command):
    """Run Azure CLI command and return JSON output"""
    try:
        result = subprocess.run(command.split(), capture_output=True, text=True)
        if result.returncode == 0:
            return json.loads(result.stdout)
        else:
            print(f"Error running command: {command}")
            print(result.stderr)
            return None
    except Exception as e:
        print(f"Exception running command: {e}")
        return None

def monitor_resources():
    """Monitor Azure resources"""
    print(f"[{datetime.now()}] Starting resource monitoring...")
    
    # Monitor VM CPU usage
    cpu_command = ("az monitor metrics list "
                  "--resource /subscriptions/{subscription-id}/resourceGroups/rg-datascience-prod/providers/Microsoft.Compute/virtualMachines/your-vm-name "
                  "--metric 'Percentage CPU' "
                  "--interval PT5M "
                  "--start-time 2024-01-01T00:00:00Z")
    
    cpu_data = run_az_command(cpu_command)
    if cpu_data:
        print(f"CPU Usage: {cpu_data['value'][0]['timeseries'][0]['data'][-1]['average']:.2f}%")
    
    # Monitor storage usage
    storage_command = ("az storage account show-usage "
                      "--resource-group rg-datascience-prod")
    
    storage_data = run_az_command(storage_command)
    if storage_data:
        print(f"Storage Usage: {storage_data['currentValue']}/{storage_data['limit']}")
    
    print("Monitoring completed.\n")

if __name__ == "__main__":
    while True:
        monitor_resources()
        time.sleep(300)  # Monitor every 5 minutes
```

This comprehensive Azure setup guide provides everything needed to build a complete data science infrastructure on Azure, with emphasis on security, cost optimization, and best practices. The guide includes practical commands, scripts, and architectures that can be immediately implemented for production use. 
# GCP Service Accounts for Data Science Workloads
# Terraform configuration with security best practices and minimal permissions

terraform {
  required_version = ">= 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

# Variables
variable "project_id" {
  description = "The GCP project ID"
  type        = string
}

variable "region" {
  description = "The GCP region"
  type        = string
  default     = "us-central1"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging, or prod."
  }
}

# Data locals for reusability
locals {
  service_accounts = {
    r_script_runner = {
      account_id   = "r-script-runner-${var.environment}"
      display_name = "R Script Runner Service Account"
      description  = "Service account for containerized R script execution"
      roles = [
        "roles/storage.objectViewer",
        "roles/storage.objectCreator",
        "roles/bigquery.dataViewer",
        "roles/bigquery.jobUser",
        "roles/logging.logWriter",
        "roles/monitoring.metricWriter"
      ]
    }
    
    shiny_app = {
      account_id   = "shiny-app-${var.environment}"
      display_name = "Shiny Application Service Account"
      description  = "Service account for Shiny web applications"
      roles = [
        "roles/storage.objectViewer",
        "roles/bigquery.dataViewer",
        "roles/bigquery.jobUser",
        "roles/logging.logWriter",
        "roles/monitoring.metricWriter",
        "roles/cloudsql.client"
      ]
    }
    
    jupyter_notebook = {
      account_id   = "jupyter-notebook-${var.environment}"
      display_name = "Jupyter Notebook Service Account"
      description  = "Service account for Jupyter notebook environments"
      roles = [
        "roles/storage.objectAdmin",
        "roles/bigquery.dataEditor",
        "roles/bigquery.jobUser",
        "roles/aiplatform.user",
        "roles/logging.logWriter",
        "roles/monitoring.metricWriter"
      ]
    }
    
    airflow_scheduler = {
      account_id   = "airflow-scheduler-${var.environment}"
      display_name = "Airflow Scheduler Service Account"
      description  = "Service account for Airflow scheduler and workers"
      roles = [
        "roles/storage.admin",
        "roles/bigquery.admin",
        "roles/composer.worker",
        "roles/iam.serviceAccountUser",
        "roles/logging.logWriter",
        "roles/monitoring.metricWriter",
        "roles/cloudscheduler.admin"
      ]
    }
    
    bigquery_etl = {
      account_id   = "bigquery-etl-${var.environment}"
      display_name = "BigQuery ETL Service Account"
      description  = "Service account for BigQuery ETL operations"
      roles = [
        "roles/bigquery.dataEditor",
        "roles/bigquery.jobUser",
        "roles/storage.objectViewer",
        "roles/logging.logWriter",
        "roles/monitoring.metricWriter"
      ]
    }
    
    data_pipeline = {
      account_id   = "data-pipeline-${var.environment}"
      display_name = "Data Pipeline Service Account"
      description  = "Service account for general data pipeline operations"
      roles = [
        "roles/storage.objectAdmin",
        "roles/bigquery.dataEditor",
        "roles/bigquery.jobUser",
        "roles/pubsub.publisher",
        "roles/pubsub.subscriber",
        "roles/dataflow.worker",
        "roles/logging.logWriter",
        "roles/monitoring.metricWriter"
      ]
    }
    
    looker_studio = {
      account_id   = "looker-studio-${var.environment}"
      display_name = "Looker Studio Service Account"
      description  = "Service account for Looker Studio data access"
      roles = [
        "roles/bigquery.dataViewer",
        "roles/bigquery.metadataViewer",
        "roles/storage.objectViewer"
      ]
    }
  }
}

# Create service accounts
resource "google_service_account" "service_accounts" {
  for_each = local.service_accounts
  
  account_id   = each.value.account_id
  display_name = each.value.display_name
  description  = each.value.description
  project      = var.project_id
}

# Assign IAM roles to service accounts
resource "google_project_iam_member" "service_account_roles" {
  for_each = {
    for sa_role in flatten([
      for sa_key, sa_config in local.service_accounts : [
        for role in sa_config.roles : {
          sa_key = sa_key
          role   = role
        }
      ]
    ]) : "${sa_role.sa_key}-${sa_role.role}" => sa_role
  }
  
  project = var.project_id
  role    = each.value.role
  member  = "serviceAccount:${google_service_account.service_accounts[each.value.sa_key].email}"
}

# Create service account keys (use with caution in production)
resource "google_service_account_key" "service_account_keys" {
  for_each = var.environment == "dev" ? local.service_accounts : {}
  
  service_account_id = google_service_account.service_accounts[each.key].name
  public_key_type    = "TYPE_X509_PEM_FILE"
  
  # Only create keys for development environment
  lifecycle {
    prevent_destroy = true
  }
}

# Custom IAM roles for fine-grained permissions
resource "google_project_iam_custom_role" "data_scientist_role" {
  role_id     = "dataScienceCustomRole${title(var.environment)}"
  title       = "Data Science Custom Role - ${var.environment}"
  description = "Custom role with minimal permissions for data science workloads"
  
  permissions = [
    # BigQuery permissions
    "bigquery.datasets.get",
    "bigquery.tables.get",
    "bigquery.tables.list",
    "bigquery.tables.getData",
    "bigquery.jobs.create",
    "bigquery.jobs.get",
    "bigquery.jobs.list",
    
    # Storage permissions
    "storage.objects.get",
    "storage.objects.list",
    "storage.objects.create",
    "storage.buckets.get",
    
    # Monitoring and logging
    "logging.logEntries.create",
    "monitoring.timeSeries.create",
    
    # Cloud Run permissions
    "run.services.get",
    "run.services.list"
  ]
}

# Workload Identity for GKE (if using Kubernetes)
resource "google_service_account_iam_binding" "workload_identity_binding" {
  for_each = var.environment == "prod" ? local.service_accounts : {}
  
  service_account_id = google_service_account.service_accounts[each.key].name
  role               = "roles/iam.workloadIdentityUser"
  
  members = [
    "serviceAccount:${var.project_id}.svc.id.goog[${each.key}/${each.key}]"
  ]
}

# Output service account information
output "service_account_emails" {
  description = "Map of service account emails"
  value = {
    for key, sa in google_service_account.service_accounts : key => sa.email
  }
}

output "service_account_keys" {
  description = "Service account private keys (base64 encoded)"
  value = {
    for key, key_resource in google_service_account_key.service_account_keys : 
    key => key_resource.private_key
  }
  sensitive = true
}

# Security recommendations output
output "security_recommendations" {
  description = "Security best practices and recommendations"
  value = {
    service_account_usage = "Use Workload Identity for GKE workloads instead of service account keys"
    key_rotation = "Rotate service account keys regularly (every 90 days)"
    principle_of_least_privilege = "Review and minimize permissions regularly"
    monitoring = "Enable audit logging and monitor service account usage"
    environment_separation = "Use separate service accounts for different environments"
  }
}

# Data source for existing project
data "google_project" "project" {
  project_id = var.project_id
}

# Enable required APIs
resource "google_project_service" "required_apis" {
  for_each = toset([
    "bigquery.googleapis.com",
    "storage.googleapis.com",
    "logging.googleapis.com",
    "monitoring.googleapis.com",
    "run.googleapis.com",
    "cloudbuild.googleapis.com",
    "container.googleapis.com",
    "aiplatform.googleapis.com",
    "dataflow.googleapis.com",
    "pubsub.googleapis.com"
  ])
  
  project = var.project_id
  service = each.value
  
  disable_on_destroy = false
}

# Cloud Storage buckets for different use cases
resource "google_storage_bucket" "data_buckets" {
  for_each = {
    raw_data    = "raw-data-${var.project_id}-${var.environment}"
    processed   = "processed-data-${var.project_id}-${var.environment}"
    outputs     = "outputs-${var.project_id}-${var.environment}"
    logs        = "logs-${var.project_id}-${var.environment}"
    models      = "ml-models-${var.project_id}-${var.environment}"
  }
  
  name     = each.value
  location = var.region
  project  = var.project_id
  
  # Security settings
  uniform_bucket_level_access = true
  
  # Lifecycle management
  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type = "Delete"
    }
  }
  
  # Versioning for important buckets
  versioning {
    enabled = contains(["processed", "models"], each.key)
  }
  
  # Encryption
  encryption {
    default_kms_key_name = google_kms_crypto_key.bucket_key.id
  }
}

# KMS key for bucket encryption
resource "google_kms_key_ring" "data_key_ring" {
  name     = "data-science-keyring-${var.environment}"
  location = var.region
  project  = var.project_id
}

resource "google_kms_crypto_key" "bucket_key" {
  name     = "bucket-encryption-key"
  key_ring = google_kms_key_ring.data_key_ring.id
  
  lifecycle {
    prevent_destroy = true
  }
}

# IAM binding for KMS key
resource "google_kms_crypto_key_iam_binding" "bucket_key_binding" {
  crypto_key_id = google_kms_crypto_key.bucket_key.id
  role          = "roles/cloudkms.cryptoKeyEncrypterDecrypter"
  
  members = [
    for sa in google_service_account.service_accounts :
    "serviceAccount:${sa.email}"
  ]
} 
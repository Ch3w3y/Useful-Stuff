#!/bin/bash

# GCP Deployment Templates - Master Deployment Script
# Comprehensive deployment automation with security checks and environment management

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/config/deployment-config.yaml"
LOG_FILE="${SCRIPT_DIR}/logs/deployment-$(date +%Y%m%d_%H%M%S).log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "${LOG_FILE}"
}

info() {
    log "${BLUE}INFO: $1${NC}"
}

success() {
    log "${GREEN}SUCCESS: $1${NC}"
}

warning() {
    log "${YELLOW}WARNING: $1${NC}"
}

error() {
    log "${RED}ERROR: $1${NC}"
}

# Help function
show_help() {
    cat << EOF
GCP Deployment Templates - Deployment Script

Usage: $0 [OPTIONS] TEMPLATE_NAME

TEMPLATES:
  r-script-runner      Deploy containerized R script runner
  shiny-app           Deploy Shiny application to Cloud Run
  jupyter-notebook    Deploy Jupyter notebook environment
  airflow             Deploy Apache Airflow for orchestration
  bigquery-etl        Deploy BigQuery ETL pipeline
  python-etl          Deploy Python-based ETL pipeline
  security            Deploy service accounts and IAM configuration
  all                 Deploy all components

OPTIONS:
  -e, --environment ENV    Environment (dev, staging, prod) [required]
  -p, --project PROJECT    GCP Project ID [required]
  -r, --region REGION      GCP Region (default: us-central1)
  -f, --force             Force deployment without confirmation
  -d, --dry-run           Show what would be deployed without executing
  -v, --verbose           Enable verbose logging
  -h, --help              Show this help message

EXAMPLES:
  $0 -e dev -p my-project shiny-app
  $0 -e prod -p my-project --force all
  $0 -e staging -p my-project --dry-run bigquery-etl

SECURITY REQUIREMENTS:
  - gcloud CLI authenticated with appropriate permissions
  - Service account with deployment permissions
  - Terraform installed (for security templates)
  - Docker installed and authenticated with gcr.io
EOF
}

# Parse command line arguments
ENVIRONMENT=""
PROJECT_ID=""
REGION="us-central1"
TEMPLATE=""
FORCE=false
DRY_RUN=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -p|--project)
            PROJECT_ID="$2"
            shift 2
            ;;
        -r|--region)
            REGION="$2"
            shift 2
            ;;
        -f|--force)
            FORCE=true
            shift
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            set -x
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            if [[ -z "$TEMPLATE" ]]; then
                TEMPLATE="$1"
            else
                error "Unknown option: $1"
                show_help
                exit 1
            fi
            shift
            ;;
    esac
done

# Validate required parameters
if [[ -z "$ENVIRONMENT" ]]; then
    error "Environment is required. Use -e or --environment"
    show_help
    exit 1
fi

if [[ -z "$PROJECT_ID" ]]; then
    error "Project ID is required. Use -p or --project"
    show_help
    exit 1
fi

if [[ -z "$TEMPLATE" ]]; then
    error "Template name is required"
    show_help
    exit 1
fi

# Validate environment
if [[ ! "$ENVIRONMENT" =~ ^(dev|staging|prod)$ ]]; then
    error "Environment must be dev, staging, or prod"
    exit 1
fi

# Create logs directory
mkdir -p "${SCRIPT_DIR}/logs"

info "Starting deployment for template: $TEMPLATE"
info "Environment: $ENVIRONMENT"
info "Project: $PROJECT_ID"
info "Region: $REGION"

# Pre-flight checks
preflight_checks() {
    info "Running pre-flight checks..."
    
    # Check if gcloud is installed and authenticated
    if ! command -v gcloud &> /dev/null; then
        error "gcloud CLI is not installed"
        exit 1
    fi
    
    # Check if authenticated
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
        error "Not authenticated with gcloud. Run 'gcloud auth login'"
        exit 1
    fi
    
    # Check if project exists and is accessible
    if ! gcloud projects describe "$PROJECT_ID" &> /dev/null; then
        error "Cannot access project $PROJECT_ID or project does not exist"
        exit 1
    fi
    
    # Set default project
    gcloud config set project "$PROJECT_ID"
    
    # Check Docker if needed
    if [[ "$TEMPLATE" =~ ^(r-script-runner|shiny-app|jupyter-notebook|airflow|python-etl)$ ]]; then
        if ! command -v docker &> /dev/null; then
            error "Docker is not installed but required for template: $TEMPLATE"
            exit 1
        fi
        
        # Configure Docker for GCR
        gcloud auth configure-docker --quiet
    fi
    
    # Check Terraform for security templates
    if [[ "$TEMPLATE" == "security" ]] || [[ "$TEMPLATE" == "all" ]]; then
        if ! command -v terraform &> /dev/null; then
            error "Terraform is not installed but required for security template"
            exit 1
        fi
    fi
    
    success "Pre-flight checks completed"
}

# Deploy R Script Runner
deploy_r_script_runner() {
    info "Deploying R Script Runner..."
    
    local template_dir="${SCRIPT_DIR}/r-environments/r-script-runner"
    local image_name="gcr.io/${PROJECT_ID}/r-script-runner:${ENVIRONMENT}-$(date +%Y%m%d-%H%M%S)"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        info "DRY RUN: Would build and deploy R script runner"
        return 0
    fi
    
    # Build Docker image
    info "Building Docker image..."
    docker build -t "$image_name" "$template_dir"
    
    # Push to GCR
    info "Pushing image to Container Registry..."
    docker push "$image_name"
    
    # Deploy to Cloud Run
    info "Deploying to Cloud Run..."
    gcloud run deploy "r-script-runner-${ENVIRONMENT}" \
        --image="$image_name" \
        --platform=managed \
        --region="$REGION" \
        --allow-unauthenticated \
        --memory=2Gi \
        --cpu=2 \
        --timeout=3600 \
        --max-instances=10 \
        --set-env-vars="ENVIRONMENT=${ENVIRONMENT},GCP_PROJECT_ID=${PROJECT_ID}"
    
    success "R Script Runner deployed successfully"
}

# Deploy Shiny App
deploy_shiny_app() {
    info "Deploying Shiny Application..."
    
    local template_dir="${SCRIPT_DIR}/r-environments/shiny-app"
    local image_name="gcr.io/${PROJECT_ID}/shiny-app:${ENVIRONMENT}-$(date +%Y%m%d-%H%M%S)"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        info "DRY RUN: Would build and deploy Shiny application"
        return 0
    fi
    
    # Build and deploy
    docker build -t "$image_name" "$template_dir"
    docker push "$image_name"
    
    gcloud run deploy "shiny-app-${ENVIRONMENT}" \
        --image="$image_name" \
        --platform=managed \
        --region="$REGION" \
        --allow-unauthenticated \
        --memory=4Gi \
        --cpu=2 \
        --port=8080 \
        --timeout=3600 \
        --set-env-vars="ENVIRONMENT=${ENVIRONMENT},GCP_PROJECT_ID=${PROJECT_ID}"
    
    success "Shiny Application deployed successfully"
}

# Deploy Jupyter Notebook
deploy_jupyter_notebook() {
    info "Deploying Jupyter Notebook Environment..."
    
    local template_dir="${SCRIPT_DIR}/jupyter-notebook"
    local image_name="gcr.io/${PROJECT_ID}/jupyter-notebook:${ENVIRONMENT}-$(date +%Y%m%d-%H%M%S)"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        info "DRY RUN: Would build and deploy Jupyter notebook"
        return 0
    fi
    
    docker build -t "$image_name" "$template_dir"
    docker push "$image_name"
    
    # Deploy to Cloud Run with persistent disk
    gcloud run deploy "jupyter-notebook-${ENVIRONMENT}" \
        --image="$image_name" \
        --platform=managed \
        --region="$REGION" \
        --memory=8Gi \
        --cpu=4 \
        --port=8888 \
        --timeout=3600 \
        --set-env-vars="ENVIRONMENT=${ENVIRONMENT},GCP_PROJECT_ID=${PROJECT_ID}"
    
    success "Jupyter Notebook deployed successfully"
}

# Deploy Security Configuration
deploy_security() {
    info "Deploying Security Configuration..."
    
    local terraform_dir="${SCRIPT_DIR}/security/service-accounts/terraform"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        info "DRY RUN: Would deploy security configuration with Terraform"
        return 0
    fi
    
    cd "$terraform_dir"
    
    # Initialize Terraform
    terraform init
    
    # Plan deployment
    terraform plan \
        -var="project_id=${PROJECT_ID}" \
        -var="region=${REGION}" \
        -var="environment=${ENVIRONMENT}" \
        -out=tfplan
    
    # Apply if not force mode
    if [[ "$FORCE" == "true" ]]; then
        terraform apply tfplan
    else
        warning "Security configuration planned. Run with --force to apply"
    fi
    
    cd - > /dev/null
    success "Security configuration processed"
}

# Deploy BigQuery ETL
deploy_bigquery_etl() {
    info "Deploying BigQuery ETL Pipeline..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        info "DRY RUN: Would deploy BigQuery ETL pipeline"
        return 0
    fi
    
    # Create dataset if it doesn't exist
    local dataset_id="data_pipeline_${ENVIRONMENT}"
    
    if ! bq ls -d "${PROJECT_ID}:${dataset_id}" &> /dev/null; then
        info "Creating BigQuery dataset: $dataset_id"
        bq mk --dataset \
            --location="$REGION" \
            --description="ETL pipeline dataset for $ENVIRONMENT" \
            "${PROJECT_ID}:${dataset_id}"
    fi
    
    # Deploy stored procedures
    local sql_file="${SCRIPT_DIR}/sql-pipelines/bigquery-sql-etl/main.sql"
    
    # Replace template variables
    sed -e "s/\${project_id}/$PROJECT_ID/g" \
        -e "s/\${dataset_id}/$dataset_id/g" \
        -e "s/\${pipeline_run_id}/$(date +%Y%m%d_%H%M%S)/g" \
        "$sql_file" | bq query --use_legacy_sql=false
    
    success "BigQuery ETL pipeline deployed successfully"
}

# Main deployment function
main() {
    preflight_checks
    
    case "$TEMPLATE" in
        r-script-runner)
            deploy_r_script_runner
            ;;
        shiny-app)
            deploy_shiny_app
            ;;
        jupyter-notebook)
            deploy_jupyter_notebook
            ;;
        security)
            deploy_security
            ;;
        bigquery-etl)
            deploy_bigquery_etl
            ;;
        all)
            deploy_security
            deploy_bigquery_etl
            deploy_r_script_runner
            deploy_shiny_app
            deploy_jupyter_notebook
            ;;
        *)
            error "Unknown template: $TEMPLATE"
            show_help
            exit 1
            ;;
    esac
    
    success "Deployment completed successfully!"
    info "Log file: $LOG_FILE"
}

# Cleanup function
cleanup() {
    if [[ "$VERBOSE" == "true" ]]; then
        set +x
    fi
}

# Set trap for cleanup
trap cleanup EXIT

# Run main function
main "$@" 
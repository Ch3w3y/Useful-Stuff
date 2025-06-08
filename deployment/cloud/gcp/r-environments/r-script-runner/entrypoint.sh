#!/bin/bash
set -euo pipefail

# R Script Runner Entrypoint
# Secure execution environment with proper authentication and logging

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "[$(date +'%Y-%m-%d %H:%M:%S')] $1" >&2
}

error_exit() {
    log "${RED}ERROR: $1${NC}"
    exit 1
}

success() {
    log "${GREEN}SUCCESS: $1${NC}"
}

warning() {
    log "${YELLOW}WARNING: $1${NC}"
}

# Validate required environment variables
required_vars=(
    "GCP_PROJECT_ID"
    "GCS_OUTPUT_BUCKET"
    "TARGET_SCRIPT"
)

log "Validating environment variables..."
for var in "${required_vars[@]}"; do
    if [[ -z "${!var:-}" ]]; then
        error_exit "Required environment variable $var is not set"
    fi
done

# Set default values for optional variables
export LOG_LEVEL=${LOG_LEVEL:-"INFO"}
export ENVIRONMENT=${ENVIRONMENT:-"production"}
export DEBUG_MODE=${DEBUG_MODE:-"false"}
export ENABLE_LOGGING=${ENABLE_LOGGING:-"true"}

# Authenticate with Google Cloud
log "Authenticating with Google Cloud..."
if [[ -f "${GOOGLE_APPLICATION_CREDENTIALS}" ]]; then
    gcloud auth activate-service-account --key-file="${GOOGLE_APPLICATION_CREDENTIALS}"
    gcloud config set project "${GCP_PROJECT_ID}"
    success "Authenticated with service account"
else
    warning "Service account key not found, attempting default authentication"
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
        error_exit "No active Google Cloud authentication found"
    fi
fi

# Verify access to required resources
log "Verifying access to Google Cloud Storage..."
if ! gsutil ls "gs://${GCS_OUTPUT_BUCKET}" >/dev/null 2>&1; then
    error_exit "Cannot access output bucket: ${GCS_OUTPUT_BUCKET}"
fi

# Create necessary directories
mkdir -p /app/logs /app/temp /app/output

# Set up R environment
log "Setting up R environment..."
export R_LIBS_USER=/app/R_libs
mkdir -p "${R_LIBS_USER}"

# Check if target script exists
if [[ ! -f "/app/${TARGET_SCRIPT}" ]]; then
    error_exit "Target script not found: ${TARGET_SCRIPT}"
fi

# Create log file with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="/app/logs/execution_${TIMESTAMP}.log"

# Execute R script with proper error handling
log "Starting R script execution: ${TARGET_SCRIPT}"
start_time=$(date +%s)

if [[ "${DEBUG_MODE}" == "true" ]]; then
    # Debug mode - more verbose output
    Rscript --vanilla "/app/${TARGET_SCRIPT}" 2>&1 | tee "${LOG_FILE}"
    exit_code=${PIPESTATUS[0]}
else
    # Production mode - captured output
    if ! Rscript --vanilla "/app/${TARGET_SCRIPT}" >"${LOG_FILE}" 2>&1; then
        exit_code=$?
        error_exit "R script execution failed with exit code: ${exit_code}"
    else
        exit_code=0
    fi
fi

end_time=$(date +%s)
execution_time=$((end_time - start_time))

if [[ ${exit_code} -eq 0 ]]; then
    success "R script completed successfully in ${execution_time} seconds"
else
    error_exit "R script failed with exit code: ${exit_code}"
fi

# Upload logs to GCS if bucket is specified
if [[ -n "${GCS_LOGS_BUCKET:-}" ]]; then
    log "Uploading logs to GCS..."
    if gsutil cp "${LOG_FILE}" "gs://${GCS_LOGS_BUCKET}/logs/"; then
        success "Logs uploaded to gs://${GCS_LOGS_BUCKET}/logs/"
    else
        warning "Failed to upload logs to GCS"
    fi
fi

# Upload output files to GCS
log "Uploading output files to GCS..."
if [[ -d "/app/output" ]] && [[ "$(ls -A /app/output)" ]]; then
    if gsutil -m cp -r "/app/output/*" "gs://${GCS_OUTPUT_BUCKET}/"; then
        success "Output files uploaded to gs://${GCS_OUTPUT_BUCKET}/"
    else
        warning "Failed to upload some output files"
    fi
else
    warning "No output files found in /app/output/"
fi

# Send completion notification if configured
if [[ -n "${NOTIFICATION_EMAIL:-}" ]]; then
    log "Sending completion notification..."
    # This would typically integrate with Cloud Functions or Pub/Sub
    # For now, we'll just log the notification
    success "Script execution completed. Notification would be sent to: ${NOTIFICATION_EMAIL}"
fi

log "Container execution completed successfully"
exit 0 
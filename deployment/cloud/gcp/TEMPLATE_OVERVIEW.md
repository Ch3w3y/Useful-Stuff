# GCP Deployment Templates - Complete Overview

A comprehensive collection of production-ready deployment templates for data science workloads on Google Cloud Platform. Each template follows security best practices, implements proper monitoring, and supports multiple environments.

## 🏗️ Template Architecture

### Environment Strategy
- **Development**: Full functionality with relaxed security for experimentation
- **Staging**: Production-like environment for testing and validation
- **Production**: Hardened security, monitoring, and performance optimization

### Security Framework
- **Service Account Separation**: Dedicated service accounts with minimal permissions
- **Network Security**: VPC configurations and firewall rules
- **Encryption**: KMS encryption for storage and data in transit
- **IAM**: Role-based access control with custom roles
- **Audit Logging**: Comprehensive logging and monitoring

## 📊 Available Templates

### R Environments

#### R Script Runner (`r-environments/r-script-runner/`)
**Purpose**: Containerized execution of R scripts with GCP integration

**Features**:
- Secure container with non-root user
- GCS input/output integration
- BigQuery connectivity
- Comprehensive logging and error handling
- Configurable resource allocation
- Cloud Run deployment ready

**Use Cases**:
- Automated data analysis scripts
- Scheduled report generation
- ETL processing with R
- Statistical model execution

**Deployment**:
```bash
./deploy.sh -e prod -p your-project-id r-script-runner
```

**Configuration Files**:
- `Dockerfile` - Container configuration
- `entrypoint.sh` - Secure startup script
- `run_script.R` - Main execution framework
- `env-template.txt` - Environment configuration

#### Shiny Application (`r-environments/shiny-app/`)
**Purpose**: Production Shiny web applications with cloud data sources

**Features**:
- Modern responsive UI with shinydashboard
- BigQuery and GCS data integration
- Interactive visualizations (plotly, leaflet)
- User authentication framework
- Session logging and monitoring
- Auto-scaling on Cloud Run

**Use Cases**:
- Executive dashboards
- Data exploration tools
- Real-time monitoring interfaces
- Client-facing analytics applications

**Key Components**:
- `server.R` - Backend logic with GCP integration
- `ui.R` - Modern dashboard interface
- Authentication and session management
- Data caching and performance optimization

### Python Environments

#### Jupyter Notebook (`jupyter-notebook/`)
**Purpose**: Secure Jupyter environment with comprehensive data science stack

**Features**:
- Pre-installed data science packages (pandas, scikit-learn, tensorflow)
- GCP SDK and authentication
- R kernel support via IRkernel
- JupyterLab extensions (Git, LSP, plotting)
- Persistent storage integration
- Custom startup scripts

**Included Packages**:
- **GCP**: `google-cloud-storage`, `google-cloud-bigquery`, `pandas-gbq`
- **ML/AI**: `scikit-learn`, `tensorflow`, `pytorch`, `optuna`, `mlflow`
- **Visualization**: `plotly`, `bokeh`, `altair`, `seaborn`
- **Development**: `black`, `isort`, `flake8`, `mypy`, `jupyter-lsp`

**Use Cases**:
- Exploratory data analysis
- Machine learning experimentation
- Collaborative research
- Educational environments

### Orchestration

#### Apache Airflow (`airflow/`)
**Purpose**: Workflow orchestration for complex data pipelines

**Features**:
- GCP providers pre-installed
- Security hardened configuration
- Custom operators for common tasks
- Monitoring and alerting integration
- Multi-environment support
- DAG version control

**Included Providers**:
- Google Cloud Platform
- PostgreSQL and BigQuery
- Slack notifications
- Email alerts
- HTTP sensors

**Use Cases**:
- ETL pipeline orchestration
- ML model training workflows
- Data quality monitoring
- Automated reporting pipelines

### SQL Pipelines

#### BigQuery ETL (`sql-pipelines/bigquery-sql-etl/`)
**Purpose**: Pure SQL-based ETL pipelines with comprehensive error handling

**Features**:
- Stored procedure architecture
- Data quality validation
- Error handling and logging
- Performance monitoring
- Incremental processing
- Data lineage tracking

**Pipeline Components**:
- **Data Validation**: Schema and quality checks
- **Transformation**: Cleaning and standardization
- **Aggregation**: Summary tables and metrics
- **Monitoring**: Execution logs and alerts

**Quality Checks**:
- Required field validation
- Data type verification
- Referential integrity
- Statistical outlier detection
- Completeness assessment

### Security Infrastructure

#### Service Accounts (`security/service-accounts/terraform/`)
**Purpose**: Terraform-managed IAM configuration with minimal permissions

**Service Accounts Created**:
- `r-script-runner` - R script execution
- `shiny-app` - Shiny application hosting
- `jupyter-notebook` - Jupyter environment
- `airflow-scheduler` - Airflow orchestration
- `bigquery-etl` - ETL operations
- `data-pipeline` - General data processing
- `looker-studio` - BI tool integration

**Security Features**:
- Principle of least privilege
- Environment separation
- KMS encryption keys
- Workload Identity support
- Custom IAM roles
- Audit logging configuration

## 🚀 Quick Start Guide

### Prerequisites
```bash
# Install required tools
gcloud components install beta
terraform --version
docker --version

# Authenticate
gcloud auth login
gcloud auth application-default login
```

### Basic Deployment
```bash
# Clone and navigate to templates
cd gcp-deployment-templates

# Deploy security infrastructure first
./deploy.sh -e dev -p your-project-id security

# Deploy specific template
./deploy.sh -e dev -p your-project-id shiny-app

# Deploy everything
./deploy.sh -e prod -p your-project-id --force all
```

### Environment Configuration
```bash
# Copy environment template
cp r-environments/r-script-runner/env-template.txt .env

# Edit configuration
vim .env

# Set required variables
export GCP_PROJECT_ID="your-project-id"
export ENVIRONMENT="dev"
```

## 📈 Use Case Examples

### Data Science Workflow
```bash
# 1. Set up infrastructure
./deploy.sh -e dev -p project security
./deploy.sh -e dev -p project jupyter-notebook

# 2. Develop analysis in Jupyter
# 3. Convert to R script for automation
./deploy.sh -e dev -p project r-script-runner

# 4. Create dashboard for results
./deploy.sh -e dev -p project shiny-app

# 5. Orchestrate with Airflow
./deploy.sh -e prod -p project airflow
```

### ETL Pipeline Setup
```bash
# 1. Deploy BigQuery ETL
./deploy.sh -e prod -p project bigquery-etl

# 2. Set up orchestration
./deploy.sh -e prod -p project airflow

# 3. Create monitoring dashboard
./deploy.sh -e prod -p project shiny-app
```

### ML Model Deployment
```bash
# 1. Develop in Jupyter
./deploy.sh -e dev -p project jupyter-notebook

# 2. Create prediction service
./deploy.sh -e prod -p project r-script-runner

# 3. Build monitoring dashboard
./deploy.sh -e prod -p project shiny-app
```

## 🔧 Customization Guide

### Adding New Templates
1. Create template directory structure
2. Include Dockerfile and configuration files
3. Add deployment function to `deploy.sh`
4. Update documentation and examples

### Modifying Existing Templates
1. Update Dockerfile for package changes
2. Modify configuration templates
3. Test in development environment
4. Update production deployment

### Environment-Specific Configuration
```yaml
# config/deployment-config.yaml
environments:
  dev:
    resources:
      memory: "2Gi"
      cpu: "1"
  prod:
    resources:
      memory: "8Gi"
      cpu: "4"
```

## 📊 Monitoring and Observability

### Built-in Monitoring
- **Cloud Logging**: Structured application logs
- **Cloud Monitoring**: Custom metrics and alerts
- **Cloud Trace**: Request tracing for web applications
- **Error Reporting**: Automatic error detection and grouping

### Custom Dashboards
Each template includes:
- Resource utilization metrics
- Application-specific KPIs
- Error rate monitoring
- Performance benchmarks

### Alerting Configuration
```yaml
alerts:
  - name: "High Error Rate"
    condition: "error_rate > 5%"
    notification: "slack://data-team"
  
  - name: "Resource Exhaustion"
    condition: "memory_usage > 90%"
    notification: "email://ops-team"
```

## 🔒 Security Best Practices

### Authentication
- Service account-based authentication
- Workload Identity for GKE workloads
- OAuth2 for user-facing applications
- API key rotation policies

### Network Security
- VPC network isolation
- Private Google Access
- Cloud NAT for outbound traffic
- Firewall rule management

### Data Protection
- Encryption at rest and in transit
- Customer-managed encryption keys (CMEK)
- Data Loss Prevention (DLP) scanning
- Access logging and audit trails

## 📝 Troubleshooting

### Common Issues

#### Authentication Errors
```bash
# Fix: Re-authenticate
gcloud auth login
gcloud auth application-default login
```

#### Permission Denied
```bash
# Fix: Check IAM roles
gcloud projects get-iam-policy your-project-id
```

#### Container Build Failures
```bash
# Fix: Clear Docker cache
docker system prune -a
```

#### Terraform State Issues
```bash
# Fix: Refresh state
terraform refresh
```

### Debug Mode
```bash
# Enable verbose logging
./deploy.sh -v -e dev -p project template-name

# Dry run mode
./deploy.sh -d -e dev -p project template-name
```

## 🔄 Maintenance and Updates

### Regular Tasks
- Service account key rotation (90 days)
- Container image updates (monthly)
- Dependency updates (quarterly)
- Security audit reviews (quarterly)

### Backup Procedures
- Terraform state backup
- Configuration versioning
- Database exports
- Code repository mirroring

### Disaster Recovery
- Multi-region deployment options
- Automated failover procedures
- Data replication strategies
- Recovery time objectives (RTO)

## 📚 Additional Resources

### Documentation
- [Google Cloud Documentation](https://cloud.google.com/docs)
- [Terraform GCP Provider](https://registry.terraform.io/providers/hashicorp/google/latest/docs)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)

### Training
- Google Cloud Professional Data Engineer
- Terraform Associate Certification
- Kubernetes Administrator (CKA)

### Community
- [GCP Slack Community](https://gcp-community.slack.com)
- [Terraform Community Forum](https://discuss.hashicorp.com/c/terraform-core)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/google-cloud-platform) 
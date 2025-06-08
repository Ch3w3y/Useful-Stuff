# Jupyter Docker Deployment for Analytics

Comprehensive Docker deployment configurations for Jupyter environments optimized for data science and analytics workflows. This setup provides both single-user Jupyter Lab instances and multi-user JupyterHub deployments with pre-configured data science libraries and tools.

## Table of Contents

- [Overview](#overview)
- [Architecture Options](#architecture-options)
- [Quick Start](#quick-start)
- [Single-User Jupyter Lab](#single-user-jupyter-lab)
- [Multi-User JupyterHub](#multi-user-jupyterhub)
- [Custom Images](#custom-images)
- [Data Persistence](#data-persistence)
- [Security Configuration](#security-configuration)
- [Performance Optimization](#performance-optimization)
- [Extensions & Plugins](#extensions--plugins)
- [Integration with Analytics Stack](#integration-with-analytics-stack)
- [Monitoring & Logging](#monitoring--logging)
- [Troubleshooting](#troubleshooting)

## Overview

This deployment provides production-ready Jupyter environments with:

- **Pre-configured Data Science Stack**: Python, R, Julia kernels with essential libraries
- **Scalable Architecture**: Single-user and multi-user configurations
- **Persistent Storage**: Volume mounting for notebooks and data
- **Security Features**: Authentication, HTTPS, user isolation
- **Analytics Integration**: Database connections, cloud storage, visualization tools
- **Resource Management**: CPU/memory limits, GPU support

### Key Features
- Docker Compose orchestration
- Multiple kernel support (Python, R, Julia, Scala)
- Pre-installed data science libraries
- Database connectivity (PostgreSQL, MySQL, MongoDB)
- Cloud storage integration (AWS S3, GCS, Azure)
- Visualization tools (Plotly, Bokeh, D3.js)
- Version control integration (Git, DVC)
- Collaborative features with JupyterHub

## Architecture Options

### Single-User Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Host                              │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │   Jupyter Lab   │    │   Data Volume   │                │
│  │   (Port 8888)   │────│   /home/jovyan  │                │
│  │                 │    │                 │                │
│  └─────────────────┘    └─────────────────┘                │
│           │                                                 │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │   PostgreSQL    │    │     Redis       │                │
│  │   Database      │    │     Cache       │                │
│  └─────────────────┘    └─────────────────┘                │
└─────────────────────────────────────────────────────────────┘
```

### Multi-User JupyterHub Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Host                              │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │   JupyterHub    │    │   User Server   │                │
│  │   (Port 8000)   │────│   Container 1   │                │
│  │                 │    │                 │                │
│  └─────────────────┘    └─────────────────┘                │
│           │              ┌─────────────────┐                │
│           │              │   User Server   │                │
│           └──────────────│   Container 2   │                │
│                          │                 │                │
│  ┌─────────────────┐    └─────────────────┘                │
│  │   Shared Data   │    ┌─────────────────┐                │
│  │   Volume        │    │   PostgreSQL    │                │
│  │                 │    │   Database      │                │
│  └─────────────────┘    └─────────────────┘                │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites
```bash
# Install Docker and Docker Compose
sudo apt-get update
sudo apt-get install docker.io docker-compose

# Create project directory
mkdir jupyter-analytics && cd jupyter-analytics
```

### Single-User Quick Start
```bash
# Create directory structure
mkdir -p notebooks data config

# Create basic docker-compose.yml
cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  jupyter:
    image: jupyter/datascience-notebook:latest
    ports:
      - "8888:8888"
    volumes:
      - ./notebooks:/home/jovyan/work
      - ./data:/home/jovyan/data
    environment:
      - JUPYTER_ENABLE_LAB=yes
      - JUPYTER_TOKEN=your-secure-token
    command: start-notebook.sh --NotebookApp.token='your-secure-token'
EOF

# Start Jupyter Lab
docker-compose up -d

# Access at http://localhost:8888 with token 'your-secure-token'
```

## Single-User Jupyter Lab

### Complete Docker Compose Configuration
```yaml
# docker-compose.yml
version: '3.8'

services:
  jupyter:
    build:
      context: .
      dockerfile: Dockerfile.jupyter
    ports:
      - "8888:8888"
    volumes:
      - ./notebooks:/home/jovyan/work
      - ./data:/home/jovyan/data
      - ./config:/home/jovyan/.jupyter
      - jupyter-data:/home/jovyan
    environment:
      - JUPYTER_ENABLE_LAB=yes
      - JUPYTER_TOKEN=${JUPYTER_TOKEN:-analytics-token}
      - GRANT_SUDO=yes
      - CHOWN_HOME=yes
      - CHOWN_HOME_OPTS='-R'
      # Database connections
      - POSTGRES_HOST=postgres
      - POSTGRES_DB=analytics
      - POSTGRES_USER=jupyter
      - POSTGRES_PASSWORD=jupyter
      # Cloud storage
      - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
      - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
      - AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION:-us-east-1}
    depends_on:
      - postgres
      - redis
    networks:
      - jupyter-network
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 8G
        reservations:
          cpus: '2.0'
          memory: 4G

  postgres:
    image: postgres:13
    environment:
      - POSTGRES_DB=analytics
      - POSTGRES_USER=jupyter
      - POSTGRES_PASSWORD=jupyter
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./sql/init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    networks:
      - jupyter-network
    restart: unless-stopped

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    networks:
      - jupyter-network
    restart: unless-stopped

  # Optional: MinIO for S3-compatible storage
  minio:
    image: minio/minio:latest
    ports:
      - "9000:9000"
      - "9001:9001"
    volumes:
      - minio-data:/data
    environment:
      - MINIO_ROOT_USER=minioadmin
      - MINIO_ROOT_PASSWORD=minioadmin
    command: server /data --console-address ":9001"
    networks:
      - jupyter-network
    restart: unless-stopped

volumes:
  jupyter-data:
  postgres-data:
  redis-data:
  minio-data:

networks:
  jupyter-network:
    driver: bridge
```

### Custom Jupyter Dockerfile
```dockerfile
# Dockerfile.jupyter
FROM jupyter/datascience-notebook:latest

USER root

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    vim \
    htop \
    tree \
    jq \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Install additional Python packages
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Install R packages
RUN R -e "install.packages(c('DBI', 'RPostgreSQL', 'mongolite', 'aws.s3', 'plotly', 'shiny', 'shinydashboard'), repos='https://cran.rstudio.com/')"

# Install Julia packages
RUN julia -e 'using Pkg; Pkg.add(["DataFrames", "CSV", "Plots", "StatsPlots", "MLJ", "Flux"])'

# Install Jupyter extensions
RUN pip install --no-cache-dir \
    jupyterlab-git \
    jupyterlab-lsp \
    python-lsp-server \
    jupyterlab_code_formatter \
    black \
    isort \
    nbdime \
    jupyterlab-drawio \
    jupyterlab-spreadsheet \
    jupyterlab_widgets \
    ipywidgets

# Enable extensions
RUN jupyter labextension install --no-build \
    @jupyterlab/git \
    @krassowski/jupyterlab-lsp \
    @ryantam626/jupyterlab_code_formatter \
    jupyterlab-drawio \
    jupyterlab-spreadsheet

RUN jupyter lab build --dev-build=False --minimize=False

# Configure Jupyter
COPY jupyter_lab_config.py /home/jovyan/.jupyter/jupyter_lab_config.py
COPY custom.css /home/jovyan/.jupyter/custom/custom.css

# Set up Git configuration
RUN git config --global user.name "Jupyter User" && \
    git config --global user.email "jupyter@analytics.local"

# Create directories
RUN mkdir -p /home/jovyan/work/notebooks \
             /home/jovyan/work/data \
             /home/jovyan/work/scripts \
             /home/jovyan/work/models

# Set ownership
RUN chown -R jovyan:users /home/jovyan

USER jovyan

# Set working directory
WORKDIR /home/jovyan/work
```

### Python Requirements
```txt
# requirements.txt
# Data manipulation and analysis
pandas>=1.5.0
numpy>=1.21.0
scipy>=1.9.0
polars>=0.15.0
dask[complete]>=2022.8.0

# Machine learning
scikit-learn>=1.1.0
xgboost>=1.6.0
lightgbm>=3.3.0
catboost>=1.1.0
tensorflow>=2.10.0
torch>=1.12.0
transformers>=4.21.0

# Deep learning utilities
pytorch-lightning>=1.7.0
optuna>=3.0.0
mlflow>=1.28.0
wandb>=0.13.0

# Visualization
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.10.0
bokeh>=2.4.0
altair>=4.2.0
holoviews>=1.15.0

# Database connectivity
sqlalchemy>=1.4.0
psycopg2-binary>=2.9.0
pymongo>=4.2.0
redis>=4.3.0
clickhouse-driver>=0.2.0

# Cloud and storage
boto3>=1.24.0
google-cloud-storage>=2.5.0
azure-storage-blob>=12.12.0
s3fs>=2022.8.0
gcsfs>=2022.8.0

# Web scraping and APIs
requests>=2.28.0
beautifulsoup4>=4.11.0
scrapy>=2.6.0
selenium>=4.4.0

# Time series
statsmodels>=0.13.0
prophet>=1.1.0
sktime>=0.13.0

# Natural language processing
spacy>=3.4.0
nltk>=3.7.0
gensim>=4.2.0
textblob>=0.17.0

# Computer vision
opencv-python>=4.6.0
pillow>=9.2.0
imageio>=2.21.0

# Utilities
tqdm>=4.64.0
joblib>=1.1.0
python-dotenv>=0.20.0
pyyaml>=6.0.0
click>=8.1.0
rich>=12.5.0

# Jupyter specific
ipywidgets>=8.0.0
ipykernel>=6.15.0
jupyter-dash>=0.4.0
voila>=0.3.0
```

### Jupyter Configuration
```python
# jupyter_lab_config.py
c = get_config()

# Server configuration
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = 8888
c.ServerApp.open_browser = False
c.ServerApp.allow_root = True
c.ServerApp.token = ''
c.ServerApp.password = ''

# Security settings
c.ServerApp.allow_origin = '*'
c.ServerApp.allow_remote_access = True
c.ServerApp.disable_check_xsrf = True

# File management
c.FileContentsManager.delete_to_trash = False
c.ServerApp.contents_manager_class = 'jupyter_server.services.contents.largefilemanager.LargeFileManager'

# Resource limits
c.MappingKernelManager.cull_idle_timeout = 3600  # 1 hour
c.MappingKernelManager.cull_interval = 300       # 5 minutes
c.MappingKernelManager.cull_connected = True

# Lab configuration
c.LabApp.default_url = '/lab'
c.LabApp.collaborative = True

# Extension configuration
c.ServerApp.jpserver_extensions = {
    'jupyterlab': True,
    'jupyterlab_git': True,
    'nbdime': True,
}

# Logging
c.Application.log_level = 'INFO'
c.ServerApp.log_level = 'INFO'

# Custom CSS and themes
c.LabApp.user_settings_dir = '/home/jovyan/.jupyter/lab/user-settings'
c.LabApp.workspaces_dir = '/home/jovyan/.jupyter/lab/workspaces'
```

## Multi-User JupyterHub

### JupyterHub Docker Compose
```yaml
# docker-compose.hub.yml
version: '3.8'

services:
  jupyterhub:
    build:
      context: .
      dockerfile: Dockerfile.hub
    ports:
      - "8000:8000"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
      - ./jupyterhub_config.py:/srv/jupyterhub/jupyterhub_config.py
      - ./userlist:/srv/jupyterhub/userlist
      - hub-data:/srv/jupyterhub/data
      - shared-data:/srv/shared
    environment:
      - DOCKER_NETWORK_NAME=jupyter-analytics_default
      - HUB_IP=jupyterhub
      - POSTGRES_HOST=postgres
      - POSTGRES_DB=jupyterhub
      - POSTGRES_USER=jupyterhub
      - POSTGRES_PASSWORD=jupyterhub
    depends_on:
      - postgres
    networks:
      - jupyter-network
    restart: unless-stopped

  postgres:
    image: postgres:13
    environment:
      - POSTGRES_DB=jupyterhub
      - POSTGRES_USER=jupyterhub
      - POSTGRES_PASSWORD=jupyterhub
    volumes:
      - postgres-hub-data:/var/lib/postgresql/data
    networks:
      - jupyter-network
    restart: unless-stopped

  # Shared file server
  fileserver:
    image: nginx:alpine
    ports:
      - "8080:80"
    volumes:
      - shared-data:/usr/share/nginx/html
      - ./nginx.conf:/etc/nginx/nginx.conf
    networks:
      - jupyter-network
    restart: unless-stopped

volumes:
  hub-data:
  postgres-hub-data:
  shared-data:

networks:
  jupyter-network:
    external: true
```

### JupyterHub Dockerfile
```dockerfile
# Dockerfile.hub
FROM jupyterhub/jupyterhub:latest

# Install dependencies
RUN pip install --no-cache-dir \
    dockerspawner \
    psycopg2-binary \
    oauthenticator \
    jupyterhub-nativeauthenticator \
    jupyterhub-ldapauthenticator

# Copy configuration
COPY jupyterhub_config.py /srv/jupyterhub/jupyterhub_config.py

# Create directories
RUN mkdir -p /srv/jupyterhub/data

WORKDIR /srv/jupyterhub

CMD ["jupyterhub", "-f", "/srv/jupyterhub/jupyterhub_config.py"]
```

### JupyterHub Configuration
```python
# jupyterhub_config.py
import os
from dockerspawner import DockerSpawner
from jupyterhub.auth import PAMAuthenticator

c = get_config()

# Hub configuration
c.JupyterHub.hub_ip = '0.0.0.0'
c.JupyterHub.hub_port = 8081
c.JupyterHub.port = 8000

# Database configuration
c.JupyterHub.db_url = 'postgresql://jupyterhub:jupyterhub@postgres:5432/jupyterhub'

# Spawner configuration
c.JupyterHub.spawner_class = DockerSpawner

# Docker spawner settings
c.DockerSpawner.image = 'jupyter/datascience-notebook:latest'
c.DockerSpawner.network_name = os.environ.get('DOCKER_NETWORK_NAME', 'jupyter-analytics_default')
c.DockerSpawner.remove = True
c.DockerSpawner.debug = True

# Resource limits
c.DockerSpawner.mem_limit = '4G'
c.DockerSpawner.cpu_limit = 2.0

# Volume mounts
c.DockerSpawner.volumes = {
    'shared-data': '/home/jovyan/shared',
    'jupyterhub-user-{username}': '/home/jovyan/work',
}

# Environment variables for user containers
c.DockerSpawner.environment = {
    'JUPYTER_ENABLE_LAB': '1',
    'GRANT_SUDO': 'yes',
    'CHOWN_HOME': 'yes',
}

# Authentication
c.JupyterHub.authenticator_class = PAMAuthenticator

# Admin users
c.Authenticator.admin_users = {'admin', 'data-admin'}

# User whitelist (optional)
# c.Authenticator.whitelist = {'user1', 'user2', 'user3'}

# Idle server culling
c.JupyterHub.services = [
    {
        'name': 'idle-culler',
        'admin': True,
        'command': [
            'python3', '-m', 'jupyterhub_idle_culler',
            '--timeout=3600',  # 1 hour
            '--cull-every=300',  # 5 minutes
        ],
    }
]

# SSL configuration (for production)
# c.JupyterHub.ssl_cert = '/path/to/cert.pem'
# c.JupyterHub.ssl_key = '/path/to/key.pem'

# Logging
c.Application.log_level = 'INFO'
c.JupyterHub.log_level = 'INFO'

# Custom logo and templates
# c.JupyterHub.logo_file = '/srv/jupyterhub/logo.png'
# c.JupyterHub.template_paths = ['/srv/jupyterhub/templates']
```

## Custom Images

### Data Science Image with GPU Support
```dockerfile
# Dockerfile.gpu
FROM jupyter/tensorflow-notebook:latest

USER root

# Install CUDA toolkit and cuDNN
RUN apt-get update && apt-get install -y \
    cuda-toolkit-11-8 \
    libcudnn8 \
    libcudnn8-dev \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install additional GPU-accelerated libraries
RUN pip install --no-cache-dir \
    cupy-cuda11x \
    rapids-cudf \
    rapids-cuml \
    rapids-cugraph

USER jovyan
```

### R-focused Analytics Image
```dockerfile
# Dockerfile.r-analytics
FROM jupyter/r-notebook:latest

USER root

# Install system dependencies for R packages
RUN apt-get update && apt-get install -y \
    libxml2-dev \
    libssl-dev \
    libcurl4-openssl-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

USER jovyan

# Install R packages for analytics
RUN R -e "install.packages(c( \
    'tidyverse', 'data.table', 'dtplyr', \
    'ggplot2', 'plotly', 'shiny', 'shinydashboard', \
    'caret', 'randomForest', 'xgboost', \
    'DBI', 'RPostgreSQL', 'mongolite', \
    'reticulate', 'rmarkdown', 'knitr', \
    'forecast', 'prophet', 'lubridate', \
    'stringr', 'rvest', 'httr', \
    'devtools', 'roxygen2', 'testthat' \
), repos='https://cran.rstudio.com/')"

# Install IRkernel for Jupyter
RUN R -e "IRkernel::installspec(user = FALSE)"
```

## Data Persistence

### Volume Management
```bash
#!/bin/bash
# scripts/manage_volumes.sh

# Create named volumes
docker volume create jupyter-notebooks
docker volume create jupyter-data
docker volume create shared-datasets

# Backup volumes
docker run --rm -v jupyter-notebooks:/data -v $(pwd):/backup alpine \
    tar czf /backup/notebooks-backup-$(date +%Y%m%d).tar.gz -C /data .

# Restore volumes
docker run --rm -v jupyter-notebooks:/data -v $(pwd):/backup alpine \
    tar xzf /backup/notebooks-backup-20240101.tar.gz -C /data

# List volumes and their sizes
docker system df -v
```

### Shared Data Configuration
```yaml
# docker-compose.shared.yml
version: '3.8'

services:
  jupyter:
    # ... other configuration
    volumes:
      - ./notebooks:/home/jovyan/work/notebooks
      - shared-datasets:/home/jovyan/work/data/shared:ro
      - user-data:/home/jovyan/work/data/private
      - ./scripts:/home/jovyan/work/scripts:ro
      - ./models:/home/jovyan/work/models

  data-loader:
    image: alpine:latest
    volumes:
      - shared-datasets:/data
    command: |
      sh -c "
        wget -O /data/sample_data.csv https://example.com/dataset.csv
        echo 'Data loaded successfully'
      "

volumes:
  shared-datasets:
  user-data:
```

## Security Configuration

### HTTPS with Let's Encrypt
```yaml
# docker-compose.secure.yml
version: '3.8'

services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
      - certbot-data:/var/www/certbot
    depends_on:
      - jupyter
    networks:
      - jupyter-network

  certbot:
    image: certbot/certbot
    volumes:
      - certbot-data:/var/www/certbot
      - ./ssl:/etc/letsencrypt
    command: certonly --webroot --webroot-path=/var/www/certbot --email admin@example.com --agree-tos --no-eff-email -d jupyter.example.com

  jupyter:
    # ... existing configuration
    ports: []  # Remove direct port exposure
    networks:
      - jupyter-network

volumes:
  certbot-data:
```

### Authentication Configuration
```python
# auth_config.py
from oauthenticator.github import GitHubOAuthenticator
from oauthenticator.google import GoogleOAuthenticator

# GitHub OAuth
c.JupyterHub.authenticator_class = GitHubOAuthenticator
c.GitHubOAuthenticator.oauth_callback_url = 'https://jupyter.example.com/hub/oauth_callback'
c.GitHubOAuthenticator.client_id = 'your-github-client-id'
c.GitHubOAuthenticator.client_secret = 'your-github-client-secret'

# Google OAuth
# c.JupyterHub.authenticator_class = GoogleOAuthenticator
# c.GoogleOAuthenticator.oauth_callback_url = 'https://jupyter.example.com/hub/oauth_callback'
# c.GoogleOAuthenticator.client_id = 'your-google-client-id'
# c.GoogleOAuthenticator.client_secret = 'your-google-client-secret'

# LDAP Authentication
# from ldapauthenticator import LDAPAuthenticator
# c.JupyterHub.authenticator_class = LDAPAuthenticator
# c.LDAPAuthenticator.server_address = 'ldap.example.com'
# c.LDAPAuthenticator.bind_dn_template = 'uid={username},ou=people,dc=example,dc=com'
```

## Performance Optimization

### Resource Monitoring
```python
# monitoring/resource_monitor.py
import psutil
import docker
import time
import json
from datetime import datetime

def monitor_containers():
    client = docker.from_env()
    
    while True:
        stats = []
        
        for container in client.containers.list():
            if 'jupyter' in container.name:
                container_stats = container.stats(stream=False)
                
                # CPU usage
                cpu_delta = container_stats['cpu_stats']['cpu_usage']['total_usage'] - \
                           container_stats['precpu_stats']['cpu_usage']['total_usage']
                system_delta = container_stats['cpu_stats']['system_cpu_usage'] - \
                              container_stats['precpu_stats']['system_cpu_usage']
                cpu_percent = (cpu_delta / system_delta) * 100.0
                
                # Memory usage
                memory_usage = container_stats['memory_stats']['usage']
                memory_limit = container_stats['memory_stats']['limit']
                memory_percent = (memory_usage / memory_limit) * 100.0
                
                stats.append({
                    'container': container.name,
                    'timestamp': datetime.now().isoformat(),
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory_percent,
                    'memory_usage_mb': memory_usage / 1024 / 1024,
                    'memory_limit_mb': memory_limit / 1024 / 1024
                })
        
        # Log stats
        with open('/var/log/jupyter_stats.json', 'a') as f:
            for stat in stats:
                f.write(json.dumps(stat) + '\n')
        
        time.sleep(60)  # Monitor every minute

if __name__ == '__main__':
    monitor_containers()
```

### Performance Tuning
```python
# performance_config.py
# Jupyter performance optimizations

# Kernel management
c.MappingKernelManager.cull_idle_timeout = 1800  # 30 minutes
c.MappingKernelManager.cull_interval = 300       # 5 minutes
c.MappingKernelManager.cull_connected = True
c.MappingKernelManager.cull_busy = False

# Memory management
c.NotebookApp.max_buffer_size = 1073741824  # 1GB

# File handling
c.FileContentsManager.delete_to_trash = False
c.ContentsManager.untitled_notebook = 'Untitled'
c.ContentsManager.untitled_file = 'untitled'
c.ContentsManager.untitled_directory = 'Untitled Folder'

# WebSocket configuration
c.NotebookApp.tornado_settings = {
    'websocket_ping_interval': 30,
    'websocket_ping_timeout': 30,
}
```

## Extensions & Plugins

### Essential Extensions Installation
```bash
#!/bin/bash
# scripts/install_extensions.sh

# Install JupyterLab extensions
jupyter labextension install --no-build \
    @jupyterlab/git \
    @krassowski/jupyterlab-lsp \
    @ryantam626/jupyterlab_code_formatter \
    jupyterlab-drawio \
    jupyterlab-spreadsheet \
    @jupyter-widgets/jupyterlab-manager \
    plotlywidget \
    @bokeh/jupyter_bokeh \
    @pyviz/jupyterlab_pyviz

# Install server extensions
pip install --no-cache-dir \
    jupyterlab-git \
    jupyterlab-lsp \
    python-lsp-server[all] \
    jupyterlab_code_formatter \
    black \
    isort \
    nbdime \
    jupyterlab-drawio \
    jupyterlab-spreadsheet

# Build JupyterLab
jupyter lab build --dev-build=False --minimize=False

# Enable server extensions
jupyter server extension enable jupyterlab_git
jupyter server extension enable jupyterlab_code_formatter
jupyter server extension enable nbdime
```

### Custom Extension Development
```python
# custom_extension/setup.py
from setuptools import setup, find_packages

setup(
    name="jupyter-analytics-extension",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "jupyterlab>=3.0.0",
        "jupyter-server>=1.0.0",
    ],
    entry_points={
        "jupyter_serverproxy_servers": [
            "analytics = jupyter_analytics_extension:load_jupyter_server_extension",
        ]
    },
)
```

## Integration with Analytics Stack

### Database Connections
```python
# notebooks/database_connections.py
import pandas as pd
import sqlalchemy as sa
from sqlalchemy import create_engine
import psycopg2
import pymongo
import redis

# PostgreSQL connection
def get_postgres_connection():
    engine = create_engine('postgresql://jupyter:jupyter@postgres:5432/analytics')
    return engine

# MongoDB connection
def get_mongo_connection():
    client = pymongo.MongoClient('mongodb://mongo:27017/')
    return client['analytics']

# Redis connection
def get_redis_connection():
    return redis.Redis(host='redis', port=6379, db=0)

# Example usage
def load_data_from_postgres(query):
    engine = get_postgres_connection()
    return pd.read_sql(query, engine)

def save_data_to_postgres(df, table_name):
    engine = get_postgres_connection()
    df.to_sql(table_name, engine, if_exists='append', index=False)
```

### Cloud Storage Integration
```python
# notebooks/cloud_storage.py
import boto3
import pandas as pd
from google.cloud import storage as gcs
from azure.storage.blob import BlobServiceClient

# AWS S3
def load_from_s3(bucket, key):
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(obj['Body'])

def save_to_s3(df, bucket, key):
    s3 = boto3.client('s3')
    csv_buffer = df.to_csv(index=False)
    s3.put_object(Bucket=bucket, Key=key, Body=csv_buffer)

# Google Cloud Storage
def load_from_gcs(bucket_name, blob_name):
    client = gcs.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    return pd.read_csv(blob.open())

# Azure Blob Storage
def load_from_azure(account_name, container, blob_name):
    blob_service = BlobServiceClient(account_url=f"https://{account_name}.blob.core.windows.net")
    blob_client = blob_service.get_blob_client(container=container, blob=blob_name)
    return pd.read_csv(blob_client.download_blob())
```

## Monitoring & Logging

### Prometheus Metrics
```python
# monitoring/prometheus_exporter.py
from prometheus_client import start_http_server, Gauge, Counter
import docker
import time

# Metrics
jupyter_containers = Gauge('jupyter_containers_total', 'Number of Jupyter containers')
jupyter_cpu_usage = Gauge('jupyter_cpu_usage_percent', 'CPU usage percentage', ['container'])
jupyter_memory_usage = Gauge('jupyter_memory_usage_bytes', 'Memory usage in bytes', ['container'])
jupyter_notebook_opens = Counter('jupyter_notebook_opens_total', 'Number of notebook opens')

def collect_metrics():
    client = docker.from_env()
    
    while True:
        containers = client.containers.list(filters={'name': 'jupyter'})
        jupyter_containers.set(len(containers))
        
        for container in containers:
            stats = container.stats(stream=False)
            
            # CPU usage calculation
            cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                       stats['precpu_stats']['cpu_usage']['total_usage']
            system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                          stats['precpu_stats']['system_cpu_usage']
            cpu_percent = (cpu_delta / system_delta) * 100.0
            
            jupyter_cpu_usage.labels(container=container.name).set(cpu_percent)
            jupyter_memory_usage.labels(container=container.name).set(
                stats['memory_stats']['usage']
            )
        
        time.sleep(30)

if __name__ == '__main__':
    start_http_server(8000)
    collect_metrics()
```

### Log Aggregation
```yaml
# logging/docker-compose.logging.yml
version: '3.8'

services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:7.15.0
    environment:
      - discovery.type=single-node
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
    volumes:
      - elasticsearch-data:/usr/share/elasticsearch/data
    networks:
      - logging

  kibana:
    image: docker.elastic.co/kibana/kibana:7.15.0
    ports:
      - "5601:5601"
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
    depends_on:
      - elasticsearch
    networks:
      - logging

  filebeat:
    image: docker.elastic.co/beats/filebeat:7.15.0
    volumes:
      - ./filebeat.yml:/usr/share/filebeat/filebeat.yml
      - /var/lib/docker/containers:/var/lib/docker/containers:ro
      - /var/run/docker.sock:/var/run/docker.sock:ro
    depends_on:
      - elasticsearch
    networks:
      - logging

volumes:
  elasticsearch-data:

networks:
  logging:
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Container Memory Issues
```bash
# Check container memory usage
docker stats

# Increase memory limits
# In docker-compose.yml:
services:
  jupyter:
    deploy:
      resources:
        limits:
          memory: 8G
```

#### 2. Kernel Connection Issues
```bash
# Check kernel status
docker exec jupyter-container jupyter kernelspec list

# Restart kernels
docker exec jupyter-container jupyter lab clean

# Check logs
docker logs jupyter-container
```

#### 3. Extension Installation Problems
```bash
# Rebuild JupyterLab
docker exec jupyter-container jupyter lab build --dev-build=False

# Clear cache
docker exec jupyter-container jupyter lab clean
docker exec jupyter-container rm -rf ~/.cache/yarn

# Check extension status
docker exec jupyter-container jupyter labextension list
```

#### 4. Database Connection Issues
```python
# Test database connectivity
import psycopg2
try:
    conn = psycopg2.connect(
        host="postgres",
        database="analytics",
        user="jupyter",
        password="jupyter"
    )
    print("Database connection successful")
except Exception as e:
    print(f"Database connection failed: {e}")
```

### Health Check Scripts
```bash
#!/bin/bash
# scripts/health_check.sh

# Check container status
echo "Checking container status..."
docker-compose ps

# Check Jupyter accessibility
echo "Checking Jupyter accessibility..."
curl -f http://localhost:8888/lab || echo "Jupyter Lab not accessible"

# Check database connectivity
echo "Checking database connectivity..."
docker-compose exec postgres pg_isready -U jupyter || echo "Database not ready"

# Check disk space
echo "Checking disk space..."
df -h

# Check memory usage
echo "Checking memory usage..."
free -h

# Check container logs for errors
echo "Checking for errors in logs..."
docker-compose logs --tail=50 jupyter | grep -i error
```

### Performance Diagnostics
```python
# diagnostics/performance_check.py
import psutil
import pandas as pd
import time

def system_diagnostics():
    """Run system performance diagnostics"""
    
    # CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_count = psutil.cpu_count()
    
    # Memory usage
    memory = psutil.virtual_memory()
    
    # Disk usage
    disk = psutil.disk_usage('/')
    
    # Network I/O
    network = psutil.net_io_counters()
    
    diagnostics = {
        'timestamp': time.time(),
        'cpu_percent': cpu_percent,
        'cpu_count': cpu_count,
        'memory_total_gb': memory.total / (1024**3),
        'memory_used_gb': memory.used / (1024**3),
        'memory_percent': memory.percent,
        'disk_total_gb': disk.total / (1024**3),
        'disk_used_gb': disk.used / (1024**3),
        'disk_percent': (disk.used / disk.total) * 100,
        'network_bytes_sent': network.bytes_sent,
        'network_bytes_recv': network.bytes_recv
    }
    
    return diagnostics

# Run diagnostics
if __name__ == '__main__':
    diag = system_diagnostics()
    df = pd.DataFrame([diag])
    print(df.to_string(index=False))
```

## Best Practices

### Security Best Practices
1. **Use strong authentication** (OAuth, LDAP, or multi-factor)
2. **Enable HTTPS** with proper SSL certificates
3. **Implement user isolation** with proper container security
4. **Regular security updates** for base images and packages
5. **Network segmentation** with Docker networks

### Performance Best Practices
1. **Resource limits** to prevent resource exhaustion
2. **Kernel culling** to free up unused resources
3. **Efficient data loading** with chunking and streaming
4. **Caching strategies** for frequently accessed data
5. **Monitoring and alerting** for resource usage

### Development Best Practices
1. **Version control** integration with Git
2. **Code formatting** with Black and isort
3. **Documentation** with markdown and docstrings
4. **Testing** with pytest and nbval
5. **Reproducible environments** with requirements.txt

---

*This Jupyter Docker deployment provides a comprehensive analytics environment suitable for both individual data scientists and teams. The configuration supports scalability, security, and integration with modern data infrastructure.* 
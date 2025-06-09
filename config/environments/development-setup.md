# Development Environment Setup

Comprehensive guide for setting up modern data science and machine learning development environments across different platforms and use cases.

## Table of Contents

- [Overview](#overview)
- [Python Environment Management](#python-environment-management)
- [IDE and Editor Setup](#ide-and-editor-setup)
- [Version Control Configuration](#version-control-configuration)
- [Docker Development](#docker-development)
- [Cloud Development](#cloud-development)
- [Team Collaboration](#team-collaboration)

## Overview

Modern development environments require sophisticated tooling for data science, ML, and analytics workflows. This guide covers best practices for scalable, reproducible development setups.

### Core Requirements

- **Reproducible environments** across team members
- **Version control** for code, data, and models
- **Container support** for consistent deployment
- **Cloud integration** for scalable compute
- **Collaboration tools** for team productivity

## Python Environment Management

### 1. Poetry for Dependency Management

```toml
# pyproject.toml
[tool.poetry]
name = "ml-project"
version = "0.1.0"
description = "Machine learning project template"
authors = ["Your Name <your.email@example.com>"]
readme = "README.md"
packages = [{include = "src"}]

[tool.poetry.dependencies]
python = "^3.9"
pandas = "^2.0.0"
numpy = "^1.24.0"
scikit-learn = "^1.3.0"
torch = "^2.0.0"
transformers = "^4.30.0"
datasets = "^2.12.0"
wandb = "^0.15.0"
jupyter = "^1.0.0"
ipykernel = "^6.23.0"

[tool.poetry.group.dev.dependencies]
pytest = "^7.3.0"
black = "^23.3.0"
isort = "^5.12.0"
flake8 = "^6.0.0"
mypy = "^1.3.0"
pre-commit = "^3.3.0"
jupyter-lab = "^4.0.0"

[tool.poetry.group.docs.dependencies]
mkdocs = "^1.4.0"
mkdocs-material = "^9.1.0"
mkdocstrings = {extras = ["python"], version = "^0.22.0"}

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"

[tool.black]
line-length = 88
target-version = ['py39']
include = '\.pyi?$'
extend-exclude = '''
/(
  # directories
  \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | build
  | dist
)/
'''

[tool.isort]
profile = "black"
multi_line_output = 3
line_length = 88
known_third_party = ["numpy", "pandas", "sklearn", "torch"]

[tool.mypy]
python_version = "3.9"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
```

### Setup Script

```bash
#!/bin/bash
# setup-dev-env.sh

set -e

echo "🚀 Setting up development environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.9"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python $required_version or higher required. Found: $python_version"
    exit 1
fi

# Install Poetry if not present
if ! command -v poetry &> /dev/null; then
    echo "📦 Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="$HOME/.local/bin:$PATH"
fi

# Configure Poetry
echo "⚙️  Configuring Poetry..."
poetry config virtualenvs.in-project true
poetry config virtualenvs.prefer-active-python true

# Install dependencies
echo "📚 Installing dependencies..."
poetry install --with dev,docs

# Setup pre-commit hooks
echo "🔒 Setting up pre-commit hooks..."
poetry run pre-commit install

# Setup Jupyter kernel
echo "📓 Setting up Jupyter kernel..."
poetry run python -m ipykernel install --user --name ml-project --display-name "ML Project"

# Create directory structure
echo "📁 Creating project structure..."
mkdir -p {data/{raw,processed,external},notebooks,src/{models,features,visualization},tests,docs,configs}

# Create .env template
cat > .env.template << EOF
# Environment Variables Template
# Copy to .env and fill in your values

# API Keys
OPENAI_API_KEY=your_openai_api_key_here
HUGGINGFACE_TOKEN=your_hf_token_here
WANDB_API_KEY=your_wandb_api_key_here

# Database URLs
DATABASE_URL=postgresql://user:password@localhost:5432/dbname
REDIS_URL=redis://localhost:6379/0

# Cloud Configuration
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_DEFAULT_REGION=us-west-2

# Model Configuration
MODEL_NAME=gpt-3.5-turbo
MAX_TOKENS=2048
TEMPERATURE=0.7

# Data Paths
DATA_DIR=./data
MODELS_DIR=./models
LOGS_DIR=./logs
EOF

echo "✅ Development environment setup complete!"
echo "📋 Next steps:"
echo "  1. Copy .env.template to .env and fill in your API keys"
echo "  2. Run 'poetry shell' to activate the virtual environment"
echo "  3. Run 'jupyter lab' to start Jupyter Lab"
```

### 2. Conda with conda-lock

```yaml
# environment.yml
name: ml-project
channels:
  - conda-forge
  - pytorch
  - nvidia
dependencies:
  - python=3.9
  - pip
  - numpy
  - pandas
  - scikit-learn
  - pytorch
  - torchvision
  - cudatoolkit=11.8
  - jupyter
  - jupyterlab
  - pip:
    - transformers
    - datasets
    - wandb
    - mlflow
    - optuna
    - streamlit
```

```bash
# Generate lock file for reproducible environments
conda-lock -f environment.yml --platform linux-64 --platform osx-64 --platform win-64

# Create environment from lock file
conda-lock install --name ml-project conda-lock.yml
```

## IDE and Editor Setup

### 1. VS Code Configuration

```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": "./.venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length", "88"],
    "python.sortImports.args": ["--profile", "black"],
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    },
    "jupyter.askForKernelRestart": false,
    "jupyter.interactiveWindow.textEditor.executeSelection": true,
    "files.exclude": {
        "**/.git": true,
        "**/.mypy_cache": true,
        "**/__pycache__": true,
        "**/.*cache": true
    },
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": [
        "tests"
    ],
    "python.testing.unittestEnabled": false,
    "autoDocstring.docstringFormat": "google",
    "jupyter.variableExplorer.executionHistoryLimit": 100
}
```

```json
// .vscode/extensions.json
{
    "recommendations": [
        "ms-python.python",
        "ms-python.vscode-pylance",
        "ms-toolsai.jupyter",
        "ms-python.black-formatter",
        "ms-python.isort",
        "ms-python.flake8",
        "njpwerner.autodocstring",
        "ms-vscode.vscode-json",
        "redhat.vscode-yaml",
        "ms-vscode-remote.remote-containers",
        "github.copilot",
        "github.copilot-chat",
        "ms-vsliveshare.vsliveshare"
    ]
}
```

```json
// .vscode/launch.json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "envFile": "${workspaceFolder}/.env"
        },
        {
            "name": "Python: Train Model",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/src/train.py",
            "console": "integratedTerminal",
            "envFile": "${workspaceFolder}/.env",
            "args": ["--config", "configs/train_config.yaml"]
        },
        {
            "name": "Python: FastAPI",
            "type": "python",
            "request": "launch",
            "module": "uvicorn",
            "console": "integratedTerminal",
            "args": ["src.api.main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"],
            "envFile": "${workspaceFolder}/.env"
        }
    ]
}
```

### 2. JupyterLab Configuration

```python
# ~/.jupyter/jupyter_lab_config.py

import os

# Server configuration
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = 8888
c.ServerApp.open_browser = False
c.ServerApp.token = ''
c.ServerApp.password = ''

# Resource limits
c.ResourceUseDisplay.mem_limit = 16 * 1024**3  # 16GB
c.ResourceUseDisplay.track_cpu_percent = True

# Extensions
c.ServerApp.jpserver_extensions = {
    'jupyterlab': True,
    'jupyter_server_proxy': True,
    'tensorboard': True
}

# Git integration
c.ContentsManager.checkpoints_class = 'notebook.services.contents.checkpoints.Checkpoints'
```

```json
# ~/.jupyter/lab/user-settings/@jupyterlab/notebook-extension/tracker.jupyterlab-settings
{
    "codeCellConfig": {
        "lineNumbers": true,
        "lineWrap": "on"
    },
    "kernelShutdown": false,
    "recordTiming": true,
    "scrollPastEnd": true
}
```

## Version Control Configuration

### 1. Git Configuration

```bash
# .gitconfig (global)
[user]
    name = Your Name
    email = your.email@example.com

[core]
    editor = code --wait
    autocrlf = input
    excludesfile = ~/.gitignore_global

[init]
    defaultBranch = main

[pull]
    rebase = true

[push]
    default = simple
    autoSetupRemote = true

[merge]
    conflictstyle = diff3

[diff]
    tool = vscode

[difftool "vscode"]
    cmd = code --wait --diff $LOCAL $REMOTE

[alias]
    st = status
    co = checkout
    br = branch
    ci = commit
    unstage = reset HEAD --
    last = log -1 HEAD
    visual = !gitk
    lg = log --oneline --graph --decorate --all
    amend = commit --amend --no-edit
```

```gitignore
# .gitignore
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/
cover/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
.pybuilder/
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# pipenv
Pipfile.lock

# poetry
poetry.lock

# pdm
.pdm.toml

# PEP 582
__pypackages__/

# Celery stuff
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# Environments
.env
.env.local
.env.development
.env.test
.env.production
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# pytype static type analyzer
.pytype/

# Cython debug symbols
cython_debug/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Data files
data/raw/
data/processed/
data/external/
!data/raw/.gitkeep
!data/processed/.gitkeep
!data/external/.gitkeep

# Model files
models/
!models/.gitkeep
*.pkl
*.joblib
*.h5
*.pth
*.ckpt

# Logs
logs/
!logs/.gitkeep
*.log

# Temporary files
tmp/
temp/
.tmp/

# ML/AI specific
wandb/
mlruns/
.mlflow/
lightning_logs/
tensorboard_logs/

# Jupyter checkpoints
.ipynb_checkpoints/

# Secrets
secrets/
.secrets/
```

### 2. Pre-commit Configuration

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-json
      - id: check-toml
      - id: check-merge-conflict
      - id: check-added-large-files
        args: ['--maxkb=1000']
      - id: debug-statements

  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3.9

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ["--profile", "black"]

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        additional_dependencies: [flake8-docstrings, flake8-typing-imports]
        args: [--max-line-length=88, --extend-ignore=E203,W503]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.3.0
    hooks:
      - id: mypy
        additional_dependencies: [types-requests, types-PyYAML]

  - repo: https://github.com/nbQA-dev/nbQA
    rev: 1.7.0
    hooks:
      - id: nbqa-black
      - id: nbqa-isort
        args: ["--profile", "black"]
      - id: nbqa-flake8
        args: [--max-line-length=88, --extend-ignore=E203,W503]

  - repo: https://github.com/Yelp/detect-secrets
    rev: v1.4.0
    hooks:
      - id: detect-secrets
        args: ['--baseline', '.secrets.baseline']
        exclude: package.lock.json
```

## Docker Development

### 1. Development Dockerfile

```dockerfile
# Dockerfile.dev
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry

# Configure Poetry
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Set work directory
WORKDIR /workspace

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry install --with dev && rm -rf $POETRY_CACHE_DIR

# Install Jupyter kernel
RUN poetry run python -m ipykernel install --name ml-project

# Copy source code
COPY . .

# Expose ports
EXPOSE 8888 8000 6006

# Default command
CMD ["poetry", "run", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
```

### 2. Docker Compose for Development

```yaml
# docker-compose.dev.yml
version: '3.8'

services:
  ml-dev:
    build:
      context: .
      dockerfile: Dockerfile.dev
    container_name: ml-dev-environment
    ports:
      - "8888:8888"  # Jupyter Lab
      - "8000:8000"  # FastAPI
      - "6006:6006"  # TensorBoard
      - "8501:8501"  # Streamlit
    volumes:
      - .:/workspace
      - ml-cache:/root/.cache
      - jupyter-config:/root/.jupyter
    environment:
      - JUPYTER_ENABLE_LAB=yes
      - GRANT_SUDO=yes
    env_file:
      - .env
    networks:
      - ml-network
    depends_on:
      - postgres
      - redis

  postgres:
    image: postgres:15
    container_name: ml-postgres
    environment:
      POSTGRES_DB: mldb
      POSTGRES_USER: mluser
      POSTGRES_PASSWORD: mlpassword
    ports:
      - "5432:5432"
    volumes:
      - postgres-data:/var/lib/postgresql/data
    networks:
      - ml-network

  redis:
    image: redis:7-alpine
    container_name: ml-redis
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    networks:
      - ml-network

  mlflow:
    image: python:3.9-slim
    container_name: ml-mlflow
    command: >
      bash -c "pip install mlflow psycopg2-binary &&
               mlflow server --host 0.0.0.0 --port 5000
               --backend-store-uri postgresql://mluser:mlpassword@postgres:5432/mldb
               --default-artifact-root /artifacts"
    ports:
      - "5000:5000"
    volumes:
      - mlflow-artifacts:/artifacts
    depends_on:
      - postgres
    networks:
      - ml-network

volumes:
  ml-cache:
  jupyter-config:
  postgres-data:
  redis-data:
  mlflow-artifacts:

networks:
  ml-network:
    driver: bridge
```

### 3. Development Scripts

```bash
#!/bin/bash
# scripts/dev-up.sh

# Start development environment
echo "🚀 Starting development environment..."

# Build and start containers
docker-compose -f docker-compose.dev.yml up -d --build

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 10

# Show service URLs
echo "✅ Development environment ready!"
echo ""
echo "📊 Services available:"
echo "  🔬 Jupyter Lab:  http://localhost:8888"
echo "  🚀 FastAPI:      http://localhost:8000"
echo "  📈 TensorBoard:  http://localhost:6006"
echo "  📊 MLflow:       http://localhost:5000"
echo "  🗄️  PostgreSQL:   localhost:5432"
echo "  🔄 Redis:        localhost:6379"
echo ""
echo "🔧 To connect to the development container:"
echo "  docker exec -it ml-dev-environment bash"
```

```bash
#!/bin/bash
# scripts/dev-down.sh

# Stop development environment
echo "🛑 Stopping development environment..."

docker-compose -f docker-compose.dev.yml down

echo "✅ Development environment stopped!"
```

## Cloud Development

### 1. GitHub Codespaces

```json
// .devcontainer/devcontainer.json
{
    "name": "ML Development",
    "dockerComposeFile": "../docker-compose.dev.yml",
    "service": "ml-dev",
    "workspaceFolder": "/workspace",
    "features": {
        "ghcr.io/devcontainers/features/git:1": {},
        "ghcr.io/devcontainers/features/github-cli:1": {}
    },
    "customizations": {
        "vscode": {
            "extensions": [
                "ms-python.python",
                "ms-python.vscode-pylance",
                "ms-toolsai.jupyter",
                "ms-python.black-formatter",
                "github.copilot"
            ],
            "settings": {
                "python.defaultInterpreterPath": "/workspace/.venv/bin/python",
                "python.formatting.provider": "black"
            }
        }
    },
    "forwardPorts": [8888, 8000, 6006, 5000],
    "portsAttributes": {
        "8888": {
            "label": "Jupyter Lab",
            "onAutoForward": "openBrowser"
        },
        "8000": {
            "label": "FastAPI"
        },
        "6006": {
            "label": "TensorBoard"
        },
        "5000": {
            "label": "MLflow"
        }
    },
    "postCreateCommand": "bash scripts/setup-dev-env.sh",
    "postStartCommand": "poetry install"
}
```

### 2. AWS SageMaker Studio

```python
# sagemaker_studio_setup.py
import boto3
import sagemaker
from sagemaker.pytorch import PyTorch

def setup_sagemaker_environment():
    """Setup SageMaker Studio environment."""
    
    # Get SageMaker session
    session = sagemaker.Session()
    role = sagemaker.get_execution_role()
    
    # Create custom container
    container_uri = f"763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.0.0-gpu-py310-cu118-ubuntu20.04-sagemaker"
    
    # Create estimator
    estimator = PyTorch(
        entry_point='train.py',
        source_dir='src',
        role=role,
        instance_type='ml.g4dn.xlarge',
        instance_count=1,
        framework_version='2.0.0',
        py_version='py310',
        image_uri=container_uri,
        hyperparameters={
            'epochs': 10,
            'batch_size': 32,
            'learning_rate': 0.001
        },
        environment={
            'WANDB_API_KEY': 'your_wandb_key',
            'MLFLOW_TRACKING_URI': 'your_mlflow_uri'
        }
    )
    
    return estimator

# Create lifecycle configuration
lifecycle_config = """
#!/bin/bash
set -e

# Install additional packages
pip install --upgrade pip
pip install wandb mlflow optuna

# Setup git
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Clone repository
cd /home/sagemaker-user
git clone https://github.com/your-org/your-repo.git

echo "Environment setup complete!"
"""
```

## Team Collaboration

### 1. Makefile for Common Tasks

```makefile
# Makefile
.PHONY: help install test lint format clean docker-build docker-run

help: ## Show this help message
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install dependencies
	poetry install --with dev,docs

install-pre-commit: ## Install pre-commit hooks
	poetry run pre-commit install

test: ## Run tests
	poetry run pytest tests/ -v --cov=src --cov-report=html

test-watch: ## Run tests in watch mode
	poetry run pytest-watch tests/

lint: ## Run linting
	poetry run flake8 src tests
	poetry run mypy src

format: ## Format code
	poetry run black src tests
	poetry run isort src tests

format-notebooks: ## Format Jupyter notebooks
	poetry run nbqa black notebooks/
	poetry run nbqa isort notebooks/

clean: ## Clean up build artifacts
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf build/ dist/ htmlcov/

docker-build: ## Build Docker image
	docker build -f Dockerfile.dev -t ml-project:dev .

docker-run: ## Run Docker container
	docker-compose -f docker-compose.dev.yml up -d

docker-stop: ## Stop Docker containers
	docker-compose -f docker-compose.dev.yml down

docs-serve: ## Serve documentation
	poetry run mkdocs serve

docs-build: ## Build documentation
	poetry run mkdocs build

train: ## Train model
	poetry run python src/train.py --config configs/train_config.yaml

evaluate: ## Evaluate model
	poetry run python src/evaluate.py --model-path models/latest.pkl

serve: ## Start API server
	poetry run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

streamlit: ## Start Streamlit app
	poetry run streamlit run src/app.py

jupyter: ## Start Jupyter Lab
	poetry run jupyter lab --ip=0.0.0.0 --port=8888

setup: install install-pre-commit ## Complete setup
	@echo "✅ Setup complete! Run 'make help' to see available commands."
```

### 2. Team Configuration Templates

```yaml
# .github/workflows/ci.yml
name: Continuous Integration

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]

    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install Poetry
      uses: snok/install-poetry@v1
      with:
        version: latest
        virtualenvs-create: true
        virtualenvs-in-project: true
    
    - name: Load cached venv
      id: cached-poetry-dependencies
      uses: actions/cache@v3
      with:
        path: .venv
        key: venv-${{ runner.os }}-${{ matrix.python-version }}-${{ hashFiles('**/poetry.lock') }}
    
    - name: Install dependencies
      if: steps.cached-poetry-dependencies.outputs.cache-hit != 'true'
      run: poetry install --with dev
    
    - name: Run tests
      run: |
        poetry run pytest tests/ --cov=src --cov-report=xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella
```

## Related Resources

- [Docker Development](../../deployment/docker/)
- [Cloud Infrastructure](../../deployment/cloud/)
- [Version Control Best Practices](../../devops/)
- [Team Collaboration Tools](../../tools/productivity/)

## Contributing

When adding new development setup patterns:
1. Test across different operating systems
2. Document resource requirements
3. Include troubleshooting guides
4. Provide automation scripts
5. Add team onboarding instructions 
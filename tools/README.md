# 🛠️ Development Tools & Utilities

> Essential tools and utilities for data science development, automation, and productivity.

## 📁 Directory Structure

```
tools/
├── automation/              # CI/CD, workflow automation
├── data-quality/           # Data validation, profiling
├── monitoring/             # Logging, metrics, alerting
└── productivity/           # IDE configs, shortcuts, aliases
```

## 🤖 Automation Tools

### CI/CD Pipelines
- **GitHub Actions**: Automated testing and deployment
- **Jenkins**: Enterprise CI/CD automation
- **GitLab CI**: Integrated DevOps platform
- **Azure DevOps**: Microsoft's DevOps solution

### Workflow Automation
- **Apache Airflow**: Workflow orchestration
- **Prefect**: Modern workflow management
- **Dagster**: Data orchestration platform
- **Luigi**: Python workflow management

### Example GitHub Actions Workflow
```yaml
name: ML Model CI/CD
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    - name: Run tests
      run: |
        pytest tests/
    - name: Train model
      run: |
        python src/train_model.py
```

## 📊 Data Quality Tools

### Data Validation
- **Great Expectations**: Data quality testing
- **Pandera**: Statistical data validation
- **Cerberus**: Lightweight data validation
- **Schema**: Simple data validation

### Data Profiling
- **Pandas Profiling**: Automated EDA reports
- **Sweetviz**: Visual data analysis
- **DataPrep**: Data preparation toolkit
- **Deequ**: Data quality validation (Scala/Python)

### Example Data Quality Check
```python
import great_expectations as ge
import pandas as pd

def validate_data(df):
    """Validate data quality using Great Expectations"""
    df_ge = ge.from_pandas(df)
    
    # Basic validations
    df_ge.expect_column_to_exist('target')
    df_ge.expect_column_values_to_not_be_null('target')
    df_ge.expect_column_values_to_be_between('age', 0, 120)
    
    # Get validation results
    validation_result = df_ge.validate()
    return validation_result.success
```

## 📈 Monitoring & Observability

### Application Monitoring
- **Prometheus**: Metrics collection
- **Grafana**: Visualization and dashboards
- **ELK Stack**: Elasticsearch, Logstash, Kibana
- **DataDog**: Cloud monitoring platform

### ML Model Monitoring
- **MLflow**: ML lifecycle management
- **Weights & Biases**: Experiment tracking
- **Neptune**: ML metadata store
- **Evidently**: ML model monitoring

### Example Monitoring Setup
```python
import logging
import mlflow
from prometheus_client import Counter, Histogram

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
prediction_counter = Counter('ml_predictions_total', 'Total predictions made')
prediction_latency = Histogram('ml_prediction_duration_seconds', 'Prediction latency')

def monitor_prediction(func):
    """Decorator to monitor ML predictions"""
    def wrapper(*args, **kwargs):
        with prediction_latency.time():
            result = func(*args, **kwargs)
        prediction_counter.inc()
        logger.info(f"Prediction made: {result}")
        return result
    return wrapper
```

## 💻 Productivity Tools

### IDE Configurations
- **VS Code**: Extensions and settings
- **PyCharm**: Professional Python IDE
- **Jupyter**: Interactive notebooks
- **RStudio**: R development environment

### Command Line Tools
- **tmux**: Terminal multiplexer
- **zsh/oh-my-zsh**: Enhanced shell
- **fzf**: Fuzzy finder
- **ripgrep**: Fast text search

### Development Shortcuts
```bash
# Useful aliases for data science
alias jl='jupyter lab'
alias jn='jupyter notebook'
alias py='python'
alias ipy='ipython'
alias condaenv='conda info --envs'
alias gitlog='git log --oneline --graph'

# Quick data exploration
alias head10='head -n 10'
alias tail10='tail -n 10'
alias wcl='wc -l'
```

## 🔧 Configuration Management

### Environment Management
- **Conda**: Package and environment management
- **Poetry**: Python dependency management
- **Docker**: Containerization
- **Vagrant**: Development environments

### Configuration Files
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 22.3.0
    hooks:
      - id: black
  - repo: https://github.com/pycqa/flake8
    rev: 4.0.1
    hooks:
      - id: flake8
  - repo: https://github.com/pycqa/isort
    rev: 5.10.1
    hooks:
      - id: isort
```

## 🚀 Quick Setup Scripts

### Python Environment Setup
```bash
#!/bin/bash
# setup_python_env.sh

echo "Setting up Python data science environment..."

# Create conda environment
conda create -n ds-env python=3.9 -y
conda activate ds-env

# Install core packages
conda install pandas numpy scikit-learn matplotlib seaborn jupyter -y
pip install mlflow great-expectations streamlit

echo "Environment setup complete!"
```

### R Environment Setup
```r
# setup_r_env.R

# Install essential packages
packages <- c(
  "tidyverse", "caret", "randomForest", "xgboost",
  "ggplot2", "plotly", "shiny", "rmarkdown"
)

install.packages(packages, dependencies = TRUE)

# Install Bioconductor packages
if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

BiocManager::install(c("DESeq2", "limma"))

cat("R environment setup complete!\n")
```

## 📚 Best Practices

### Development Workflow
1. **Version Control**: Use Git with meaningful commit messages
2. **Code Quality**: Implement linting and formatting
3. **Testing**: Write unit tests and integration tests
4. **Documentation**: Maintain clear documentation
5. **Automation**: Automate repetitive tasks

### Tool Selection Criteria
1. **Compatibility**: Works with existing stack
2. **Scalability**: Handles growing data and complexity
3. **Community**: Active development and support
4. **Learning Curve**: Reasonable adoption time
5. **Cost**: Fits within budget constraints

## 🔗 Additional Resources

- [Awesome Data Science Tools](https://github.com/academic/awesome-datascience)
- [MLOps Tools Landscape](https://ml-ops.org/content/state-of-mlops)
- [Data Engineering Tools](https://github.com/igorbarinov/awesome-data-engineering)
- [DevOps Tools](https://github.com/wmariuss/awesome-devops) 
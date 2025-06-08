# MLOps and Data Science DevOps

A comprehensive guide to implementing DevOps practices for data science and machine learning projects, covering the entire ML lifecycle from development to production.

## Table of Contents

1. [MLOps Fundamentals](#mlops-fundamentals)
2. [CI/CD for Data Science](#cicd-for-data-science)
3. [Model Versioning and Registry](#model-versioning-and-registry)
4. [Automated Testing for ML](#automated-testing-for-ml)
5. [Model Deployment Strategies](#model-deployment-strategies)
6. [Monitoring and Observability](#monitoring-and-observability)
7. [Infrastructure as Code](#infrastructure-as-code)
8. [Data Pipeline Orchestration](#data-pipeline-orchestration)
9. [Security and Governance](#security-and-governance)
10. [Advanced MLOps Patterns](#advanced-mlops-patterns)

## MLOps Fundamentals

### MLOps Maturity Model

```yaml
# mlops-maturity-levels.yml
maturity_levels:
  level_0_manual:
    description: "Manual, script-driven, and interactive process"
    characteristics:
      - Manual data preparation
      - Manual model training
      - Manual deployment
      - No automation
      - Notebook-driven development

  level_1_ml_pipeline:
    description: "ML pipeline automation"
    characteristics:
      - Automated training pipeline
      - Experimental tracking
      - Model registry
      - Basic CI/CD for ML code
      - Feature store integration

  level_2_training_pipeline:
    description: "Automated training pipeline"
    characteristics:
      - Source control integration
      - Automated testing
      - Containerized environments
      - Automated model validation
      - Performance monitoring

  level_3_cicd_pipeline:
    description: "Robust CI/CD pipeline"
    characteristics:
      - Automated deployment
      - A/B testing framework
      - Model performance monitoring
      - Automated rollback
      - Multi-environment deployment

  level_4_full_mlops:
    description: "Full MLOps automation"
    characteristics:
      - Automated retraining
      - Drift detection
      - Automated model updates
      - End-to-end monitoring
      - Self-healing systems
```

### Core MLOps Tools and Technologies

```bash
#!/bin/bash
# install-mlops-tools.sh

# Core MLOps platform tools
pip install --upgrade \
    mlflow \
    wandb \
    neptune-client \
    dvc \
    cml \
    great-expectations \
    evidently \
    seldon-core \
    kubeflow-pipelines \
    tfx \
    metaflow \
    prefect \
    airflow-providers-kubernetes \
    feast \
    whylogs

# Container and orchestration tools
sudo apt install -y docker.io docker-compose
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Infrastructure as Code
sudo snap install terraform
pip install pulumi

# Monitoring and observability
pip install prometheus-client grafana-api
sudo apt install -y prometheus grafana

# Data quality and validation
pip install pandera schema
```

## CI/CD for Data Science

### GitHub Actions for ML Projects

```yaml
# .github/workflows/ml-pipeline.yml
name: ML Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

env:
  PYTHON_VERSION: 3.9
  DOCKER_REGISTRY: your-registry.com
  MODEL_REGISTRY: mlflow-registry

jobs:
  data-validation:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Install dependencies
      run: |
        pip install -r requirements-dev.txt
    
    - name: Data validation
      run: |
        python scripts/validate_data.py
        
    - name: Upload data validation report
      uses: actions/upload-artifact@v3
      with:
        name: data-validation-report
        path: reports/data_validation_report.html

  model-training:
    needs: data-validation
    runs-on: ubuntu-latest
    strategy:
      matrix:
        model-type: [xgboost, lightgbm, random-forest]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    
    - name: Train model
      run: |
        python src/train_model.py \
          --model-type ${{ matrix.model-type }} \
          --experiment-name "ci-cd-${{ github.run_id }}" \
          --tracking-uri ${{ secrets.MLFLOW_TRACKING_URI }}
      env:
        MLFLOW_TRACKING_USERNAME: ${{ secrets.MLFLOW_USERNAME }}
        MLFLOW_TRACKING_PASSWORD: ${{ secrets.MLFLOW_PASSWORD }}
    
    - name: Model validation
      run: |
        python src/validate_model.py \
          --model-type ${{ matrix.model-type }} \
          --run-id ${{ github.run_id }}

  model-testing:
    needs: model-training
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Install dependencies
      run: |
        pip install -r requirements-test.txt
    
    - name: Run model tests
      run: |
        pytest tests/test_models.py -v --cov=src/models
        
    - name: Performance benchmarking
      run: |
        python tests/benchmark_models.py
    
    - name: Security scanning
      run: |
        bandit -r src/
        safety check
    
    - name: Upload test results
      uses: actions/upload-artifact@v3
      with:
        name: test-results
        path: |
          test-results.xml
          coverage.xml
          benchmark-results.json

  build-and-deploy:
    needs: model-testing
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Build Docker image
      run: |
        docker build -t ${{ env.DOCKER_REGISTRY }}/ml-model:${{ github.sha }} .
        docker build -t ${{ env.DOCKER_REGISTRY }}/ml-model:latest .
    
    - name: Run container tests
      run: |
        docker run --rm ${{ env.DOCKER_REGISTRY }}/ml-model:${{ github.sha }} \
          python -m pytest tests/test_api.py
    
    - name: Push to registry
      run: |
        echo ${{ secrets.DOCKER_PASSWORD }} | docker login ${{ env.DOCKER_REGISTRY }} -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
        docker push ${{ env.DOCKER_REGISTRY }}/ml-model:${{ github.sha }}
        docker push ${{ env.DOCKER_REGISTRY }}/ml-model:latest
    
    - name: Deploy to staging
      run: |
        kubectl apply -f k8s/staging/ --dry-run=client
        kubectl apply -f k8s/staging/
        kubectl set image deployment/ml-model-staging ml-model=${{ env.DOCKER_REGISTRY }}/ml-model:${{ github.sha }}
    
    - name: Run integration tests
      run: |
        python tests/test_integration.py --endpoint "https://ml-model-staging.example.com"
    
    - name: Deploy to production
      if: success()
      run: |
        kubectl apply -f k8s/production/
        kubectl set image deployment/ml-model-prod ml-model=${{ env.DOCKER_REGISTRY }}/ml-model:${{ github.sha }}

  model-monitoring:
    needs: build-and-deploy
    runs-on: ubuntu-latest
    
    steps:
    - name: Setup monitoring
      run: |
        python scripts/setup_monitoring.py \
          --model-version ${{ github.sha }} \
          --environment production
```

### GitLab CI/CD for Data Science

```yaml
# .gitlab-ci.yml
stages:
  - validate
  - train
  - test
  - build
  - deploy
  - monitor

variables:
  DOCKER_DRIVER: overlay2
  DOCKER_TLS_CERTDIR: "/certs"
  PYTHON_VERSION: "3.9"

before_script:
  - python --version
  - pip install --upgrade pip

data_validation:
  stage: validate
  image: python:${PYTHON_VERSION}
  script:
    - pip install -r requirements-validation.txt
    - python scripts/data_quality_check.py
    - python scripts/schema_validation.py
  artifacts:
    reports:
      junit: reports/data-validation-report.xml
    paths:
      - reports/
    expire_in: 1 week
  rules:
    - if: '$CI_PIPELINE_SOURCE == "merge_request_event"'
    - if: '$CI_COMMIT_BRANCH == "main"'

feature_engineering:
  stage: validate
  image: python:${PYTHON_VERSION}
  script:
    - pip install -r requirements.txt
    - python src/feature_engineering.py --validate
    - python tests/test_features.py
  artifacts:
    paths:
      - features/
    expire_in: 1 day

model_training:
  stage: train
  image: python:${PYTHON_VERSION}
  script:
    - pip install -r requirements.txt
    - |
      python src/train.py \
        --experiment-name "gitlab-ci-${CI_PIPELINE_ID}" \
        --tracking-uri ${MLFLOW_TRACKING_URI} \
        --model-registry ${MODEL_REGISTRY_URI}
  artifacts:
    paths:
      - models/
      - reports/training/
    expire_in: 1 week
  only:
    - main
    - develop

model_evaluation:
  stage: test
  image: python:${PYTHON_VERSION}
  script:
    - pip install -r requirements-test.txt
    - python src/evaluate_model.py
    - python tests/test_model_performance.py
  artifacts:
    reports:
      junit: reports/model-test-report.xml
    paths:
      - reports/evaluation/
    expire_in: 1 week

security_scan:
  stage: test
  image: python:${PYTHON_VERSION}
  script:
    - pip install bandit safety
    - bandit -r src/ -f json -o reports/bandit-report.json
    - safety check --json --output reports/safety-report.json
  artifacts:
    paths:
      - reports/
    expire_in: 1 week
  allow_failure: true

build_image:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  script:
    - docker build -t ${CI_REGISTRY_IMAGE}:${CI_COMMIT_SHA} .
    - docker build -t ${CI_REGISTRY_IMAGE}:latest .
    - docker login -u ${CI_REGISTRY_USER} -p ${CI_REGISTRY_PASSWORD} ${CI_REGISTRY}
    - docker push ${CI_REGISTRY_IMAGE}:${CI_COMMIT_SHA}
    - docker push ${CI_REGISTRY_IMAGE}:latest
  only:
    - main

deploy_staging:
  stage: deploy
  image: bitnami/kubectl:latest
  script:
    - kubectl config use-context staging
    - kubectl set image deployment/ml-model ml-model=${CI_REGISTRY_IMAGE}:${CI_COMMIT_SHA}
    - kubectl rollout status deployment/ml-model
  environment:
    name: staging
    url: https://ml-model-staging.example.com
  only:
    - main

integration_tests:
  stage: deploy
  image: python:${PYTHON_VERSION}
  script:
    - pip install requests pytest
    - python tests/test_api_integration.py --base-url https://ml-model-staging.example.com
  dependencies:
    - deploy_staging

deploy_production:
  stage: deploy
  image: bitnami/kubectl:latest
  script:
    - kubectl config use-context production
    - kubectl set image deployment/ml-model ml-model=${CI_REGISTRY_IMAGE}:${CI_COMMIT_SHA}
    - kubectl rollout status deployment/ml-model
  environment:
    name: production
    url: https://ml-model.example.com
  when: manual
  only:
    - main

setup_monitoring:
  stage: monitor
  image: python:${PYTHON_VERSION}
  script:
    - pip install prometheus-client grafana-api
    - python scripts/setup_model_monitoring.py
    - python scripts/create_alerts.py
  only:
    - main
```

## Model Versioning and Registry

### MLflow Model Registry Setup

```python
#!/usr/bin/env python3
# scripts/mlflow_setup.py

import mlflow
import mlflow.sklearn
import mlflow.pytorch
import mlflow.tensorflow
from mlflow.tracking import MlflowClient
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
import boto3
from datetime import datetime
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLflowModelRegistry:
    def __init__(self, tracking_uri, registry_uri=None, s3_bucket=None):
        """
        Initialize MLflow Model Registry
        
        Args:
            tracking_uri: MLflow tracking server URI
            registry_uri: Model registry URI (if different from tracking)
            s3_bucket: S3 bucket for artifact storage
        """
        self.tracking_uri = tracking_uri
        self.registry_uri = registry_uri or tracking_uri
        self.s3_bucket = s3_bucket
        
        mlflow.set_tracking_uri(self.tracking_uri)
        if self.registry_uri:
            mlflow.set_registry_uri(self.registry_uri)
        
        self.client = MlflowClient()
        
        if s3_bucket:
            self._setup_s3_artifacts()
    
    def _setup_s3_artifacts(self):
        """Setup S3 for artifact storage"""
        artifact_uri = f"s3://{self.s3_bucket}/mlflow-artifacts"
        mlflow.set_tracking_uri(f"{self.tracking_uri}")
        
    def register_model(self, model, model_name, run_id=None, tags=None, 
                      description=None, signature=None):
        """
        Register a model in the MLflow Model Registry
        
        Args:
            model: Trained model object
            model_name: Name for the registered model
            run_id: MLflow run ID (if None, creates new run)
            tags: Dictionary of tags to add
            description: Model description
            signature: Model signature
        """
        if run_id is None:
            with mlflow.start_run() as run:
                run_id = run.info.run_id
                self._log_model(model, model_name, signature)
        
        # Register the model
        model_uri = f"runs:/{run_id}/{model_name}"
        
        try:
            # Create registered model if it doesn't exist
            try:
                self.client.get_registered_model(model_name)
            except Exception:
                self.client.create_registered_model(
                    model_name, 
                    tags=tags,
                    description=description
                )
            
            # Create model version
            model_version = self.client.create_model_version(
                name=model_name,
                source=model_uri,
                run_id=run_id,
                tags=tags,
                description=description
            )
            
            logger.info(f"Registered model {model_name} version {model_version.version}")
            return model_version
            
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            raise
    
    def _log_model(self, model, model_name, signature=None):
        """Log model based on its type"""
        if hasattr(model, 'predict'):  # Sklearn-like interface
            mlflow.sklearn.log_model(
                model, 
                model_name,
                signature=signature,
                registered_model_name=model_name
            )
        elif hasattr(model, 'state_dict'):  # PyTorch
            mlflow.pytorch.log_model(
                model,
                model_name,
                signature=signature,
                registered_model_name=model_name
            )
        else:
            # Generic Python model
            mlflow.pyfunc.log_model(
                model_name,
                python_model=model,
                signature=signature,
                registered_model_name=model_name
            )
    
    def transition_model_stage(self, model_name, version, stage, 
                             archive_existing=True):
        """
        Transition model to a specific stage
        
        Args:
            model_name: Name of the registered model
            version: Version number or string
            stage: Target stage ('Staging', 'Production', 'Archived')
            archive_existing: Whether to archive existing models in target stage
        """
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage,
                archive_existing_versions=archive_existing
            )
            logger.info(f"Transitioned {model_name} v{version} to {stage}")
            
        except Exception as e:
            logger.error(f"Failed to transition model stage: {e}")
            raise
    
    def load_model(self, model_name, stage="Production", version=None):
        """
        Load a model from the registry
        
        Args:
            model_name: Name of the registered model
            stage: Model stage to load from
            version: Specific version (overrides stage)
        """
        if version:
            model_uri = f"models:/{model_name}/{version}"
        else:
            model_uri = f"models:/{model_name}/{stage}"
        
        try:
            return mlflow.pyfunc.load_model(model_uri)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def compare_models(self, model_name, version1, version2, metrics=None):
        """
        Compare two model versions
        
        Args:
            model_name: Name of the registered model
            version1: First version to compare
            version2: Second version to compare
            metrics: List of metrics to compare
        """
        mv1 = self.client.get_model_version(model_name, version1)
        mv2 = self.client.get_model_version(model_name, version2)
        
        run1 = self.client.get_run(mv1.run_id)
        run2 = self.client.get_run(mv2.run_id)
        
        comparison = {
            'model_name': model_name,
            'version1': {
                'version': version1,
                'run_id': mv1.run_id,
                'metrics': run1.data.metrics,
                'creation_timestamp': mv1.creation_timestamp
            },
            'version2': {
                'version': version2,
                'run_id': mv2.run_id,
                'metrics': run2.data.metrics,
                'creation_timestamp': mv2.creation_timestamp
            }
        }
        
        if metrics:
            comparison['metric_comparison'] = {}
            for metric in metrics:
                m1 = run1.data.metrics.get(metric)
                m2 = run2.data.metrics.get(metric)
                if m1 is not None and m2 is not None:
                    comparison['metric_comparison'][metric] = {
                        'version1': m1,
                        'version2': m2,
                        'difference': m2 - m1,
                        'improvement': m2 > m1
                    }
        
        return comparison
    
    def get_model_lineage(self, model_name, version):
        """Get model lineage information"""
        model_version = self.client.get_model_version(model_name, version)
        run = self.client.get_run(model_version.run_id)
        
        lineage = {
            'model_name': model_name,
            'version': version,
            'run_id': model_version.run_id,
            'creation_timestamp': model_version.creation_timestamp,
            'source': model_version.source,
            'run_name': run.data.tags.get('mlflow.runName'),
            'user_id': model_version.user_id,
            'description': model_version.description,
            'tags': model_version.tags,
            'params': run.data.params,
            'metrics': run.data.metrics,
            'artifacts': [f.path for f in self.client.list_artifacts(model_version.run_id)]
        }
        
        return lineage

# Example usage and model management workflows
class ModelLifecycleManager:
    def __init__(self, registry: MLflowModelRegistry):
        self.registry = registry
        
    def automated_model_promotion(self, model_name, promotion_criteria):
        """
        Automatically promote models based on criteria
        
        Args:
            model_name: Name of the model to evaluate
            promotion_criteria: Dictionary with promotion thresholds
        """
        # Get latest staging model
        staging_models = self.registry.client.get_latest_versions(
            model_name, stages=["Staging"]
        )
        
        if not staging_models:
            logger.warning(f"No staging models found for {model_name}")
            return
        
        staging_model = staging_models[0]
        run = self.registry.client.get_run(staging_model.run_id)
        
        # Check promotion criteria
        promote = True
        for metric, threshold in promotion_criteria.items():
            if metric not in run.data.metrics:
                logger.warning(f"Metric {metric} not found in model metrics")
                promote = False
                break
            
            if run.data.metrics[metric] < threshold:
                logger.info(f"Model {model_name} v{staging_model.version} does not meet {metric} threshold")
                promote = False
                break
        
        if promote:
            self.registry.transition_model_stage(
                model_name, 
                staging_model.version, 
                "Production"
            )
            logger.info(f"Promoted {model_name} v{staging_model.version} to Production")
        else:
            logger.info(f"Model {model_name} v{staging_model.version} not promoted")
    
    def model_rollback(self, model_name, target_version=None):
        """
        Rollback to previous production model version
        
        Args:
            model_name: Name of the model
            target_version: Specific version to rollback to (optional)
        """
        if target_version:
            # Rollback to specific version
            self.registry.transition_model_stage(
                model_name, target_version, "Production"
            )
        else:
            # Rollback to previous production version
            versions = self.registry.client.search_model_versions(
                f"name='{model_name}'"
            )
            
            # Sort by creation timestamp
            versions.sort(key=lambda x: x.creation_timestamp, reverse=True)
            
            # Find current production and previous version
            current_prod = None
            previous_version = None
            
            for version in versions:
                if version.current_stage == "Production":
                    current_prod = version
                elif current_prod and version.current_stage == "Archived":
                    previous_version = version
                    break
            
            if previous_version:
                self.registry.transition_model_stage(
                    model_name, previous_version.version, "Production"
                )
                logger.info(f"Rolled back {model_name} to version {previous_version.version}")
            else:
                logger.error(f"No previous version found for rollback of {model_name}")

# Initialize model registry
if __name__ == "__main__":
    registry = MLflowModelRegistry(
        tracking_uri="http://mlflow-server:5000",
        s3_bucket="my-mlflow-artifacts"
    )
    
    lifecycle_manager = ModelLifecycleManager(registry)
    
    # Example promotion criteria
    promotion_criteria = {
        "accuracy": 0.95,
        "precision": 0.90,
        "recall": 0.85
    }
    
    lifecycle_manager.automated_model_promotion("fraud_detection", promotion_criteria)
```

## Automated Testing for ML

### Comprehensive ML Testing Framework

```python
#!/usr/bin/env python3
# tests/test_ml_framework.py

import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib
import json
import warnings
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

@dataclass
class ModelTestConfig:
    """Configuration for model testing"""
    min_accuracy: float = 0.8
    min_precision: float = 0.7
    min_recall: float = 0.7
    max_prediction_time_ms: float = 100.0
    max_memory_usage_mb: float = 500.0
    required_features: List[str] = None
    categorical_features: List[str] = None
    numerical_features: List[str] = None

class MLTestFramework:
    """Comprehensive testing framework for ML models"""
    
    def __init__(self, config: ModelTestConfig):
        self.config = config
        self.test_results = {}
    
    def test_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Test data quality and integrity"""
        results = {
            'total_rows': len(data),
            'total_columns': len(data.columns),
            'missing_values': {},
            'data_types': {},
            'duplicates': 0,
            'outliers': {},
            'passed': True,
            'issues': []
        }
        
        # Check for missing values
        missing = data.isnull().sum()
        results['missing_values'] = missing.to_dict()
        
        if missing.sum() > 0:
            results['issues'].append(f"Found {missing.sum()} missing values")
        
        # Check data types
        results['data_types'] = data.dtypes.to_dict()
        
        # Check for duplicates
        duplicates = data.duplicated().sum()
        results['duplicates'] = duplicates
        
        if duplicates > 0:
            results['issues'].append(f"Found {duplicates} duplicate rows")
        
        # Check for outliers (using IQR method for numerical columns)
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((data[col] < (Q1 - 1.5 * IQR)) | 
                       (data[col] > (Q3 + 1.5 * IQR))).sum()
            results['outliers'][col] = outliers
        
        # Check required features
        if self.config.required_features:
            missing_features = set(self.config.required_features) - set(data.columns)
            if missing_features:
                results['passed'] = False
                results['issues'].append(f"Missing required features: {missing_features}")
        
        results['passed'] = len(results['issues']) == 0
        return results
    
    def test_model_performance(self, model, X_test: pd.DataFrame, 
                             y_test: pd.Series) -> Dict[str, Any]:
        """Test model performance metrics"""
        results = {
            'metrics': {},
            'passed': True,
            'issues': []
        }
        
        # Make predictions
        try:
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            
            results['metrics'] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall
            }
            
            # Check against thresholds
            if accuracy < self.config.min_accuracy:
                results['passed'] = False
                results['issues'].append(f"Accuracy {accuracy:.3f} below threshold {self.config.min_accuracy}")
            
            if precision < self.config.min_precision:
                results['passed'] = False
                results['issues'].append(f"Precision {precision:.3f} below threshold {self.config.min_precision}")
            
            if recall < self.config.min_recall:
                results['passed'] = False
                results['issues'].append(f"Recall {recall:.3f} below threshold {self.config.min_recall}")
                
        except Exception as e:
            results['passed'] = False
            results['issues'].append(f"Error during prediction: {str(e)}")
        
        return results
    
    def test_model_robustness(self, model, X_test: pd.DataFrame) -> Dict[str, Any]:
        """Test model robustness and stability"""
        results = {
            'noise_sensitivity': {},
            'missing_value_handling': {},
            'prediction_consistency': {},
            'passed': True,
            'issues': []
        }
        
        try:
            # Test noise sensitivity
            base_predictions = model.predict(X_test)
            
            # Add small amount of noise
            numerical_cols = X_test.select_dtypes(include=[np.number]).columns
            X_noisy = X_test.copy()
            for col in numerical_cols:
                noise = np.random.normal(0, 0.01 * X_test[col].std(), len(X_test))
                X_noisy[col] += noise
            
            noisy_predictions = model.predict(X_noisy)
            
            # Calculate prediction stability
            if hasattr(model, 'predict_proba'):
                base_proba = model.predict_proba(X_test)
                noisy_proba = model.predict_proba(X_noisy)
                proba_diff = np.mean(np.abs(base_proba - noisy_proba))
                results['noise_sensitivity']['probability_change'] = proba_diff
            
            prediction_change_rate = np.mean(base_predictions != noisy_predictions)
            results['noise_sensitivity']['prediction_change_rate'] = prediction_change_rate
            
            if prediction_change_rate > 0.1:  # More than 10% change
                results['issues'].append(f"Model too sensitive to noise: {prediction_change_rate:.3f} change rate")
            
            # Test missing value handling
            X_missing = X_test.copy()
            for col in X_test.columns[:min(3, len(X_test.columns))]:  # Test first 3 columns
                X_temp = X_missing.copy()
                X_temp[col] = np.nan
                
                try:
                    missing_predictions = model.predict(X_temp)
                    results['missing_value_handling'][col] = 'handled'
                except Exception as e:
                    results['missing_value_handling'][col] = f'error: {str(e)}'
                    results['issues'].append(f"Model cannot handle missing values in {col}")
            
            # Test prediction consistency (multiple runs should give same result)
            consistency_predictions = []
            for i in range(5):
                pred = model.predict(X_test.head(100))  # Test on subset for speed
                consistency_predictions.append(pred)
            
            # Check if all predictions are identical
            consistent = all(np.array_equal(consistency_predictions[0], pred) 
                           for pred in consistency_predictions[1:])
            results['prediction_consistency']['is_consistent'] = consistent
            
            if not consistent:
                results['issues'].append("Model predictions are not consistent across runs")
                
        except Exception as e:
            results['passed'] = False
            results['issues'].append(f"Error during robustness testing: {str(e)}")
        
        results['passed'] = len(results['issues']) == 0
        return results
    
    def test_model_fairness(self, model, X_test: pd.DataFrame, 
                          y_test: pd.Series, sensitive_features: List[str]) -> Dict[str, Any]:
        """Test model fairness across different groups"""
        results = {
            'fairness_metrics': {},
            'passed': True,
            'issues': []
        }
        
        if not sensitive_features:
            results['issues'].append("No sensitive features provided for fairness testing")
            return results
        
        try:
            y_pred = model.predict(X_test)
            
            for feature in sensitive_features:
                if feature not in X_test.columns:
                    results['issues'].append(f"Sensitive feature {feature} not found in test data")
                    continue
                
                feature_metrics = {}
                unique_values = X_test[feature].unique()
                
                for value in unique_values:
                    mask = X_test[feature] == value
                    if mask.sum() > 0:  # Ensure group has samples
                        group_accuracy = accuracy_score(y_test[mask], y_pred[mask])
                        feature_metrics[str(value)] = {
                            'accuracy': group_accuracy,
                            'sample_count': mask.sum()
                        }
                
                # Calculate fairness metrics
                accuracies = [metrics['accuracy'] for metrics in feature_metrics.values()]
                if len(accuracies) > 1:
                    max_accuracy = max(accuracies)
                    min_accuracy = min(accuracies)
                    fairness_gap = max_accuracy - min_accuracy
                    
                    feature_metrics['fairness_gap'] = fairness_gap
                    
                    # Flag if fairness gap is too large
                    if fairness_gap > 0.1:  # 10% threshold
                        results['issues'].append(
                            f"Large fairness gap for {feature}: {fairness_gap:.3f}"
                        )
                
                results['fairness_metrics'][feature] = feature_metrics
                
        except Exception as e:
            results['passed'] = False
            results['issues'].append(f"Error during fairness testing: {str(e)}")
        
        results['passed'] = len(results['issues']) == 0
        return results
    
    def test_model_performance_characteristics(self, model, X_test: pd.DataFrame) -> Dict[str, Any]:
        """Test model performance characteristics (speed, memory)"""
        results = {
            'prediction_time': {},
            'memory_usage': {},
            'passed': True,
            'issues': []
        }
        
        import time
        import psutil
        import os
        
        try:
            # Test prediction time
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Warm-up run
            _ = model.predict(X_test.head(10))
            
            # Measure prediction time
            start_time = time.time()
            predictions = model.predict(X_test)
            end_time = time.time()
            
            prediction_time_ms = (end_time - start_time) * 1000
            per_sample_time_ms = prediction_time_ms / len(X_test)
            
            results['prediction_time'] = {
                'total_time_ms': prediction_time_ms,
                'per_sample_time_ms': per_sample_time_ms,
                'samples_per_second': len(X_test) / (prediction_time_ms / 1000)
            }
            
            # Check against threshold
            if per_sample_time_ms > self.config.max_prediction_time_ms:
                results['issues'].append(
                    f"Prediction time {per_sample_time_ms:.2f}ms exceeds threshold {self.config.max_prediction_time_ms}ms"
                )
            
            # Measure memory usage
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            results['memory_usage'] = {
                'initial_memory_mb': initial_memory,
                'final_memory_mb': final_memory,
                'memory_increase_mb': memory_increase
            }
            
            if memory_increase > self.config.max_memory_usage_mb:
                results['issues'].append(
                    f"Memory usage increase {memory_increase:.2f}MB exceeds threshold {self.config.max_memory_usage_mb}MB"
                )
                
        except Exception as e:
            results['passed'] = False
            results['issues'].append(f"Error during performance testing: {str(e)}")
        
        results['passed'] = len(results['issues']) == 0
        return results
    
    def run_all_tests(self, model, X_test: pd.DataFrame, y_test: pd.Series, 
                     sensitive_features: List[str] = None) -> Dict[str, Any]:
        """Run all tests and return comprehensive results"""
        all_results = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'model_type': type(model).__name__,
            'test_data_shape': X_test.shape,
            'tests': {}
        }
        
        # Data quality tests
        print("Running data quality tests...")
        all_results['tests']['data_quality'] = self.test_data_quality(X_test)
        
        # Model performance tests
        print("Running model performance tests...")
        all_results['tests']['performance'] = self.test_model_performance(model, X_test, y_test)
        
        # Model robustness tests
        print("Running model robustness tests...")
        all_results['tests']['robustness'] = self.test_model_robustness(model, X_test)
        
        # Model fairness tests
        if sensitive_features:
            print("Running model fairness tests...")
            all_results['tests']['fairness'] = self.test_model_fairness(
                model, X_test, y_test, sensitive_features
            )
        
        # Performance characteristics tests
        print("Running performance characteristics tests...")
        all_results['tests']['performance_characteristics'] = self.test_model_performance_characteristics(
            model, X_test
        )
        
        # Overall pass/fail
        all_results['overall_passed'] = all(
            test_result.get('passed', True) 
            for test_result in all_results['tests'].values()
        )
        
        # Collect all issues
        all_issues = []
        for test_name, test_result in all_results['tests'].items():
            if 'issues' in test_result:
                all_issues.extend([f"{test_name}: {issue}" for issue in test_result['issues']])
        
        all_results['all_issues'] = all_issues
        
        return all_results

# Pytest integration
class TestMLModel:
    """Pytest test class for ML models"""
    
    @pytest.fixture
    def model_and_data(self):
        """Fixture to load model and test data"""
        # Load your model and test data here
        model = joblib.load('models/trained_model.pkl')
        
        # Load test data
        test_data = pd.read_csv('data/test_data.csv')
        X_test = test_data.drop('target', axis=1)
        y_test = test_data['target']
        
        return model, X_test, y_test
    
    @pytest.fixture
    def test_config(self):
        """Test configuration fixture"""
        return ModelTestConfig(
            min_accuracy=0.8,
            min_precision=0.7,
            min_recall=0.7,
            max_prediction_time_ms=50.0,
            max_memory_usage_mb=200.0
        )
    
    def test_model_performance_meets_requirements(self, model_and_data, test_config):
        """Test that model meets performance requirements"""
        model, X_test, y_test = model_and_data
        
        framework = MLTestFramework(test_config)
        results = framework.test_model_performance(model, X_test, y_test)
        
        assert results['passed'], f"Performance test failed: {results['issues']}"
        assert results['metrics']['accuracy'] >= test_config.min_accuracy
        assert results['metrics']['precision'] >= test_config.min_precision
        assert results['metrics']['recall'] >= test_config.min_recall
    
    def test_model_robustness(self, model_and_data, test_config):
        """Test model robustness"""
        model, X_test, y_test = model_and_data
        
        framework = MLTestFramework(test_config)
        results = framework.test_model_robustness(model, X_test)
        
        assert results['passed'], f"Robustness test failed: {results['issues']}"
    
    def test_prediction_performance(self, model_and_data, test_config):
        """Test prediction speed and memory usage"""
        model, X_test, y_test = model_and_data
        
        framework = MLTestFramework(test_config)
        results = framework.test_model_performance_characteristics(model, X_test)
        
        assert results['passed'], f"Performance characteristics test failed: {results['issues']}"
        assert results['prediction_time']['per_sample_time_ms'] <= test_config.max_prediction_time_ms

# CLI for running tests
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run ML model tests')
    parser.add_argument('--model-path', required=True, help='Path to trained model')
    parser.add_argument('--test-data-path', required=True, help='Path to test data')
    parser.add_argument('--output-path', default='test_results.json', help='Output path for results')
    parser.add_argument('--config-path', help='Path to test configuration JSON')
    
    args = parser.parse_args()
    
    # Load model
    model = joblib.load(args.model_path)
    
    # Load test data
    test_data = pd.read_csv(args.test_data_path)
    X_test = test_data.drop('target', axis=1)
    y_test = test_data['target']
    
    # Load or create config
    if args.config_path:
        with open(args.config_path, 'r') as f:
            config_dict = json.load(f)
        config = ModelTestConfig(**config_dict)
    else:
        config = ModelTestConfig()
    
    # Run tests
    framework = MLTestFramework(config)
    results = framework.run_all_tests(model, X_test, y_test)
    
    # Save results
    with open(args.output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Test results saved to {args.output_path}")
    print(f"Overall passed: {results['overall_passed']}")
    
    if not results['overall_passed']:
        print("Issues found:")
        for issue in results['all_issues']:
            print(f"  - {issue}")
        exit(1)
    else:
        print("All tests passed!")
```

This comprehensive MLOps guide provides enterprise-ready tools and practices for implementing DevOps in data science projects. The content focuses on practical, immediately implementable solutions that scale from small teams to large organizations. 
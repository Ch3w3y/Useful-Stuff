# ML Model Serving with Docker

## Overview

Comprehensive guide to deploying and serving machine learning models using Docker containers, covering TensorFlow Serving, PyTorch serving, MLflow, batch processing, and production ML pipelines.

## Table of Contents

- [TensorFlow Serving](#tensorflow-serving)
- [PyTorch Serving](#pytorch-serving)
- [MLflow Model Serving](#mlflow-model-serving)
- [Custom Model APIs](#custom-model-apis)
- [Batch Processing](#batch-processing)
- [Model Management](#model-management)
- [Monitoring & Observability](#monitoring--observability)
- [Auto-scaling & Load Balancing](#auto-scaling--load-balancing)

## TensorFlow Serving

### TensorFlow Serving Deployment

```yaml
# docker-compose.tf-serving.yml
version: '3.8'

services:
  tensorflow-serving:
    image: tensorflow/serving:latest
    container_name: tf-serving
    restart: unless-stopped
    environment:
      MODEL_NAME: ${MODEL_NAME:-my_model}
      MODEL_BASE_PATH: /models
    volumes:
      - ./models:/models:ro
      - ./tf-serving/config:/config:ro
    ports:
      - "8501:8501"  # REST API
      - "8500:8500"  # gRPC API
    networks:
      - ml-network
    command: >
      tensorflow_model_server
      --port=8500
      --rest_api_port=8501
      --model_config_file=/config/models.config
      --monitoring_config_file=/config/monitoring.config
      --allow_version_labels_for_unavailable_models=true
      --file_system_poll_wait_seconds=1
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/v1/models/${MODEL_NAME:-my_model}"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  tf-serving-proxy:
    image: nginx:alpine
    container_name: tf-serving-proxy
    restart: unless-stopped
    volumes:
      - ./nginx/tf-serving.conf:/etc/nginx/conf.d/default.conf:ro
    ports:
      - "80:80"
    networks:
      - ml-network
    depends_on:
      tensorflow-serving:
        condition: service_healthy

networks:
  ml-network:
    driver: bridge
```

### TensorFlow Serving Configuration

```protobuf
# tf-serving/config/models.config
model_config_list {
  config {
    name: 'my_model'
    base_path: '/models/my_model'
    model_platform: 'tensorflow'
    model_version_policy {
      latest {
        num_versions: 2
      }
    }
    version_labels {
      key: 'stable'
      value: 1
    }
    version_labels {
      key: 'canary'
      value: 2
    }
  }
  config {
    name: 'text_classifier'
    base_path: '/models/text_classifier'
    model_platform: 'tensorflow'
    model_version_policy {
      specific {
        versions: 1
        versions: 2
        versions: 3
      }
    }
  }
}
```

```yaml
# tf-serving/config/monitoring.config
prometheus_config {
  enable: true
  path: "/monitoring/prometheus/metrics"
}
```

### Model Preparation Script

```python
#!/usr/bin/env python3
# scripts/prepare_tf_model.py

import tensorflow as tf
import numpy as np
import os
import shutil
from datetime import datetime

def create_sample_model():
    """Create a sample TensorFlow model for demonstration"""
    # Simple sequential model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Generate sample data
    X = np.random.random((1000, 10))
    y = np.random.randint(2, size=(1000, 1))
    
    # Train the model
    model.fit(X, y, epochs=5, batch_size=32, verbose=1)
    
    return model

def export_model_for_serving(model, model_name, version=1):
    """Export model in TensorFlow Serving format"""
    export_path = f"./models/{model_name}/{version}"
    
    # Remove existing version if it exists
    if os.path.exists(export_path):
        shutil.rmtree(export_path)
    
    # Export the model
    tf.saved_model.save(model, export_path)
    
    print(f"Model exported to {export_path}")
    
    # Create signature for serving
    @tf.function
    def predict(input_data):
        return model(input_data)
    
    # Test the exported model
    imported = tf.saved_model.load(export_path)
    test_input = tf.constant(np.random.random((1, 10)), dtype=tf.float32)
    prediction = imported.signatures['serving_default'](test_input)
    print(f"Test prediction: {prediction}")

def create_model_metadata(model_name, version, description=""):
    """Create metadata file for the model"""
    metadata = {
        "name": model_name,
        "version": version,
        "description": description,
        "created_at": datetime.now().isoformat(),
        "input_shape": [None, 10],
        "output_shape": [None, 1],
        "model_type": "classification"
    }
    
    metadata_path = f"./models/{model_name}/{version}/metadata.json"
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    model_name = "my_model"
    version = 1
    
    # Create and export model
    model = create_sample_model()
    export_model_for_serving(model, model_name, version)
    create_model_metadata(model_name, version, "Sample binary classification model")
    
    print(f"Model {model_name} v{version} is ready for serving!")
```

## PyTorch Serving

### TorchServe Deployment

```yaml
# docker-compose.torchserve.yml
version: '3.8'

services:
  torchserve:
    image: pytorch/torchserve:latest
    container_name: torchserve
    restart: unless-stopped
    environment:
      TS_CONFIG_FILE: /home/model-server/config.properties
    volumes:
      - ./pytorch-models:/home/model-server/model-store
      - ./torchserve/config:/home/model-server/config:ro
      - ./torchserve/logs:/home/model-server/logs
    ports:
      - "8080:8080"  # Inference API
      - "8081:8081"  # Management API
      - "8082:8082"  # Metrics API
    networks:
      - ml-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/ping"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

networks:
  ml-network:
    driver: bridge
```

### TorchServe Configuration

```properties
# torchserve/config/config.properties
inference_address=http://0.0.0.0:8080
management_address=http://0.0.0.0:8081
metrics_address=http://0.0.0.0:8082

number_of_netty_threads=4
job_queue_size=10
number_of_gpu=0
model_store=/home/model-server/model-store
model_snapshot={"name":"startup.cfg","modelCount":1,"models":{"my_pytorch_model":{"1.0":{"defaultVersion":true,"marName":"my_pytorch_model.mar","minWorkers":1,"maxWorkers":3,"batchSize":1,"maxBatchDelay":5000,"responseTimeout":120}}}}

enable_envvars_config=true
install_py_dep_per_model=true
enable_metrics_api=true
metrics_format=prometheus
number_of_netty_threads=4
netty_client_threads=0
default_workers_per_model=1
blacklist_env_vars=
default_response_timeout=120
unregister_model_timeout=120
decode_input_request=true
```

### PyTorch Model Handler

```python
# torchserve/handlers/custom_handler.py

import torch
import torch.nn as nn
import numpy as np
import json
import logging
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)

class CustomModelHandler(BaseHandler):
    """
    Custom handler for PyTorch models
    """
    
    def __init__(self):
        super(CustomModelHandler, self).__init__()
        self.model = None
        self.device = None
        self.initialized = False
    
    def initialize(self, context):
        """Initialize the model"""
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        model_file = context.manifest['model']['serializedFile']
        model_path = f"{model_dir}/{model_file}"
        
        if model_path.endswith('.pt') or model_path.endswith('.pth'):
            self.model = torch.jit.load(model_path, map_location=self.device)
        else:
            # Load state dict
            self.model = self._load_model_from_state_dict(model_path)
        
        self.model.eval()
        self.initialized = True
        logger.info(f"Model loaded successfully on {self.device}")
    
    def _load_model_from_state_dict(self, model_path):
        """Load model from state dict"""
        # Define your model architecture here
        class SimpleModel(nn.Module):
            def __init__(self):
                super(SimpleModel, self).__init__()
                self.fc1 = nn.Linear(10, 64)
                self.fc2 = nn.Linear(64, 32)
                self.fc3 = nn.Linear(32, 1)
                self.dropout = nn.Dropout(0.2)
                self.relu = nn.ReLU()
                self.sigmoid = nn.Sigmoid()
            
            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.relu(self.fc2(x))
                x = self.sigmoid(self.fc3(x))
                return x
        
        model = SimpleModel()
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        return model.to(self.device)
    
    def preprocess(self, data):
        """Preprocess input data"""
        preprocessed_data = []
        
        for row in data:
            # Extract input from request
            input_data = row.get("body")
            if input_data is None:
                input_data = row.get("data")
            
            if isinstance(input_data, (bytes, bytearray)):
                input_data = input_data.decode('utf-8')
            
            if isinstance(input_data, str):
                input_data = json.loads(input_data)
            
            # Convert to tensor
            if isinstance(input_data, dict):
                features = input_data.get("instances", input_data.get("features", input_data))
            else:
                features = input_data
            
            tensor = torch.FloatTensor(features).to(self.device)
            preprocessed_data.append(tensor)
        
        return torch.stack(preprocessed_data)
    
    def inference(self, data):
        """Run inference"""
        with torch.no_grad():
            predictions = self.model(data)
        return predictions
    
    def postprocess(self, data):
        """Postprocess output"""
        predictions = data.cpu().numpy().tolist()
        
        # Format output
        output = []
        for prediction in predictions:
            if isinstance(prediction, list) and len(prediction) == 1:
                prediction = prediction[0]
            
            output.append({
                "prediction": prediction,
                "probability": float(prediction) if isinstance(prediction, (int, float)) else prediction
            })
        
        return output

# Custom model class for demonstration
class SimpleClassifier(nn.Module):
    def __init__(self, input_size=10, hidden_size=64, output_size=1):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 32)
        self.fc3 = nn.Linear(32, output_size)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x
```

### Model Archive Creation Script

```bash
#!/bin/bash
# scripts/create_torch_archive.sh

set -euo pipefail

MODEL_NAME="my_pytorch_model"
MODEL_VERSION="1.0"
HANDLER_FILE="torchserve/handlers/custom_handler.py"
MODEL_FILE="pytorch-models/model.pth"
REQUIREMENTS_FILE="torchserve/requirements.txt"

# Create model archive
torch-model-archiver \
    --model-name $MODEL_NAME \
    --version $MODEL_VERSION \
    --model-file $MODEL_FILE \
    --handler $HANDLER_FILE \
    --requirements-file $REQUIREMENTS_FILE \
    --export-path ./pytorch-models/ \
    --force

echo "Model archive created: ${MODEL_NAME}.mar"

# Register model with TorchServe (if running)
if curl -s http://localhost:8081/ping > /dev/null; then
    echo "Registering model with TorchServe..."
    curl -X POST "http://localhost:8081/models?url=${MODEL_NAME}.mar&initial_workers=1&synchronous=true"
    echo "Model registered successfully!"
else
    echo "TorchServe not running. Start TorchServe to register the model."
fi
```

## MLflow Model Serving

### MLflow Deployment

```yaml
# docker-compose.mlflow.yml
version: '3.8'

services:
  mlflow-server:
    image: python:3.9-slim
    container_name: mlflow-server
    restart: unless-stopped
    environment:
      MLFLOW_BACKEND_STORE_URI: postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}
      MLFLOW_DEFAULT_ARTIFACT_ROOT: s3://${S3_BUCKET}/mlflow-artifacts
      AWS_ACCESS_KEY_ID: ${AWS_ACCESS_KEY_ID}
      AWS_SECRET_ACCESS_KEY: ${AWS_SECRET_ACCESS_KEY}
    volumes:
      - ./mlflow:/app
    ports:
      - "5000:5000"
    networks:
      - ml-network
    command: >
      sh -c "
        pip install mlflow[extras] psycopg2-binary boto3 &&
        mlflow server
        --backend-store-uri postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}
        --default-artifact-root s3://${S3_BUCKET}/mlflow-artifacts
        --host 0.0.0.0
        --port 5000
      "
    depends_on:
      - postgres
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  mlflow-model-server:
    image: python:3.9-slim
    container_name: mlflow-model-server
    restart: unless-stopped
    environment:
      MLFLOW_TRACKING_URI: http://mlflow-server:5000
      MODEL_URI: ${MODEL_URI:-models:/my_model/latest}
    ports:
      - "5001:5001"
    networks:
      - ml-network
    command: >
      sh -c "
        pip install mlflow[extras] &&
        mlflow models serve
        --model-uri ${MODEL_URI:-models:/my_model/latest}
        --host 0.0.0.0
        --port 5001
        --no-conda
      "
    depends_on:
      mlflow-server:
        condition: service_healthy

networks:
  ml-network:
    driver: bridge
```

### MLflow Model Registration Script

```python
#!/usr/bin/env python3
# scripts/register_mlflow_model.py

import mlflow
import mlflow.sklearn
import mlflow.tensorflow
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

def train_and_register_sklearn_model():
    """Train and register a scikit-learn model"""
    
    # Generate sample data
    np.random.seed(42)
    X = np.random.random((1000, 10))
    y = np.random.randint(2, size=1000)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Start MLflow run
    with mlflow.start_run(run_name="sklearn_random_forest"):
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Log parameters and metrics
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", 42)
        mlflow.log_metric("accuracy", accuracy)
        
        # Log model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="sklearn_classifier",
            input_example=X_train[:5],
            signature=mlflow.models.infer_signature(X_train, y_pred)
        )
        
        print(f"Model accuracy: {accuracy:.4f}")
        print(f"Model registered as 'sklearn_classifier'")

def register_model_version(model_name, model_uri, version_description=""):
    """Register a new version of an existing model"""
    
    client = mlflow.tracking.MlflowClient()
    
    # Create or get model
    try:
        model = client.create_registered_model(model_name)
        print(f"Created new registered model: {model_name}")
    except mlflow.exceptions.RestException:
        model = client.get_registered_model(model_name)
        print(f"Found existing registered model: {model_name}")
    
    # Create new model version
    model_version = client.create_model_version(
        name=model_name,
        source=model_uri,
        description=version_description
    )
    
    print(f"Created model version {model_version.version} for {model_name}")
    return model_version

def promote_model_to_production(model_name, version):
    """Promote a model version to production"""
    
    client = mlflow.tracking.MlflowClient()
    
    # Transition to production
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Production",
        archive_existing_versions=True
    )
    
    print(f"Model {model_name} version {version} promoted to Production")

def create_model_deployment_config(model_name, stage="Production"):
    """Create deployment configuration for model serving"""
    
    config = {
        "model_name": model_name,
        "model_stage": stage,
        "serving_config": {
            "port": 5001,
            "host": "0.0.0.0",
            "workers": 4,
            "timeout": 60
        },
        "resource_requirements": {
            "cpu": "500m",
            "memory": "1Gi"
        },
        "health_check": {
            "endpoint": "/health",
            "interval": 30,
            "timeout": 10
        }
    }
    
    config_path = f"./mlflow/configs/{model_name}_deployment.json"
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    import json
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Deployment config saved to {config_path}")

if __name__ == "__main__":
    # Train and register model
    train_and_register_sklearn_model()
    
    # Create deployment config
    create_model_deployment_config("sklearn_classifier")
    
    print("MLflow model registration completed!")
```

## Custom Model APIs

### FastAPI Model Serving

```python
# ml-api/main.py

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
import numpy as np
import joblib
import asyncio
import logging
from typing import List, Dict, Any, Optional
import time
from datetime import datetime
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ML Model Serving API",
    description="Production-ready ML model serving with FastAPI",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class PredictionRequest(BaseModel):
    features: List[float]
    model_name: Optional[str] = "default"
    
    @validator('features')
    def validate_features(cls, v):
        if len(v) != 10:  # Expected feature count
            raise ValueError('Features must contain exactly 10 values')
        return v

class BatchPredictionRequest(BaseModel):
    instances: List[List[float]]
    model_name: Optional[str] = "default"

class PredictionResponse(BaseModel):
    prediction: float
    probability: float
    model_name: str
    timestamp: datetime
    processing_time_ms: float

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    batch_size: int
    total_processing_time_ms: float

class HealthResponse(BaseModel):
    status: str
    models_loaded: List[str]
    uptime_seconds: float

# Global variables
models = {}
model_metadata = {}
start_time = time.time()

class ModelManager:
    """Manage model loading and prediction"""
    
    def __init__(self):
        self.models = {}
        self.metadata = {}
    
    async def load_model(self, model_name: str, model_path: str):
        """Load a model asynchronously"""
        try:
            logger.info(f"Loading model {model_name} from {model_path}")
            
            # Simulate async loading
            await asyncio.sleep(0.1)
            
            if model_path.endswith('.joblib') or model_path.endswith('.pkl'):
                model = joblib.load(model_path)
            else:
                # Mock model for demonstration
                model = MockModel()
            
            self.models[model_name] = model
            self.metadata[model_name] = {
                "loaded_at": datetime.now(),
                "model_path": model_path,
                "version": "1.0.0"
            }
            
            logger.info(f"Model {model_name} loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            return False
    
    def get_model(self, model_name: str):
        """Get a loaded model"""
        return self.models.get(model_name)
    
    def list_models(self):
        """List all loaded models"""
        return list(self.models.keys())

class MockModel:
    """Mock model for demonstration"""
    
    def predict(self, X):
        """Mock prediction"""
        if isinstance(X, list):
            X = np.array(X)
        return np.random.random(X.shape[0])
    
    def predict_proba(self, X):
        """Mock probability prediction"""
        if isinstance(X, list):
            X = np.array(X)
        probs = np.random.random((X.shape[0], 2))
        return probs / probs.sum(axis=1, keepdims=True)

# Initialize model manager
model_manager = ModelManager()

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    # Load default model
    await model_manager.load_model("default", "models/default_model.joblib")
    logger.info("Application startup completed")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    uptime = time.time() - start_time
    
    return HealthResponse(
        status="healthy",
        models_loaded=model_manager.list_models(),
        uptime_seconds=uptime
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Single prediction endpoint"""
    start_time_ms = time.time() * 1000
    
    # Get model
    model = model_manager.get_model(request.model_name)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model {request.model_name} not found")
    
    try:
        # Make prediction
        features = np.array([request.features])
        prediction = model.predict(features)[0]
        
        # Get probability if available
        try:
            probabilities = model.predict_proba(features)[0]
            probability = float(probabilities[1]) if len(probabilities) > 1 else float(prediction)
        except:
            probability = float(prediction)
        
        processing_time = (time.time() * 1000) - start_time_ms
        
        return PredictionResponse(
            prediction=float(prediction),
            probability=probability,
            model_name=request.model_name,
            timestamp=datetime.now(),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/batch-predict", response_model=BatchPredictionResponse)
async def batch_predict(request: BatchPredictionRequest):
    """Batch prediction endpoint"""
    start_time_ms = time.time() * 1000
    
    # Get model
    model = model_manager.get_model(request.model_name)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model {request.model_name} not found")
    
    try:
        # Make batch predictions
        features = np.array(request.instances)
        predictions = model.predict(features)
        
        # Get probabilities if available
        try:
            probabilities = model.predict_proba(features)
        except:
            probabilities = predictions.reshape(-1, 1)
        
        # Format responses
        prediction_responses = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            prob_value = float(prob[1]) if len(prob) > 1 else float(pred)
            
            prediction_responses.append(PredictionResponse(
                prediction=float(pred),
                probability=prob_value,
                model_name=request.model_name,
                timestamp=datetime.now(),
                processing_time_ms=0  # Individual time not calculated for batch
            ))
        
        total_processing_time = (time.time() * 1000) - start_time_ms
        
        return BatchPredictionResponse(
            predictions=prediction_responses,
            batch_size=len(request.instances),
            total_processing_time_ms=total_processing_time
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/models")
async def list_models():
    """List available models"""
    models_info = {}
    for model_name in model_manager.list_models():
        metadata = model_manager.metadata.get(model_name, {})
        models_info[model_name] = {
            "name": model_name,
            "loaded_at": metadata.get("loaded_at"),
            "version": metadata.get("version"),
            "status": "ready"
        }
    
    return {"models": models_info}

@app.post("/models/{model_name}/load")
async def load_model(model_name: str, model_path: str, background_tasks: BackgroundTasks):
    """Load a new model"""
    background_tasks.add_task(model_manager.load_model, model_name, model_path)
    return {"message": f"Loading model {model_name} in background"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Docker Configuration for Custom API

```dockerfile
# ml-api/Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories
RUN mkdir -p models logs

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

```yaml
# docker-compose.custom-api.yml
version: '3.8'

services:
  ml-api:
    build:
      context: ./ml-api
      dockerfile: Dockerfile
    container_name: ml-api
    restart: unless-stopped
    environment:
      MODEL_PATH: /app/models
      LOG_LEVEL: INFO
    volumes:
      - ./models:/app/models:ro
      - ./ml-api/logs:/app/logs
    ports:
      - "8000:8000"
    networks:
      - ml-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  nginx:
    image: nginx:alpine
    container_name: ml-api-nginx
    restart: unless-stopped
    volumes:
      - ./nginx/ml-api.conf:/etc/nginx/conf.d/default.conf:ro
    ports:
      - "80:80"
    networks:
      - ml-network
    depends_on:
      ml-api:
        condition: service_healthy

networks:
  ml-network:
    driver: bridge
```

---

*This comprehensive ML model serving guide provides production-ready containerized deployment solutions for TensorFlow, PyTorch, MLflow, and custom model APIs with monitoring and scaling capabilities.* 
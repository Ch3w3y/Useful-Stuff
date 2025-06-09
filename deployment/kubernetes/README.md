# Kubernetes for Data Science and Machine Learning

This directory contains resources for deploying and managing ML/AI workloads on Kubernetes, from basic containerization to advanced inference serving patterns.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Core Concepts](#core-concepts)
- [Deployment Patterns](#deployment-patterns)
- [ML-Specific Features](#ml-specific-features)
- [Production Considerations](#production-considerations)
- [Examples](#examples)

## Overview

Kubernetes has become the de facto standard for orchestrating machine learning workloads at scale. This guide covers deployment patterns specifically optimized for data science and ML use cases.

### Why Kubernetes for ML?

- **Scalability**: Auto-scaling for training and inference
- **Resource Management**: GPU/CPU allocation and scheduling
- **Fault Tolerance**: Self-healing and high availability
- **Portability**: Deploy anywhere - cloud, on-premises, edge
- **Multi-tenancy**: Isolated workspaces for different teams

## Prerequisites

- Basic Kubernetes knowledge
- Docker containerization experience
- Understanding of ML workflows (training, inference, monitoring)

## Core Concepts

### Pods and Containers for ML
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: ml-training-pod
spec:
  containers:
  - name: training-container
    image: tensorflow/tensorflow:2.15.0-gpu
    resources:
      requests:
        memory: "8Gi"
        cpu: "4"
        nvidia.com/gpu: 1
      limits:
        memory: "16Gi"
        cpu: "8"
        nvidia.com/gpu: 1
    volumeMounts:
    - name: data-volume
      mountPath: /data
    - name: model-volume
      mountPath: /models
  volumes:
  - name: data-volume
    persistentVolumeClaim:
      claimName: training-data-pvc
  - name model-volume
    persistentVolumeClaim:
      claimName: model-storage-pvc
```

### Services for Model Serving
```yaml
apiVersion: v1
kind: Service
metadata:
  name: model-serving-service
spec:
  selector:
    app: model-server
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
  type: LoadBalancer
```

## Deployment Patterns

### 1. Batch Training Jobs
```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: model-training-job
spec:
  template:
    spec:
      containers:
      - name: trainer
        image: my-training-image:latest
        command: ["python", "train.py"]
        env:
        - name: EPOCHS
          value: "100"
        - name: BATCH_SIZE
          value: "32"
        resources:
          requests:
            nvidia.com/gpu: 2
          limits:
            nvidia.com/gpu: 2
      restartPolicy: Never
  backoffLimit: 3
```

### 2. Real-time Inference Serving
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: inference-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: inference-server
  template:
    metadata:
      labels:
        app: inference-server
    spec:
      containers:
      - name: server
        image: tensorflow/serving:2.15.0
        ports:
        - containerPort: 8501
        env:
        - name: MODEL_NAME
          value: "my_model"
        - name: MODEL_BASE_PATH
          value: "/models"
        volumeMounts:
        - name: model-storage
          mountPath: /models
        readinessProbe:
          httpGet:
            path: /v1/models/my_model
            port: 8501
          initialDelaySeconds: 30
          periodSeconds: 10
        livenessProbe:
          httpGet:
            path: /v1/models/my_model
            port: 8501
          initialDelaySeconds: 60
          periodSeconds: 30
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-pvc
```

### 3. Horizontal Pod Autoscaler
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: inference-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: inference-server
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## ML-Specific Features

### GPU Node Scheduling
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: gpu-workload
spec:
  nodeSelector:
    accelerator: nvidia-tesla-v100
  containers:
  - name: gpu-container
    image: pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel
    resources:
      limits:
        nvidia.com/gpu: 1
```

### Priority Classes for ML Workloads
```yaml
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: ml-training-priority
value: 1000
globalDefault: false
description: "Priority class for ML training jobs"
---
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: ml-inference-priority
value: 2000
globalDefault: false
description: "Priority class for ML inference serving"
```

### Resource Quotas
```yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: ml-team-quota
  namespace: ml-team
spec:
  hard:
    requests.cpu: "50"
    requests.memory: 200Gi
    requests.nvidia.com/gpu: "8"
    limits.cpu: "100"
    limits.memory: 400Gi
    limits.nvidia.com/gpu: "8"
    persistentvolumeclaims: "20"
```

## Production Considerations

### Security
- Use Pod Security Standards
- Implement RBAC for ML teams
- Secure model artifacts and data
- Network policies for isolation

### Monitoring
- Prometheus metrics for model performance
- Grafana dashboards for ML workloads
- Custom metrics for inference latency
- Resource utilization tracking

### Storage
- Use persistent volumes for model storage
- Configure storage classes for different performance needs
- Implement backup strategies for training data and models

### Networking
- Service mesh for advanced traffic management
- Ingress controllers for external access
- Load balancing strategies for inference

## Examples

See the following subdirectories for complete examples:
- `./training-jobs/` - Distributed training examples
- `./inference-serving/` - Model serving patterns
- `./mlops-pipelines/` - End-to-end ML pipelines
- `./monitoring/` - Observability setup
- `./security/` - Security configurations

## Best Practices

1. **Resource Management**
   - Set appropriate resource requests and limits
   - Use node affinity for GPU workloads
   - Implement resource quotas per team

2. **Scalability**
   - Design stateless inference services
   - Use horizontal pod autoscaling
   - Implement cluster autoscaling for dynamic workloads

3. **Reliability**
   - Use health checks and readiness probes
   - Implement retry strategies
   - Plan for graceful degradation

4. **Cost Optimization**
   - Use spot instances for training
   - Scale down idle resources
   - Optimize resource allocation

## Related Resources

- [Cloud Provider Integration](../cloud/)
- [Container Registry Setup](../docker/)
- [Infrastructure as Code](../infrastructure/)
- [MLOps Pipelines](../../devops/mlops-pipelines.md)

## Contributing

When adding new Kubernetes examples:
1. Include resource specifications
2. Add monitoring and health checks
3. Document any prerequisites
4. Provide cleanup instructions 
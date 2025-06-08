# PyTorch Deep Learning Framework

## Overview

Comprehensive guide to PyTorch for deep learning, neural networks, and AI model development. Covers fundamentals through advanced techniques with practical examples and production-ready patterns.

## Table of Contents

- [PyTorch Fundamentals](#pytorch-fundamentals)
- [Neural Network Architectures](#neural-network-architectures)
- [Training Workflows](#training-workflows)
- [Computer Vision](#computer-vision)
- [Natural Language Processing](#natural-language-processing)
- [Model Optimization](#model-optimization)
- [Production Deployment](#production-deployment)
- [Best Practices](#best-practices)

## PyTorch Fundamentals

### Tensor Operations

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Basic tensor operations
def tensor_basics():
    """Essential tensor operations in PyTorch"""
    
    # Create tensors
    x = torch.randn(3, 4)  # Random tensor
    y = torch.zeros(3, 4)  # Zero tensor
    z = torch.ones(3, 4)   # Ones tensor
    
    # Tensor operations
    result = torch.matmul(x, y.T)  # Matrix multiplication
    element_wise = x * y            # Element-wise multiplication
    
    # GPU operations
    if torch.cuda.is_available():
        device = torch.device('cuda')
        x_gpu = x.to(device)
        y_gpu = y.to(device)
        result_gpu = torch.matmul(x_gpu, y_gpu.T)
    
    # Automatic differentiation
    x.requires_grad_(True)
    loss = torch.sum(x ** 2)
    loss.backward()
    gradients = x.grad
    
    return {
        'tensor': x,
        'result': result,
        'gradients': gradients
    }

# Custom dataset class
class CustomDataset(Dataset):
    """Template for custom PyTorch datasets"""
    
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        target = self.targets[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, target

# Data loading utilities
def create_dataloader(dataset, batch_size=32, shuffle=True, num_workers=4):
    """Create optimized data loader"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
```

## Neural Network Architectures

### Basic Neural Networks

```python
class MLP(nn.Module):
    """Multi-Layer Perceptron"""
    
    def __init__(self, input_size, hidden_sizes, num_classes, dropout=0.2):
        super(MLP, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# Convolutional Neural Network
class CNN(nn.Module):
    """Convolutional Neural Network for image classification"""
    
    def __init__(self, num_classes=10, num_channels=3):
        super(CNN, self).__init__()
        
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(num_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Residual Block for ResNet
class ResidualBlock(nn.Module):
    """Residual block with skip connections"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
```

### Advanced Architectures

```python
# Transformer architecture
class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""
    
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention
        attention_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        return self.W_o(attention_output)
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)
        
        return attention_output

# LSTM for sequence modeling
class LSTMModel(nn.Module):
    """LSTM for sequence classification/regression"""
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.2):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout
        )
        
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward pass
        out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Use last time step output
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        
        return out
```

## Training Workflows

### Training Loop Template

```python
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt

class ModelTrainer:
    """Comprehensive model training class"""
    
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100. * correct / total:.2f}%'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(self, num_epochs, save_best=True, patience=10):
        """Complete training loop"""
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch+1}/{num_epochs}')
            print('-' * 50)
            
            # Training
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            # Validation
            val_loss, val_acc = self.validate()
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Save best model
            if save_best and val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_model.pth')
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f'Early stopping after {epoch+1} epochs')
                break
            
            # Log to wandb if available
            try:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc
                })
            except:
                pass
    
    def plot_metrics(self):
        """Plot training metrics"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        
        # Accuracy plot
        ax2.plot(self.train_accuracies, label='Train Acc')
        ax2.plot(self.val_accuracies, label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()

# Learning rate scheduling
def get_scheduler(optimizer, scheduler_type='cosine', **kwargs):
    """Get learning rate scheduler"""
    if scheduler_type == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
    elif scheduler_type == 'step':
        return optim.lr_scheduler.StepLR(optimizer, **kwargs)
    elif scheduler_type == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
```

## Computer Vision

### Image Classification

```python
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, ImageFolder
from PIL import Image

# Data augmentation for computer vision
def get_transforms(mode='train'):
    """Get image transformations"""
    if mode == 'train':
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

# Transfer learning with pretrained models
def create_transfer_model(model_name='resnet18', num_classes=10, pretrained=True):
    """Create transfer learning model"""
    import torchvision.models as models
    
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'vgg16':
        model = models.vgg16(pretrained=pretrained)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name == 'efficientnet':
        model = models.efficientnet_b0(pretrained=pretrained)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    return model

# Object detection utilities
class BoundingBoxUtils:
    """Utilities for bounding box operations"""
    
    @staticmethod
    def calculate_iou(box1, box2):
        """Calculate Intersection over Union"""
        x1, y1, x2, y2 = box1
        x1_p, y1_p, x2_p, y2_p = box2
        
        # Intersection coordinates
        xi1 = max(x1, x1_p)
        yi1 = max(y1, y1_p)
        xi2 = min(x2, x2_p)
        yi2 = min(y2, y2_p)
        
        # Intersection area
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # Union area
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2_p - x1_p) * (y2_p - y1_p)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    @staticmethod
    def non_max_suppression(boxes, scores, iou_threshold=0.5):
        """Non-maximum suppression"""
        indices = torch.argsort(scores, descending=True)
        keep = []
        
        while len(indices) > 0:
            current = indices[0]
            keep.append(current.item())
            
            if len(indices) == 1:
                break
            
            # Calculate IoU with remaining boxes
            ious = torch.tensor([
                BoundingBoxUtils.calculate_iou(boxes[current], boxes[idx])
                for idx in indices[1:]
            ])
            
            # Keep boxes with IoU below threshold
            indices = indices[1:][ious < iou_threshold]
        
        return keep
```

## Model Optimization

### Quantization and Pruning

```python
import torch.quantization as quantization
from torch.nn.utils import prune

class ModelOptimizer:
    """Model optimization utilities"""
    
    @staticmethod
    def quantize_model(model, example_input):
        """Post-training quantization"""
        model.eval()
        
        # Prepare model for quantization
        model_fused = torch.quantization.fuse_modules(model, [['conv', 'bn', 'relu']])
        model_fused.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        
        # Prepare and calibrate
        model_prepared = torch.quantization.prepare(model_fused)
        
        # Calibration (run some data through the model)
        with torch.no_grad():
            model_prepared(example_input)
        
        # Convert to quantized model
        quantized_model = torch.quantization.convert(model_prepared)
        
        return quantized_model
    
    @staticmethod
    def prune_model(model, pruning_amount=0.3):
        """Prune model weights"""
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                prune.l1_unstructured(module, name='weight', amount=pruning_amount)
                prune.remove(module, 'weight')
        
        return model
    
    @staticmethod
    def get_model_size(model):
        """Calculate model size in MB"""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb

# Mixed precision training
from torch.cuda.amp import autocast, GradScaler

class MixedPrecisionTrainer(ModelTrainer):
    """Training with automatic mixed precision"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scaler = GradScaler()
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for data, target in self.train_loader:
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with autocast
            with autocast():
                output = self.model(data)
                loss = self.criterion(output, target)
            
            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Statistics
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        return total_loss / len(self.train_loader), 100. * correct / total
```

## Production Deployment

### Model Serving

```python
import torch.jit as jit
from torch.utils.mobile_optimizer import optimize_for_mobile

class ModelDeployment:
    """Model deployment utilities"""
    
    @staticmethod
    def export_torchscript(model, example_input, save_path):
        """Export model to TorchScript"""
        model.eval()
        
        # Trace the model
        traced_model = jit.trace(model, example_input)
        
        # Save the traced model
        traced_model.save(save_path)
        
        return traced_model
    
    @staticmethod
    def export_onnx(model, example_input, save_path):
        """Export model to ONNX format"""
        model.eval()
        
        torch.onnx.export(
            model,
            example_input,
            save_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
    
    @staticmethod
    def optimize_for_mobile(model, example_input):
        """Optimize model for mobile deployment"""
        model.eval()
        traced_model = jit.trace(model, example_input)
        optimized_model = optimize_for_mobile(traced_model)
        
        return optimized_model

# FastAPI model serving
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io

app = FastAPI()

class ModelServer:
    """FastAPI model server"""
    
    def __init__(self, model_path):
        self.model = torch.jit.load(model_path)
        self.model.eval()
        self.transform = get_transforms('val')
    
    async def predict(self, file: UploadFile):
        """Predict from uploaded image"""
        # Read image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Preprocess
        input_tensor = self.transform(image).unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            predicted_class = output.argmax(dim=1).item()
        
        return {
            'predicted_class': predicted_class,
            'confidence': probabilities[0][predicted_class].item(),
            'all_probabilities': probabilities[0].tolist()
        }

# Initialize server
model_server = ModelServer('model.pt')

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    return await model_server.predict(file)
```

## Best Practices

### Training Tips

```python
# 1. Reproducibility
def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 2. Model initialization
def init_weights(m):
    """Initialize model weights"""
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

# 3. Gradient clipping
def clip_gradients(model, max_norm=1.0):
    """Clip gradients to prevent exploding gradients"""
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

# 4. Model checkpointing
class ModelCheckpoint:
    """Save and load model checkpoints"""
    
    @staticmethod
    def save_checkpoint(model, optimizer, epoch, loss, path):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }
        torch.save(checkpoint, path)
    
    @staticmethod
    def load_checkpoint(model, optimizer, path):
        """Load training checkpoint"""
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        
        return model, optimizer, epoch, loss

# 5. Memory optimization
def clear_cache():
    """Clear GPU cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# 6. Model profiling
def profile_model(model, input_tensor):
    """Profile model performance"""
    from torch.profiler import profile, record_function, ProfilerActivity
    
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with record_function("model_inference"):
            model(input_tensor)
    
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

---

*This comprehensive PyTorch guide provides everything needed for deep learning development from fundamentals to production deployment.* 
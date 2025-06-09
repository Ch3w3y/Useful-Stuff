# PyTorch Distributed Training

Comprehensive guide for scaling PyTorch models using modern distributed training techniques, from single-node multi-GPU to large-scale cluster training.

## Table of Contents

- [Overview](#overview)
- [Distribution Strategies](#distribution-strategies)
- [Single-Node Multi-GPU](#single-node-multi-gpu)
- [Multi-Node Training](#multi-node-training)
- [Advanced Techniques](#advanced-techniques)
- [Infrastructure Setup](#infrastructure-setup)
- [Performance Optimization](#performance-optimization)

## Overview

PyTorch distributed training has evolved significantly with new techniques for scaling deep learning models across multiple GPUs and nodes.

### Key Concepts

- **Data Parallelism**: Distribute data across devices
- **Model Parallelism**: Split model across devices
- **Pipeline Parallelism**: Stage-wise model execution
- **Tensor Parallelism**: Split tensors across devices
- **Gradient Accumulation**: Simulate larger batch sizes

## Distribution Strategies

### 1. DistributedDataParallel (DDP)

```python
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import os

class ResNetModel(nn.Module):
    """Example ResNet model for distributed training."""
    
    def __init__(self, num_classes=1000):
        super().__init__()
        self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)

def setup_distributed(rank, world_size):
    """Initialize distributed training environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group
    dist.init_process_group(
        backend='nccl',  # Use 'gloo' for CPU
        rank=rank,
        world_size=world_size
    )
    
    # Set device for current process
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """Clean up distributed training."""
    dist.destroy_process_group()

def train_distributed(rank, world_size, epochs=10, batch_size=32):
    """Main distributed training function."""
    
    # Setup distributed environment
    setup_distributed(rank, world_size)
    
    # Create model and move to GPU
    model = ResNetModel(num_classes=10)
    model = model.to(rank)
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[rank], output_device=rank)
    
    # Create dataset and sampler
    from torchvision import datasets, transforms
    
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    # Distributed sampler ensures each process gets different data
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )
    
    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        # Set epoch for sampler to ensure different shuffling
        sampler.set_epoch(epoch)
        
        epoch_loss = 0.0
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(rank), target.to(rank)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 100 == 0 and rank == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        # Synchronize metrics across processes
        avg_loss = epoch_loss / len(dataloader)
        if rank == 0:
            print(f'Epoch {epoch} completed, Average Loss: {avg_loss:.4f}')
    
    # Cleanup
    cleanup_distributed()

def launch_distributed_training():
    """Launch distributed training on multiple GPUs."""
    world_size = torch.cuda.device_count()
    print(f"Launching distributed training on {world_size} GPUs")
    
    mp.spawn(
        train_distributed,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    launch_distributed_training()
```

### 2. Fully Sharded Data Parallel (FSDP)

```python
import torch
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)

class TransformerModel(nn.Module):
    """Large transformer model for FSDP training."""
    
    def __init__(self, vocab_size=50000, hidden_size=4096, num_layers=24, num_heads=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # Create transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(hidden_size, num_heads)
            for _ in range(num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.ln_f(x)
        return self.head(x)

class TransformerLayer(nn.Module):
    """Individual transformer layer."""
    
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
    
    def forward(self, x):
        # Self-attention with residual connection
        attn_out, _ = self.attention(x, x, x)
        x = self.ln1(x + attn_out)
        
        # MLP with residual connection
        mlp_out = self.mlp(x)
        x = self.ln2(x + mlp_out)
        
        return x

def setup_fsdp_model(model, rank):
    """Setup FSDP with optimized configuration."""
    
    # Auto-wrap policy based on parameter count
    auto_wrap_policy = size_based_auto_wrap_policy(
        min_num_params=1e6  # Wrap modules with >1M parameters
    )
    
    # FSDP configuration
    fsdp_model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        cpu_offload=CPUOffload(offload_params=True),
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        mixed_precision=torch.distributed.fsdp.MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        ),
        device_id=rank,
    )
    
    return fsdp_model

def train_with_fsdp(rank, world_size):
    """Training with FSDP for large models."""
    
    setup_distributed(rank, world_size)
    
    # Create large model
    model = TransformerModel(
        vocab_size=50000,
        hidden_size=4096,
        num_layers=24,
        num_heads=32
    )
    
    # Setup FSDP
    model = setup_fsdp_model(model, rank)
    
    # Optimizer with FSDP-compatible settings
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.1
    )
    
    # Training loop
    model.train()
    for epoch in range(10):
        for batch_idx in range(100):  # Mock training steps
            # Generate random data
            input_ids = torch.randint(0, 50000, (8, 512)).to(rank)
            targets = torch.randint(0, 50000, (8, 512)).to(rank)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(input_ids)
            loss = nn.CrossEntropyLoss()(
                outputs.view(-1, outputs.size(-1)),
                targets.view(-1)
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            if batch_idx % 10 == 0 and rank == 0:
                print(f'Epoch {epoch}, Step {batch_idx}, Loss: {loss.item():.4f}')
    
    cleanup_distributed()
```

## Single-Node Multi-GPU

### torchrun Integration

```python
# train_multigpu.py
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import os

def main():
    """Main training function for torchrun."""
    
    # Get distributed training info from environment
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    
    # Initialize distributed training
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    
    print(f"Rank {rank}/{world_size}, Local Rank: {local_rank}")
    
    # Your training code here
    model = YourModel().to(local_rank)
    model = DDP(model, device_ids=[local_rank])
    
    # Training loop
    train_model(model, rank, world_size, local_rank)
    
    # Cleanup
    dist.destroy_process_group()

def train_model(model, rank, world_size, device):
    """Training function."""
    
    # Setup data loading
    dataset = YourDataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=32)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    for epoch in range(10):
        sampler.set_epoch(epoch)
        
        for batch in dataloader:
            # Training step
            loss = train_step(model, batch, device)
            
            if rank == 0 and batch_idx % 100 == 0:
                print(f"Loss: {loss:.4f}")

if __name__ == "__main__":
    main()
```

Launch with torchrun:
```bash
# Single node, 4 GPUs
torchrun --nproc_per_node=4 train_multigpu.py

# With additional arguments
torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29500 \
    train_multigpu.py
```

## Multi-Node Training

### Kubernetes Deployment

```yaml
# pytorch-distributed-training.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: pytorch-distributed-training
spec:
  completions: 2  # Number of nodes
  parallelism: 2  # Run in parallel
  template:
    metadata:
      labels:
        app: pytorch-training
    spec:
      subdomain: pytorch-training
      containers:
      - name: pytorch-worker
        image: pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel
        command: ["torchrun"]
        args:
          - "--nnodes=2"
          - "--nproc_per_node=4"  # 4 GPUs per node
          - "--node_rank=$(NODE_RANK)"
          - "--master_addr=pytorch-training-0.pytorch-training"
          - "--master_port=29500"
          - "distributed_training.py"
        env:
        - name: NODE_RANK
          valueFrom:
            fieldRef:
              fieldPath: metadata.annotations['batch.kubernetes.io/job-completion-index']
        - name: CUDA_VISIBLE_DEVICES
          value: "0,1,2,3"
        resources:
          requests:
            nvidia.com/gpu: 4
            memory: "64Gi"
            cpu: "16"
          limits:
            nvidia.com/gpu: 4
            memory: "128Gi"
            cpu: "32"
        volumeMounts:
        - name: training-data
          mountPath: /data
        - name: shared-storage
          mountPath: /shared
      volumes:
      - name: training-data
        persistentVolumeClaim:
          claimName: training-data-pvc
      - name: shared-storage
        persistentVolumeClaim:
          claimName: shared-storage-pvc
      restartPolicy: Never

---
# Headless service for pod discovery
apiVersion: v1
kind: Service
metadata:
  name: pytorch-training
spec:
  clusterIP: None
  selector:
    app: pytorch-training
  ports:
  - port: 29500
    targetPort: 29500
```

### Docker Compose Setup

```yaml
# docker-compose.yml
version: '3.8'

services:
  master:
    image: pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel
    command: >
      torchrun
      --nnodes=2
      --nproc_per_node=2
      --node_rank=0
      --master_addr=master
      --master_port=29500
      /workspace/distributed_training.py
    volumes:
      - ./:/workspace
      - data:/data
    environment:
      - CUDA_VISIBLE_DEVICES=0,1
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 2
              capabilities: [gpu]
    networks:
      - training-network

  worker:
    image: pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel
    command: >
      torchrun
      --nnodes=2
      --nproc_per_node=2
      --node_rank=1
      --master_addr=master
      --master_port=29500
      /workspace/distributed_training.py
    volumes:
      - ./:/workspace
      - data:/data
    environment:
      - CUDA_VISIBLE_DEVICES=0,1
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 2
              capabilities: [gpu]
    networks:
      - training-network
    depends_on:
      - master

volumes:
  data:

networks:
  training-network:
    driver: bridge
```

## Advanced Techniques

### Mixed Precision Training

```python
import torch
from torch.cuda.amp import GradScaler, autocast

class MixedPrecisionTrainer:
    """Trainer with automatic mixed precision."""
    
    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.scaler = GradScaler()
    
    def train_step(self, batch):
        """Single training step with mixed precision."""
        
        data, targets = batch
        data, targets = data.to(self.device), targets.to(self.device)
        
        self.optimizer.zero_grad()
        
        # Forward pass in mixed precision
        with autocast():
            outputs = self.model(data)
            loss = torch.nn.functional.cross_entropy(outputs, targets)
        
        # Backward pass with gradient scaling
        self.scaler.scale(loss).backward()
        
        # Gradient clipping (unscale first)
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Optimizer step
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss.item()

def setup_mixed_precision_training():
    """Setup training with mixed precision and distributed training."""
    
    # Model setup
    model = YourModel()
    model = model.to(device)
    model = DDP(model, device_ids=[device])
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Mixed precision trainer
    trainer = MixedPrecisionTrainer(model, optimizer, device)
    
    return trainer
```

### Gradient Accumulation

```python
class GradientAccumulationTrainer:
    """Trainer with gradient accumulation for large effective batch sizes."""
    
    def __init__(self, model, optimizer, accumulation_steps=4):
        self.model = model
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.scaler = GradScaler()
        self.step_count = 0
    
    def train_step(self, batch):
        """Training step with gradient accumulation."""
        
        data, targets = batch
        data, targets = data.to(device), targets.to(device)
        
        # Scale loss by accumulation steps
        with autocast():
            outputs = self.model(data)
            loss = torch.nn.functional.cross_entropy(outputs, targets)
            loss = loss / self.accumulation_steps
        
        # Backward pass
        self.scaler.scale(loss).backward()
        
        self.step_count += 1
        
        # Only update optimizer every accumulation_steps
        if self.step_count % self.accumulation_steps == 0:
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        
        return loss.item() * self.accumulation_steps  # Return unscaled loss

def train_with_gradient_accumulation():
    """Training loop with gradient accumulation."""
    
    trainer = GradientAccumulationTrainer(model, optimizer, accumulation_steps=8)
    
    for epoch in range(num_epochs):
        for batch in dataloader:
            loss = trainer.train_step(batch)
            
            if trainer.step_count % trainer.accumulation_steps == 0:
                print(f"Step {trainer.step_count // trainer.accumulation_steps}, Loss: {loss:.4f}")
```

### Model Checkpointing

```python
import torch
import os
from pathlib import Path

class DistributedCheckpointing:
    """Distributed training checkpointing utilities."""
    
    def __init__(self, model, optimizer, scheduler=None, checkpoint_dir="./checkpoints"):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
    
    def save_checkpoint(self, epoch, loss, rank=0, save_optimizer=True):
        """Save model checkpoint."""
        
        if rank == 0:  # Only save from rank 0 to avoid conflicts
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.module.state_dict(),  # Remove DDP wrapper
                'loss': loss,
                'timestamp': torch.tensor(time.time())
            }
            
            if save_optimizer:
                checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
                
                if self.scheduler:
                    checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save(checkpoint, checkpoint_path)
            
            # Also save as latest
            latest_path = self.checkpoint_dir / "latest_checkpoint.pt"
            torch.save(checkpoint, latest_path)
            
            print(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path=None, load_optimizer=True):
        """Load model checkpoint."""
        
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_dir / "latest_checkpoint.pt"
        
        if not os.path.exists(checkpoint_path):
            print(f"No checkpoint found at {checkpoint_path}")
            return 0, float('inf')
        
        checkpoint = torch.load(checkpoint_path, map_location=f'cuda:{torch.cuda.current_device()}')
        
        # Load model state
        self.model.module.load_state_dict(checkpoint['model_state_dict'])
        
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if self.scheduler and 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        
        print(f"Checkpoint loaded: epoch {epoch}, loss {loss:.4f}")
        return epoch, loss
    
    def save_best_model(self, current_loss, best_loss, rank=0):
        """Save model if it's the best so far."""
        
        if current_loss < best_loss and rank == 0:
            best_model_path = self.checkpoint_dir / "best_model.pt"
            torch.save({
                'model_state_dict': self.model.module.state_dict(),
                'loss': current_loss
            }, best_model_path)
            print(f"New best model saved with loss: {current_loss:.4f}")
            return current_loss
        
        return best_loss

# Usage in training loop
def train_with_checkpointing():
    """Training with checkpointing."""
    
    checkpointer = DistributedCheckpointing(model, optimizer, scheduler)
    
    # Load existing checkpoint if available
    start_epoch, best_loss = checkpointer.load_checkpoint()
    
    for epoch in range(start_epoch, num_epochs):
        # Training loop
        epoch_loss = train_epoch(model, dataloader, optimizer)
        
        # Save checkpoint every few epochs
        if epoch % 5 == 0:
            checkpointer.save_checkpoint(epoch, epoch_loss, rank)
        
        # Save best model
        best_loss = checkpointer.save_best_model(epoch_loss, best_loss, rank)
```

## Performance Optimization

### Communication Optimization

```python
import torch.distributed as dist

class CommunicationOptimizer:
    """Optimize distributed communication patterns."""
    
    def __init__(self, model, bucket_cap_mb=25):
        self.model = model
        self.bucket_cap_mb = bucket_cap_mb
        
        # Setup gradient compression
        self.setup_gradient_compression()
    
    def setup_gradient_compression(self):
        """Setup gradient compression to reduce communication."""
        
        # Register hooks for gradient compression
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.register_hook(self.compress_gradient_hook)
    
    def compress_gradient_hook(self, grad):
        """Compress gradients before communication."""
        
        # Top-k sparsification
        k = int(0.1 * grad.numel())  # Keep top 10% of gradients
        _, indices = torch.topk(grad.abs().flatten(), k)
        
        compressed_grad = torch.zeros_like(grad.flatten())
        compressed_grad[indices] = grad.flatten()[indices]
        
        return compressed_grad.reshape(grad.shape)
    
    def all_reduce_coalesced(self, tensors):
        """Perform coalesced all-reduce for better bandwidth utilization."""
        
        # Flatten and concatenate tensors
        flat_tensors = []
        tensor_shapes = []
        
        for tensor in tensors:
            flat_tensors.append(tensor.flatten())
            tensor_shapes.append(tensor.shape)
        
        coalesced_tensor = torch.cat(flat_tensors)
        
        # All-reduce the coalesced tensor
        dist.all_reduce(coalesced_tensor)
        
        # Split back to original tensors
        split_tensors = torch.split(coalesced_tensor, [t.numel() for t in flat_tensors])
        
        result_tensors = []
        for tensor, shape in zip(split_tensors, tensor_shapes):
            result_tensors.append(tensor.reshape(shape))
        
        return result_tensors

def optimize_communication_backend():
    """Optimize communication backend settings."""
    
    # Set optimal NCCL parameters
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_TREE_THRESHOLD'] = '0'  # Use ring algorithm
    os.environ['NCCL_IB_DISABLE'] = '0'      # Enable InfiniBand if available
    os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'  # Specify network interface
    
    # Set memory allocation strategy
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

def profile_communication():
    """Profile communication patterns."""
    
    import torch.profiler
    
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=1,
            warmup=1,
            active=3,
            repeat=2
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_logs'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        
        # Your training code here
        for step in range(10):
            # Training step
            train_step()
            prof.step()
```

### Memory Optimization

```python
class MemoryOptimizer:
    """Memory optimization techniques for distributed training."""
    
    def __init__(self, model):
        self.model = model
        self.activation_checkpointing = False
    
    def enable_activation_checkpointing(self):
        """Enable gradient checkpointing to save memory."""
        
        from torch.utils.checkpoint import checkpoint
        
        def checkpoint_wrapper(module):
            def wrapper(*args, **kwargs):
                return checkpoint(module, *args, **kwargs)
            return wrapper
        
        # Apply checkpointing to specific layers
        for name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.TransformerEncoderLayer, TransformerLayer)):
                module.forward = checkpoint_wrapper(module.forward)
        
        self.activation_checkpointing = True
        print("Activation checkpointing enabled")
    
    def optimize_memory_allocation(self):
        """Optimize CUDA memory allocation."""
        
        # Set memory fraction to avoid OOM
        torch.cuda.set_per_process_memory_fraction(0.9)
        
        # Enable memory pool
        torch.cuda.empty_cache()
        
        # Use memory efficient attention if available
        try:
            from torch.nn.functional import scaled_dot_product_attention
            print("Using memory efficient attention")
        except ImportError:
            print("Memory efficient attention not available")
    
    def monitor_memory_usage(self):
        """Monitor GPU memory usage."""
        
        allocated = torch.cuda.memory_allocated() / 1e9
        cached = torch.cuda.memory_reserved() / 1e9
        max_allocated = torch.cuda.max_memory_allocated() / 1e9
        
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, "
              f"Cached: {cached:.2f}GB, Max: {max_allocated:.2f}GB")
        
        return {
            'allocated': allocated,
            'cached': cached,
            'max_allocated': max_allocated
        }

# Usage in training
def memory_efficient_training():
    """Training with memory optimizations."""
    
    model = YourLargeModel()
    optimizer = MemoryOptimizer(model)
    
    # Enable optimizations
    optimizer.enable_activation_checkpointing()
    optimizer.optimize_memory_allocation()
    
    # Training loop with memory monitoring
    for epoch in range(num_epochs):
        for batch in dataloader:
            # Training step
            loss = train_step(model, batch)
            
            # Monitor memory every 100 steps
            if step % 100 == 0:
                memory_stats = optimizer.monitor_memory_usage()
```

## Related Resources

- [Kubernetes Deployment](../../deployment/kubernetes/)
- [Performance Monitoring](../../tools/monitoring/)
- [Infrastructure Setup](../../deployment/infrastructure/)
- [Model Serving](../serving/)

## Contributing

When adding new distributed training patterns:
1. Include performance benchmarks
2. Document hardware requirements
3. Provide monitoring setup
4. Add troubleshooting guides
5. Include cost optimization tips 
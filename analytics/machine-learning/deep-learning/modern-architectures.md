# Modern Deep Learning Architectures

Comprehensive guide to state-of-the-art deep learning architectures, including transformers, diffusion models, and emerging techniques for 2024-2025.

## Table of Contents

- [Overview](#overview)
- [Transformer Architectures](#transformer-architectures)
- [Diffusion Models](#diffusion-models)
- [Multimodal Models](#multimodal-models)
- [Efficient Architectures](#efficient-architectures)
- [Implementation Examples](#implementation-examples)

## Overview

Modern deep learning has shifted toward attention-based architectures, with transformers becoming the dominant paradigm across domains. Key trends include:

- **Unified architectures** across modalities
- **Parameter-efficient** training methods
- **Multimodal fusion** techniques
- **Self-supervised** learning approaches
- **Architecture search** automation

## Transformer Architectures

### 1. Vision Transformer (ViT)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math

class PatchEmbedding(nn.Module):
    """Convert image to sequence of patches."""
    
    def __init__(self, image_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e h w -> b (h w) e')
        )
    
    def forward(self, x):
        return self.projection(x)

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention computation
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        return x

class TransformerBlock(nn.Module):
    """Transformer encoder block."""
    
    def __init__(self, embed_dim, num_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # Self-attention with residual connection
        x = x + self.attn(self.norm1(x))
        
        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        
        return x

class VisionTransformer(nn.Module):
    """Vision Transformer implementation."""
    
    def __init__(self, image_size=224, patch_size=16, in_channels=3, 
                 num_classes=1000, embed_dim=768, depth=12, num_heads=12,
                 mlp_ratio=4, dropout=0.1):
        super().__init__()
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # Learnable position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize model weights."""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Classification
        x = self.norm(x)
        cls_token_final = x[:, 0]  # Use class token for classification
        
        return self.head(cls_token_final)

# Usage example
def create_vision_transformer():
    """Create Vision Transformer model."""
    
    model = VisionTransformer(
        image_size=224,
        patch_size=16,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12
    )
    
    return model

# Example training setup
def train_vision_transformer():
    """Training setup for Vision Transformer."""
    
    model = create_vision_transformer()
    
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=0.3
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=300  # Total epochs
    )
    
    return model, optimizer, scheduler
```

### 2. Swin Transformer

```python
class WindowAttention(nn.Module):
    """Window-based multi-head self-attention."""
    
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        
        # Get pair-wise relative position indices
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock(nn.Module):
    """Swin Transformer block."""
    
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        
        self.drop_path = nn.Identity()  # Could implement stochastic depth
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
        
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                       slice(-self.window_size, -self.shift_size),
                       slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                       slice(-self.window_size, -self.shift_size),
                       slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            
            mask_windows = self.window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        
        self.register_buffer("attn_mask", attn_mask)
    
    def window_partition(self, x, window_size):
        """Partition into windows."""
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
        return windows
    
    def window_reverse(self, windows, window_size, H, W):
        """Reverse window partitioning."""
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x
    
    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        
        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        
        # Partition windows
        x_windows = self.window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        
        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = self.window_reverse(attn_windows, self.window_size, H, W)
        
        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        
        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x
```

## Diffusion Models

### 1. Denoising Diffusion Probabilistic Models (DDPM)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for time steps."""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResidualBlock(nn.Module):
    """Residual block with time embedding."""
    
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x, time_emb):
        h = self.block1(x)
        
        # Add time embedding
        h += self.time_mlp(time_emb)[:, :, None, None]
        
        h = self.block2(h)
        
        return h + self.residual_conv(x)

class AttentionBlock(nn.Module):
    """Self-attention block for diffusion models."""
    
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape for attention
        q = q.view(B, C, H * W).transpose(1, 2)
        k = k.view(B, C, H * W).transpose(1, 2)
        v = v.view(B, C, H * W).transpose(1, 2)
        
        # Attention
        scale = (C // 8) ** -0.5
        attn = torch.bmm(q, k.transpose(1, 2)) * scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.bmm(attn, v)
        out = out.transpose(1, 2).view(B, C, H, W)
        
        out = self.proj(out)
        
        return x + out

class UNet(nn.Module):
    """U-Net architecture for diffusion models."""
    
    def __init__(self, in_channels=3, out_channels=3, features=[64, 128, 256, 512], 
                 time_emb_dim=256, num_res_blocks=2):
        super().__init__()
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.GELU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )
        
        # Encoder
        self.encoder = nn.ModuleList()
        self.pool = nn.ModuleList()
        
        # Initial convolution
        self.init_conv = nn.Conv2d(in_channels, features[0], 3, padding=1)
        
        # Encoder blocks
        in_ch = features[0]
        for feature in features:
            # Residual blocks
            blocks = []
            for _ in range(num_res_blocks):
                blocks.append(ResidualBlock(in_ch, feature, time_emb_dim))
                in_ch = feature
            
            # Add attention for higher resolution features
            if feature >= 256:
                blocks.append(AttentionBlock(feature))
            
            self.encoder.append(nn.ModuleList(blocks))
            self.pool.append(nn.Conv2d(feature, feature, 3, stride=2, padding=1))
        
        # Bottleneck
        self.bottleneck = nn.ModuleList([
            ResidualBlock(features[-1], features[-1] * 2, time_emb_dim),
            AttentionBlock(features[-1] * 2),
            ResidualBlock(features[-1] * 2, features[-1], time_emb_dim)
        ])
        
        # Decoder
        self.decoder = nn.ModuleList()
        self.upconv = nn.ModuleList()
        
        for feature in reversed(features):
            self.upconv.append(nn.ConvTranspose2d(feature * 2, feature, 2, stride=2))
            
            blocks = []
            for _ in range(num_res_blocks):
                blocks.append(ResidualBlock(feature * 2, feature, time_emb_dim))
            
            if feature >= 256:
                blocks.append(AttentionBlock(feature))
            
            self.decoder.append(nn.ModuleList(blocks))
        
        # Final convolution
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, features[0]),
            nn.SiLU(),
            nn.Conv2d(features[0], out_channels, 3, padding=1)
        )
    
    def forward(self, x, timestep):
        # Time embedding
        t = self.time_mlp(timestep)
        
        # Initial convolution
        x = self.init_conv(x)
        
        # Encoder
        skip_connections = []
        for encoder_block, pool in zip(self.encoder, self.pool):
            for block in encoder_block:
                if isinstance(block, ResidualBlock):
                    x = block(x, t)
                else:
                    x = block(x)
            
            skip_connections.append(x)
            x = pool(x)
        
        # Bottleneck
        for block in self.bottleneck:
            if isinstance(block, ResidualBlock):
                x = block(x, t)
            else:
                x = block(x)
        
        # Decoder
        for decoder_block, upconv, skip in zip(self.decoder, self.upconv, reversed(skip_connections)):
            x = upconv(x)
            x = torch.cat([x, skip], dim=1)
            
            for block in decoder_block:
                if isinstance(block, ResidualBlock):
                    x = block(x, t)
                else:
                    x = block(x)
        
        return self.final_conv(x)

class DDPM(nn.Module):
    """Denoising Diffusion Probabilistic Model."""
    
    def __init__(self, model, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        
        # Linear schedule for betas
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
    
    def q_sample(self, x_start, t, noise=None):
        """Sample from q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_losses(self, x_start, t, noise=None):
        """Calculate training losses."""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = self.model(x_noisy, t)
        
        loss = F.mse_loss(noise, predicted_noise)
        return loss
    
    @torch.no_grad()
    def p_sample(self, x, t, t_index):
        """Sample from p(x_{t-1} | x_t)."""
        betas_t = self.betas[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        sqrt_recip_alphas_t = torch.sqrt(1.0 / self.alphas[t])[:, None, None, None]
        
        # Predict noise
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * self.model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )
        
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance[t][:, None, None, None]
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    @torch.no_grad()
    def sample(self, image_size, batch_size=1, channels=3):
        """Generate samples from the model."""
        device = next(self.model.parameters()).device
        
        # Start from random noise
        img = torch.randn((batch_size, channels, image_size, image_size), device=device)
        
        # Reverse diffusion process
        for i in reversed(range(0, self.timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            img = self.p_sample(img, t, i)
        
        return img
    
    def forward(self, x):
        """Forward pass for training."""
        device = x.device
        t = torch.randint(0, self.timesteps, (x.shape[0],), device=device).long()
        return self.p_losses(x, t)

# Usage example
def train_ddpm():
    """Training setup for DDPM."""
    
    # Create model
    unet = UNet(in_channels=3, out_channels=3)
    ddpm = DDPM(unet, timesteps=1000)
    
    # Optimizer
    optimizer = torch.optim.Adam(ddpm.parameters(), lr=2e-4)
    
    # Training loop
    for epoch in range(100):
        for batch in dataloader:
            optimizer.zero_grad()
            
            loss = ddpm(batch)
            loss.backward()
            
            optimizer.step()
            
            if step % 100 == 0:
                print(f"Loss: {loss.item():.4f}")
        
        # Generate samples
        if epoch % 10 == 0:
            samples = ddpm.sample(image_size=64, batch_size=4)
            # Save samples
```

## Multimodal Models

### 1. CLIP-style Contrastive Learning

```python
class CLIPModel(nn.Module):
    """CLIP-style multimodal model."""
    
    def __init__(self, image_encoder, text_encoder, embed_dim=512, temperature=0.07):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.temperature = temperature
        
        # Projection heads
        self.image_projection = nn.Linear(image_encoder.embed_dim, embed_dim)
        self.text_projection = nn.Linear(text_encoder.embed_dim, embed_dim)
        
        # Learnable temperature parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
        
    def encode_image(self, image):
        """Encode image to embedding space."""
        image_features = self.image_encoder(image)
        image_features = self.image_projection(image_features)
        return F.normalize(image_features, dim=-1)
    
    def encode_text(self, text):
        """Encode text to embedding space."""
        text_features = self.text_encoder(text)
        text_features = self.text_projection(text_features)
        return F.normalize(text_features, dim=-1)
    
    def forward(self, image, text):
        """Forward pass with contrastive loss."""
        # Get embeddings
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        
        # Compute logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        
        return logits_per_image, logits_per_text
    
    def contrastive_loss(self, image, text):
        """Compute contrastive loss."""
        logits_per_image, logits_per_text = self(image, text)
        
        # Ground truth: diagonal entries should be high
        batch_size = image.shape[0]
        labels = torch.arange(batch_size).to(image.device)
        
        # Symmetric loss
        loss_img = F.cross_entropy(logits_per_image, labels)
        loss_txt = F.cross_entropy(logits_per_text, labels)
        
        return (loss_img + loss_txt) / 2

class MultimodalTransformer(nn.Module):
    """Transformer with cross-modal attention."""
    
    def __init__(self, dim, num_heads, num_layers, vocab_size, max_seq_len=512):
        super().__init__()
        self.dim = dim
        
        # Text embedding
        self.text_embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(max_seq_len, dim)
        
        # Image patch embedding
        self.image_patch_embed = PatchEmbedding(embed_dim=dim)
        
        # Modality type embeddings
        self.text_type_embed = nn.Parameter(torch.zeros(1, 1, dim))
        self.image_type_embed = nn.Parameter(torch.zeros(1, 1, dim))
        
        # Transformer layers with cross-attention
        self.layers = nn.ModuleList([
            MultimodalTransformerLayer(dim, num_heads)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, text_ids, image, text_mask=None):
        """Forward pass with text and image inputs."""
        
        # Text encoding
        text_embeds = self.text_embed(text_ids)
        seq_len = text_ids.shape[1]
        pos_ids = torch.arange(seq_len, device=text_ids.device).expand_as(text_ids)
        text_embeds += self.pos_embed(pos_ids)
        text_embeds += self.text_type_embed
        
        # Image encoding
        image_embeds = self.image_patch_embed(image)
        image_embeds += self.image_type_embed
        
        # Combine modalities
        combined_embeds = torch.cat([text_embeds, image_embeds], dim=1)
        
        # Create attention mask
        if text_mask is not None:
            image_mask = torch.ones(image_embeds.shape[:2], device=image.device)
            combined_mask = torch.cat([text_mask, image_mask], dim=1)
        else:
            combined_mask = None
        
        # Apply transformer layers
        for layer in self.layers:
            combined_embeds = layer(combined_embeds, combined_mask)
        
        return self.norm(combined_embeds)

class MultimodalTransformerLayer(nn.Module):
    """Transformer layer with cross-modal capabilities."""
    
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, mask=None):
        # Self-attention
        attn_mask = self._create_attention_mask(mask) if mask is not None else None
        x2, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x + x2)
        
        # MLP
        x2 = self.mlp(x)
        x = self.norm2(x + x2)
        
        return x
    
    def _create_attention_mask(self, mask):
        """Create attention mask for transformer."""
        # mask: (batch_size, seq_len)
        # output: (seq_len, seq_len) for each batch
        seq_len = mask.shape[1]
        mask = mask.unsqueeze(1).expand(-1, seq_len, -1)  # (batch, seq_len, seq_len)
        mask = mask & mask.transpose(1, 2)  # Bidirectional mask
        return ~mask  # Invert for attention mask
```

## Related Resources

- [PyTorch Distributed Training](../../machine-learning/pytorch/distributed-training.md)
- [Model Serving](../../machine-learning/serving/)
- [Performance Optimization](../../deployment/kubernetes/)
- [Data Pipelines](../../analytics/machine-learning/)

## Contributing

When adding new architecture patterns:
1. Include theoretical background
2. Provide complete implementations
3. Add training examples
4. Document computational requirements
5. Include evaluation metrics 
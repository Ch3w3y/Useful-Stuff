# TensorFlow Deep Learning Framework

## Overview

Comprehensive guide to TensorFlow for deep learning, neural networks, and production AI systems. Covers TensorFlow 2.x fundamentals through advanced techniques with practical examples and deployment strategies.

## Table of Contents

- [TensorFlow Fundamentals](#tensorflow-fundamentals)
- [Neural Network Architectures](#neural-network-architectures)
- [Advanced Deep Learning](#advanced-deep-learning)
- [Computer Vision](#computer-vision)
- [Natural Language Processing](#natural-language-processing)
- [Time Series & Forecasting](#time-series--forecasting)
- [Model Optimization](#model-optimization)
- [Production Deployment](#production-deployment)

## TensorFlow Fundamentals

### Core Operations and Tensors

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")

# Basic tensor operations
def tensor_fundamentals():
    """Essential TensorFlow tensor operations"""
    
    # Create tensors
    scalar = tf.constant(42)
    vector = tf.constant([1, 2, 3, 4])
    matrix = tf.constant([[1, 2], [3, 4]])
    tensor_3d = tf.random.normal((2, 3, 4))
    
    print(f"Scalar: {scalar}")
    print(f"Vector shape: {vector.shape}")
    print(f"Matrix:\n{matrix}")
    print(f"3D tensor shape: {tensor_3d.shape}")
    
    # Tensor operations
    a = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    b = tf.constant([[5, 6], [7, 8]], dtype=tf.float32)
    
    # Mathematical operations
    addition = tf.add(a, b)
    multiplication = tf.multiply(a, b)
    matrix_mult = tf.matmul(a, b)
    
    # Reduction operations
    sum_all = tf.reduce_sum(a)
    mean_axis0 = tf.reduce_mean(a, axis=0)
    max_axis1 = tf.reduce_max(a, axis=1)
    
    return {
        'addition': addition,
        'matrix_mult': matrix_mult,
        'sum_all': sum_all,
        'mean_axis0': mean_axis0
    }

# Variables and gradients
class GradientExample:
    """Demonstrate automatic differentiation"""
    
    def __init__(self):
        self.w = tf.Variable(2.0)
        self.b = tf.Variable(1.0)
    
    def forward(self, x):
        return self.w * x + self.b
    
    def loss_function(self, x, y_true):
        y_pred = self.forward(x)
        return tf.reduce_mean(tf.square(y_true - y_pred))
    
    def train_step(self, x, y_true, learning_rate=0.01):
        with tf.GradientTape() as tape:
            loss = self.loss_function(x, y_true)
        
        # Compute gradients
        gradients = tape.gradient(loss, [self.w, self.b])
        
        # Update variables
        self.w.assign_sub(learning_rate * gradients[0])
        self.b.assign_sub(learning_rate * gradients[1])
        
        return loss

# Data pipeline with tf.data
def create_data_pipeline(batch_size=32):
    """Create efficient data pipeline"""
    
    # Generate sample data
    x_data = np.random.randn(1000, 10).astype(np.float32)
    y_data = np.random.randint(0, 2, (1000, 1)).astype(np.float32)
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))
    
    # Apply transformations
    dataset = (dataset
               .shuffle(buffer_size=1000)
               .batch(batch_size)
               .prefetch(tf.data.AUTOTUNE))
    
    return dataset

# Example usage
results = tensor_fundamentals()
print(f"Matrix multiplication result:\n{results['matrix_mult']}")

# Gradient example
grad_example = GradientExample()
x_train = tf.constant([1.0, 2.0, 3.0, 4.0])
y_train = tf.constant([3.0, 5.0, 7.0, 9.0])  # y = 2x + 1

for epoch in range(100):
    loss = grad_example.train_step(x_train, y_train)
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}, w: {grad_example.w:.4f}, b: {grad_example.b:.4f}")
```

## Neural Network Architectures

### Sequential and Functional API

```python
# Sequential API - Simple models
def create_sequential_model(input_shape, num_classes):
    """Create model using Sequential API"""
    
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=input_shape),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

# Functional API - Complex architectures
def create_functional_model(input_shape, num_classes):
    """Create model using Functional API"""
    
    # Input layer
    inputs = keras.Input(shape=input_shape)
    
    # Feature extraction branch
    x1 = layers.Dense(64, activation='relu')(inputs)
    x1 = layers.Dropout(0.3)(x1)
    x1 = layers.Dense(32, activation='relu')(x1)
    
    # Alternative branch
    x2 = layers.Dense(32, activation='relu')(inputs)
    x2 = layers.Dropout(0.2)(x2)
    
    # Concatenate branches
    merged = layers.concatenate([x1, x2])
    
    # Final layers
    x = layers.Dense(64, activation='relu')(merged)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

# Custom layers
class CustomDenseLayer(layers.Layer):
    """Custom dense layer with batch normalization"""
    
    def __init__(self, units, activation='relu', **kwargs):
        super(CustomDenseLayer, self).__init__(**kwargs)
        self.units = units
        self.activation = activation
        
    def build(self, input_shape):
        self.dense = layers.Dense(self.units)
        self.batch_norm = layers.BatchNormalization()
        self.activation_layer = layers.Activation(self.activation)
        super(CustomDenseLayer, self).build(input_shape)
    
    def call(self, inputs, training=None):
        x = self.dense(inputs)
        x = self.batch_norm(x, training=training)
        return self.activation_layer(x)
    
    def get_config(self):
        config = super(CustomDenseLayer, self).get_config()
        config.update({
            'units': self.units,
            'activation': self.activation
        })
        return config

# Residual connections
def residual_block(x, filters, kernel_size=3, stride=1):
    """Residual block for ResNet-style architectures"""
    
    # Main path
    shortcut = x
    
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.Conv2D(filters, kernel_size, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Adjust shortcut if needed
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    
    # Add shortcut
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    
    return x

# Attention mechanism
class MultiHeadAttention(layers.Layer):
    """Multi-head attention layer"""
    
    def __init__(self, d_model, num_heads, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % self.num_heads == 0
        
        self.depth = d_model // self.num_heads
        
        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)
        
        self.dense = layers.Dense(d_model)
    
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q, mask=None):
        batch_size = tf.shape(q)[0]
        
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        # Scaled dot-product attention
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(output, (batch_size, -1, self.d_model))
        
        output = self.dense(concat_attention)
        return output, attention_weights
```

## Advanced Deep Learning

### Transfer Learning and Fine-tuning

```python
# Transfer learning with pre-trained models
def create_transfer_model(base_model_name='ResNet50', num_classes=10, input_shape=(224, 224, 3)):
    """Create transfer learning model"""
    
    # Load pre-trained base model
    if base_model_name == 'ResNet50':
        base_model = keras.applications.ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
    elif base_model_name == 'EfficientNetB0':
        base_model = keras.applications.EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
    elif base_model_name == 'VGG16':
        base_model = keras.applications.VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Add custom head
    inputs = keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    return model, base_model

# Fine-tuning strategy
def fine_tune_model(model, base_model, fine_tune_at=100):
    """Fine-tune pre-trained model"""
    
    # Unfreeze top layers
    base_model.trainable = True
    
    # Fine-tune from this layer onwards
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    
    # Use lower learning rate for fine-tuning
    model.compile(
        optimizer=keras.optimizers.Adam(1e-5/10),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Model ensembling
class ModelEnsemble:
    """Ensemble multiple models for better performance"""
    
    def __init__(self, models):
        self.models = models
    
    def predict(self, x):
        """Average predictions from all models"""
        predictions = []
        for model in self.models:
            pred = model.predict(x)
            predictions.append(pred)
        
        # Average predictions
        ensemble_pred = tf.reduce_mean(predictions, axis=0)
        return ensemble_pred
    
    def predict_with_uncertainty(self, x, num_samples=100):
        """Monte Carlo dropout for uncertainty estimation"""
        predictions = []
        
        for model in self.models:
            model_preds = []
            for _ in range(num_samples):
                # Enable dropout during inference
                pred = model(x, training=True)
                model_preds.append(pred)
            predictions.append(tf.stack(model_preds))
        
        # Calculate mean and variance
        all_preds = tf.concat(predictions, axis=0)
        mean_pred = tf.reduce_mean(all_preds, axis=0)
        var_pred = tf.math.reduce_variance(all_preds, axis=0)
        
        return mean_pred, var_pred

# Advanced training techniques
class AdvancedTrainer:
    """Advanced training utilities"""
    
    def __init__(self, model):
        self.model = model
        self.history = {'loss': [], 'val_loss': [], 'lr': []}
    
    def cosine_decay_schedule(self, epoch, lr):
        """Cosine annealing learning rate schedule"""
        import math
        
        total_epochs = 100
        min_lr = 1e-6
        max_lr = 1e-3
        
        if epoch < total_epochs:
            cosine_decay = 0.5 * (1 + math.cos(math.pi * epoch / total_epochs))
            return min_lr + (max_lr - min_lr) * cosine_decay
        return min_lr
    
    def train_with_mixed_precision(self, train_data, val_data, epochs=10):
        """Train with mixed precision for faster training"""
        
        # Enable mixed precision
        policy = keras.mixed_precision.Policy('mixed_float16')
        keras.mixed_precision.set_global_policy(policy)
        
        # Compile with loss scaling
        optimizer = keras.optimizers.Adam()
        optimizer = keras.mixed_precision.LossScaleOptimizer(optimizer)
        
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks
        callbacks = [
            keras.callbacks.LearningRateScheduler(self.cosine_decay_schedule),
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
        ]
        
        # Train model
        history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def gradual_unfreezing(self, base_model, train_data, val_data, stages=3):
        """Gradually unfreeze layers during training"""
        
        total_layers = len(base_model.layers)
        layers_per_stage = total_layers // stages
        
        for stage in range(stages):
            print(f"Training stage {stage + 1}/{stages}")
            
            # Unfreeze more layers
            unfreeze_from = total_layers - (stage + 1) * layers_per_stage
            for i, layer in enumerate(base_model.layers):
                layer.trainable = i >= unfreeze_from
            
            # Adjust learning rate
            lr = 1e-4 / (2 ** stage)
            self.model.compile(
                optimizer=keras.optimizers.Adam(lr),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Train for a few epochs
            self.model.fit(
                train_data,
                validation_data=val_data,
                epochs=5,
                verbose=1
            )
```

## Computer Vision

### CNN Architectures and Applications

```python
# Advanced CNN architectures
def create_efficientnet_custom(input_shape, num_classes):
    """Custom EfficientNet-inspired architecture"""
    
    inputs = keras.Input(shape=input_shape)
    
    # Stem
    x = layers.Conv2D(32, 3, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # MBConv blocks
    def mbconv_block(x, filters, kernel_size, strides, expand_ratio):
        # Expansion
        expanded_filters = int(x.shape[-1] * expand_ratio)
        if expand_ratio != 1:
            x = layers.Conv2D(expanded_filters, 1, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
        
        # Depthwise convolution
        x = layers.DepthwiseConv2D(kernel_size, strides=strides, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        # Squeeze and excitation
        se = layers.GlobalAveragePooling2D()(x)
        se = layers.Dense(expanded_filters // 4, activation='relu')(se)
        se = layers.Dense(expanded_filters, activation='sigmoid')(se)
        se = layers.Reshape((1, 1, expanded_filters))(se)
        x = layers.Multiply()([x, se])
        
        # Output projection
        x = layers.Conv2D(filters, 1, padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        return x
    
    # Build blocks
    x = mbconv_block(x, 64, 3, 1, 6)
    x = mbconv_block(x, 128, 3, 2, 6)
    x = mbconv_block(x, 256, 5, 2, 6)
    
    # Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    return model

# Object detection utilities
class ObjectDetectionUtils:
    """Utilities for object detection tasks"""
    
    @staticmethod
    def create_yolo_model(input_shape, num_classes, num_anchors=3):
        """Simplified YOLO-style model"""
        
        inputs = keras.Input(shape=input_shape)
        
        # Backbone
        x = layers.Conv2D(32, 3, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D(2)(x)
        
        x = layers.Conv2D(64, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D(2)(x)
        
        x = layers.Conv2D(128, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D(2)(x)
        
        # Detection head
        x = layers.Conv2D(256, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        # Output: (batch, grid_h, grid_w, anchors * (5 + num_classes))
        # 5 = x, y, w, h, confidence
        outputs = layers.Conv2D(num_anchors * (5 + num_classes), 1)(x)
        
        model = keras.Model(inputs, outputs)
        return model
    
    @staticmethod
    def yolo_loss(y_true, y_pred, num_classes=80, num_anchors=3):
        """YOLO loss function"""
        
        # Reshape predictions
        grid_h, grid_w = tf.shape(y_pred)[1], tf.shape(y_pred)[2]
        y_pred = tf.reshape(y_pred, (-1, grid_h, grid_w, num_anchors, 5 + num_classes))
        
        # Split predictions
        pred_xy = tf.sigmoid(y_pred[..., 0:2])
        pred_wh = y_pred[..., 2:4]
        pred_conf = tf.sigmoid(y_pred[..., 4:5])
        pred_class = tf.sigmoid(y_pred[..., 5:])
        
        # Calculate losses (simplified)
        xy_loss = tf.reduce_sum(tf.square(y_true[..., 0:2] - pred_xy))
        wh_loss = tf.reduce_sum(tf.square(y_true[..., 2:4] - pred_wh))
        conf_loss = tf.reduce_sum(tf.square(y_true[..., 4:5] - pred_conf))
        class_loss = tf.reduce_sum(tf.square(y_true[..., 5:] - pred_class))
        
        total_loss = xy_loss + wh_loss + conf_loss + class_loss
        return total_loss

# Image segmentation
def create_unet_model(input_shape, num_classes):
    """U-Net architecture for image segmentation"""
    
    inputs = keras.Input(shape=input_shape)
    
    # Encoder
    c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D(2)(c1)
    
    c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D(2)(c2)
    
    c3 = layers.Conv2D(256, 3, activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, 3, activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D(2)(c3)
    
    # Bridge
    c4 = layers.Conv2D(512, 3, activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, 3, activation='relu', padding='same')(c4)
    
    # Decoder
    u5 = layers.UpSampling2D(2)(c4)
    u5 = layers.concatenate([u5, c3])
    c5 = layers.Conv2D(256, 3, activation='relu', padding='same')(u5)
    c5 = layers.Conv2D(256, 3, activation='relu', padding='same')(c5)
    
    u6 = layers.UpSampling2D(2)(c5)
    u6 = layers.concatenate([u6, c2])
    c6 = layers.Conv2D(128, 3, activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(128, 3, activation='relu', padding='same')(c6)
    
    u7 = layers.UpSampling2D(2)(c6)
    u7 = layers.concatenate([u7, c1])
    c7 = layers.Conv2D(64, 3, activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(64, 3, activation='relu', padding='same')(c7)
    
    outputs = layers.Conv2D(num_classes, 1, activation='softmax')(c7)
    
    model = keras.Model(inputs, outputs)
    return model
```

---

*This comprehensive TensorFlow guide provides complete coverage from fundamentals to advanced deep learning and production deployment.* 
# DeepLearning2 Usage Examples

This document provides comprehensive examples of how to use the DeepLearning2 framework for various deep learning experiments.

## Table of Contents
1. [Basic Usage](#basic-usage)
2. [Creating Custom Architectures](#creating-custom-architectures)
3. [Creating Custom Optimizers](#creating-custom-optimizers)
4. [Training Models](#training-models)
5. [Comparing Experiments](#comparing-experiments)

## Basic Usage

### 1. Check GPU Availability

```python
from src.utils import get_device, print_gpu_info

# Get the best available device
device = get_device()
print(f"Using device: {device}")

# Print detailed GPU information
print_gpu_info()
```

### 2. Create a Simple Model

```python
from src.architectures import SimpleCNN

# Create a CNN for MNIST (28x28 grayscale images, 10 classes)
model = SimpleCNN(input_channels=1, num_classes=10)

# Print model information
model.print_model_info()
```

### 3. Use Custom Optimizers

```python
from src.optimizers import CustomAdam, CustomSGD

# Create CustomAdam optimizer
optimizer = CustomAdam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

# Or use CustomSGD with momentum
optimizer = CustomSGD(model.parameters(), lr=0.01, momentum=0.9)
```

## Creating Custom Architectures

### Example 1: Simple Custom CNN

```python
from src.architectures.base import BaseArchitecture
import torch.nn as nn
import torch.nn.functional as F

class MyCustomCNN(BaseArchitecture):
    def __init__(self, input_channels=3, num_classes=10):
        super().__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
        # Regularization
        self.dropout = nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
    
    def forward(self, x):
        # Conv block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # Conv block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # Conv block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
```

### Example 2: Custom Attention-Based Architecture

```python
import torch
import torch.nn as nn
from src.architectures.base import BaseArchitecture

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        batch, channels, height, width = x.size()
        
        # Compute attention
        query = self.query(x).view(batch, -1, height * width).permute(0, 2, 1)
        key = self.key(x).view(batch, -1, height * width)
        attention = F.softmax(torch.bmm(query, key), dim=-1)
        
        value = self.value(x).view(batch, -1, height * width)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch, channels, height, width)
        
        return self.gamma * out + x

class AttentionNet(BaseArchitecture):
    def __init__(self, input_channels=1, num_classes=10):
        super().__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 64, 3, padding=1)
        self.attention = AttentionBlock(64)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.attention(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
```

## Creating Custom Optimizers

### Example 1: SGD with Nesterov Momentum

```python
import torch
from torch.optim import Optimizer

class NesterovSGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.9):
        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                param_state = self.state[p]
                
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                else:
                    buf = param_state['momentum_buffer']
                
                # Nesterov momentum
                buf.mul_(momentum).add_(grad)
                p.data.add_(buf.mul(momentum).add(grad), alpha=-lr)
        
        return loss
```

### Example 2: Custom Learning Rate Warmup Optimizer

```python
from torch.optim import Optimizer

class WarmupAdam(Optimizer):
    def __init__(self, params, lr=0.001, warmup_steps=1000, betas=(0.9, 0.999), eps=1e-8):
        defaults = dict(lr=lr, warmup_steps=warmup_steps, betas=betas, eps=eps)
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            base_lr = group['lr']
            warmup_steps = group['warmup_steps']
            beta1, beta2 = group['betas']
            eps = group['eps']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                param_state = self.state[p]
                
                if len(param_state) == 0:
                    param_state['step'] = 0
                    param_state['exp_avg'] = torch.zeros_like(p.data)
                    param_state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = param_state['exp_avg'], param_state['exp_avg_sq']
                param_state['step'] += 1
                step = param_state['step']
                
                # Warmup learning rate
                if step <= warmup_steps:
                    lr = base_lr * (step / warmup_steps)
                else:
                    lr = base_lr
                
                # Adam update
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                
                step_size = lr / bias_correction1
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
        
        return loss
```

## Training Models

### Example 1: Simple Training Loop

```python
import torch.nn as nn
from src.architectures import SimpleCNN
from src.optimizers import CustomAdam
from src.utils import Trainer, get_mnist_loaders, get_device

# Setup
device = get_device()
model = SimpleCNN(input_channels=1, num_classes=10)
optimizer = CustomAdam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Create trainer
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    device=device
)

# Load data
train_loader, val_loader, test_loader = get_mnist_loaders(batch_size=64)

# Train
history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=10
)

# Evaluate
test_loss, test_acc = trainer.validate(test_loader)
print(f"Test Accuracy: {test_acc:.2f}%")
```

### Example 2: Training with Learning Rate Scheduling

```python
import torch.optim as optim
from src.architectures import ResNet
from src.utils import Trainer, get_mnist_loaders, get_device

# Setup
device = get_device()
model = ResNet(input_channels=1, num_classes=10, num_blocks=[2, 2, 2])
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Create trainer with scheduler
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    criterion=nn.CrossEntropyLoss(),
    device=device,
    scheduler=scheduler,
    gradient_clip=1.0  # Gradient clipping
)

# Load data and train
train_loader, val_loader, _ = get_mnist_loaders()
history = trainer.train(train_loader, epochs=15, val_loader=val_loader)
```

### Example 3: Save and Load Checkpoints

```python
# Save checkpoint
trainer.save_checkpoint(
    filepath='checkpoints/best_model.pth',
    epoch=10,
    test_accuracy=98.5
)

# Load checkpoint
checkpoint = trainer.load_checkpoint('checkpoints/best_model.pth')
print(f"Loaded model from epoch {checkpoint['epoch']}")
```

## Comparing Experiments

### Example 1: Compare Multiple Architectures

```python
from src.experiments import compare_architectures

# Compare SimpleCNN, ResNet, and FullyConnectedNet
results = compare_architectures(epochs=5)

# Results will show performance metrics for each architecture
```

### Example 2: Compare Multiple Optimizers

```python
from src.experiments import compare_optimizers

# Compare SGD, Adam, CustomSGD, and CustomAdam
results = compare_optimizers(epochs=5)

# Results will show performance metrics for each optimizer
```

### Example 3: Custom Experiment Comparison

```python
import torch.nn as nn
from src.architectures import SimpleCNN, ResNet
from src.optimizers import CustomAdam, CustomSGD
from src.utils import Trainer, get_mnist_loaders, get_device

# Define experiments
experiments = {
    'CNN + Adam': {
        'model': SimpleCNN(input_channels=1, num_classes=10),
        'optimizer_fn': lambda params: CustomAdam(params, lr=0.001)
    },
    'ResNet + SGD': {
        'model': ResNet(input_channels=1, num_classes=10),
        'optimizer_fn': lambda params: CustomSGD(params, lr=0.01, momentum=0.9)
    },
}

# Run experiments
device = get_device()
train_loader, val_loader, test_loader = get_mnist_loaders()
results = {}

for name, config in experiments.items():
    print(f"\nRunning: {name}")
    
    model = config['model']
    optimizer = config['optimizer_fn'](model.parameters())
    
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=nn.CrossEntropyLoss(),
        device=device
    )
    
    history = trainer.train(train_loader, epochs=5, val_loader=val_loader)
    test_loss, test_acc = trainer.validate(test_loader)
    
    results[name] = {'test_loss': test_loss, 'test_acc': test_acc}

# Print comparison
print("\n" + "=" * 60)
for name, result in results.items():
    print(f"{name}: Accuracy = {result['test_acc']:.2f}%, Loss = {result['test_loss']:.4f}")
```

## Running Examples

You can run the provided example experiments:

```bash
# Basic MNIST experiment with SimpleCNN and Adam
python -m src.experiments.mnist_example --arch cnn --optimizer adam --epochs 5

# Compare different architectures
python -m src.experiments.compare_architectures --epochs 3

# Compare different optimizers
python -m src.experiments.compare_optimizers --epochs 5

# Run quick start demo
python quick_start.py
```

## Tips and Best Practices

1. **GPU Usage**: Always check GPU availability before training
   ```python
   from src.utils import print_gpu_info
   print_gpu_info()
   ```

2. **Batch Size**: Adjust based on GPU memory
   - Start with 64 for small models
   - Reduce to 32 or 16 for larger models like ResNet

3. **Learning Rate**: 
   - Adam: typically 0.001
   - SGD: typically 0.01 with momentum 0.9

4. **Monitoring**: Check validation loss to detect overfitting

5. **Checkpointing**: Save models regularly during training
   ```python
   trainer.save_checkpoint('model.pth', epoch=i)
   ```

6. **Gradient Clipping**: Use for RNNs or unstable training
   ```python
   trainer = Trainer(..., gradient_clip=1.0)
   ```

# DeepLearning2

A flexible Python framework for experimenting with different deep learning architectures and optimizers on a single GPU using PyTorch and CUDA.

## Features

- **Modular Architecture Design**: Easily design and test different neural network architectures
- **Custom Optimizers**: Implement and test various optimization algorithms
- **GPU Acceleration**: Full CUDA support for single GPU training
- **Extensible Framework**: Simple interfaces for adding new architectures and optimizers

## Installation

1. Clone the repository:
```bash
git clone https://github.com/intrinsicD/DeepLearning2.git
cd DeepLearning2
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
DeepLearning2/
├── src/
│   ├── architectures/     # Neural network architectures
│   ├── optimizers/        # Custom optimizers
│   ├── utils/             # Training utilities and helpers
│   └── experiments/       # Example experiments
├── requirements.txt       # Python dependencies
└── README.md
```

## Quick Start

### 1. Define a Custom Architecture

```python
from src.architectures.base import BaseArchitecture
import torch.nn as nn

class MyCustomNet(BaseArchitecture):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

### 2. Use a Custom Optimizer

```python
from src.optimizers.custom_sgd import CustomSGD

optimizer = CustomSGD(model.parameters(), lr=0.01, momentum=0.9)
```

### 3. Train Your Model

```python
from src.utils.trainer import Trainer

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    device='cuda',  # Use GPU
    criterion=nn.CrossEntropyLoss()
)

trainer.train(train_loader, epochs=10)
```

## Example Experiments

Run the example MNIST classification experiment:
```bash
python -m src.experiments.mnist_example
```

## GPU Support

The framework automatically detects and uses CUDA if available. You can check GPU availability:

```python
from src.utils.device import get_device, print_gpu_info

device = get_device()  # Returns 'cuda' if available, else 'cpu'
print_gpu_info()  # Prints GPU information
```

## Adding New Architectures

1. Create a new file in `src/architectures/`
2. Inherit from `BaseArchitecture`
3. Implement the `__init__` and `forward` methods

## Adding New Optimizers

1. Create a new file in `src/optimizers/`
2. Inherit from `torch.optim.Optimizer`
3. Implement the `step` method

## Requirements

- Python 3.8+
- PyTorch 2.0+ with CUDA support
- NVIDIA GPU with CUDA capability (recommended)

## License

MIT License
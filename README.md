# DeepLearning2

A flexible PyTorch framework for rapid experimentation with multimodal architectures, custom optimizers, and unified training utilities.

## Features

- **Modular architecture design**: plug-and-play building blocks in `architectures/` and `models/`
- **Optimizer zoo**: custom optimizers and schedulers collected under `optimizers/`
- **Experiment hub**: runnable demos and ablations inside `modules/`
- **Training scripts**: reproducible pipelines collected under `training/scripts/`
- **Results archive**: curated metrics, figures, and reports in `results/`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/intrinsicD/DeepLearning2.git
   cd DeepLearning2
   ```
2. (Recommended) create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Project layout

```
DeepLearning2/
├── architectures/             # Core architecture definitions
├── archive/                   # Historical notes, reports, and retired scripts
├── data/                      # Placeholder for locally downloaded datasets
├── datasets/                  # Dataset preparation utilities and scripts
│   └── scripts/
├── models/                    # Model wrappers and shared modules
├── modules/                   # Experiments, demos, and the nl_mm package
│   ├── demos/
│   ├── experiments/
│   └── nl_mm/
├── optimizers/                # Custom optimizers and scheduler utilities
├── results/
│   ├── analysis/              # Result inspection notebooks and scripts
│   ├── figures/               # Generated plots and visual assets
│   └── folder_per_model/      # Run artifacts grouped by model
├── tests/                     # Automated and manual test suites
├── training/
│   ├── diagnostics/           # Debugging helpers for training runs
│   └── scripts/               # Command-line entry points for experiments
├── utils/                     # Shared utility modules (device, data, viz, ...)
├── venv/                      # Local virtual environment (optional)
├── requirements.txt
└── README.md
```

## Quick start

Run the MNIST comparison experiment:
```bash
python -m modules.experiments.compare_architectures --epochs 3
```

Or launch the multimodal Flickr8k training pipeline:
```bash
python -m training.scripts.train_flickr8k --data-root /path/to/flickr8k
```

### Using utilities

```python
from architectures import SimpleCNN
from optimizers import CustomAdam
from utils import Trainer, get_device, get_mnist_loaders

model = SimpleCNN(input_channels=1, num_classes=10)
device = get_device()
optimizer = CustomAdam(model.parameters(), lr=1e-3)
trainer = Trainer(model=model, optimizer=optimizer, device=device)
train_loader, val_loader, _ = get_mnist_loaders(batch_size=64)
trainer.train(train_loader, epochs=5, val_loader=val_loader)
```

## License

MIT License

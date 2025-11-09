"""Test different MuonFast and AndersonGDA configurations on ViT."""

import torch
import torch.nn as nn
from src.architectures import VisionTransformer
from src.optimizers import AndersonGDA, MuonFast
from src.utils import get_device, get_mnist_loaders, Trainer

device = get_device()
print(f"Device: {device}\n")

train_loader, val_loader, test_loader = get_mnist_loaders(batch_size=128)
criterion = nn.CrossEntropyLoss()

configs = [
    {
        "name": "MuonFast (current config)",
        "optimizer": lambda model: MuonFast(
            model.parameters(),
            lr=0.0001,
            momentum=0.95,
            nesterov=True,
            model=model,
            scale_mode="spectral",
            scale_extra=0.1,
            ns_coefficients="simple",
            ns_tol=1e-3,
        ),
    },
    {
        "name": "MuonFast (lr=0.001, scale_extra=1.0)",
        "optimizer": lambda model: MuonFast(
            model.parameters(),
            lr=0.001,
            momentum=0.95,
            nesterov=True,
            model=model,
            scale_mode="spectral",
            scale_extra=1.0,
            ns_coefficients="simple",
            ns_tol=1e-3,
        ),
    },
    {
        "name": "MuonFast (lr=0.002, scale_extra=1.0)",
        "optimizer": lambda model: MuonFast(
            model.parameters(),
            lr=0.002,
            momentum=0.95,
            nesterov=True,
            model=model,
            scale_mode="spectral",
            scale_extra=1.0,
            ns_coefficients="simple",
            ns_tol=1e-3,
        ),
    },
    {
        "name": "AndersonGDA (lr=0.001)",
        "optimizer": lambda model: AndersonGDA(model.parameters(), lr=0.001),
    },
    {
        "name": "AndersonGDA (lr=0.01)",
        "optimizer": lambda model: AndersonGDA(model.parameters(), lr=0.01),
    },
    {
        "name": "AndersonGDA (lr=0.01, beta=0.5)",
        "optimizer": lambda model: AndersonGDA(model.parameters(), lr=0.01, beta=0.5),
    },
    {
        "name": "Adam (baseline)",
        "optimizer": lambda model: torch.optim.Adam(model.parameters(), lr=0.001),
    },
]

print("=" * 80)
print("TESTING DIFFERENT CONFIGURATIONS (2 epochs each)")
print("=" * 80)

results = []

for config in configs:
    print(f"\n{'='*80}")
    print(f"Testing: {config['name']}")
    print(f"{'='*80}")
    
    model = VisionTransformer(
        image_size=28,
        patch_size=4,
        in_channels=1,
        num_classes=10,
        embed_dim=64,
        depth=4,
        num_heads=4,
        mlp_dim=128,
        dropout=0.1,
    ).to(device)
    
    optimizer = config["optimizer"](model)
    
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
    )
    
    history = trainer.train(
        train_loader=train_loader,
        epochs=2,
        val_loader=val_loader,
    )
    
    test_loss, test_acc = trainer.validate(test_loader)
    
    results.append({
        "name": config["name"],
        "test_acc": test_acc,
        "test_loss": test_loss,
        "final_val_acc": history["val_acc"][-1],
        "final_val_loss": history["val_loss"][-1],
    })
    
    print(f"Results: Val Acc={history['val_acc'][-1]:.2f}%, Test Acc={test_acc:.2f}%")

print("\n" + "=" * 80)
print("SUMMARY OF RESULTS (2 epochs)")
print("=" * 80)
print(f"{'Configuration':<45} {'Val Acc':<12} {'Test Acc':<12} {'Test Loss':<12}")
print("-" * 80)

for result in results:
    print(f"{result['name']:<45} {result['final_val_acc']:<12.2f} {result['test_acc']:<12.2f} {result['test_loss']:<12.4f}")

print("=" * 80)


"""Test universal optimizers on multiple architectures.

Tests:
1. Vision Transformer (small matrices, attention)
2. ResNet-like CNN (convolutional layers)
3. Simple MLP (dense layers)
4. Mixed architecture (various layer types)
"""

import torch
import torch.nn as nn
from optimizers import UniversalMuon, UniversalAndersonGDA
from architectures import VisionTransformer, SimpleCNN, FullyConnectedNet
from utils import get_device, get_mnist_loaders, Trainer

device = get_device()
print(f"Device: {device}\n")

train_loader, val_loader, test_loader = get_mnist_loaders(batch_size=128)
criterion = nn.CrossEntropyLoss()

# Test configurations
configs = [
    # Vision Transformer (small matrices)
    {
        "name": "ViT",
        "model": VisionTransformer(
            image_size=28, patch_size=4, in_channels=1, num_classes=10,
            embed_dim=64, depth=4, num_heads=4, mlp_dim=128, dropout=0.1,
        ),
        "architecture_type": "transformer",
    },
    # CNN (convolutional layers)
    {
        "name": "CNN",
        "model": SimpleCNN(input_channels=1, num_classes=10),
        "architecture_type": "cnn",
    },
    # MLP (dense layers)
    {
        "name": "MLP",
        "model": FullyConnectedNet(input_size=784, hidden_sizes=[256, 128], num_classes=10),
        "architecture_type": "mlp",
    },
]

# Optimizer configurations
optimizer_configs = [
    {
        "name": "Adam (baseline)",
        "factory": lambda model: torch.optim.Adam(model.parameters(), lr=1e-3),
    },
    {
        "name": "UniversalMuon (auto)",
        "factory": lambda model: UniversalMuon(
            model.parameters(),
            lr=1e-3,
            ortho_mode="auto",  # Automatically decides when to orthogonalize
            ortho_threshold=128,
            scale_mode="adaptive",
        ),
    },
    {
        "name": "UniversalMuon (preserve_mag)",
        "factory": lambda model: UniversalMuon(
            model.parameters(),
            lr=1e-3,
            ortho_mode="preserve_magnitude",  # Orthogonalize but preserve magnitude
            ortho_threshold=64,
            scale_mode="adaptive",
        ),
    },
    {
        "name": "UniversalAndersonGDA",
        "factory": lambda model: UniversalAndersonGDA(
            model.parameters(),
            lr=1e-3,
            anderson_m=3,
            anderson_reg=1e-3,
            trust_region=1.5,
            use_weighting=True,
        ),
    },
]

print("=" * 80)
print("TESTING UNIVERSAL OPTIMIZERS ON MULTIPLE ARCHITECTURES")
print("=" * 80)

results = {}

for arch_config in configs:
    arch_name = arch_config["name"]
    arch_type = arch_config["architecture_type"]
    
    print(f"\n{'='*80}")
    print(f"ARCHITECTURE: {arch_name} ({arch_type})")
    print(f"{'='*80}\n")
    
    results[arch_name] = {}
    
    for opt_config in optimizer_configs:
        opt_name = opt_config["name"]
        
        print(f"Testing {opt_name} on {arch_name}...")
        
        # Recreate model properly
        if arch_name == "ViT":
            model = VisionTransformer(
                image_size=28, patch_size=4, in_channels=1, num_classes=10,
                embed_dim=64, depth=4, num_heads=4, mlp_dim=128, dropout=0.1,
            )
        elif arch_name == "CNN":
            model = SimpleCNN(input_channels=1, num_classes=10)
        else:  # MLP
            model = FullyConnectedNet(input_size=784, hidden_sizes=[256, 128], num_classes=10)

        model = model.to(device)
        
        # Create optimizer
        optimizer = opt_config["factory"](model)
        
        # Train for 3 epochs
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )
        
        try:
            history = trainer.train(
                train_loader=train_loader,
                epochs=3,
                val_loader=val_loader,
            )
            
            test_loss, test_acc = trainer.validate(test_loader)
            
            results[arch_name][opt_name] = {
                "test_acc": test_acc,
                "test_loss": test_loss,
                "final_val_acc": history["val_acc"][-1],
                "converged": test_acc > 50.0,  # Basic convergence check
                "status": "✅" if test_acc > 80.0 else ("⚠️" if test_acc > 50.0 else "❌"),
            }
            
            print(f"  → Val Acc: {history['val_acc'][-1]:.2f}%, Test Acc: {test_acc:.2f}% {results[arch_name][opt_name]['status']}")
            
        except Exception as e:
            print(f"  → FAILED: {str(e)[:100]}")
            results[arch_name][opt_name] = {
                "test_acc": 0.0,
                "test_loss": 999.0,
                "final_val_acc": 0.0,
                "converged": False,
                "status": "❌ CRASH",
                "error": str(e),
            }

print("\n" + "=" * 80)
print("SUMMARY: UNIVERSAL OPTIMIZERS ON ALL ARCHITECTURES")
print("=" * 80)

# Print summary table
for arch_name in results:
    print(f"\n{arch_name}:")
    print(f"{'Optimizer':<30} {'Val Acc':<12} {'Test Acc':<12} {'Status':<10}")
    print("-" * 70)
    
    for opt_name, result in results[arch_name].items():
        status = result.get("status", "❌")
        val_acc = result.get("final_val_acc", 0.0)
        test_acc = result.get("test_acc", 0.0)
        
        print(f"{opt_name:<30} {val_acc:<12.2f} {test_acc:<12.2f} {status:<10}")

# Check if universal optimizers work on all architectures
print("\n" + "=" * 80)
print("UNIVERSAL OPTIMIZER VERDICT")
print("=" * 80)

for opt_name in ["UniversalMuon (auto)", "UniversalMuon (preserve_mag)", "UniversalAndersonGDA"]:
    works_on_all = True
    architectures_tested = []
    
    for arch_name in results:
        if opt_name in results[arch_name]:
            result = results[arch_name][opt_name]
            converged = result.get("converged", False)
            architectures_tested.append(f"{arch_name}: {result['test_acc']:.1f}%")
            
            if not converged:
                works_on_all = False
    
    status = "✅ UNIVERSAL" if works_on_all else "⚠️ PARTIAL"
    print(f"\n{opt_name}: {status}")
    print(f"  Tested on: {', '.join(architectures_tested)}")

print("\n" + "=" * 80)
print("Key:")
print("  ✅ = >80% accuracy (excellent)")
print("  ⚠️  = 50-80% accuracy (works but needs tuning)")
print("  ❌ = <50% accuracy (failed)")
print("=" * 80)


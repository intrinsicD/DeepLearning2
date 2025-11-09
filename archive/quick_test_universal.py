"""Quick test of UniversalMuon on ViT (2 epochs)."""

import torch
import torch.nn as nn
from src.optimizers import UniversalMuon, UniversalAndersonGDA
from src.architectures import VisionTransformer
from src.utils import get_device, get_mnist_loaders, Trainer

device = get_device()
print(f"Device: {device}\n")

train_loader, val_loader, test_loader = get_mnist_loaders(batch_size=128)
criterion = nn.CrossEntropyLoss()

print("=" * 80)
print("QUICK TEST: Universal Optimizers on Vision Transformer")
print("=" * 80)

# Test configurations
tests = [
    ("Adam (baseline)", lambda m: torch.optim.Adam(m.parameters(), lr=1e-3)),
    ("UniversalMuon (auto)", lambda m: UniversalMuon(
        m.parameters(), lr=1e-3, ortho_mode="auto", ortho_threshold=128
    )),
    ("UniversalMuon (preserve)", lambda m: UniversalMuon(
        m.parameters(), lr=1e-3, ortho_mode="preserve_magnitude", ortho_threshold=128
    )),
    ("UniversalAndersonGDA", lambda m: UniversalAndersonGDA(
        m.parameters(), lr=1e-3, anderson_m=3, use_weighting=True
    )),
]

for name, opt_factory in tests:
    print(f"\n{'='*80}")
    print(f"Testing: {name}")
    print(f"{'='*80}")
    
    model = VisionTransformer(
        image_size=28, patch_size=4, in_channels=1, num_classes=10,
        embed_dim=64, depth=4, num_heads=4, mlp_dim=128, dropout=0.1,
    ).to(device)
    
    optimizer = opt_factory(model)
    
    trainer = Trainer(model=model, optimizer=optimizer, criterion=criterion, device=device)
    
    history = trainer.train(train_loader=train_loader, epochs=2, val_loader=val_loader)
    
    test_loss, test_acc = trainer.validate(test_loader)
    
    status = "✅" if test_acc > 80 else ("⚠️" if test_acc > 50 else "❌")
    print(f"\nResult: Val={history['val_acc'][-1]:.2f}%, Test={test_acc:.2f}% {status}")

print("\n" + "=" * 80)
print("If UniversalMuon and UniversalAndersonGDA get >80%, they're working!")
print("=" * 80)


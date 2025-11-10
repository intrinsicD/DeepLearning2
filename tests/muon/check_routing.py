"""Check which parameters go to Muon vs fallback."""

from architectures import SimpleCNN
from optimizers import MuonFast

model = SimpleCNN(input_channels=1, num_classes=10)

print("Creating MuonFast optimizer with default settings...")
optimizer = MuonFast(model.parameters(), lr=0.001, momentum=0.95)

print("\nMuon parameter groups:")
for i, group in enumerate(optimizer.param_groups):
    print(f"Group {i}:")
    for param in group['params']:
        for name, p in model.named_parameters():
            if p is param:
                print(f"  - {name}: shape {list(p.shape)}")
                break

print(f"\nFallback optimizer: {optimizer._fallback_opt}")
if optimizer._fallback_opt is not None:
    print("\nFallback parameter groups:")
    for i, group in enumerate(optimizer._fallback_opt.param_groups):
        print(f"Group {i}:")
        for param in group['params']:
            for name, p in model.named_parameters():
                if p is param:
                    print(f"  - {name}: shape {list(p.shape)}")
                    break


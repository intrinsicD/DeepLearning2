"""Test improved MuonFast with dimension-aware scaling on ViT."""

import torch
import torch.nn as nn
from src.architectures import VisionTransformer
from src.optimizers import MuonFast, AndersonGDA
from src.utils import get_device, get_mnist_loaders

device = get_device()
print(f"Device: {device}\n")

# Create ViT model
model = VisionTransformer(
    image_size=28, patch_size=4, in_channels=1, num_classes=10,
    embed_dim=64, depth=4, num_heads=4, mlp_dim=128, dropout=0.1,
).to(device)

# Test improved MuonFast
optimizer = MuonFast(
    model.parameters(),
    lr=0.002,
    momentum=0.95,
    nesterov=True,
    model=model,
    scale_mode="spectral",
    scale_extra=1.0,
    ns_coefficients="simple",
    ns_tol=1e-3,
)

train_loader, _, _ = get_mnist_loaders(batch_size=128)
criterion = nn.CrossEntropyLoss()

# Get one batch
data, target = next(iter(train_loader))
data, target = data.to(device), target.to(device)

print("=" * 80)
print("TESTING IMPROVED MUONFAST (with dimension-aware scaling)")
print("=" * 80)

# Store initial parameters
initial_params = {}
for name, param in model.named_parameters():
    initial_params[name] = param.data.clone()

# Do one training step
model.train()
optimizer.zero_grad()
output = model(data)
loss = criterion(output, target)
print(f"\nInitial loss: {loss.item():.4f}")

loss.backward()
optimizer.step()

# Check parameter updates
total_update_norm = 0.0
max_relative_update = 0.0
for name, param in model.named_parameters():
    update = param.data - initial_params[name]
    update_norm = update.norm().item()
    param_norm = param.data.norm().item()
    relative_update = update_norm / (param_norm + 1e-8)
    total_update_norm += update_norm
    max_relative_update = max(max_relative_update, relative_update)

print(f"\nTotal update norm: {total_update_norm:.4f}")
print(f"Max relative update: {max_relative_update:.4f}")

# Do forward pass again
output = model(data)
new_loss = criterion(output, target)
print(f"\nLoss after step: {new_loss.item():.4f}")
print(f"Loss change: {new_loss.item() - loss.item():.4f}")

if new_loss.item() < loss.item():
    print("✅ Loss DECREASED - Good!")
else:
    print("❌ Loss INCREASED - Still problematic")

# Compare with Adam baseline
print("\n" + "=" * 80)
print("ADAM BASELINE COMPARISON")
print("=" * 80)

model_adam = VisionTransformer(
    image_size=28, patch_size=4, in_channels=1, num_classes=10,
    embed_dim=64, depth=4, num_heads=4, mlp_dim=128, dropout=0.1,
).to(device)

optimizer_adam = torch.optim.Adam(model_adam.parameters(), lr=0.001)

initial_params_adam = {}
for name, param in model_adam.named_parameters():
    initial_params_adam[name] = param.data.clone()

model_adam.train()
optimizer_adam.zero_grad()
output = model_adam(data)
loss_adam = criterion(output, target)
print(f"\nInitial loss: {loss_adam.item():.4f}")

loss_adam.backward()
optimizer_adam.step()

total_update_norm_adam = 0.0
for name, param in model_adam.named_parameters():
    update = param.data - initial_params_adam[name]
    total_update_norm_adam += update.norm().item()

print(f"Total update norm: {total_update_norm_adam:.4f}")

output = model_adam(data)
new_loss_adam = criterion(output, target)
print(f"Loss after step: {new_loss_adam.item():.4f}")
print(f"Loss change: {new_loss_adam.item() - loss_adam.item():.4f}")

print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)
print(f"MuonFast update norm: {total_update_norm:.4f}")
print(f"Adam update norm:     {total_update_norm_adam:.4f}")
print(f"Ratio (Muon/Adam):    {total_update_norm / total_update_norm_adam:.2f}x")
print(f"\nMuonFast loss change: {new_loss.item() - loss.item():.4f}")
print(f"Adam loss change:     {new_loss_adam.item() - loss_adam.item():.4f}")

# Test improved AndersonGDA
print("\n" + "=" * 80)
print("TESTING IMPROVED ANDERSONGDA (lr=0.01, beta=0, m=2)")
print("=" * 80)

model_anderson = VisionTransformer(
    image_size=28, patch_size=4, in_channels=1, num_classes=10,
    embed_dim=64, depth=4, num_heads=4, mlp_dim=128, dropout=0.1,
).to(device)

optimizer_anderson = AndersonGDA(model_anderson.parameters(), lr=0.01, beta=0.0, m=2)

model_anderson.train()
optimizer_anderson.zero_grad()
output = model_anderson(data)
loss_anderson = criterion(output, target)
print(f"\nInitial loss: {loss_anderson.item():.4f}")

loss_anderson.backward()
optimizer_anderson.step()

output = model_anderson(data)
new_loss_anderson = criterion(output, target)
print(f"Loss after step: {new_loss_anderson.item():.4f}")
print(f"Loss change: {new_loss_anderson.item() - loss_anderson.item():.4f}")

if new_loss_anderson.item() < loss_anderson.item():
    print("✅ Loss DECREASED - Good!")
else:
    print("❌ Loss INCREASED - Still problematic")


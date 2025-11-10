"""Deep dive into what's wrong with MuonFast on ViT."""

import torch
import torch.nn as nn
from architectures import VisionTransformer
from optimizers import MuonFast
from utils import get_device, get_mnist_loaders

device = get_device()
print(f"Device: {device}\n")

# Create ViT model
model = VisionTransformer(
    image_size=28, patch_size=4, in_channels=1, num_classes=10,
    embed_dim=64, depth=4, num_heads=4, mlp_dim=128, dropout=0.1,
).to(device)

# Create MuonFast optimizer with better settings
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
print("PARAMETER UPDATE ANALYSIS")
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

# Check gradients before step
print("\nGradient statistics:")
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        grad_mean = param.grad.mean().item()
        grad_std = param.grad.std().item()
        print(f"  {name:50s}: norm={grad_norm:.4e}, mean={grad_mean:.4e}, std={grad_std:.4e}")

# Perform optimizer step
optimizer.step()

# Check parameter updates
print("\n" + "=" * 80)
print("PARAMETER UPDATE MAGNITUDES")
print("=" * 80)

total_update_norm = 0.0
for name, param in model.named_parameters():
    update = param.data - initial_params[name]
    update_norm = update.norm().item()
    param_norm = param.data.norm().item()
    relative_update = update_norm / (param_norm + 1e-8)
    total_update_norm += update_norm
    
    print(f"{name:50s}: update_norm={update_norm:.4e}, param_norm={param_norm:.4e}, relative={relative_update:.4e}")

print(f"\nTotal update norm: {total_update_norm:.4e}")

# Do forward pass again to see if loss changed
output = model(data)
new_loss = criterion(output, target)
print(f"\nLoss after step: {new_loss.item():.4f}")
print(f"Loss change: {new_loss.item() - loss.item():.4e}")

# Compare with Adam
print("\n" + "=" * 80)
print("COMPARISON WITH ADAM")
print("=" * 80)

model_adam = VisionTransformer(
    image_size=28, patch_size=4, in_channels=1, num_classes=10,
    embed_dim=64, depth=4, num_heads=4, mlp_dim=128, dropout=0.1,
).to(device)

optimizer_adam = torch.optim.Adam(model_adam.parameters(), lr=0.001)

# Store initial parameters
initial_params_adam = {}
for name, param in model_adam.named_parameters():
    initial_params_adam[name] = param.data.clone()

# Do one training step
model_adam.train()
optimizer_adam.zero_grad()
output = model_adam(data)
loss_adam = criterion(output, target)
print(f"\nInitial loss: {loss_adam.item():.4f}")

loss_adam.backward()
optimizer_adam.step()

# Check parameter updates
print("\nAdam parameter update magnitudes:")
total_update_norm_adam = 0.0
for name, param in model_adam.named_parameters():
    update = param.data - initial_params_adam[name]
    update_norm = update.norm().item()
    param_norm = param.data.norm().item()
    relative_update = update_norm / (param_norm + 1e-8)
    total_update_norm_adam += update_norm
    
    if 'encoder.layers.0' in name:  # Just show first layer for brevity
        print(f"  {name:50s}: update_norm={update_norm:.4e}, relative={relative_update:.4e}")

print(f"\nTotal update norm: {total_update_norm_adam:.4e}")

# Do forward pass again
output = model_adam(data)
new_loss_adam = criterion(output, target)
print(f"\nLoss after step: {new_loss_adam.item():.4f}")
print(f"Loss change: {new_loss_adam.item() - loss_adam.item():.4e}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"MuonFast total update norm: {total_update_norm:.4e}")
print(f"Adam total update norm:     {total_update_norm_adam:.4e}")
print(f"Ratio (MuonFast/Adam):      {total_update_norm / total_update_norm_adam:.4f}")


"""Diagnose why AndersonGDA and MuonFast perform poorly on ViT."""

import torch
import torch.nn as nn
from src.architectures import VisionTransformer
from src.optimizers import AndersonGDA, MuonFast
from src.utils import get_device, get_mnist_loaders

device = get_device()
print(f"Device: {device}\n")

# Create ViT model
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
)
model.to(device)

print("=" * 70)
print("MODEL PARAMETER ANALYSIS")
print("=" * 70)

# Analyze parameters
total_params = 0
muon_params = 0
anderson_params = 0

for name, param in model.named_parameters():
    numel = param.numel()
    shape = tuple(param.shape)
    ndim = param.ndim
    total_params += numel
    
    # Check if suitable for MuonFast (2D and min_dim >= 64)
    is_muon_eligible = ndim == 2 and min(shape) >= 64
    if is_muon_eligible:
        muon_params += numel
    
    print(f"{name:50s} shape={str(shape):20s} numel={numel:8d} ndim={ndim} muon={is_muon_eligible}")

print(f"\nTotal parameters: {total_params:,}")
print(f"MuonFast-eligible parameters: {muon_params:,} ({100*muon_params/total_params:.1f}%)")
print(f"Other parameters: {total_params - muon_params:,} ({100*(total_params-muon_params)/total_params:.1f}%)")

# Check MuonFast routing
print("\n" + "=" * 70)
print("MUONFAST ROUTING ANALYSIS")
print("=" * 70)

model_for_muon = VisionTransformer(
    image_size=28, patch_size=4, in_channels=1, num_classes=10,
    embed_dim=64, depth=4, num_heads=4, mlp_dim=128, dropout=0.1,
).to(device)

muon_optimizer = MuonFast(
    model_for_muon.parameters(),
    lr=0.0001,
    momentum=0.95,
    nesterov=True,
    model=model_for_muon,
    scale_mode="spectral",
    scale_extra=0.1,
    ns_coefficients="simple",
    ns_tol=1e-3,
    verbose=True,
)

print(f"\nNumber of parameter groups in MuonFast: {len(muon_optimizer.param_groups)}")
for i, group in enumerate(muon_optimizer.param_groups):
    print(f"  Group {i}: {len(group['params'])} parameters")

# Check if there's a fallback optimizer
if hasattr(muon_optimizer, '_fallback_opt') and muon_optimizer._fallback_opt is not None:
    print(f"\nFallback optimizer: {type(muon_optimizer._fallback_opt).__name__}")
    print(f"Fallback parameter groups: {len(muon_optimizer._fallback_opt.param_groups)}")
    for i, group in enumerate(muon_optimizer._fallback_opt.param_groups):
        print(f"  Fallback Group {i}: {len(group['params'])} parameters")
else:
    print("\nNo fallback optimizer configured!")

# Test a few training steps
print("\n" + "=" * 70)
print("TRAINING STEP ANALYSIS")
print("=" * 70)

train_loader, _, _ = get_mnist_loaders(batch_size=128)
criterion = nn.CrossEntropyLoss()

# Test AndersonGDA
print("\nTesting AndersonGDA...")
model_anderson = VisionTransformer(
    image_size=28, patch_size=4, in_channels=1, num_classes=10,
    embed_dim=64, depth=4, num_heads=4, mlp_dim=128, dropout=0.1,
).to(device)

anderson_opt = AndersonGDA(model_anderson.parameters(), lr=0.001)

model_anderson.train()
data, target = next(iter(train_loader))
data, target = data.to(device), target.to(device)

losses = []
for step in range(5):
    anderson_opt.zero_grad()
    output = model_anderson(data)
    loss = criterion(output, target)
    loss.backward()
    
    # Check gradient stats
    grad_norms = []
    for p in model_anderson.parameters():
        if p.grad is not None:
            grad_norms.append(p.grad.norm().item())
    
    anderson_opt.step()
    losses.append(loss.item())
    
    print(f"  Step {step+1}: loss={loss.item():.4f}, "
          f"grad_norm_mean={sum(grad_norms)/len(grad_norms):.4e}, "
          f"grad_norm_max={max(grad_norms):.4e}")

print(f"  Loss change: {losses[0]:.4f} -> {losses[-1]:.4f} (delta={losses[-1]-losses[0]:.4f})")

# Test MuonFast
print("\nTesting MuonFast...")
model_muon = VisionTransformer(
    image_size=28, patch_size=4, in_channels=1, num_classes=10,
    embed_dim=64, depth=4, num_heads=4, mlp_dim=128, dropout=0.1,
).to(device)

muon_opt = MuonFast(
    model_muon.parameters(),
    lr=0.0001,
    momentum=0.95,
    nesterov=True,
    model=model_muon,
    scale_mode="spectral",
    scale_extra=0.1,
    ns_coefficients="simple",
    ns_tol=1e-3,
)

model_muon.train()
losses = []
for step in range(5):
    muon_opt.zero_grad()
    output = model_muon(data)
    loss = criterion(output, target)
    loss.backward()
    
    # Check gradient stats
    grad_norms = []
    for p in model_muon.parameters():
        if p.grad is not None:
            grad_norms.append(p.grad.norm().item())
    
    muon_opt.step()
    losses.append(loss.item())
    
    print(f"  Step {step+1}: loss={loss.item():.4f}, "
          f"grad_norm_mean={sum(grad_norms)/len(grad_norms):.4e}, "
          f"grad_norm_max={max(grad_norms):.4e}")

print(f"  Loss change: {losses[0]:.4f} -> {losses[-1]:.4f} (delta={losses[-1]-losses[0]:.4f})")

# Test standard Adam for comparison
print("\nTesting Adam (baseline)...")
model_adam = VisionTransformer(
    image_size=28, patch_size=4, in_channels=1, num_classes=10,
    embed_dim=64, depth=4, num_heads=4, mlp_dim=128, dropout=0.1,
).to(device)

adam_opt = torch.optim.Adam(model_adam.parameters(), lr=0.001)

model_adam.train()
losses = []
for step in range(5):
    adam_opt.zero_grad()
    output = model_adam(data)
    loss = criterion(output, target)
    loss.backward()
    
    adam_opt.step()
    losses.append(loss.item())
    
    print(f"  Step {step+1}: loss={loss.item():.4f}")

print(f"  Loss change: {losses[0]:.4f} -> {losses[-1]:.4f} (delta={losses[-1]-losses[0]:.4f})")


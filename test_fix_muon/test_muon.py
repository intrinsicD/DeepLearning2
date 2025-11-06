"""Test script to debug MuonFast optimizer."""

import torch
import torch.nn as nn
from src.optimizers.muon_fast import MuonFast
from src.architectures import SimpleCNN

# Create a simple model
model = SimpleCNN(input_channels=1, num_classes=10)

print("Model architecture:")
print(model)
print("\n" + "="*60)

# Check parameter shapes
print("\nParameter shapes in the model:")
for name, param in model.named_parameters():
    print(f"{name:40s} shape={str(list(param.shape)):20s} ndim={param.ndim}")

print("\n" + "="*60)

# Try to create the optimizer
try:
    optimizer = MuonFast(model.parameters(), lr=0.001, momentum=0.95)
    print("\n✓ Optimizer created successfully")
    print(f"Muon parameter groups: {len(optimizer.param_groups)}")
    print(f"Fallback optimizer: {optimizer._fallback_opt}")
    
    # Count parameters
    muon_params = sum(len(group['params']) for group in optimizer.param_groups)
    print(f"Number of Muon parameters (2D): {muon_params}")
    
    if optimizer._fallback_opt:
        fallback_params = sum(len(group['params']) for group in optimizer._fallback_opt.param_groups)
        print(f"Number of fallback parameters (non-2D): {fallback_params}")
    
    # Test one optimization step
    print("\n" + "="*60)
    print("Testing one optimization step...")
    
    # Create dummy input and target
    dummy_input = torch.randn(2, 1, 28, 28)
    dummy_target = torch.randint(0, 10, (2,))
    
    # Forward pass
    output = model(dummy_input)
    loss = nn.CrossEntropyLoss()(output, dummy_target)
    
    print(f"Loss: {loss.item():.4f}")
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Check gradients
    print("\nGradients after backward:")
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name:40s} grad_shape={str(list(param.grad.shape)):20s} grad_norm={param.grad.norm().item():.4f}")
        else:
            print(f"{name:40s} NO GRADIENT")
    
    # Optimization step
    import time
    start = time.time()
    optimizer.step()
    end = time.time()
    
    print(f"\n✓ Optimization step completed in {(end-start)*1000:.2f}ms")
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()


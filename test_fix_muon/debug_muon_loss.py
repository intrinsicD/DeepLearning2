"""Debug why MuonFast is producing huge losses."""

import torch
import torch.nn as nn
from src.architectures import SimpleCNN
from src.optimizers.muon_fast import MuonFast

# Create a simple test
model = SimpleCNN(input_channels=1, num_classes=10)
optimizer = MuonFast(model.parameters(), lr=0.001, momentum=0.95)

# Dummy data
data = torch.randn(4, 1, 28, 28)
target = torch.randint(0, 10, (4,))

criterion = nn.CrossEntropyLoss()

print("Testing MuonFast for numerical issues...\n")

for step in range(5):
    # Forward
    output = model(data)
    loss = criterion(output, target)
    
    print(f"Step {step+1}:")
    print(f"  Loss: {loss.item():.4f}")
    
    # Check parameter values before update
    fc1_weight_norm = model.fc1.weight.norm().item()
    fc2_weight_norm = model.fc2.weight.norm().item()
    print(f"  FC1 weight norm: {fc1_weight_norm:.4f}")
    print(f"  FC2 weight norm: {fc2_weight_norm:.4f}")
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    
    # Check gradient values
    fc1_grad_norm = model.fc1.weight.grad.norm().item()
    fc2_grad_norm = model.fc2.weight.grad.norm().item()
    print(f"  FC1 grad norm: {fc1_grad_norm:.4f}")
    print(f"  FC2 grad norm: {fc2_grad_norm:.4f}")
    
    # Check momentum buffer before update
    state_fc1 = optimizer.state.get(model.fc1.weight, {})
    state_fc2 = optimizer.state.get(model.fc2.weight, {})
    
    if 'momentum_buffer' in state_fc1:
        buf_norm = state_fc1['momentum_buffer'].norm().item()
        print(f"  FC1 momentum buffer norm: {buf_norm:.4f}")
    if 'momentum_buffer' in state_fc2:
        buf_norm = state_fc2['momentum_buffer'].norm().item()
        print(f"  FC2 momentum buffer norm: {buf_norm:.4f}")
    
    # Step
    optimizer.step()
    
    # Check parameter values after update
    fc1_weight_norm_after = model.fc1.weight.norm().item()
    fc2_weight_norm_after = model.fc2.weight.norm().item()
    print(f"  FC1 weight norm after: {fc1_weight_norm_after:.4f}")
    print(f"  FC2 weight norm after: {fc2_weight_norm_after:.4f}")
    
    # Check for NaN or Inf
    if torch.isnan(model.fc1.weight).any() or torch.isinf(model.fc1.weight).any():
        print("  ⚠️  WARNING: NaN or Inf in FC1 weights!")
        break
    if torch.isnan(model.fc2.weight).any() or torch.isinf(model.fc2.weight).any():
        print("  ⚠️  WARNING: NaN or Inf in FC2 weights!")
        break
    
    print()


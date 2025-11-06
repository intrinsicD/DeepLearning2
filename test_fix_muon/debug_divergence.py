"""Debug why MuonFast diverges after epoch 2."""

import torch
import torch.nn as nn
from src.architectures import SimpleCNN
from src.utils import get_mnist_loaders, get_device
from src.optimizers import MuonFast

def debug_divergence():
    """Track what happens during training to find divergence."""
    
    device = get_device()
    train_loader, val_loader, _ = get_mnist_loaders(batch_size=64)
    
    model = SimpleCNN(input_channels=1, num_classes=10).to(device)
    optimizer = MuonFast(model.parameters(), lr=0.001, momentum=0.95)
    criterion = nn.CrossEntropyLoss()
    
    print("Tracking parameter norms and gradients...\n")
    
    for epoch in range(3):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Track norms at start of epoch
        fc1_norm_start = model.fc1.weight.norm().item()
        fc2_norm_start = model.fc2.weight.norm().item()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Check gradients before step
            fc1_grad_norm = model.fc1.weight.grad.norm().item()
            fc2_grad_norm = model.fc2.weight.grad.norm().item()
            
            # Check momentum buffer
            fc1_buf_norm = 0
            fc2_buf_norm = 0
            if model.fc1.weight in optimizer.state:
                if 'momentum_buffer' in optimizer.state[model.fc1.weight]:
                    fc1_buf_norm = optimizer.state[model.fc1.weight]['momentum_buffer'].norm().item()
            if model.fc2.weight in optimizer.state:
                if 'momentum_buffer' in optimizer.state[model.fc2.weight]:
                    fc2_buf_norm = optimizer.state[model.fc2.weight]['momentum_buffer'].norm().item()
            
            optimizer.step()
            
            # Check for NaN/Inf
            if torch.isnan(model.fc1.weight).any() or torch.isinf(model.fc1.weight).any():
                print(f"\n⚠️  Epoch {epoch+1}, Batch {batch_idx}: NaN/Inf in FC1!")
                print(f"  Loss: {loss.item():.4f}")
                print(f"  FC1 grad norm: {fc1_grad_norm:.4f}")
                print(f"  FC1 buffer norm: {fc1_buf_norm:.4f}")
                return
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Print diagnostic info for first few batches of each epoch
            if batch_idx < 3:
                print(f"Epoch {epoch+1}, Batch {batch_idx}:")
                print(f"  Loss: {loss.item():.4f}")
                print(f"  FC1 grad: {fc1_grad_norm:.4f}, FC2 grad: {fc2_grad_norm:.4f}")
                print(f"  FC1 buffer: {fc1_buf_norm:.4f}, FC2 buffer: {fc2_buf_norm:.4f}")
                print(f"  FC1 weight norm: {model.fc1.weight.norm().item():.4f}")
                print(f"  FC2 weight norm: {model.fc2.weight.norm().item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        
        fc1_norm_end = model.fc1.weight.norm().item()
        fc2_norm_end = model.fc2.weight.norm().item()
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        print(f"  FC1 norm: {fc1_norm_start:.4f} → {fc1_norm_end:.4f} (change: {fc1_norm_end - fc1_norm_start:+.4f})")
        print(f"  FC2 norm: {fc2_norm_start:.4f} → {fc2_norm_end:.4f} (change: {fc2_norm_end - fc2_norm_start:+.4f})")
        print()

if __name__ == "__main__":
    debug_divergence()


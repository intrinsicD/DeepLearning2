"""Quick training test to verify MuonFast works correctly."""

import torch
import torch.nn as nn
from src.architectures import SimpleCNN
from src.utils import get_mnist_loaders, get_device
from src.optimizers import MuonFast, CustomAdam
import time

def quick_training_test(epochs=2):
    """Quick test to compare MuonFast with Adam."""
    
    device = get_device()
    print(f"Using device: {device}\n")
    
    # Load a small subset of data
    train_loader, val_loader, _ = get_mnist_loaders(batch_size=128)
    
    results = {}
    
    for opt_name, opt_class, opt_kwargs in [
        ("Adam", torch.optim.Adam, {"lr": 0.001}),
        ("MuonFast", MuonFast, {"lr": 0.001, "momentum": 0.95}),
    ]:
        print("=" * 60)
        print(f"Testing {opt_name}")
        print("=" * 60)
        
        # Create model
        model = SimpleCNN(input_channels=1, num_classes=10).to(device)
        optimizer = opt_class(model.parameters(), **opt_kwargs)
        criterion = nn.CrossEntropyLoss()
        
        # Training
        epoch_times = []
        losses = []
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            start_time = time.time()
            
            # Train on first 100 batches only for speed
            for batch_idx, (data, target) in enumerate(train_loader):
                if batch_idx >= 100:
                    break
                    
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
            
            epoch_time = time.time() - start_time
            epoch_times.append(epoch_time)
            avg_loss = epoch_loss / min(100, len(train_loader))
            accuracy = 100.0 * correct / total
            losses.append(avg_loss)
            
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Loss={avg_loss:.4f}, "
                  f"Acc={accuracy:.2f}%, "
                  f"Time={epoch_time:.2f}s")
        
        avg_time = sum(epoch_times) / len(epoch_times)
        final_loss = losses[-1]
        
        results[opt_name] = {
            "avg_epoch_time": avg_time,
            "final_loss": final_loss,
            "all_losses": losses,
        }
        
        print(f"\nAverage epoch time: {avg_time:.2f}s")
        print(f"Final loss: {final_loss:.4f}\n")
    
    # Compare results
    print("=" * 60)
    print("COMPARISON")
    print("=" * 60)
    
    adam_time = results["Adam"]["avg_epoch_time"]
    muon_time = results["MuonFast"]["avg_epoch_time"]
    
    print(f"Adam average epoch time:     {adam_time:.2f}s")
    print(f"MuonFast average epoch time: {muon_time:.2f}s")
    print(f"Speed ratio: {muon_time/adam_time:.2f}x")
    
    print(f"\nAdam final loss:     {results['Adam']['final_loss']:.4f}")
    print(f"MuonFast final loss: {results['MuonFast']['final_loss']:.4f}")
    
    # Check if MuonFast is competitive
    if muon_time < adam_time * 3:  # Allow some overhead
        print("\n✓ MuonFast speed is COMPETITIVE!")
    else:
        print(f"\n✗ MuonFast is still too slow ({muon_time/adam_time:.1f}x slower)")
    
    if results['MuonFast']['final_loss'] < results['Adam']['final_loss'] * 2:
        print("✓ MuonFast convergence looks REASONABLE!")
    else:
        print("✗ MuonFast convergence may have issues")

if __name__ == "__main__":
    quick_training_test(epochs=2)


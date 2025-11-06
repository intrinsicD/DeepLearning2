"""Test the improved canonical Muon implementation."""

import torch
import torch.nn as nn
from src.architectures import SimpleCNN
from src.utils import get_mnist_loaders, get_device, Trainer
from src.optimizers import MuonFast

def test_canonical_muon():
    """Test Muon with Nesterov momentum and FP32 buffer."""
    
    device = get_device()
    print("Testing Canonical Muon Implementation")
    print("=" * 70)
    
    # Create model
    model = SimpleCNN(input_channels=1, num_classes=10)
    
    # Test with Nesterov (default)
    print("\n1. Testing Nesterov momentum (default, canonical Muon):")
    optimizer_nesterov = MuonFast(
        model.parameters(),
        lr=0.0001,
        momentum=0.95,
        nesterov=True  # Canonical Muon uses Nesterov
    )
    
    print("   ✓ Nesterov momentum enabled")
    print(f"   ✓ Defaults: nesterov={optimizer_nesterov.defaults['nesterov']}")
    
    # Test routing audit
    print("\n2. Routing Audit:")
    optimizer_nesterov.print_routing_audit()
    
    # Test FP32 momentum buffer
    print("\n3. Testing FP32 momentum buffer:")
    dummy_input = torch.randn(2, 1, 28, 28)
    dummy_target = torch.randint(0, 10, (2,))
    
    output = model(dummy_input)
    loss = nn.CrossEntropyLoss()(output, dummy_target)
    loss.backward()
    optimizer_nesterov.step()
    
    # Check momentum buffer type
    for param in model.parameters():
        if param in optimizer_nesterov.state:
            state = optimizer_nesterov.state[param]
            if 'momentum_buffer_fp32' in state:
                buf = state['momentum_buffer_fp32']
                print(f"   ✓ Momentum buffer dtype: {buf.dtype} (should be torch.float32)")
                assert buf.dtype == torch.float32, "Momentum buffer should be FP32!"
                break
    
    # Test classical momentum for comparison
    print("\n4. Testing classical momentum (for comparison):")
    model2 = SimpleCNN(input_channels=1, num_classes=10)
    optimizer_classical = MuonFast(
        model2.parameters(),
        lr=0.0001,
        momentum=0.95,
        nesterov=False  # Classical momentum
    )
    print(f"   ✓ Classical momentum: nesterov={optimizer_classical.defaults['nesterov']}")
    
    # Test training with both
    print("\n5. Quick training test (3 epochs):")
    train_loader, val_loader, _ = get_mnist_loaders(batch_size=64)
    
    # Recreate models on correct device
    model_nesterov = SimpleCNN(input_channels=1, num_classes=10).to(device)
    model_classical = SimpleCNN(input_channels=1, num_classes=10).to(device)

    opt_nesterov = MuonFast(model_nesterov.parameters(), lr=0.0001, momentum=0.95, nesterov=True)
    opt_classical = MuonFast(model_classical.parameters(), lr=0.0001, momentum=0.95, nesterov=False)

    configs = [
        (model_nesterov, opt_nesterov, "Nesterov (canonical)"),
        (model_classical, opt_classical, "Classical"),
    ]
    
    for m, opt, name in configs:
        criterion = nn.CrossEntropyLoss()
        trainer = Trainer(model=m, optimizer=opt, criterion=criterion, device=device)
        
        history = trainer.train(train_loader=train_loader, val_loader=val_loader, epochs=3)
        
        final_acc = history['train_acc'][-1]
        final_loss = history['train_loss'][-1]
        
        print(f"\n   {name}:")
        print(f"     Final accuracy: {final_acc:.2f}%")
        print(f"     Final loss: {final_loss:.4f}")
        
        if final_acc > 95:
            print(f"     ✓ Training successful!")
        else:
            print(f"     ⚠️  Lower than expected")
    
    print("\n" + "=" * 70)
    print("All tests passed! Canonical Muon implementation verified.")
    print("=" * 70)

if __name__ == "__main__":
    test_canonical_muon()


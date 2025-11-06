"""
Template for creating custom neural network architectures.

Copy this file and modify it to create your own custom architecture.
"""

import torch.nn as nn
import torch.nn.functional as F
from src.architectures.base import BaseArchitecture


class CustomArchitectureTemplate(BaseArchitecture):
    """
    Template for custom architecture.
    
    Replace this with your architecture description.
    """
    
    def __init__(self, input_channels=1, num_classes=10, **kwargs):
        """
        Initialize your custom architecture.
        
        Args:
            input_channels (int): Number of input channels
            num_classes (int): Number of output classes
            **kwargs: Additional architecture-specific parameters
        """
        super(CustomArchitectureTemplate, self).__init__()
        
        # Define your layers here
        # Example:
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        """
        Define the forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # Example forward pass
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        x = x.view(x.size(0), -1)  # Flatten
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


# Example usage:
if __name__ == '__main__':
    import torch
    
    # Create model
    model = CustomArchitectureTemplate(input_channels=1, num_classes=10)
    
    # Print model info
    model.print_model_info()
    
    # Test forward pass
    dummy_input = torch.randn(4, 1, 28, 28)
    output = model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

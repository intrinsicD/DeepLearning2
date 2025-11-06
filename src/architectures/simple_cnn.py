"""
Simple Convolutional Neural Network architecture.
"""

import torch.nn as nn
import torch.nn.functional as F
from .base import BaseArchitecture


class SimpleCNN(BaseArchitecture):
    """
    A simple CNN for image classification.
    
    Architecture:
    - 2 Convolutional layers with ReLU and MaxPooling
    - 2 Fully connected layers
    """
    
    def __init__(self, input_channels=1, num_classes=10):
        """
        Initialize SimpleCNN.
        
        Args:
            input_channels (int): Number of input channels (1 for grayscale, 3 for RGB)
            num_classes (int): Number of output classes
        """
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Assuming 28x28 input -> 7x7 after pooling
        self.fc2 = nn.Linear(128, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Conv layers with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

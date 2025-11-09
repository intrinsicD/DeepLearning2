"""
Fully Connected Neural Network architecture.
"""

import torch.nn as nn
from .base import BaseArchitecture


class FullyConnectedNet(BaseArchitecture):
    """
    A flexible fully connected neural network.
    
    Allows specification of arbitrary hidden layer sizes.
    """
    
    def __init__(self, input_size, hidden_sizes, num_classes, dropout_rate=0.5):
        """
        Initialize FullyConnectedNet.
        
        Args:
            input_size (int): Size of input features
            hidden_sizes (list): List of hidden layer sizes
            num_classes (int): Number of output classes
            dropout_rate (float): Dropout probability
        """
        super(FullyConnectedNet, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Build hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Flatten input if needed
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        
        return self.network(x)

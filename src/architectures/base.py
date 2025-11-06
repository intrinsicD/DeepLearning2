"""
Base architecture class for all neural network models.
"""

import torch.nn as nn


class BaseArchitecture(nn.Module):
    """
    Base class for all neural network architectures.
    
    All custom architectures should inherit from this class and implement
    the forward method.
    """
    
    def __init__(self):
        super(BaseArchitecture, self).__init__()
    
    def forward(self, x):
        """
        Forward pass of the network.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        raise NotImplementedError("Subclasses must implement forward method")
    
    def get_num_parameters(self):
        """
        Get the total number of trainable parameters in the model.
        
        Returns:
            int: Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def print_model_info(self):
        """Print model architecture and parameter count."""
        print(f"\n{self.__class__.__name__} Architecture:")
        print("=" * 60)
        print(self)
        print("=" * 60)
        print(f"Total trainable parameters: {self.get_num_parameters():,}")
        print("=" * 60)

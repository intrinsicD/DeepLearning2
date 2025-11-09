"""
Template for creating custom optimizers.

Copy this file and modify it to create your own custom optimizer.
"""

import torch
from torch.optim import Optimizer


class CustomOptimizerTemplate(Optimizer):
    """
    Template for custom optimizer implementation.
    
    Replace this with your optimizer description.
    """
    
    def __init__(self, params, lr=0.001, custom_param=0.9):
        """
        Initialize your custom optimizer.
        
        Args:
            params: Model parameters to optimize
            lr (float): Learning rate
            custom_param (float): Your custom parameter
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        
        # Define default hyperparameters
        defaults = dict(lr=lr, custom_param=custom_param)
        super(CustomOptimizerTemplate, self).__init__(params, defaults)
    
    def step(self, closure=None):
        """
        Perform a single optimization step.
        
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
                
        Returns:
            loss (optional): The loss value if closure is provided
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        # Iterate over parameter groups
        for group in self.param_groups:
            lr = group['lr']
            custom_param = group['custom_param']
            
            # Iterate over parameters
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Get gradient
                grad = p.grad.data
                
                # Get or initialize optimizer state for this parameter
                param_state = self.state[p]
                
                # Initialize state on first step
                if len(param_state) == 0:
                    param_state['step'] = 0
                    # Add any state variables you need
                    param_state['momentum_buffer'] = torch.zeros_like(p.data)
                
                # Increment step counter
                param_state['step'] += 1
                
                # Implement your optimization logic here
                # Example: Simple gradient descent with momentum
                momentum_buffer = param_state['momentum_buffer']
                momentum_buffer.mul_(custom_param).add_(grad)
                
                # Update parameters
                p.data.add_(momentum_buffer, alpha=-lr)
        
        return loss


# Example usage:
if __name__ == '__main__':
    import torch.nn as nn
    
    # Create a simple model
    model = nn.Linear(10, 5)
    
    # Create optimizer
    optimizer = CustomOptimizerTemplate(model.parameters(), lr=0.01, custom_param=0.9)
    
    print("Custom optimizer created successfully!")
    print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
    print(f"Custom parameter: {optimizer.param_groups[0]['custom_param']}")
    
    # Test optimization step
    dummy_input = torch.randn(4, 10)
    dummy_target = torch.randn(4, 5)
    
    # Forward pass
    output = model(dummy_input)
    loss = nn.MSELoss()(output, dummy_target)
    
    print(f"\nInitial loss: {loss.item():.4f}")
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Check new loss
    output = model(dummy_input)
    new_loss = nn.MSELoss()(output, dummy_target)
    print(f"Loss after one step: {new_loss.item():.4f}")

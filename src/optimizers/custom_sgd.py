"""
Custom SGD optimizer with momentum.
"""

import torch
from torch.optim import Optimizer


class CustomSGD(Optimizer):
    """
    Custom implementation of Stochastic Gradient Descent with momentum.
    
    This is an example of how to create custom optimizers that can be
    tested and compared with standard PyTorch optimizers.
    """
    
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        """
        Initialize CustomSGD optimizer.
        
        Args:
            params: Model parameters to optimize
            lr (float): Learning rate
            momentum (float): Momentum factor
            weight_decay (float): L2 regularization factor
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super(CustomSGD, self).__init__(params, defaults)
    
    def step(self, closure=None):
        """
        Perform a single optimization step.
        
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                
                # Add weight decay
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)
                
                # Apply momentum
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(grad)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(grad)
                    grad = buf
                
                # Update parameters
                p.data.add_(grad, alpha=-lr)
        
        return loss

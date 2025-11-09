"""
Custom Adam optimizer implementation.
"""

import math
import torch
from torch.optim import Optimizer


class CustomAdam(Optimizer):
    """
    Custom implementation of Adam optimizer.
    
    Adam: A Method for Stochastic Optimization
    (https://arxiv.org/abs/1412.6980)
    """
    
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        """
        Initialize CustomAdam optimizer.
        
        Args:
            params: Model parameters to optimize
            lr (float): Learning rate
            betas (tuple): Coefficients for computing running averages of gradient
                          and its square
            eps (float): Term added to denominator for numerical stability
            weight_decay (float): L2 regularization factor
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(CustomAdam, self).__init__(params, defaults)
    
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
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                
                # Add weight decay
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)
                
                param_state = self.state[p]
                
                # State initialization
                if len(param_state) == 0:
                    param_state['step'] = 0
                    # Exponential moving average of gradient values
                    param_state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    param_state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = param_state['exp_avg'], param_state['exp_avg_sq']
                param_state['step'] += 1
                
                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # Update biased second raw moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Compute bias-corrected first moment estimate
                bias_correction1 = 1 - beta1 ** param_state['step']
                # Compute bias-corrected second raw moment estimate
                bias_correction2 = 1 - beta2 ** param_state['step']
                
                step_size = lr / bias_correction1
                bias_correction2_sqrt = math.sqrt(bias_correction2)
                
                # Update parameters
                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
        
        return loss

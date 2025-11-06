"""
Training utilities and trainer class.
"""

import torch
import torch.nn as nn
from tqdm import tqdm
import time


class Trainer:
    """
    Trainer class for training neural networks.
    
    Handles the training loop, validation, and provides utilities for
    monitoring training progress.
    """
    
    def __init__(self, model, optimizer, criterion, device='cuda', 
                 scheduler=None, gradient_clip=None):
        """
        Initialize Trainer.
        
        Args:
            model (nn.Module): Neural network model
            optimizer: Optimizer instance
            criterion: Loss function
            device (str or torch.device): Device to train on
            scheduler: Learning rate scheduler (optional)
            gradient_clip (float): Max gradient norm for clipping (optional)
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = torch.device(device) if isinstance(device, str) else device
        self.scheduler = scheduler
        self.gradient_clip = gradient_clip
        
        # Move model to device
        self.model.to(self.device)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
        }
    
    def train_epoch(self, train_loader):
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            
        Returns:
            tuple: (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc='Training', leave=False)
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                              self.gradient_clip)
            
            # Optimizer step
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader):
        """
        Validate the model.
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            tuple: (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc='Validation', leave=False):
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # Statistics
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader, epochs, val_loader=None, verbose=True):
        """
        Train the model for multiple epochs.
        
        Args:
            train_loader: DataLoader for training data
            epochs (int): Number of epochs to train
            val_loader: DataLoader for validation data (optional)
            verbose (bool): Print training progress
            
        Returns:
            dict: Training history
        """
        print(f"\nTraining on device: {self.device}")
        print(f"Number of epochs: {epochs}")
        print("=" * 60)
        
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            
            # Validate
            if val_loader is not None:
                val_loss, val_acc = self.validate(val_loader)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
            
            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()
            
            epoch_time = time.time() - start_time
            
            if verbose:
                print(f"\nEpoch {epoch}/{epochs} - {epoch_time:.2f}s")
                print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
                if val_loader is not None:
                    print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
                if self.scheduler is not None:
                    print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
        
        print("\n" + "=" * 60)
        print("Training completed!")
        print("=" * 60)
        
        return self.history
    
    def save_checkpoint(self, filepath, epoch=None, **kwargs):
        """
        Save model checkpoint.
        
        Args:
            filepath (str): Path to save checkpoint
            epoch (int): Current epoch number
            **kwargs: Additional data to save
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
        }
        
        if epoch is not None:
            checkpoint['epoch'] = epoch
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        checkpoint.update(kwargs)
        
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath):
        """
        Load model checkpoint.
        
        Args:
            filepath (str): Path to checkpoint file
            
        Returns:
            dict: Checkpoint data
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Checkpoint loaded from {filepath}")
        return checkpoint

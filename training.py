import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional, List
from models import BinaryClassifier

class Trainer:
    def __init__(self, model: nn.Module, learning_rate: float = 1.0):
        self.model = model
        self.optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        # Use BCELoss for binary classification if the model is a BinaryClassifier
        self.is_binary = isinstance(model, BinaryClassifier)
        self.criterion = nn.BCELoss() if self.is_binary else nn.MSELoss()
        self.batch_gradients = {}
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []
        
    def training_step(self, batch_x: torch.Tensor, batch_y: torch.Tensor) -> Tuple[float, Dict]:
        """Perform a single training step and return loss and gradients."""
        self.optimizer.zero_grad()
        
        # Forward pass on batch
        outputs = self.model(batch_x)
        if self.is_binary:
            # Ensure targets are float for BCE loss
            batch_y = batch_y.float()
        batch_loss = self.criterion(outputs, batch_y)
        
        # Backward pass
        batch_loss.backward()
        
        # Store gradients before optimizer step
        gradients = {
            name: param.grad.clone() 
            for name, param in self.model.named_parameters()
        }
        
        self.optimizer.step()
        
        # Compute full training loss (without gradients)
        with torch.no_grad():
            full_outputs = self.model(batch_x)
            batch_loss = self.criterion(full_outputs, batch_y)
        
        return batch_loss.item(), gradients
    
    def validate(self, val_x: torch.Tensor, val_y: torch.Tensor) -> float:
        """Compute validation loss."""
        with torch.no_grad():
            outputs = self.model(val_x)
            if self.is_binary:
                val_y = val_y.float()
            loss = self.criterion(outputs, val_y)
        return loss.item()
    
    def train_epoch(self, train_data: Tuple[torch.Tensor, torch.Tensor], 
                   val_data: Tuple[torch.Tensor, torch.Tensor],
                   batch_size: int) -> Tuple[float, float, bool]:
        """Train for one epoch and return training loss, validation loss, and whether to stop."""
        x, y = train_data
        val_x, val_y = val_data
        indices = torch.randperm(len(x))
        total_loss = 0
        n_batches = 0
        
        for i in range(0, len(x), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_x = x[batch_indices]
            batch_y = y[batch_indices]
            
            loss, _ = self.training_step(batch_x, batch_y)
            total_loss += loss
            n_batches += 1
        
        train_loss = total_loss / n_batches
        val_loss = self.validate(val_x, val_y)
        
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        
        # Early stopping logic
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            
        should_stop = self.patience_counter >= 10
        
        return train_loss, val_loss, should_stop 

    def compute_loss(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Compute loss on full dataset."""
        with torch.no_grad():
            outputs = self.model(x)
            loss = self.criterion(outputs, y)
        return loss.item() 
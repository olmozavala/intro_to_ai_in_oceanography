import torch
import numpy as np
from typing import Tuple

def split_data(x: torch.Tensor, y: torch.Tensor, 
               train_ratio=0.7, val_ratio=0.15) -> Tuple[Tuple[torch.Tensor, torch.Tensor], 
                                                       Tuple[torch.Tensor, torch.Tensor],
                                                       Tuple[torch.Tensor, torch.Tensor]]:
    """Split data into training, validation and test sets."""
    n = len(x)
    indices = torch.randperm(n)
    
    train_size = int(train_ratio * n)
    val_size = int(val_ratio * n)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    return (x[train_indices], y[train_indices]), \
           (x[val_indices], y[val_indices]), \
           (x[test_indices], y[test_indices])

def generate_linear_data(n_points=100, noise_std=5.0):
    """Generate synthetic linear data with noise."""
    x = torch.linspace(-5, 5, n_points).reshape(-1, 1)
    y = 2 * x + 1
    y += torch.randn_like(y) * noise_std
    return split_data(x, y)

def generate_nonlinear_data(n_points=100, noise_std=0.3):
    """Generate synthetic nonlinear data with noise."""
    x = torch.linspace(-5, 5, n_points).reshape(-1, 1)
    y = torch.sin(x) + x**2 * 0.1
    y += torch.randn_like(y) * noise_std
    return split_data(x, y)

def generate_classification_data(n_points=100, noise_std=1.0):
    """Generate synthetic classification data from two Gaussians."""
    # Generate first Gaussian (class 0)
    n_points_per_class = n_points // 2
    x0 = torch.normal(mean=-2.0, std=noise_std, size=(n_points_per_class, 1))
    y0 = torch.zeros((n_points_per_class, 1))
    
    # Generate second Gaussian (class 1)
    x1 = torch.normal(mean=2.0, std=noise_std, size=(n_points_per_class, 1))
    y1 = torch.ones((n_points_per_class, 1))
    
    # Combine the data
    x = torch.cat([x0, x1], dim=0)
    y = torch.cat([y0, y1], dim=0)
    
    # Shuffle the data
    indices = torch.randperm(n_points)
    x = x[indices]
    y = y[indices]
    
    return split_data(x, y)

def get_dataset(dataset_type: str, n_points: int = 100):
    """Factory function to get different types of datasets."""
    datasets = {
        'linear': lambda: generate_linear_data(n_points=n_points),
        'nonlinear': lambda: generate_nonlinear_data(n_points=n_points),
        'classification': lambda: generate_classification_data(n_points=n_points)
    }
    return datasets[dataset_type]() 
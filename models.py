import torch
import torch.nn as nn

class SingleNeuronModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
    
    def forward(self, x):
        return self.linear(x)

class MultiLayerModel(nn.Module):
    def __init__(self, hidden_size=10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x):
        return self.network(x)

class BinaryClassifier(nn.Module):
    def __init__(self, hidden_size=10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid()  # Add sigmoid for binary classification
        )
    
    def forward(self, x):
        return self.network(x)

class DeepMultiLayerModel(nn.Module):
    def __init__(self, hidden_size=10, n_layers=20):
        super().__init__()
        
        # Create list of layers
        layers = []
        # First layer (input to hidden)
        layers.append(nn.Linear(1, hidden_size))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        
        # Output layer
        layers.append(nn.Linear(hidden_size, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class DeepBatchNormModel(nn.Module):
    def __init__(self, hidden_size=10, n_layers=20):
        super().__init__()
        
        # Create list of layers
        layers = []
        # First layer (input to hidden)
        layers.append(nn.Linear(1, hidden_size))
        layers.append(nn.BatchNorm1d(hidden_size))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
        
        # Output layer
        layers.append(nn.Linear(hidden_size, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

def get_model(model_type: str) -> nn.Module:
    """Factory function to create models based on type."""
    models = {
        'single_neuron': SingleNeuronModel,
        'multi_layer': MultiLayerModel,
        'binary_classifier': BinaryClassifier,
        'deep_multi_layer': DeepMultiLayerModel,
        'deep_batchnorm': DeepBatchNormModel
    }
    return models[model_type]() 
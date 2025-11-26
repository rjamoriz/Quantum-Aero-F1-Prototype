"""
Geometric Convolutional Network for Aerodynamic Prediction
Based on Graph Neural Networks for mesh-based geometry
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class GeoConvLayer(nn.Module):
    """
    Geometric Convolutional Layer
    Processes mesh geometry with spatial awareness
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features [batch, nodes, in_channels]
        Returns:
            Output features [batch, nodes, out_channels]
        """
        batch_size, num_nodes, _ = x.shape
        
        # Reshape for batch norm
        x = x.view(batch_size * num_nodes, -1)
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        
        # Reshape back
        x = x.view(batch_size, num_nodes, -1)
        return x


class AeroSurrogateNet(nn.Module):
    """
    Neural Network Surrogate for Aerodynamic Prediction
    
    Architecture:
    - Geometric encoder (processes mesh)
    - Parameter encoder (processes flow conditions)
    - Fusion layer
    - Multiple prediction heads (pressure, forces, etc.)
    """
    
    def __init__(
        self,
        mesh_features: int = 3,  # x, y, z coordinates
        param_features: int = 3,  # velocity, alpha, yaw
        hidden_dim: int = 128,
        num_layers: int = 3,
        output_pressure: bool = True,
        output_forces: bool = True
    ):
        super().__init__()
        
        self.output_pressure = output_pressure
        self.output_forces = output_forces
        
        # Geometric encoder
        self.geo_encoder = nn.ModuleList([
            GeoConvLayer(
                mesh_features if i == 0 else hidden_dim,
                hidden_dim
            )
            for i in range(num_layers)
        ])
        
        # Parameter encoder
        self.param_encoder = nn.Sequential(
            nn.Linear(param_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Prediction heads
        if output_pressure:
            self.pressure_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)  # Pressure coefficient per node
            )
        
        if output_forces:
            self.force_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 3)  # CL, CD, CM
            )
        
        # Confidence estimation head
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Confidence score [0, 1]
        )
        
        logger.info(f"AeroSurrogateNet initialized: {self.count_parameters()} parameters")
    
    def forward(
        self,
        mesh: torch.Tensor,
        parameters: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            mesh: Mesh coordinates [batch, nodes, 3]
            parameters: Flow parameters [batch, 3] (velocity, alpha, yaw)
            
        Returns:
            Dictionary with predictions
        """
        batch_size = mesh.shape[0]
        num_nodes = mesh.shape[1]
        
        # Encode geometry
        geo_features = mesh
        for layer in self.geo_encoder:
            geo_features = layer(geo_features)
        
        # Global pooling (mean over nodes)
        geo_global = geo_features.mean(dim=1)  # [batch, hidden_dim]
        
        # Encode parameters
        param_features = self.param_encoder(parameters)  # [batch, hidden_dim]
        
        # Fuse features
        fused = torch.cat([geo_global, param_features], dim=1)
        fused = self.fusion(fused)  # [batch, hidden_dim]
        
        # Predictions
        outputs = {}
        
        if self.output_pressure:
            # Broadcast fused features to all nodes
            fused_expanded = fused.unsqueeze(1).expand(-1, num_nodes, -1)
            # Combine with local geometric features
            combined = fused_expanded + geo_features
            # Predict pressure at each node
            pressure = self.pressure_head(combined)  # [batch, nodes, 1]
            outputs['pressure'] = pressure.squeeze(-1)  # [batch, nodes]
        
        if self.output_forces:
            forces = self.force_head(fused)  # [batch, 3]
            outputs['cl'] = forces[:, 0]
            outputs['cd'] = forces[:, 1]
            outputs['cm'] = forces[:, 2]
        
        # Confidence estimation
        confidence = self.confidence_head(fused)  # [batch, 1]
        outputs['confidence'] = confidence.squeeze(-1)  # [batch]
        
        return outputs
    
    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def predict_with_uncertainty(
        self,
        mesh: torch.Tensor,
        parameters: torch.Tensor,
        n_samples: int = 10
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Predict with uncertainty estimation using MC Dropout
        
        Args:
            mesh: Mesh coordinates
            parameters: Flow parameters
            n_samples: Number of MC samples
            
        Returns:
            (mean_predictions, std_predictions)
        """
        self.train()  # Enable dropout
        
        predictions = []
        for _ in range(n_samples):
            with torch.no_grad():
                pred = self.forward(mesh, parameters)
                predictions.append(pred)
        
        self.eval()  # Disable dropout
        
        # Compute mean and std
        mean_pred = {}
        std_pred = {}
        
        for key in predictions[0].keys():
            stacked = torch.stack([p[key] for p in predictions])
            mean_pred[key] = stacked.mean(dim=0)
            std_pred[key] = stacked.std(dim=0)
        
        return mean_pred, std_pred


class LightweightSurrogate(nn.Module):
    """
    Lightweight surrogate for fast inference
    Simpler architecture for real-time predictions
    """
    
    def __init__(
        self,
        input_dim: int = 6,  # Simplified geometry features + parameters
        hidden_dim: int = 64,
        output_dim: int = 3   # CL, CD, CM
    ):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        logger.info(f"LightweightSurrogate initialized: {self.count_parameters()} parameters")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features [batch, input_dim]
        Returns:
            Predictions [batch, output_dim]
        """
        return self.network(x)
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(
    model_type: str = 'full',
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    **kwargs
) -> nn.Module:
    """
    Factory function to create models
    
    Args:
        model_type: 'full' or 'lightweight'
        device: 'cuda' or 'cpu'
        **kwargs: Model-specific arguments
        
    Returns:
        Model instance on specified device
    """
    if model_type == 'full':
        model = AeroSurrogateNet(**kwargs)
    elif model_type == 'lightweight':
        model = LightweightSurrogate(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(device)
    logger.info(f"Model created on {device}")
    
    return model


if __name__ == "__main__":
    # Test model
    logging.basicConfig(level=logging.INFO)
    
    # Create model
    model = create_model('full', device='cpu')
    
    # Test forward pass
    batch_size = 2
    num_nodes = 100
    
    mesh = torch.randn(batch_size, num_nodes, 3)
    parameters = torch.randn(batch_size, 3)
    
    outputs = model(mesh, parameters)
    
    print("\nModel Test:")
    print(f"Input mesh shape: {mesh.shape}")
    print(f"Input parameters shape: {parameters.shape}")
    print(f"\nOutputs:")
    for key, value in outputs.items():
        print(f"  {key}: {value.shape}")
    
    print(f"\nModel parameters: {model.count_parameters():,}")

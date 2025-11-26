"""
AeroTransformer: Vision Transformer + U-Net Hybrid for 3D Flow Field Prediction

Based on Evolution Roadmap - Phase 1: Advanced AI Surrogates
Target: <50ms inference for full 3D pressure/velocity fields
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class PatchEmbed3D(nn.Module):
    """
    3D Patch Embedding for volumetric mesh data
    Converts 3D mesh into sequence of tokens
    """
    
    def __init__(
        self,
        volume_size: Tuple[int, int, int] = (64, 64, 64),
        patch_size: int = 8,
        in_channels: int = 3,  # x, y, z coordinates
        embed_dim: int = 768
    ):
        super().__init__()
        self.volume_size = volume_size
        self.patch_size = patch_size
        self.num_patches = (volume_size[0] // patch_size) ** 3
        
        # 3D Convolution for patch embedding
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # Position embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, D, H, W) - Batch, Channels, Depth, Height, Width
        Returns:
            (B, num_patches, embed_dim)
        """
        B = x.shape[0]
        
        # Patch embedding
        x = self.proj(x)  # (B, embed_dim, D', H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        
        # Add position embedding
        x = x + self.pos_embed
        
        return x


class MultiHeadAttention3D(nn.Module):
    """
    Multi-Head Self-Attention for capturing long-range aerodynamic dependencies
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, embed_dim)
        Returns:
            (B, N, embed_dim)
        """
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        
        return x


class TransformerBlock(nn.Module):
    """
    Transformer block with multi-head attention and MLP
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention3D(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # MLP
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention with residual
        x = x + self.attn(self.norm1(x))
        
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        
        return x


class UNetDecoder3D(nn.Module):
    """
    3D U-Net Decoder for high-resolution field reconstruction
    """
    
    def __init__(
        self,
        in_channels: int = 768,
        out_channels: int = 7,  # p, u, v, w, k, omega, nut
        base_channels: int = 64
    ):
        super().__init__()
        
        # Decoder blocks (upsampling)
        self.up1 = nn.ConvTranspose3d(in_channels, base_channels * 8, 2, stride=2)
        self.conv1 = self._conv_block(base_channels * 8, base_channels * 8)
        
        self.up2 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.conv2 = self._conv_block(base_channels * 4, base_channels * 4)
        
        self.up3 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.conv3 = self._conv_block(base_channels * 2, base_channels * 2)
        
        self.up4 = nn.ConvTranspose3d(base_channels * 2, base_channels, 2, stride=2)
        self.conv4 = self._conv_block(base_channels, base_channels)
        
        # Output layer
        self.out_conv = nn.Conv3d(base_channels, out_channels, 1)
        
    def _conv_block(self, in_ch: int, out_ch: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x: torch.Tensor, volume_shape: Tuple[int, int, int]) -> torch.Tensor:
        """
        Args:
            x: (B, num_patches, embed_dim)
            volume_shape: Target output shape (D, H, W)
        Returns:
            (B, out_channels, D, H, W)
        """
        B, N, C = x.shape
        
        # Reshape to 3D volume
        D = H = W = int(round(N ** (1/3)))
        x = x.transpose(1, 2).reshape(B, C, D, H, W)
        
        # Decoder path
        x = self.up1(x)
        x = self.conv1(x)
        
        x = self.up2(x)
        x = self.conv2(x)
        
        x = self.up3(x)
        x = self.conv3(x)
        
        x = self.up4(x)
        x = self.conv4(x)
        
        # Output
        x = self.out_conv(x)
        
        # Interpolate to target shape if needed
        if x.shape[2:] != volume_shape:
            x = F.interpolate(x, size=volume_shape, mode='trilinear', align_corners=False)
        
        return x


class PhysicsInformedLoss(nn.Module):
    """
    Physics-Informed Loss Function
    Enforces continuity equation and momentum conservation
    """
    
    def __init__(
        self,
        lambda_data: float = 1.0,
        lambda_continuity: float = 0.1,
        lambda_momentum: float = 0.1,
        lambda_boundary: float = 0.05
    ):
        super().__init__()
        self.lambda_data = lambda_data
        self.lambda_continuity = lambda_continuity
        self.lambda_momentum = lambda_momentum
        self.lambda_boundary = lambda_boundary
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mesh_info: Optional[dict] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            pred: (B, 7, D, H, W) - Predicted fields [p, u, v, w, k, omega, nut]
            target: (B, 7, D, H, W) - Ground truth fields
            mesh_info: Optional mesh metadata
        Returns:
            Total loss and loss components dict
        """
        # Data loss (MSE)
        loss_data = F.mse_loss(pred, target)
        
        # Extract velocity components
        u = pred[:, 1:2]  # u velocity
        v = pred[:, 2:3]  # v velocity
        w = pred[:, 3:4]  # w velocity
        
        # Continuity loss: ∇·u = 0
        du_dx = self._gradient(u, dim=2)
        dv_dy = self._gradient(v, dim=3)
        dw_dz = self._gradient(w, dim=4)
        divergence = du_dx + dv_dy + dw_dz
        loss_continuity = torch.mean(divergence ** 2)
        
        # Momentum loss (simplified)
        # In practice, would include full Navier-Stokes residual
        velocity_magnitude = torch.sqrt(u**2 + v**2 + w**2 + 1e-8)
        loss_momentum = torch.mean((velocity_magnitude - torch.mean(velocity_magnitude)) ** 2)
        
        # Boundary loss (enforce no-slip at walls if boundary info available)
        loss_boundary = torch.tensor(0.0, device=pred.device)
        if mesh_info and 'boundary_mask' in mesh_info:
            boundary_mask = mesh_info['boundary_mask']
            loss_boundary = F.mse_loss(
                pred[:, 1:4] * boundary_mask,
                torch.zeros_like(pred[:, 1:4]) * boundary_mask
            )
        
        # Total loss
        total_loss = (
            self.lambda_data * loss_data +
            self.lambda_continuity * loss_continuity +
            self.lambda_momentum * loss_momentum +
            self.lambda_boundary * loss_boundary
        )
        
        loss_dict = {
            'total': total_loss.item(),
            'data': loss_data.item(),
            'continuity': loss_continuity.item(),
            'momentum': loss_momentum.item(),
            'boundary': loss_boundary.item()
        }
        
        return total_loss, loss_dict
    
    def _gradient(self, field: torch.Tensor, dim: int) -> torch.Tensor:
        """Compute gradient along specified dimension"""
        return torch.gradient(field, dim=dim)[0]


class AeroTransformer(nn.Module):
    """
    Complete AeroTransformer Model
    
    Vision Transformer + U-Net Hybrid for 3D Flow Field Prediction
    Target: <50ms inference on RTX 4090
    """
    
    def __init__(
        self,
        volume_size: Tuple[int, int, int] = (64, 64, 64),
        patch_size: int = 8,
        in_channels: int = 3,
        out_channels: int = 7,
        embed_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        super().__init__()
        self.volume_size = volume_size
        
        # Patch embedding
        self.patch_embed = PatchEmbed3D(
            volume_size, patch_size, in_channels, embed_dim
        )
        
        # Transformer encoder
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # U-Net decoder
        self.decoder = UNetDecoder3D(embed_dim, out_channels)
        
        # Physics-informed loss
        self.physics_loss = PhysicsInformedLoss()
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, D, H, W) - Input mesh geometry
        Returns:
            (B, out_channels, D, H, W) - Predicted flow fields
        """
        # Patch embedding
        x = self.patch_embed(x)
        
        # Transformer encoding
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # U-Net decoding
        x = self.decoder(x, self.volume_size)
        
        return x
    
    def compute_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mesh_info: Optional[dict] = None
    ) -> Tuple[torch.Tensor, dict]:
        """Compute physics-informed loss"""
        return self.physics_loss(pred, target, mesh_info)


def create_aero_transformer(
    model_size: str = 'base',
    volume_size: Tuple[int, int, int] = (64, 64, 64)
) -> AeroTransformer:
    """
    Factory function to create AeroTransformer models
    
    Args:
        model_size: 'tiny', 'small', 'base', 'large'
        volume_size: 3D volume dimensions
    """
    configs = {
        'tiny': {'embed_dim': 192, 'num_layers': 6, 'num_heads': 3},
        'small': {'embed_dim': 384, 'num_layers': 8, 'num_heads': 6},
        'base': {'embed_dim': 768, 'num_layers': 12, 'num_heads': 12},
        'large': {'embed_dim': 1024, 'num_layers': 24, 'num_heads': 16}
    }
    
    config = configs[model_size]
    
    return AeroTransformer(
        volume_size=volume_size,
        embed_dim=config['embed_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads']
    )


if __name__ == "__main__":
    # Test model
    model = create_aero_transformer('base', volume_size=(64, 64, 64))
    
    # Dummy input
    x = torch.randn(2, 3, 64, 64, 64)  # Batch of 2
    
    # Forward pass
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

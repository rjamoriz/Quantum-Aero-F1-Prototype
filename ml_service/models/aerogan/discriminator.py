"""
AeroGAN Discriminator
Physics-informed discriminator for aerodynamic geometries
Evaluates both realism and aerodynamic validity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class ResidualBlock3D(nn.Module):
    """3D Residual block for discriminator"""
    
    def __init__(self, in_channels: int, out_channels: int, downsample: bool = True):
        super().__init__()
        
        stride = 2 if downsample else 1
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        
        self.norm1 = nn.InstanceNorm3d(out_channels)
        self.norm2 = nn.InstanceNorm3d(out_channels)
        
        self.activation = nn.LeakyReLU(0.2)
        
        # Skip connection
        if downsample or in_channels != out_channels:
            self.skip = nn.Conv3d(in_channels, out_channels, 1, stride=stride)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        
        out = out + residual
        out = self.activation(out)
        
        return out


class PhysicsHead(nn.Module):
    """
    Physics-informed head
    Predicts aerodynamic coefficients from geometry
    """
    
    def __init__(self, feature_dim: int = 512):
        super().__init__()
        
        self.predictor = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 3)  # Cl, Cd, Cm
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Predict aerodynamic coefficients
        
        Returns:
            (B, 3) - [Cl, Cd, Cm]
        """
        return self.predictor(features)


class AeroGANDiscriminator(nn.Module):
    """
    Physics-informed discriminator for aerodynamic geometries
    
    Dual objectives:
    1. Adversarial: Real vs Fake
    2. Physics: Aerodynamic validity
    """
    
    def __init__(
        self,
        resolution: int = 64,
        base_channels: int = 64,
        max_channels: int = 512,
        condition_dim: int = 16
    ):
        super().__init__()
        
        self.resolution = resolution
        
        # Input layer
        self.from_sdf = nn.Sequential(
            nn.Conv3d(1, base_channels, 3, padding=1),
            nn.LeakyReLU(0.2)
        )
        
        # Condition encoder
        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2)
        )
        
        # Downsampling blocks
        self.blocks = nn.ModuleList()
        
        num_blocks = int(np.log2(resolution // 4))
        in_channels = base_channels
        
        for i in range(num_blocks):
            out_channels = min(base_channels * (2 ** i), max_channels)
            
            block = ResidualBlock3D(in_channels, out_channels, downsample=True)
            self.blocks.append(block)
            
            in_channels = out_channels
        
        # Final block (no downsampling)
        self.blocks.append(ResidualBlock3D(in_channels, max_channels, downsample=False))
        
        # Global pooling
        self.pool = nn.AdaptiveAvgPool3d(1)
        
        # Adversarial head
        self.adversarial_head = nn.Sequential(
            nn.Linear(max_channels + 512, 256),  # +512 for condition
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )
        
        # Physics head
        self.physics_head = PhysicsHead(max_channels)
        
        print(f"AeroGAN Discriminator initialized")
        print(f"  Resolution: {resolution}³")
        print(f"  Downsampling blocks: {len(self.blocks)}")
    
    def forward(
        self,
        sdf: torch.Tensor,
        condition: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Discriminate geometry and predict physics
        
        Args:
            sdf: (B, 1, resolution, resolution, resolution) - Signed distance field
            condition: (B, condition_dim) - Aerodynamic conditions
        
        Returns:
            (validity, physics) where:
                validity: (B, 1) - Real/Fake score
                physics: (B, 3) - Predicted [Cl, Cd, Cm]
        """
        # Extract features from SDF
        x = self.from_sdf(sdf)
        
        # Downsampling
        for block in self.blocks:
            x = block(x)
        
        # Global pooling
        features = self.pool(x).squeeze(-1).squeeze(-1).squeeze(-1)  # (B, C)
        
        # Encode condition
        c = self.condition_encoder(condition)
        
        # Adversarial prediction
        combined = torch.cat([features, c], dim=1)
        validity = self.adversarial_head(combined)
        
        # Physics prediction
        physics = self.physics_head(features)
        
        return validity, physics


class AeroGANLoss:
    """
    Combined loss for AeroGAN
    
    Components:
    1. Adversarial loss (WGAN-GP)
    2. Physics loss (MSE on aerodynamic coefficients)
    3. Gradient penalty
    """
    
    def __init__(
        self,
        lambda_gp: float = 10.0,
        lambda_physics: float = 1.0
    ):
        self.lambda_gp = lambda_gp
        self.lambda_physics = lambda_physics
    
    def discriminator_loss(
        self,
        real_validity: torch.Tensor,
        fake_validity: torch.Tensor,
        real_physics: torch.Tensor,
        target_physics: torch.Tensor,
        gradient_penalty: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Discriminator loss
        
        Returns:
            (total_loss, loss_dict)
        """
        # Adversarial loss (WGAN)
        adv_loss = fake_validity.mean() - real_validity.mean()
        
        # Physics loss
        physics_loss = F.mse_loss(real_physics, target_physics)
        
        # Total loss
        total_loss = adv_loss + self.lambda_gp * gradient_penalty + self.lambda_physics * physics_loss
        
        loss_dict = {
            'adv_loss': adv_loss.item(),
            'physics_loss': physics_loss.item(),
            'gradient_penalty': gradient_penalty.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_dict
    
    def generator_loss(
        self,
        fake_validity: torch.Tensor,
        fake_physics: torch.Tensor,
        target_physics: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generator loss
        
        Returns:
            (total_loss, loss_dict)
        """
        # Adversarial loss
        adv_loss = -fake_validity.mean()
        
        # Physics loss (encourage physically valid geometries)
        physics_loss = F.mse_loss(fake_physics, target_physics)
        
        # Total loss
        total_loss = adv_loss + self.lambda_physics * physics_loss
        
        loss_dict = {
            'adv_loss': adv_loss.item(),
            'physics_loss': physics_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_dict
    
    def compute_gradient_penalty(
        self,
        discriminator: nn.Module,
        real_sdf: torch.Tensor,
        fake_sdf: torch.Tensor,
        condition: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute gradient penalty for WGAN-GP
        
        Returns:
            gradient_penalty: scalar
        """
        B = real_sdf.shape[0]
        device = real_sdf.device
        
        # Random interpolation weight
        alpha = torch.rand(B, 1, 1, 1, 1, device=device)
        
        # Interpolate
        interpolated = alpha * real_sdf + (1 - alpha) * fake_sdf
        interpolated.requires_grad_(True)
        
        # Discriminator output
        validity, _ = discriminator(interpolated, condition)
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=validity,
            inputs=interpolated,
            grad_outputs=torch.ones_like(validity),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Gradient penalty
        gradients = gradients.view(B, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return gradient_penalty


if __name__ == "__main__":
    # Test discriminator
    print("Testing AeroGAN Discriminator\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Create discriminator
    discriminator = AeroGANDiscriminator(
        resolution=64,
        base_channels=64,
        max_channels=512,
        condition_dim=16
    ).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in discriminator.parameters())
    print(f"Discriminator parameters: {num_params:,}\n")
    
    # Test forward pass
    batch_size = 2
    sdf = torch.randn(batch_size, 1, 64, 64, 64).to(device)
    condition = torch.randn(batch_size, 16).to(device)
    
    print("Forward pass:")
    with torch.no_grad():
        validity, physics = discriminator(sdf, condition)
    
    print(f"  Input SDF shape: {sdf.shape}")
    print(f"  Input condition shape: {condition.shape}")
    print(f"  Output validity shape: {validity.shape}")
    print(f"  Output physics shape: {physics.shape}")
    print(f"  Validity range: [{validity.min().item():.3f}, {validity.max().item():.3f}]")
    print(f"  Physics (Cl, Cd, Cm): {physics[0].detach().cpu().numpy()}")
    
    # Test loss
    print("\nTesting loss computation:")
    loss_fn = AeroGANLoss()
    
    real_sdf = torch.randn(2, 1, 64, 64, 64).to(device)
    fake_sdf = torch.randn(2, 1, 64, 64, 64).to(device)
    target_physics = torch.tensor([[2.8, 0.4, -0.1], [2.5, 0.45, -0.08]]).to(device)
    
    real_validity, real_physics = discriminator(real_sdf, condition)
    fake_validity, fake_physics = discriminator(fake_sdf, condition)
    
    gp = loss_fn.compute_gradient_penalty(discriminator, real_sdf, fake_sdf, condition)
    
    d_loss, d_dict = loss_fn.discriminator_loss(
        real_validity, fake_validity, real_physics, target_physics, gp
    )
    
    g_loss, g_dict = loss_fn.generator_loss(
        fake_validity, fake_physics, target_physics
    )
    
    print(f"  Discriminator loss: {d_loss.item():.4f}")
    print(f"  Generator loss: {g_loss.item():.4f}")
    print(f"  Gradient penalty: {gp.item():.4f}")
    
    print("\n✓ Discriminator test complete!")

"""
AeroGAN Generator
StyleGAN3-inspired generator for aerodynamic geometries
Generates 3D SDF (Signed Distance Field) representations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List


class MappingNetwork(nn.Module):
    """
    Mapping network: z -> w
    Maps latent code to intermediate latent space
    """
    
    def __init__(
        self,
        z_dim: int = 512,
        w_dim: int = 512,
        num_layers: int = 8,
        condition_dim: int = 16
    ):
        super().__init__()
        
        self.z_dim = z_dim
        self.w_dim = w_dim
        
        # Condition encoder
        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2)
        )
        
        # Mapping layers
        layers = []
        for i in range(num_layers):
            in_dim = z_dim + 512 if i == 0 else w_dim
            layers.extend([
                nn.Linear(in_dim, w_dim),
                nn.LeakyReLU(0.2)
            ])
        
        self.mapping = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, z_dim) - Random latent code
            condition: (B, condition_dim) - Aerodynamic conditions
        
        Returns:
            w: (B, w_dim) - Intermediate latent code
        """
        c = self.condition_encoder(condition)
        x = torch.cat([z, c], dim=1)
        w = self.mapping(x)
        return w


class ModulatedConv3d(nn.Module):
    """
    Modulated 3D convolution with style modulation
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        w_dim: int = 512,
        demodulate: bool = True
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.demodulate = demodulate
        
        # Weight
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size)
        )
        
        # Style modulation
        self.affine = nn.Linear(w_dim, in_channels)
        
        # Bias
        self.bias = nn.Parameter(torch.zeros(out_channels))
    
    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, D, H, W)
            w: (B, w_dim)
        
        Returns:
            (B, C_out, D, H, W)
        """
        B, C_in, D, H, W = x.shape
        
        # Get style
        style = self.affine(w)  # (B, C_in)
        
        # Modulate weight
        weight = self.weight.unsqueeze(0)  # (1, C_out, C_in, K, K, K)
        weight = weight * style.view(B, 1, C_in, 1, 1, 1)  # (B, C_out, C_in, K, K, K)
        
        # Demodulation
        if self.demodulate:
            d = torch.rsqrt(weight.pow(2).sum([2, 3, 4, 5]) + 1e-8)  # (B, C_out)
            weight = weight * d.view(B, self.out_channels, 1, 1, 1, 1)
        
        # Reshape for grouped convolution
        x = x.reshape(1, B * C_in, D, H, W)
        weight = weight.reshape(B * self.out_channels, C_in, self.kernel_size, self.kernel_size, self.kernel_size)
        
        # Convolution
        out = F.conv3d(x, weight, padding=self.kernel_size // 2, groups=B)
        out = out.reshape(B, self.out_channels, D, H, W)
        
        # Add bias
        out = out + self.bias.view(1, -1, 1, 1, 1)
        
        return out


class SynthesisBlock(nn.Module):
    """
    Synthesis block with style modulation
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        w_dim: int = 512,
        resolution: int = 4
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.resolution = resolution
        
        # Upsampling
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        
        # Modulated convolutions
        self.conv1 = ModulatedConv3d(in_channels, out_channels, 3, w_dim)
        self.conv2 = ModulatedConv3d(out_channels, out_channels, 3, w_dim)
        
        # Activation
        self.activation = nn.LeakyReLU(0.2)
        
        # Noise
        self.noise_strength1 = nn.Parameter(torch.zeros(1))
        self.noise_strength2 = nn.Parameter(torch.zeros(1))
    
    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, D, H, W)
            w: (B, w_dim)
        
        Returns:
            (B, C_out, D*2, H*2, W*2)
        """
        # Upsample
        x = self.upsample(x)
        
        # First conv
        x = self.conv1(x, w)
        x = self.activation(x)
        
        # Add noise
        noise = torch.randn_like(x) * self.noise_strength1
        x = x + noise
        
        # Second conv
        x = self.conv2(x, w)
        x = self.activation(x)
        
        # Add noise
        noise = torch.randn_like(x) * self.noise_strength2
        x = x + noise
        
        return x


class AeroGANGenerator(nn.Module):
    """
    StyleGAN3-inspired generator for aerodynamic geometries
    
    Generates 3D SDF (Signed Distance Field) representations
    Conditioned on aerodynamic targets (Cl, Cd, Cm)
    """
    
    def __init__(
        self,
        z_dim: int = 512,
        w_dim: int = 512,
        condition_dim: int = 16,
        base_channels: int = 512,
        max_channels: int = 512,
        resolution: int = 64
    ):
        super().__init__()
        
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.resolution = resolution
        
        # Mapping network
        self.mapping = MappingNetwork(z_dim, w_dim, num_layers=8, condition_dim=condition_dim)
        
        # Constant input
        self.const_input = nn.Parameter(torch.randn(1, base_channels, 4, 4, 4))
        
        # Synthesis blocks
        self.blocks = nn.ModuleList()
        
        # Calculate number of blocks needed
        num_blocks = int(np.log2(resolution // 4))
        
        in_channels = base_channels
        for i in range(num_blocks):
            out_channels = min(base_channels // (2 ** i), max_channels)
            out_channels = max(out_channels, 64)
            
            block = SynthesisBlock(
                in_channels,
                out_channels,
                w_dim,
                resolution=4 * (2 ** (i + 1))
            )
            self.blocks.append(block)
            
            in_channels = out_channels
        
        # Output layer (SDF)
        self.to_sdf = nn.Sequential(
            nn.Conv3d(in_channels, 32, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(32, 1, 1),
            nn.Tanh()  # SDF in [-1, 1]
        )
        
        print(f"AeroGAN Generator initialized")
        print(f"  Resolution: {resolution}³")
        print(f"  Synthesis blocks: {len(self.blocks)}")
    
    def forward(
        self,
        z: torch.Tensor,
        condition: torch.Tensor,
        truncation_psi: float = 1.0
    ) -> torch.Tensor:
        """
        Generate 3D geometry
        
        Args:
            z: (B, z_dim) - Random latent code
            condition: (B, condition_dim) - Aerodynamic conditions
            truncation_psi: Truncation trick parameter
        
        Returns:
            sdf: (B, 1, resolution, resolution, resolution) - Signed distance field
        """
        B = z.shape[0]
        
        # Map to intermediate latent space
        w = self.mapping(z, condition)
        
        # Truncation trick
        if truncation_psi < 1.0:
            w = w * truncation_psi
        
        # Start from constant
        x = self.const_input.repeat(B, 1, 1, 1, 1)
        
        # Synthesis
        for block in self.blocks:
            x = block(x, w)
        
        # Convert to SDF
        sdf = self.to_sdf(x)
        
        return sdf
    
    def generate(
        self,
        batch_size: int,
        condition: torch.Tensor,
        truncation_psi: float = 0.7,
        device: str = 'cuda'
    ) -> torch.Tensor:
        """
        Generate batch of geometries
        
        Args:
            batch_size: Number of samples
            condition: (B, condition_dim) - Aerodynamic conditions
            truncation_psi: Truncation for quality/diversity tradeoff
            device: Device
        
        Returns:
            sdf: (B, 1, resolution, resolution, resolution)
        """
        # Sample latent codes
        z = torch.randn(batch_size, self.z_dim, device=device)
        
        # Generate
        with torch.no_grad():
            sdf = self.forward(z, condition, truncation_psi)
        
        return sdf


if __name__ == "__main__":
    # Test generator
    print("Testing AeroGAN Generator\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Create generator
    generator = AeroGANGenerator(
        z_dim=512,
        w_dim=512,
        condition_dim=16,
        resolution=64
    ).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in generator.parameters())
    print(f"Generator parameters: {num_params:,}\n")
    
    # Test generation
    batch_size = 2
    z = torch.randn(batch_size, 512).to(device)
    condition = torch.randn(batch_size, 16).to(device)
    
    print("Generating geometries...")
    with torch.no_grad():
        sdf = generator(z, condition)
    
    print(f"  Input z shape: {z.shape}")
    print(f"  Input condition shape: {condition.shape}")
    print(f"  Output SDF shape: {sdf.shape}")
    print(f"  SDF range: [{sdf.min().item():.3f}, {sdf.max().item():.3f}]")
    
    # Test batch generation
    print("\nBatch generation test:")
    sdf_batch = generator.generate(
        batch_size=4,
        condition=torch.randn(4, 16).to(device),
        truncation_psi=0.7,
        device=device
    )
    print(f"  Generated batch shape: {sdf_batch.shape}")
    
    print("\n✓ Generator test complete!")

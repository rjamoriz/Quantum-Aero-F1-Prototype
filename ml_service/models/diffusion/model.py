"""
Diffusion Model for 3D Aerodynamic Geometry Generation
Conditional generation of F1 wing geometries
Target: 5-second generation, 1000+ candidates/day
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple
import time


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for timestep encoding"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """Residual block with time and condition embedding"""
    
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, cond_emb_dim: int):
        super().__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.cond_mlp = nn.Linear(cond_emb_dim, out_channels)
        
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        if in_channels != out_channels:
            self.residual_conv = nn.Conv3d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        residual = self.residual_conv(x)
        
        # First conv
        h = self.conv1(x)
        h = self.norm1(h)
        
        # Add time embedding
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, :, None, None, None]
        
        # Add condition embedding
        cond_emb = self.cond_mlp(cond_emb)
        h = h + cond_emb[:, :, None, None, None]
        
        h = F.silu(h)
        
        # Second conv
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)
        
        return h + residual


class AttentionBlock(nn.Module):
    """Self-attention block for 3D features"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv3d(channels, channels * 3, 1)
        self.proj = nn.Conv3d(channels, channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        residual = x
        
        x = self.norm(x)
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape for attention
        q = q.reshape(B, C, -1).transpose(1, 2)
        k = k.reshape(B, C, -1).transpose(1, 2)
        v = v.reshape(B, C, -1).transpose(1, 2)
        
        # Scaled dot-product attention
        scale = 1.0 / np.sqrt(C)
        attn = torch.softmax(torch.bmm(q, k.transpose(1, 2)) * scale, dim=-1)
        out = torch.bmm(attn, v)
        
        # Reshape back
        out = out.transpose(1, 2).reshape(B, C, D, H, W)
        out = self.proj(out)
        
        return out + residual


class AeroDiffusionUNet(nn.Module):
    """
    3D U-Net for diffusion-based geometry generation
    
    Conditional on:
    - Target aerodynamic coefficients (Cl, Cd, Cm)
    - Design constraints (volume, thickness, etc.)
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        time_emb_dim: int = 256,
        cond_dim: int = 16
    ):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )
        
        # Condition encoder (aerodynamic targets)
        self.cond_encoder = nn.Sequential(
            nn.Linear(cond_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, time_emb_dim)
        )
        
        # Encoder (downsampling)
        self.enc1 = ResidualBlock(in_channels, base_channels, time_emb_dim, time_emb_dim)
        self.enc2 = ResidualBlock(base_channels, base_channels * 2, time_emb_dim, time_emb_dim)
        self.enc3 = ResidualBlock(base_channels * 2, base_channels * 4, time_emb_dim, time_emb_dim)
        self.enc4 = ResidualBlock(base_channels * 4, base_channels * 8, time_emb_dim, time_emb_dim)
        
        # Bottleneck with attention
        self.bottleneck = nn.Sequential(
            ResidualBlock(base_channels * 8, base_channels * 8, time_emb_dim, time_emb_dim),
            AttentionBlock(base_channels * 8),
            ResidualBlock(base_channels * 8, base_channels * 8, time_emb_dim, time_emb_dim)
        )
        
        # Decoder (upsampling)
        self.dec4 = ResidualBlock(base_channels * 16, base_channels * 4, time_emb_dim, time_emb_dim)
        self.dec3 = ResidualBlock(base_channels * 8, base_channels * 2, time_emb_dim, time_emb_dim)
        self.dec2 = ResidualBlock(base_channels * 4, base_channels, time_emb_dim, time_emb_dim)
        self.dec1 = ResidualBlock(base_channels * 2, base_channels, time_emb_dim, time_emb_dim)
        
        # Output
        self.out = nn.Conv3d(base_channels, out_channels, 1)
        
        self.pool = nn.MaxPool3d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
    
    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        condition: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: (B, 1, D, H, W) - Noisy geometry
            timestep: (B,) - Diffusion timestep
            condition: (B, cond_dim) - Aerodynamic targets
        
        Returns:
            (B, 1, D, H, W) - Predicted noise
        """
        # Encode timestep and condition
        t_emb = self.time_mlp(timestep)
        c_emb = self.cond_encoder(condition)
        
        # Encoder
        e1 = self.enc1(x, t_emb, c_emb)
        e2 = self.enc2(self.pool(e1), t_emb, c_emb)
        e3 = self.enc3(self.pool(e2), t_emb, c_emb)
        e4 = self.enc4(self.pool(e3), t_emb, c_emb)
        
        # Bottleneck
        b = self.bottleneck[0](self.pool(e4), t_emb, c_emb)
        b = self.bottleneck[1](b)
        b = self.bottleneck[2](b, t_emb, c_emb)
        
        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.upsample(b), e4], dim=1), t_emb, c_emb)
        d3 = self.dec3(torch.cat([self.upsample(d4), e3], dim=1), t_emb, c_emb)
        d2 = self.dec2(torch.cat([self.upsample(d3), e2], dim=1), t_emb, c_emb)
        d1 = self.dec1(torch.cat([self.upsample(d2), e1], dim=1), t_emb, c_emb)
        
        # Output
        out = self.out(d1)
        
        return out


class DiffusionScheduler:
    """
    DDPM (Denoising Diffusion Probabilistic Models) scheduler
    """
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02
    ):
        self.num_timesteps = num_timesteps
        
        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
    
    def add_noise(
        self,
        x_0: torch.Tensor,
        timestep: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Add noise to clean data"""
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timestep].reshape(-1, 1, 1, 1, 1)
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timestep].reshape(-1, 1, 1, 1, 1)
        
        return sqrt_alpha_prod.to(x_0.device) * x_0 + sqrt_one_minus_alpha_prod.to(x_0.device) * noise
    
    def denoise_step(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        timestep: int,
        condition: torch.Tensor
    ) -> torch.Tensor:
        """Single denoising step"""
        t = torch.full((x_t.shape[0],), timestep, device=x_t.device, dtype=torch.long)
        
        # Predict noise
        predicted_noise = model(x_t, t, condition)
        
        # Get coefficients
        alpha = self.alphas[timestep].to(x_t.device)
        alpha_cumprod = self.alphas_cumprod[timestep].to(x_t.device)
        beta = self.betas[timestep].to(x_t.device)
        
        # Compute x_{t-1}
        x_0_pred = (x_t - torch.sqrt(1 - alpha_cumprod) * predicted_noise) / torch.sqrt(alpha_cumprod)
        
        if timestep > 0:
            noise = torch.randn_like(x_t)
            variance = self.posterior_variance[timestep].to(x_t.device)
            x_t_prev = torch.sqrt(alpha) * x_0_pred + torch.sqrt(beta) * noise
        else:
            x_t_prev = x_0_pred
        
        return x_t_prev


if __name__ == "__main__":
    # Test diffusion model
    print("Testing Aerodynamic Diffusion Model\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Create model
    model = AeroDiffusionUNet(
        in_channels=1,
        out_channels=1,
        base_channels=32,  # Reduced for testing
        time_emb_dim=128,
        cond_dim=16
    ).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 1, 32, 32, 32).to(device)
    timestep = torch.randint(0, 1000, (batch_size,)).to(device)
    condition = torch.randn(batch_size, 16).to(device)
    
    print("\nForward pass:")
    print(f"  Input shape: {x.shape}")
    print(f"  Timestep shape: {timestep.shape}")
    print(f"  Condition shape: {condition.shape}")
    
    with torch.no_grad():
        start = time.time()
        output = model(x, timestep, condition)
        elapsed = time.time() - start
    
    print(f"  Output shape: {output.shape}")
    print(f"  Inference time: {elapsed*1000:.2f}ms")
    
    # Test scheduler
    print("\nTesting diffusion scheduler:")
    scheduler = DiffusionScheduler(num_timesteps=1000)
    
    # Add noise
    x_0 = torch.randn(1, 1, 32, 32, 32).to(device)
    t = torch.tensor([500]).to(device)
    x_t = scheduler.add_noise(x_0, t)
    
    print(f"  Clean data shape: {x_0.shape}")
    print(f"  Noisy data shape: {x_t.shape}")
    print(f"  Noise level (t=500): {torch.mean((x_t - x_0)**2).item():.4f}")
    
    print("\n✓ Diffusion model test complete!")

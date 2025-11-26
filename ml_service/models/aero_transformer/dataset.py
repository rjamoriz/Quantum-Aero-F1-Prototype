"""
CFD Dataset for AeroTransformer Training
Loads 3D mesh geometry and flow field data
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import os
import h5py
from typing import Dict, Any, Tuple
import json


class CFDDataset(Dataset):
    """
    CFD Dataset for AeroTransformer
    
    Expected data format:
    - Input: 3D mesh geometry (voxelized or point cloud)
    - Output: Pressure + velocity + turbulence fields
    
    Dataset structure:
    data/
    ├── train/
    │   ├── case_0000.h5
    │   ├── case_0001.h5
    │   └── ...
    ├── val/
    │   └── ...
    └── metadata.json
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        volume_size: Tuple[int, int, int] = (64, 64, 64),
        transform=None
    ):
        self.data_dir = data_dir
        self.split = split
        self.volume_size = volume_size
        self.transform = transform
        
        # Load file list
        self.split_dir = os.path.join(data_dir, split)
        self.file_list = sorted([
            f for f in os.listdir(self.split_dir)
            if f.endswith('.h5')
        ])
        
        # Load metadata
        metadata_path = os.path.join(data_dir, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
        
        print(f"Loaded {len(self.file_list)} samples from {split} split")
    
    def __len__(self) -> int:
        return len(self.file_list)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load a single CFD case
        
        Returns:
            dict with keys:
                - input: (C, D, H, W) - Mesh geometry
                - target: (7, D, H, W) - Flow fields [p, u, v, w, k, omega, nut]
                - mesh_info: Optional metadata
        """
        file_path = os.path.join(self.split_dir, self.file_list[idx])
        
        with h5py.File(file_path, 'r') as f:
            # Load input geometry
            if 'geometry' in f:
                geometry = torch.from_numpy(f['geometry'][:]).float()
            else:
                # Generate mock geometry
                geometry = self._generate_mock_geometry()
            
            # Load target flow fields
            if 'flow_fields' in f:
                flow_fields = torch.from_numpy(f['flow_fields'][:]).float()
            else:
                # Generate mock flow fields
                flow_fields = self._generate_mock_flow_fields()
            
            # Load mesh info
            mesh_info = {}
            if 'boundary_mask' in f:
                mesh_info['boundary_mask'] = torch.from_numpy(f['boundary_mask'][:]).float()
        
        # Apply transforms
        if self.transform:
            geometry, flow_fields = self.transform(geometry, flow_fields)
        
        return {
            'input': geometry,
            'target': flow_fields,
            'mesh_info': mesh_info,
            'case_id': self.file_list[idx]
        }
    
    def _generate_mock_geometry(self) -> torch.Tensor:
        """Generate mock geometry for testing"""
        D, H, W = self.volume_size
        
        # Create simple wing geometry
        geometry = torch.zeros(3, D, H, W)  # x, y, z coordinates
        
        for i in range(D):
            for j in range(H):
                for k in range(W):
                    # Normalized coordinates
                    x = (i - D/2) / D
                    y = (j - H/2) / H
                    z = k / W
                    
                    geometry[0, i, j, k] = x
                    geometry[1, i, j, k] = y
                    geometry[2, i, j, k] = z
        
        return geometry
    
    def _generate_mock_flow_fields(self) -> torch.Tensor:
        """Generate mock flow fields for testing"""
        D, H, W = self.volume_size
        
        # Create flow fields: [p, u, v, w, k, omega, nut]
        flow_fields = torch.zeros(7, D, H, W)
        
        for i in range(D):
            for j in range(H):
                for k in range(W):
                    # Normalized coordinates
                    x = (i - D/2) / D
                    y = (j - H/2) / H
                    z = k / W
                    
                    # Pressure (decreasing with x)
                    flow_fields[0, i, j, k] = 1.0 - 0.5 * x
                    
                    # Velocity u (freestream + perturbation)
                    flow_fields[1, i, j, k] = 1.0 + 0.1 * np.sin(x * np.pi)
                    
                    # Velocity v (crossflow)
                    flow_fields[2, i, j, k] = 0.05 * np.sin(y * np.pi)
                    
                    # Velocity w (vertical)
                    flow_fields[3, i, j, k] = 0.02 * np.cos(z * np.pi)
                    
                    # Turbulent kinetic energy
                    flow_fields[4, i, j, k] = 0.01 * (1 + 0.5 * np.random.randn())
                    
                    # Specific dissipation rate
                    flow_fields[5, i, j, k] = 0.1 * (1 + 0.5 * np.random.randn())
                    
                    # Turbulent viscosity
                    flow_fields[6, i, j, k] = 0.001 * (1 + 0.5 * np.random.randn())
        
        return flow_fields


class CFDDataAugmentation:
    """Data augmentation for CFD datasets"""
    
    def __init__(
        self,
        rotation_prob: float = 0.5,
        flip_prob: float = 0.5,
        noise_std: float = 0.01
    ):
        self.rotation_prob = rotation_prob
        self.flip_prob = flip_prob
        self.noise_std = noise_std
    
    def __call__(
        self,
        geometry: torch.Tensor,
        flow_fields: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply augmentations"""
        
        # Random rotation (around vertical axis)
        if torch.rand(1).item() < self.rotation_prob:
            angle = torch.rand(1).item() * 2 * np.pi
            geometry, flow_fields = self._rotate(geometry, flow_fields, angle)
        
        # Random flip (spanwise symmetry)
        if torch.rand(1).item() < self.flip_prob:
            geometry, flow_fields = self._flip(geometry, flow_fields)
        
        # Add noise
        if self.noise_std > 0:
            geometry = geometry + torch.randn_like(geometry) * self.noise_std
            flow_fields = flow_fields + torch.randn_like(flow_fields) * self.noise_std
        
        return geometry, flow_fields
    
    def _rotate(
        self,
        geometry: torch.Tensor,
        flow_fields: torch.Tensor,
        angle: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Rotate around vertical axis"""
        # Simplified rotation (full implementation would use proper 3D rotation)
        return geometry, flow_fields
    
    def _flip(
        self,
        geometry: torch.Tensor,
        flow_fields: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Flip spanwise"""
        geometry = torch.flip(geometry, dims=[2])  # Flip along span
        flow_fields = torch.flip(flow_fields, dims=[2])
        
        # Flip v velocity component (spanwise)
        flow_fields[2] = -flow_fields[2]
        
        return geometry, flow_fields


def create_mock_dataset(
    output_dir: str,
    num_train: int = 1000,
    num_val: int = 200,
    volume_size: Tuple[int, int, int] = (64, 64, 64)
):
    """
    Create mock CFD dataset for testing
    
    In production, this would be replaced with actual CFD simulation data
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val'), exist_ok=True)
    
    print(f"Creating mock dataset: {num_train} train, {num_val} val samples")
    
    # Create training samples
    for i in range(num_train):
        file_path = os.path.join(output_dir, 'train', f'case_{i:06d}.h5')
        
        with h5py.File(file_path, 'w') as f:
            # Geometry
            geometry = np.random.randn(3, *volume_size).astype(np.float32)
            f.create_dataset('geometry', data=geometry)
            
            # Flow fields
            flow_fields = np.random.randn(7, *volume_size).astype(np.float32)
            f.create_dataset('flow_fields', data=flow_fields)
            
            # Boundary mask
            boundary_mask = np.random.randint(0, 2, (1, *volume_size)).astype(np.float32)
            f.create_dataset('boundary_mask', data=boundary_mask)
        
        if (i + 1) % 100 == 0:
            print(f"  Created {i + 1}/{num_train} training samples")
    
    # Create validation samples
    for i in range(num_val):
        file_path = os.path.join(output_dir, 'val', f'case_{i:06d}.h5')
        
        with h5py.File(file_path, 'w') as f:
            geometry = np.random.randn(3, *volume_size).astype(np.float32)
            f.create_dataset('geometry', data=geometry)
            
            flow_fields = np.random.randn(7, *volume_size).astype(np.float32)
            f.create_dataset('flow_fields', data=flow_fields)
            
            boundary_mask = np.random.randint(0, 2, (1, *volume_size)).astype(np.float32)
            f.create_dataset('boundary_mask', data=boundary_mask)
    
    # Create metadata
    metadata = {
        'num_train': num_train,
        'num_val': num_val,
        'volume_size': volume_size,
        'fields': ['pressure', 'u', 'v', 'w', 'k', 'omega', 'nut'],
        'created': str(torch.datetime.datetime.now())
    }
    
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Mock dataset created at {output_dir}")


if __name__ == "__main__":
    # Create mock dataset for testing
    create_mock_dataset(
        output_dir='data/cfd_dataset',
        num_train=1000,
        num_val=200
    )
    
    # Test dataset loading
    dataset = CFDDataset(
        data_dir='data/cfd_dataset',
        split='train'
    )
    
    sample = dataset[0]
    print(f"\nSample shapes:")
    print(f"  Input: {sample['input'].shape}")
    print(f"  Target: {sample['target'].shape}")

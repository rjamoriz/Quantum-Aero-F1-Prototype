"""
AeroTransformer Training Script
Train Vision Transformer + U-Net hybrid on CFD dataset
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import json
from datetime import datetime
from typing import Dict, Any
import numpy as np

from .model import AeroTransformer, create_aero_transformer
from .dataset import CFDDataset


class AeroTransformerTrainer:
    """
    Trainer for AeroTransformer model
    
    Target: <50ms inference on RTX 4090
    Dataset: 100K+ RANS/LES simulations
    """
    
    def __init__(
        self,
        model: AeroTransformer,
        train_dataset: CFDDataset,
        val_dataset: CFDDataset,
        config: Dict[str, Any]
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # Data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.get('batch_size', 4),
            shuffle=True,
            num_workers=config.get('num_workers', 4),
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.get('batch_size', 4),
            shuffle=False,
            num_workers=config.get('num_workers', 4),
            pin_memory=True
        )
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('epochs', 100),
            eta_min=1e-6
        )
        
        # TensorBoard
        self.writer = SummaryWriter(
            log_dir=config.get('log_dir', 'runs/aerotransformer')
        )
        
        # Checkpointing
        self.checkpoint_dir = config.get('checkpoint_dir', 'checkpoints/aerotransformer')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.global_step = 0
        
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        epoch_losses = {
            'total': 0.0,
            'data': 0.0,
            'continuity': 0.0,
            'momentum': 0.0,
            'boundary': 0.0
        }
        
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            inputs = batch['input'].to(self.device)
            targets = batch['target'].to(self.device)
            mesh_info = batch.get('mesh_info', None)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # Compute loss
            loss, loss_dict = self.model.compute_loss(outputs, targets, mesh_info)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.get('grad_clip', 1.0)
            )
            
            self.optimizer.step()
            
            # Accumulate losses
            for key, value in loss_dict.items():
                epoch_losses[key] += value
            
            num_batches += 1
            self.global_step += 1
            
            # Log to TensorBoard
            if batch_idx % self.config.get('log_interval', 10) == 0:
                for key, value in loss_dict.items():
                    self.writer.add_scalar(
                        f'train/{key}_loss',
                        value,
                        self.global_step
                    )
                
                print(f"Epoch {self.current_epoch} [{batch_idx}/{len(self.train_loader)}] "
                      f"Loss: {loss_dict['total']:.4f} "
                      f"(Data: {loss_dict['data']:.4f}, "
                      f"Continuity: {loss_dict['continuity']:.4f}, "
                      f"Momentum: {loss_dict['momentum']:.4f})")
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def validate(self) -> Dict[str, float]:
        """Validate on validation set"""
        self.model.eval()
        
        val_losses = {
            'total': 0.0,
            'data': 0.0,
            'continuity': 0.0,
            'momentum': 0.0,
            'boundary': 0.0
        }
        
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)
                mesh_info = batch.get('mesh_info', None)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Compute loss
                loss, loss_dict = self.model.compute_loss(outputs, targets, mesh_info)
                
                # Accumulate losses
                for key, value in loss_dict.items():
                    val_losses[key] += value
                
                num_batches += 1
        
        # Average losses
        for key in val_losses:
            val_losses[key] /= num_batches
        
        return val_losses
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f'checkpoint_epoch_{self.current_epoch}.pt'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best model (val_loss: {self.best_val_loss:.4f})")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"✓ Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self, num_epochs: int):
        """Main training loop"""
        print("=" * 60)
        print("Starting AeroTransformer Training")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M")
        print(f"Training samples: {len(self.train_dataset)}")
        print(f"Validation samples: {len(self.val_dataset)}")
        print(f"Batch size: {self.config.get('batch_size', 4)}")
        print(f"Epochs: {num_epochs}")
        print("=" * 60)
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_losses = self.train_epoch()
            
            # Validate
            val_losses = self.validate()
            
            # Learning rate step
            self.scheduler.step()
            
            # Log to TensorBoard
            for key in train_losses:
                self.writer.add_scalar(f'epoch/train_{key}_loss', train_losses[key], epoch)
                self.writer.add_scalar(f'epoch/val_{key}_loss', val_losses[key], epoch)
            
            self.writer.add_scalar('epoch/learning_rate', self.scheduler.get_last_lr()[0], epoch)
            
            # Print epoch summary
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_losses['total']:.4f}")
            print(f"  Val Loss: {val_losses['total']:.4f}")
            print(f"  LR: {self.scheduler.get_last_lr()[0]:.2e}")
            
            # Save checkpoint
            is_best = val_losses['total'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_losses['total']
            
            if epoch % self.config.get('save_interval', 10) == 0 or is_best:
                self.save_checkpoint(is_best)
        
        print("\n" + "=" * 60)
        print("Training Complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print("=" * 60)
        
        self.writer.close()


def main():
    """Example training script"""
    
    # Configuration
    config = {
        'model_size': 'base',
        'volume_size': (64, 64, 64),
        'batch_size': 4,
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'epochs': 100,
        'num_workers': 4,
        'grad_clip': 1.0,
        'log_interval': 10,
        'save_interval': 10,
        'log_dir': 'runs/aerotransformer',
        'checkpoint_dir': 'checkpoints/aerotransformer',
        'dataset_path': 'data/cfd_dataset'
    }
    
    # Create model
    model = create_aero_transformer(
        model_size=config['model_size'],
        volume_size=config['volume_size']
    )
    
    # Create datasets (placeholder - implement CFDDataset)
    from .dataset import CFDDataset
    train_dataset = CFDDataset(
        data_dir=config['dataset_path'],
        split='train'
    )
    val_dataset = CFDDataset(
        data_dir=config['dataset_path'],
        split='val'
    )
    
    # Create trainer
    trainer = AeroTransformerTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config
    )
    
    # Train
    trainer.train(num_epochs=config['epochs'])


if __name__ == "__main__":
    main()

"""
ML Surrogate Model Training Pipeline
Trains neural network on synthetic F1 aerodynamic data
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from tqdm import tqdm
import json
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.geo_conv_net import AeroSurrogateNet, create_model
from training.dataloader import create_dataloaders
from inference.predictor import AeroPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AeroTrainer:
    """
    Trainer for aerodynamic surrogate models.
    
    Features:
    - Multi-output training (pressure + forces)
    - Learning rate scheduling
    - Early stopping
    - Model checkpointing
    - TensorBoard logging
    - ONNX export
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate: float = 1e-3,
        output_dir: str = 'outputs'
    ):
        """
        Initialize trainer.
        
        Args:
            model: Neural network model
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device for training
            learning_rate: Initial learning rate
            output_dir: Output directory for checkpoints
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        
        # Output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard
        self.writer = SummaryWriter(self.output_dir / 'tensorboard')
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        logger.info(f"Trainer initialized on {device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def train_epoch(self) -> dict:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        losses = {'cl': 0, 'cd': 0, 'cm': 0, 'pressure': 0}
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(self.device)
            
            # Move targets to device
            targets_device = {}
            for key, value in targets.items():
                targets_device[key] = value.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs, inputs)  # Simplified - needs proper mesh input
            
            # Compute losses
            loss_cl = self.mse_loss(outputs['cl'], targets_device['cl'])
            loss_cd = self.mse_loss(outputs['cd'], targets_device['cd'])
            loss_cm = self.mse_loss(outputs['cm'], targets_device['cm'])
            loss_pressure = self.mse_loss(outputs['pressure'], targets_device['pressure'])
            
            # Combined loss
            loss = loss_cl + loss_cd + loss_cm + 0.1 * loss_pressure
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track losses
            total_loss += loss.item()
            losses['cl'] += loss_cl.item()
            losses['cd'] += loss_cd.item()
            losses['cm'] += loss_cm.item()
            losses['pressure'] += loss_pressure.item()
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        # Average losses
        n_batches = len(self.train_loader)
        avg_loss = total_loss / n_batches
        for key in losses:
            losses[key] /= n_batches
        
        return {'total': avg_loss, **losses}
    
    def validate(self) -> dict:
        """Validate model"""
        self.model.eval()
        
        total_loss = 0
        losses = {'cl': 0, 'cd': 0, 'cm': 0, 'pressure': 0}
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device)
                
                targets_device = {}
                for key, value in targets.items():
                    targets_device[key] = value.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs, inputs)
                
                # Compute losses
                loss_cl = self.mse_loss(outputs['cl'], targets_device['cl'])
                loss_cd = self.mse_loss(outputs['cd'], targets_device['cd'])
                loss_cm = self.mse_loss(outputs['cm'], targets_device['cm'])
                loss_pressure = self.mse_loss(outputs['pressure'], targets_device['pressure'])
                
                loss = loss_cl + loss_cd + loss_cm + 0.1 * loss_pressure
                
                total_loss += loss.item()
                losses['cl'] += loss_cl.item()
                losses['cd'] += loss_cd.item()
                losses['cm'] += loss_cm.item()
                losses['pressure'] += loss_pressure.item()
        
        # Average losses
        n_batches = len(self.val_loader)
        avg_loss = total_loss / n_batches
        for key in losses:
            losses[key] /= n_batches
        
        return {'total': avg_loss, **losses}
    
    def train(
        self,
        n_epochs: int = 100,
        early_stopping_patience: int = 15
    ):
        """
        Train model.
        
        Args:
            n_epochs: Number of epochs
            early_stopping_patience: Patience for early stopping
        """
        logger.info(f"Starting training: {n_epochs} epochs")
        
        for epoch in range(n_epochs):
            self.epoch = epoch
            
            # Train
            train_losses = self.train_epoch()
            
            # Validate
            val_losses = self.validate()
            
            # Learning rate scheduling
            self.scheduler.step(val_losses['total'])
            
            # Log to TensorBoard
            self.writer.add_scalar('Loss/train', train_losses['total'], epoch)
            self.writer.add_scalar('Loss/val', val_losses['total'], epoch)
            self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)
            
            for key in ['cl', 'cd', 'cm', 'pressure']:
                self.writer.add_scalar(f'Loss/train_{key}', train_losses[key], epoch)
                self.writer.add_scalar(f'Loss/val_{key}', val_losses[key], epoch)
            
            # Log
            logger.info(
                f"Epoch {epoch}: "
                f"train_loss={train_losses['total']:.6f}, "
                f"val_loss={val_losses['total']:.6f}"
            )
            
            # Save best model
            if val_losses['total'] < self.best_val_loss:
                self.best_val_loss = val_losses['total']
                self.patience_counter = 0
                self.save_checkpoint('best_model.pt')
                logger.info(f"✓ New best model saved (val_loss={self.best_val_loss:.6f})")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
            
            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt')
        
        logger.info("Training complete!")
        
        # Export to ONNX
        self.export_onnx()
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
        }
        
        filepath = self.output_dir / filename
        torch.save(checkpoint, filepath)
        logger.debug(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        filepath = self.output_dir / filename
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        
        logger.info(f"Checkpoint loaded: {filepath}")
    
    def export_onnx(self, filename: str = 'aero_surrogate.onnx'):
        """Export model to ONNX format"""
        self.model.eval()
        
        # Dummy input
        dummy_input = torch.randn(1, 12).to(self.device)  # Adjust size as needed
        
        output_path = self.output_dir / filename
        
        torch.onnx.export(
            self.model,
            (dummy_input, dummy_input),
            output_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['parameters', 'mesh'],
            output_names=['cl', 'cd', 'cm', 'pressure', 'confidence'],
            dynamic_axes={
                'parameters': {0: 'batch_size'},
                'mesh': {0: 'batch_size'},
                'cl': {0: 'batch_size'},
                'cd': {0: 'batch_size'},
                'cm': {0: 'batch_size'},
                'pressure': {0: 'batch_size'},
                'confidence': {0: 'batch_size'}
            }
        )
        
        logger.info(f"Model exported to ONNX: {output_path}")


def main():
    """Main training script"""
    print("F1 Aerodynamic Surrogate Training")
    print("=" * 60)
    
    # Configuration
    config = {
        'dataset_path': 'data/training-datasets/f1_aero_dataset.h5',
        'batch_size': 32,
        'learning_rate': 1e-3,
        'n_epochs': 100,
        'early_stopping_patience': 15,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'output_dir': 'outputs/training'
    }
    
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Check if dataset exists
    if not Path(config['dataset_path']).exists():
        print(f"\n❌ Dataset not found: {config['dataset_path']}")
        print("   Please run generate_dataset.py first")
        return
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        config['dataset_path'],
        batch_size=config['batch_size'],
        device=config['device']
    )
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Create model
    print("\nCreating model...")
    model = create_model(
        model_type='full',
        device=config['device'],
        mesh_features=3,
        param_features=12,
        hidden_dim=128,
        num_layers=3
    )
    
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = AeroTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=config['device'],
        learning_rate=config['learning_rate'],
        output_dir=config['output_dir']
    )
    
    # Train
    print("\nStarting training...")
    trainer.train(
        n_epochs=config['n_epochs'],
        early_stopping_patience=config['early_stopping_patience']
    )
    
    print("\n✅ Training complete!")
    print(f"   Best validation loss: {trainer.best_val_loss:.6f}")
    print(f"   Model saved: {config['output_dir']}/best_model.pt")
    print(f"   ONNX exported: {config['output_dir']}/aero_surrogate.onnx")


if __name__ == "__main__":
    main()

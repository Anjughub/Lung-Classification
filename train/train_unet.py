import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import time
from collections import defaultdict
import matplotlib.pyplot as plt
from pathlib import Path
import json

class SegmentationTrainer:
    def __init__(self, model, optimizer, criterion, device='cuda', 
                 scheduler=None, early_stopping_patience=None):
        
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.early_stopping_patience = early_stopping_patience
        
        self.model.to(device)
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_iou': [],
            'val_iou': [],
            'train_dice': [],
            'val_dice': [],
            'learning_rates': []
        }
        
        # Early stopping variables
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_model_state = None
        
    def calculate_iou(self, pred, target, threshold=0.5):
        """Calculate Intersection over Union (IoU)"""
        pred = (pred > threshold).float()
        target = (target > threshold).float()
        
        intersection = (pred * target).sum(dim=(1, 2, 3))
        union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
        
        iou = (intersection + 1e-8) / (union + 1e-8)
        return iou.mean().item()
    
    def calculate_dice(self, pred, target, threshold=0.5):
        pred = (pred > threshold).float()
        target = (target > threshold).float()
        
        intersection = (pred * target).sum(dim=(1, 2, 3))
        dice = (2 * intersection + 1e-8) / (pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1e-8)
        
        return dice.mean().item()
    
    def train_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0
        running_iou = 0.0
        running_dice = 0.0
        
        # Create progress bar
        pbar = tqdm(train_loader, desc='Training', leave=False)
        
        for batch_idx, (images, masks) in enumerate(pbar):
            # Move to device
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            
            # Calculate loss
            loss = self.criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                # Apply sigmoid if using BCEWithLogitsLoss
                if isinstance(self.criterion, nn.BCEWithLogitsLoss):
                    pred_probs = torch.sigmoid(outputs)
                else:
                    pred_probs = outputs
                
                iou = self.calculate_iou(pred_probs, masks)
                dice = self.calculate_dice(pred_probs, masks)
            
            # Update running metrics
            running_loss += loss.item()
            running_iou += iou
            running_dice += dice
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'IoU': f'{iou:.4f}',
                'Dice': f'{dice:.4f}'
            })
        
        # Calculate epoch averages
        epoch_loss = running_loss / len(train_loader)
        epoch_iou = running_iou / len(train_loader)
        epoch_dice = running_dice / len(train_loader)
        
        return epoch_loss, epoch_iou, epoch_dice
    
    def validate_epoch(self, val_loader):
        self.model.eval()
        running_loss = 0.0
        running_iou = 0.0
        running_dice = 0.0
        

        pbar = tqdm(val_loader, desc='Validation', leave=False)
        
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(pbar):
                # Move to device
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Calculate loss
                loss = self.criterion(outputs, masks)
                
                # Calculate metrics
                if isinstance(self.criterion, nn.BCEWithLogitsLoss):
                    pred_probs = torch.sigmoid(outputs)
                else:
                    pred_probs = outputs
                
                iou = self.calculate_iou(pred_probs, masks)
                dice = self.calculate_dice(pred_probs, masks)
                
                # Update running metrics
                running_loss += loss.item()
                running_iou += iou
                running_dice += dice
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'IoU': f'{iou:.4f}',
                    'Dice': f'{dice:.4f}'
                })
        

        epoch_loss = running_loss / len(val_loader)
        epoch_iou = running_iou / len(val_loader)
        epoch_dice = running_dice / len(val_loader)
        
        return epoch_loss, epoch_iou, epoch_dice
    
    def train(self, train_loader, val_loader, num_epochs, 
              save_dir='checkpoints', save_best=True, print_every=1):
        
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print("-" * 80)
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # Train epoch
            train_loss, train_iou, train_dice = self.train_epoch(train_loader)
            
            # Validate epoch
            val_loss, val_iou, val_dice = self.validate_epoch(val_loader)
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_iou'].append(train_iou)
            self.history['val_iou'].append(val_iou)
            self.history['train_dice'].append(train_dice)
            self.history['val_dice'].append(val_dice)
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Print epoch results
            if (epoch + 1) % print_every == 0:
                print(f"\nEpoch [{epoch+1}/{num_epochs}] - Time: {epoch_time:.2f}s")
                print(f"Train Loss: {train_loss:.4f} | Train IoU: {train_iou:.4f} | Train Dice: {train_dice:.4f}")
                print(f"Val Loss:   {val_loss:.4f} | Val IoU:   {val_iou:.4f} | Val Dice:   {val_dice:.4f}")
                print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
                
                # Show improvement indicators
                if val_loss < self.best_val_loss:
                    print("🎉 New best validation loss!")
                print("-" * 80)
            
            # Save best model
            if save_best and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                self.patience_counter = 0
                
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_iou': val_iou,
                    'val_dice': val_dice,
                    'history': self.history
                }, save_dir / 'best_model.pth')
            
            # Early stopping
            if self.early_stopping_patience:
                if val_loss >= self.best_val_loss:
                    self.patience_counter += 1
                
                if self.patience_counter >= self.early_stopping_patience:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    print(f"Best validation loss: {self.best_val_loss:.4f}")
                    break
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                    'history': self.history
                }, save_dir / f'checkpoint_epoch_{epoch+1}.pth')
        
        # Training completed
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/3600:.2f} hours")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        # Load best model
        if save_best and self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
            print("Loaded best model weights")
        
        # Save training history
        with open(save_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        
        return self.history
    
    def plot_training_history(self, save_path=None):

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss', color='blue')
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss', color='red')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # IoU plot
        axes[0, 1].plot(self.history['train_iou'], label='Train IoU', color='blue')
        axes[0, 1].plot(self.history['val_iou'], label='Val IoU', color='red')
        axes[0, 1].set_title('Training and Validation IoU')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('IoU')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Dice plot
        axes[1, 0].plot(self.history['train_dice'], label='Train Dice', color='blue')
        axes[1, 0].plot(self.history['val_dice'], label='Val Dice', color='red')
        axes[1, 0].set_title('Training and Validation Dice')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Dice')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate plot
        axes[1, 1].plot(self.history['learning_rates'], color='green')
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

def train_segmentation_model(model, train_loader, val_loader, num_epochs=50, 
                           learning_rate=1e-3, device='cuda'):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    trainer = SegmentationTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        early_stopping_patience=15
    )
    
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        save_dir='checkpoints',
        save_best=True,
        print_every=1
    )
    
    trainer.plot_training_history('training_plots.png')
    
    return trainer, history

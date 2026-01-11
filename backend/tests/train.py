import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import NAFNet, download_pretrained_weights

class DeblurDataset(Dataset):
    """Dataset for image deblurring"""
    def __init__(self, blur_dir, sharp_dir, transform=None):
        self.blur_dir = blur_dir
        self.sharp_dir = sharp_dir
        self.transform = transform
        self.image_files = sorted(os.listdir(blur_dir))
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        blur_path = os.path.join(self.blur_dir, self.image_files[idx])
        sharp_path = os.path.join(self.sharp_dir, self.image_files[idx])
        
        blur_img = cv2.imread(blur_path)
        sharp_img = cv2.imread(sharp_path)
        
        blur_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2RGB)
        sharp_img = cv2.cvtColor(sharp_img, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            blur_img = self.transform(blur_img)
            sharp_img = self.transform(sharp_img)
        
        return blur_img, sharp_img

class PSNRLoss(nn.Module):
    """PSNR Loss"""
    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super().__init__()
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False
            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            pred, target = pred / 255., target / 255.
        
        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()

def train_nafnet(config):
    """
    Train NAFNet for deblurring
    
    Args:
        config: Dictionary with training configuration
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Datasets
    train_dataset = DeblurDataset(
        blur_dir=os.path.join(config['data_dir'], 'train', 'blur'),
        sharp_dir=os.path.join(config['data_dir'], 'train', 'sharp'),
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Model
    model = NAFNet(
        img_channel=3,
        width=config['width'],
        middle_blk_num=config['middle_blk_num'],
        enc_blk_nums=config['enc_blk_nums'],
        dec_blk_nums=config['dec_blk_nums']
    ).to(device)
    
    # Load pretrained weights if specified
    start_epoch = 0
    if config.get('use_pretrained'):
        pretrained_path = download_pretrained_weights()
        if pretrained_path and os.path.exists(pretrained_path):
            print(f"Loading pretrained weights from {pretrained_path}")
            try:
                state_dict = torch.load(pretrained_path, map_location=device)
                # Handle different state dict formats
                if 'params' in state_dict:
                    state_dict = state_dict['params']
                elif 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                
                # Load with strict=False to allow partial loading
                model.load_state_dict(state_dict, strict=False)
                print("Pretrained weights loaded successfully!")
            except Exception as e:
                print(f"Warning: Could not load pretrained weights: {e}")
                print("Training from scratch instead.")
        else:
            print("No pretrained weights available. Training from scratch.")
    
    # Resume from checkpoint if specified
    if config.get('resume_checkpoint'):
        print(f"Resuming from checkpoint {config['resume_checkpoint']}")
        checkpoint = torch.load(config['resume_checkpoint'], map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resuming from epoch {start_epoch}")
    
    # Loss function
    criterion = PSNRLoss()
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        betas=(0.9, 0.9),
        weight_decay=0.0
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['epochs'],
        eta_min=1e-6  # Changed from 1e-7 to 1e-6
    )
    
    # Training history
    history = {
        'train_loss': [],
    }
    
    # Training loop
    print(f"\nStarting training for {config['epochs']} epochs...")
    
    for epoch in range(start_epoch, start_epoch + config['epochs']):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{start_epoch + config['epochs']}")
        
        for blur_imgs, sharp_imgs in pbar:
            blur_imgs = blur_imgs.to(device)
            sharp_imgs = sharp_imgs.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            pred_imgs = model(blur_imgs)
            
            # Calculate loss
            loss = criterion(pred_imgs, sharp_imgs)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            pbar.set_postfix({'Loss': loss.item()})
        
        # Update learning rate
        scheduler.step()
        
        # Average loss
        avg_loss = epoch_loss / len(train_loader)
        history['train_loss'].append(avg_loss)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{start_epoch + config['epochs']} Summary:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.7f}")
        
        # Save checkpoint
        if (epoch + 1) % config['save_interval'] == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'history': history
            }
            torch.save(checkpoint, os.path.join(config['checkpoint_dir'], 
                                                f'checkpoint_epoch_{epoch+1}.pth'))
            print(f"Checkpoint saved at epoch {epoch+1}")
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(config['checkpoint_dir'], 'nafnet_final.pth'))
    
    # Plot training curve
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('NAFNet Training Loss')
    plt.grid(True)
    plt.savefig(os.path.join(config['checkpoint_dir'], 'training_curve.png'))
    print(f"\nTraining curve saved to {config['checkpoint_dir']}/training_curve.png")
    
    print("\nTraining completed!")

if __name__ == '__main__':
    # Training configuration
    config = {
        'data_dir': './deblur_dataset',
        'checkpoint_dir': './checkpoints',
        'batch_size': 10,
        'epochs': 5,
        'lr': 0.0005,  # Increased for fine-tuning (was 0.0001)
        'width': 32,
        'middle_blk_num': 12,
        'enc_blk_nums': [2, 2, 4, 8],
        'dec_blk_nums': [2, 2, 2, 2],
        'save_interval': 10,
        
        # Fine-tuning options
        'use_pretrained': True,  # Set to True to use pretrained weights
        'resume_checkpoint': None,  # Path to resume training
    }
    
    # Create checkpoint directory
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    # Train model
    train_nafnet(config)
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
from model import Generator, Discriminator, PerceptualLoss, weights_init

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

def train_deblur_gan(config):
    """
    Train DeblurGAN-v2
    
    Args:
        config: Dictionary with training configuration
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
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
    
    # Models
    generator = Generator(
        in_channels=3,
        out_channels=3,
        base_channels=config['base_channels'],
        num_residual_blocks=config['num_residual_blocks']
    ).to(device)
    
    discriminator = Discriminator(
        in_channels=3,
        base_channels=config['base_channels']
    ).to(device)
    
    # Initialize weights
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    
    # Loss functions
    criterion_gan = nn.MSELoss()
    criterion_pixel = nn.L1Loss()
    perceptual_loss = PerceptualLoss().to(device)
    
    # Optimizers
    optimizer_g = optim.Adam(
        generator.parameters(),
        lr=config['lr'],
        betas=(0.5, 0.999)
    )
    
    optimizer_d = optim.Adam(
        discriminator.parameters(),
        lr=config['lr'],
        betas=(0.5, 0.999)
    )
    
    # Learning rate schedulers
    scheduler_g = optim.lr_scheduler.StepLR(optimizer_g, step_size=50, gamma=0.5)
    scheduler_d = optim.lr_scheduler.StepLR(optimizer_d, step_size=50, gamma=0.5)
    
    # Training history
    history = {
        'g_loss': [],
        'd_loss': [],
        'pixel_loss': [],
        'perceptual_loss': [],
        'gan_loss': []
    }
    
    # Training loop
    print(f"\nStarting training for {config['epochs']} epochs...")
    
    for epoch in range(config['epochs']):
        generator.train()
        discriminator.train()
        
        epoch_g_loss = 0
        epoch_d_loss = 0
        epoch_pixel_loss = 0
        epoch_perceptual_loss = 0
        epoch_gan_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        
        for i, (blur_imgs, sharp_imgs) in enumerate(pbar):
            blur_imgs = blur_imgs.to(device)
            sharp_imgs = sharp_imgs.to(device)
            
            batch_size = blur_imgs.size(0)
            
            # -----------------
            # Train Generator
            # -----------------
            optimizer_g.zero_grad()
            
            # Generate deblurred images
            deblurred_imgs = generator(blur_imgs)
            
            # Adversarial loss
            pred_fake = discriminator(deblurred_imgs)
            valid = torch.ones_like(pred_fake).to(device)
            loss_gan = criterion_gan(pred_fake, valid)
            
            # Pixel loss
            loss_pixel = criterion_pixel(deblurred_imgs, sharp_imgs)
            
            # Perceptual loss
            loss_perceptual = perceptual_loss(deblurred_imgs, sharp_imgs)
            
            # Total generator loss
            loss_g = (config['lambda_pixel'] * loss_pixel + 
                     config['lambda_perceptual'] * loss_perceptual +
                     config['lambda_gan'] * loss_gan)
            
            loss_g.backward()
            optimizer_g.step()
            
            # ---------------------
            # Train Discriminator
            # ---------------------
            optimizer_d.zero_grad()
            
            # Real loss
            pred_real = discriminator(sharp_imgs)
            valid = torch.ones_like(pred_real).to(device)
            loss_real = criterion_gan(pred_real, valid)
            
            # Fake loss
            pred_fake = discriminator(deblurred_imgs.detach())
            fake = torch.zeros_like(pred_fake).to(device)
            loss_fake = criterion_gan(pred_fake, fake)
            
            # Total discriminator loss
            loss_d = (loss_real + loss_fake) / 2
            
            loss_d.backward()
            optimizer_d.step()
            
            # Update metrics
            epoch_g_loss += loss_g.item()
            epoch_d_loss += loss_d.item()
            epoch_pixel_loss += loss_pixel.item()
            epoch_perceptual_loss += loss_perceptual.item()
            epoch_gan_loss += loss_gan.item()
            
            pbar.set_postfix({
                'G_loss': loss_g.item(),
                'D_loss': loss_d.item(),
                'Pixel': loss_pixel.item()
            })
        
        # Average losses
        num_batches = len(train_loader)
        history['g_loss'].append(epoch_g_loss / num_batches)
        history['d_loss'].append(epoch_d_loss / num_batches)
        history['pixel_loss'].append(epoch_pixel_loss / num_batches)
        history['perceptual_loss'].append(epoch_perceptual_loss / num_batches)
        history['gan_loss'].append(epoch_gan_loss / num_batches)
        
        # Update learning rates
        scheduler_g.step()
        scheduler_d.step()
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{config['epochs']} Summary:")
        print(f"  G_loss: {history['g_loss'][-1]:.4f}")
        print(f"  D_loss: {history['d_loss'][-1]:.4f}")
        print(f"  Pixel_loss: {history['pixel_loss'][-1]:.4f}")
        print(f"  Perceptual_loss: {history['perceptual_loss'][-1]:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % config['save_interval'] == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_g_state_dict': optimizer_g.state_dict(),
                'optimizer_d_state_dict': optimizer_d.state_dict(),
                'history': history
            }
            torch.save(checkpoint, os.path.join(config['checkpoint_dir'], 
                                                f'checkpoint_epoch_{epoch+1}.pth'))
            print(f"Checkpoint saved at epoch {epoch+1}")
    
    # Save final model
    torch.save(generator.state_dict(), os.path.join(config['checkpoint_dir'], 'generator_final.pth'))
    torch.save(discriminator.state_dict(), os.path.join(config['checkpoint_dir'], 'discriminator_final.pth'))
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['g_loss'], label='Generator Loss')
    plt.plot(history['d_loss'], label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('GAN Losses')
    
    plt.subplot(1, 3, 2)
    plt.plot(history['pixel_loss'], label='Pixel Loss (L1)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Pixel-wise Loss')
    
    plt.subplot(1, 3, 3)
    plt.plot(history['perceptual_loss'], label='Perceptual Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Perceptual Loss')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config['checkpoint_dir'], 'training_curves.png'))
    print(f"\nTraining curves saved to {config['checkpoint_dir']}/training_curves.png")
    
    print("\nTraining completed!")

if __name__ == '__main__':
    # Training configuration
    config = {
        'data_dir': './deblur_dataset',
        'checkpoint_dir': './checkpoints',
        'batch_size': 4,
        'epochs': 1,
        'lr': 0.0001,
        'base_channels': 64,
        'num_residual_blocks': 9,
        'lambda_pixel': 100,
        'lambda_perceptual': 10,
        'lambda_gan': 1,
        'save_interval': 10
    }
    
    # Create checkpoint directory
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    # Train model
    train_deblur_gan(config)
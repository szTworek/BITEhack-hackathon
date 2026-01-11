import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from model import NAFNet

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
        
        return blur_img, sharp_img, self.image_files[idx]

def tensor_to_image(tensor):
    """Convert tensor to numpy image"""
    img = tensor.cpu().clone()
    if img.dim() == 4:
        img = img.squeeze(0)
    img = img.permute(1, 2, 0).numpy()
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return img

def calculate_metrics(deblurred, sharp):
    """Calculate PSNR and SSIM metrics"""
    deblurred_np = tensor_to_image(deblurred)
    sharp_np = tensor_to_image(sharp)
    
    psnr_value = psnr(sharp_np, deblurred_np, data_range=255)
    ssim_value = ssim(sharp_np, deblurred_np, channel_axis=2, data_range=255)
    
    return psnr_value, ssim_value

def test_model(config):
    """
    Test NAFNet model on test set
    
    Args:
        config: Dictionary with test configuration
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Test dataset
    test_dataset = DeblurDataset(
        blur_dir=os.path.join(config['data_dir'], 'test', 'blur'),
        sharp_dir=os.path.join(config['data_dir'], 'test', 'sharp'),
        transform=transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2
    )
    
    # Load model
    model = NAFNet(
        img_channel=3,
        width=config['width'],
        middle_blk_num=config['middle_blk_num'],
        enc_blk_nums=config['enc_blk_nums'],
        dec_blk_nums=config['dec_blk_nums']
    ).to(device)
    
    # Load checkpoint
    checkpoint_path = os.path.join(config['checkpoint_dir'], config['model_name'])
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    
    print(f"Loading model from {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    # Create output directory
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # Test metrics
    total_psnr = 0
    total_ssim = 0
    num_samples = 0
    
    print("\nTesting model on test set...")
    
    with torch.no_grad():
        for blur_imgs, sharp_imgs, filenames in tqdm(test_loader):
            blur_imgs = blur_imgs.to(device)
            sharp_imgs = sharp_imgs.to(device)
            
            # Generate deblurred images
            deblurred_imgs = model(blur_imgs)
            
            # Calculate metrics
            for i in range(blur_imgs.size(0)):
                psnr_val, ssim_val = calculate_metrics(deblurred_imgs[i], sharp_imgs[i])
                total_psnr += psnr_val
                total_ssim += ssim_val
                num_samples += 1
                
                # Save visualization
                if config['save_visualizations']:
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    
                    # Blurred image
                    blur_np = tensor_to_image(blur_imgs[i])
                    axes[0].imshow(blur_np)
                    axes[0].set_title('Blurred Input')
                    axes[0].axis('off')
                    
                    # Deblurred image
                    deblurred_np = tensor_to_image(deblurred_imgs[i])
                    axes[1].imshow(deblurred_np)
                    axes[1].set_title(f'Deblurred Output\nPSNR: {psnr_val:.2f} dB\nSSIM: {ssim_val:.4f}')
                    axes[1].axis('off')
                    
                    # Ground truth
                    sharp_np = tensor_to_image(sharp_imgs[i])
                    axes[2].imshow(sharp_np)
                    axes[2].set_title('Ground Truth')
                    axes[2].axis('off')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f'result_{filenames[i]}'), 
                               bbox_inches='tight', dpi=150)
                    plt.close()
    
    # Calculate average metrics
    avg_psnr = total_psnr / num_samples
    avg_ssim = total_ssim / num_samples
    
    print("\n" + "="*50)
    print("Test Results:")
    print("="*50)
    print(f"Number of test samples: {num_samples}")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print("="*50)
    
    # Save metrics to file
    with open(os.path.join(output_dir, 'test_metrics.txt'), 'w') as f:
        f.write(f"Test Results\n")
        f.write(f"{'='*50}\n")
        f.write(f"Model: {config['model_name']}\n")
        f.write(f"Number of test samples: {num_samples}\n")
        f.write(f"Average PSNR: {avg_psnr:.2f} dB\n")
        f.write(f"Average SSIM: {avg_ssim:.4f}\n")
    
    print(f"\nTest results saved to {output_dir}")
    
    # Create comparison grid
    if config['save_visualizations'] and num_samples > 0:
        print("\nCreating comparison grid...")
        create_comparison_grid(test_loader, model, device, output_dir, num_samples=min(9, num_samples))

def create_comparison_grid(test_loader, model, device, output_dir, num_samples=9):
    """Create a grid of comparisons"""
    model.eval()
    
    samples_collected = 0
    blur_list = []
    deblurred_list = []
    sharp_list = []
    
    with torch.no_grad():
        for blur_imgs, sharp_imgs, _ in test_loader:
            if samples_collected >= num_samples:
                break
            
            blur_imgs = blur_imgs.to(device)
            sharp_imgs = sharp_imgs.to(device)
            deblurred_imgs = model(blur_imgs)
            
            blur_list.append(tensor_to_image(blur_imgs[0]))
            deblurred_list.append(tensor_to_image(deblurred_imgs[0]))
            sharp_list.append(tensor_to_image(sharp_imgs[0]))
            samples_collected += 1
    
    # Create grid
    rows = int(np.ceil(np.sqrt(num_samples)))
    cols = int(np.ceil(num_samples / rows))
    
    fig, axes = plt.subplots(rows, cols * 3, figsize=(cols * 9, rows * 3))
    
    for i in range(num_samples):
        row = i // cols
        col = (i % cols) * 3
        
        if rows == 1:
            ax_blur = axes[col] if cols > 1 else axes
            ax_deblur = axes[col + 1] if cols > 1 else axes
            ax_sharp = axes[col + 2] if cols > 1 else axes
        else:
            ax_blur = axes[row, col]
            ax_deblur = axes[row, col + 1]
            ax_sharp = axes[row, col + 2]
        
        ax_blur.imshow(blur_list[i])
        ax_blur.set_title('Blurred')
        ax_blur.axis('off')
        
        ax_deblur.imshow(deblurred_list[i])
        ax_deblur.set_title('Deblurred')
        ax_deblur.axis('off')
        
        ax_sharp.imshow(sharp_list[i])
        ax_sharp.set_title('Sharp')
        ax_sharp.axis('off')
    
    # Hide unused subplots
    if rows > 1 or cols > 1:
        for i in range(num_samples, rows * cols):
            row = i // cols
            for j in range(3):
                col = (i % cols) * 3 + j
                if rows == 1:
                    if cols > 1:
                        axes[col].axis('off')
                else:
                    axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_grid.png'), 
               bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"Comparison grid saved to {output_dir}/comparison_grid.png")

if __name__ == '__main__':
    # Test configuration
    config = {
        'data_dir': './deblur_dataset',
        'checkpoint_dir': './checkpoints',
        'model_name': 'nafnet_final.pth',
        'output_dir': './test_results',
        'width': 32,
        'middle_blk_num': 12,
        'enc_blk_nums': [2, 2, 4, 8],
        'dec_blk_nums': [2, 2, 2, 2],
        'save_visualizations': True
    }
    
    # Test model
    test_model(config)
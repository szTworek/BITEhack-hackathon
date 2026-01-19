import os
import cv2
import numpy as np
from pathlib import Path
import random
from tqdm import tqdm

KERNEL_SIZE = 101

def apply_gaussian_blur(img, kernel_size=KERNEL_SIZE):
    """Apply Gaussian blur"""
    sigma = random.uniform(1, 5)
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)

def create_blurred_dataset(input_path, output_dir, train_ratio=0.8, img_size=256):
    """
    Create blurred dataset with train/test split
    
    Args:
        input_path: Path containing boat*.png images
        output_dir: Output directory for dataset
        train_ratio: Ratio for train/test split
        img_size: Size to resize images
    """
    
    # Create directory structure
    dirs = {
        'train_sharp': os.path.join(output_dir, 'train', 'sharp'),
        'train_blur': os.path.join(output_dir, 'train', 'blur'),
        'test_sharp': os.path.join(output_dir, 'test', 'sharp'),
        'test_blur': os.path.join(output_dir, 'test', 'blur'),
    }
    
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    
    # Get all boat images
    img_files = sorted([f for f in os.listdir(input_path) if f.startswith('boat') and f.endswith('.png')])
    
    if not img_files:
        raise ValueError(f"No boat*.png images found in {input_path}")
    
    print(f"Found {len(img_files)} images")
    
    # Shuffle and split
    random.shuffle(img_files)
    split_idx = int(len(img_files) * train_ratio)
    train_files = img_files[:split_idx]
    test_files = img_files[split_idx:]
    
    print(f"Train: {len(train_files)}, Test: {len(test_files)}")
    
    # Process train images
    print("\nProcessing train images...")
    for idx, fname in enumerate(tqdm(train_files)):
        img_path = os.path.join(input_path, fname)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Warning: Could not read {img_path}")
            continue
        
        # Resize
        img = cv2.resize(img, (img_size, img_size))
        
        # Apply Gaussian blur
        blurred = apply_gaussian_blur(img.copy())
        
        # Save
        cv2.imwrite(os.path.join(dirs['train_sharp'], f'{idx:04d}.png'), img)
        cv2.imwrite(os.path.join(dirs['train_blur'], f'{idx:04d}.png'), blurred)
    
    # Process test images
    print("\nProcessing test images...")
    for idx, fname in enumerate(tqdm(test_files)):
        img_path = os.path.join(input_path, fname)
        img = cv2.imread(img_path)
        
        if img is None:
            continue
        
        img = cv2.resize(img, (img_size, img_size))
        
        # Apply Gaussian blur
        blurred = apply_gaussian_blur(img.copy())
        
        cv2.imwrite(os.path.join(dirs['test_sharp'], f'{idx:04d}.png'), img)
        cv2.imwrite(os.path.join(dirs['test_blur'], f'{idx:04d}.png'), blurred)
    
    print(f"\nDataset created successfully in {output_dir}")
    print(f"Train samples: {len(train_files)}")
    print(f"Test samples: {len(test_files)}")

if __name__ == '__main__':
    # Configure paths
    INPUT_PATH = "/home/albert/.cache/kagglehub/datasets/andrewmvd/ship-detection/versions/1/images"
    OUTPUT_DIR = './deblur_dataset'
    
    create_blurred_dataset(INPUT_PATH, OUTPUT_DIR, train_ratio=0.8, img_size=256)
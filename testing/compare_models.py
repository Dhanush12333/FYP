# testing/compare_models.py
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

from models.generator_3d_unet import Generator3D_UNet
from dataloaders.dataset_3d import MRI3DDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# Paths
root_dir = r"D:\FYP\Processed_MRI"
val_csv = r"D:\FYP\MRI_GAN_Project\data\val_volumes.csv"

# Model paths
ORIGINAL_MODEL = r"checkpoints/generator_epoch_35.pth"  # Original trained model
ENHANCED_MODEL = r"checkpoints/generator_epoch_35.pth"   # Enhanced model (same for now)

stack_depth = 16
batch_size = 1

def load_model(checkpoint_path, use_cbam=False):
    """Load model with specified CBAM setting"""
    model = Generator3D_UNet(
        in_channels=1, out_channels=1, base_filters=32,
        num_levels=4, preserve_depth=True,
        use_cbam=use_cbam, cbam_position='both'
    ).to(device)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Remove 'module.' prefix
        new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)
        print(f"✅ Loaded: {checkpoint_path}")
    else:
        print(f"❌ Not found: {checkpoint_path}")
        return None
    
    model.eval()
    return model

def compute_metrics(real_slice, gen_slice):
    """Compute metrics"""
    real_slice = np.clip(real_slice, 0, 1)
    gen_slice = np.clip(gen_slice, 0, 1)
    
    try:
        ssim_val = ssim(real_slice, gen_slice, data_range=1.0)
    except:
        ssim_val = 0.0
    
    try:
        psnr_val = psnr(real_slice, gen_slice, data_range=1.0)
    except:
        psnr_val = 0.0
    
    mae_val = np.mean(np.abs(real_slice - gen_slice))
    mse_val = np.mean((real_slice - gen_slice) ** 2)
    
    return {'ssim': ssim_val, 'psnr': psnr_val, 'mae': mae_val, 'mse': mse_val}

def compare_models():
    """Compare original vs enhanced model"""
    print("\n" + "="*60)
    print("MODEL COMPARISON: Original vs Enhanced")
    print("="*60)
    
    # Load models
    model_original = load_model(ORIGINAL_MODEL, use_cbam=False)
    model_enhanced = load_model(ENHANCED_MODEL, use_cbam=True)
    
    if model_original is None or model_enhanced is None:
        print("❌ Could not load both models")
        return
    
    # Load validation data
    val_dataset = MRI3DDataset(val_csv, root_dir, stack_depth=stack_depth)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Store results
    results = {'original': [], 'enhanced': []}
    
    print("\n🔍 Comparing models...")
    with torch.no_grad():
        for batch in tqdm(val_loader):
            if isinstance(batch, (list, tuple)) and len(batch) >= 4:
                input_vol, target_vol, missing_idx, meta = batch[:4]
            else:
                input_vol, target_vol = batch[0], batch[1]
                missing_idx = batch[2] if len(batch) > 2 else 0
            
            input_vol = input_vol.to(device)
            target_vol = target_vol.to(device)
            
            # Get missing index
            if isinstance(missing_idx, torch.Tensor):
                missing_local = missing_idx.item()
            else:
                missing_local = missing_idx if isinstance(missing_idx, int) else missing_idx[0]
            
            # Generate with both models
            gen_original = model_original(input_vol)
            gen_enhanced = model_enhanced(input_vol)
            
            # Extract missing slice
            real_slice = target_vol[0, 0, missing_local, :, :].detach().cpu().numpy()
            orig_slice = gen_original[0, 0, missing_local, :, :].detach().cpu().numpy()
            enh_slice = gen_enhanced[0, 0, missing_local, :, :].detach().cpu().numpy()
            
            # Compute metrics
            results['original'].append(compute_metrics(real_slice, orig_slice))
            results['enhanced'].append(compute_metrics(real_slice, enh_slice))
    
    # Create comparison DataFrame
    orig_df = pd.DataFrame(results['original'])
    enh_df = pd.DataFrame(results['enhanced'])
    
    # Print results
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    print(f"\n{'Metric':<10} {'Original':<15} {'Enhanced':<15} {'Improvement':<15}")
    print("-"*55)
    
    for metric in ['ssim', 'psnr', 'mae', 'mse']:
        orig_mean = orig_df[metric].mean()
        enh_mean = enh_df[metric].mean()
        
        if metric in ['ssim', 'psnr']:
            improvement = ((enh_mean - orig_mean) / orig_mean) * 100
            arrow = "↑" if improvement > 0 else "↓"
        else:
            improvement = ((orig_mean - enh_mean) / orig_mean) * 100
            arrow = "↓" if improvement > 0 else "↑"
        
        print(f"{metric:<10} {orig_mean:<15.4f} {enh_mean:<15.4f} {arrow} {abs(improvement):.1f}%")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # SSIM comparison
    axes[0,0].scatter(orig_df['ssim'], enh_df['ssim'], alpha=0.5)
    axes[0,0].plot([0, 1], [0, 1], 'r--', label='Perfect')
    axes[0,0].set_xlabel('Original Model SSIM')
    axes[0,0].set_ylabel('Enhanced Model SSIM')
    axes[0,0].set_title('SSIM Comparison')
    axes[0,0].legend()
    
    # PSNR comparison
    axes[0,1].hist(orig_df['psnr'], bins=20, alpha=0.5, label='Original', color='blue')
    axes[0,1].hist(enh_df['psnr'], bins=20, alpha=0.5, label='Enhanced', color='green')
    axes[0,1].set_xlabel('PSNR (dB)')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].set_title('PSNR Distribution')
    axes[0,1].legend()
    
    # MAE comparison
    axes[1,0].hist(orig_df['mae'], bins=20, alpha=0.5, label='Original', color='blue')
    axes[1,0].hist(enh_df['mae'], bins=20, alpha=0.5, label='Enhanced', color='green')
    axes[1,0].set_xlabel('MAE')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].set_title('MAE Distribution')
    axes[1,0].legend()
    
    # Improvement histogram
    improvement = enh_df['ssim'] - orig_df['ssim']
    axes[1,1].hist(improvement, bins=20, color='purple', alpha=0.7)
    axes[1,1].axvline(0, color='red', linestyle='--')
    axes[1,1].axvline(improvement.mean(), color='green', linestyle='--', 
                      label=f'Mean Improvement: {improvement.mean():.4f}')
    axes[1,1].set_xlabel('SSIM Improvement')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].set_title('Improvement Distribution')
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig("validation_results/model_comparison.png", dpi=150)
    plt.show()
    
    # Save results
    comparison_df = pd.DataFrame({
        'Metric': ['SSIM', 'PSNR', 'MAE', 'MSE'],
        'Original': [orig_df['ssim'].mean(), orig_df['psnr'].mean(), orig_df['mae'].mean(), orig_df['mse'].mean()],
        'Enhanced': [enh
# testing/validate_model.py
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

# Project imports
from models.generator_3d_unet import Generator3D_UNet
from dataloaders.dataset_3d import MRI3DDataset
from dataloaders.dataloader_3d import get_dataloaders_3d

# -----------------------
# Configuration
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# Paths (UPDATE THESE)
root_dir = r"D:\FYP\Processed_MRI"
val_csv = r"D:\FYP\MRI_GAN_Project\data\val_volumes.csv"

# Model checkpoint to validate
CHECKPOINT_PATH = r"D:\FYP\MRI_GAN_Project\checkpoints\generator_epoch_35.pth"  # Change to your checkpoint

# Validation configuration
stack_depth = 16
batch_size = 1  # Keep 1 for validation

# Output directories
os.makedirs("validation_results", exist_ok=True)
os.makedirs("validation_results/comparisons", exist_ok=True)
os.makedirs("validation_results/best_worst", exist_ok=True)

# -----------------------
# Load Model
# -----------------------
def load_model(checkpoint_path, use_cbam=True, cbam_position='both'):
    """Load the generator model with checkpoint"""
    print(f"📦 Loading model from: {checkpoint_path}")
    
    # Create model with same configuration as training
    model = Generator3D_UNet(
        in_channels=1,
        out_channels=1,
        base_filters=32,
        num_levels=4,
        preserve_depth=True,
        use_cbam=use_cbam,
        cbam_position=cbam_position
    ).to(device)
    
    # Load checkpoint
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif isinstance(checkpoint, dict) and 'generator_state_dict' in checkpoint:
            state_dict = checkpoint['generator_state_dict']
        else:
            state_dict = checkpoint
        
        # Remove 'module.' prefix if present (from DataParallel)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        
        model.load_state_dict(new_state_dict, strict=False)
        print(f"✅ Model loaded successfully!")
        
        # Print epoch info if available
        if isinstance(checkpoint, dict) and 'epoch' in checkpoint:
            print(f"   Checkpoint epoch: {checkpoint['epoch']}")
        if isinstance(checkpoint, dict) and 'best_ssim' in checkpoint:
            print(f"   Best SSIM: {checkpoint['best_ssim']:.4f}")
    else:
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return None
    
    model.eval()
    return model

# -----------------------
# Validation Functions
# -----------------------
def compute_metrics(real_slice, generated_slice):
    """Compute comprehensive metrics between real and generated slices"""
    # Ensure values are in range [0,1]
    real_slice = np.clip(real_slice, 0, 1)
    generated_slice = np.clip(generated_slice, 0, 1)
    
    # SSIM
    try:
        ssim_value = ssim(real_slice, generated_slice, data_range=1.0)
    except:
        ssim_value = 0.0
    
    # PSNR
    try:
        psnr_value = psnr(real_slice, generated_slice, data_range=1.0)
    except:
        psnr_value = 0.0
    
    # MSE
    mse_value = np.mean((real_slice - generated_slice) ** 2)
    
    # MAE (L1)
    mae_value = np.mean(np.abs(real_slice - generated_slice))
    
    # Peak SNR (alternative)
    max_pixel = 1.0
    psnr_alt = 20 * np.log10(max_pixel / np.sqrt(mse_value)) if mse_value > 0 else 100
    
    return {
        'ssim': ssim_value,
        'psnr': psnr_value,
        'psnr_alt': psnr_alt,
        'mse': mse_value,
        'mae': mae_value
    }

def compute_volume_metrics(real_volume, generated_volume):
    """Compute metrics for entire volume"""
    depth = real_volume.shape[0]
    metrics_list = []
    
    for i in range(depth):
        metrics = compute_metrics(real_volume[i], generated_volume[i])
        metrics_list.append(metrics)
    
    # Average across all slices
    avg_metrics = {
        'ssim': np.mean([m['ssim'] for m in metrics_list]),
        'psnr': np.mean([m['psnr'] for m in metrics_list]),
        'mse': np.mean([m['mse'] for m in metrics_list]),
        'mae': np.mean([m['mae'] for m in metrics_list])
    }
    
    return avg_metrics, metrics_list

def visualize_validation_comparison(real_slice, generated_slice, missing_idx, patient_id, 
                                   metrics, save_path):
    """Create detailed validation comparison visualization"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Real slice
    axes[0,0].imshow(real_slice, cmap='gray')
    axes[0,0].set_title(f'Real Slice (Index: {missing_idx})', fontsize=12)
    axes[0,0].axis('off')
    
    # Generated slice
    axes[0,1].imshow(generated_slice, cmap='gray')
    axes[0,1].set_title(f'Generated Slice', fontsize=12)
    axes[0,1].axis('off')
    
    # Difference map
    diff = np.abs(real_slice - generated_slice)
    im = axes[0,2].imshow(diff, cmap='hot', vmin=0, vmax=0.2)
    axes[0,2].set_title(f'Difference Map', fontsize=12)
    axes[0,2].axis('off')
    plt.colorbar(im, ax=axes[0,2], fraction=0.046, pad=0.04)
    
    # Error histograms
    axes[1,0].hist(real_slice.flatten(), bins=50, alpha=0.5, label='Real', color='blue')
    axes[1,0].hist(generated_slice.flatten(), bins=50, alpha=0.5, label='Generated', color='red')
    axes[1,0].set_xlabel('Pixel Intensity')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].set_title('Pixel Intensity Distribution')
    axes[1,0].legend()
    
    # Error distribution
    axes[1,1].hist(diff.flatten(), bins=50, color='purple', alpha=0.7)
    axes[1,1].set_xlabel('Absolute Error')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].set_title(f'Error Distribution (MAE: {metrics["mae"]:.4f})')
    
    # Metrics text
    axes[1,2].axis('off')
    metrics_text = f"""
    Validation Metrics:
    
    SSIM:  {metrics['ssim']:.4f}
    PSNR:  {metrics['psnr']:.2f} dB
    MAE:   {metrics['mae']:.4f}
    MSE:   {metrics['mse']:.6f}
    """
    axes[1,2].text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center',
                   fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'Patient: {patient_id} | Missing Slice Index: {missing_idx}', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_validation_summary(metrics_df, save_path):
    """Create comprehensive validation summary plots"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. SSIM Distribution
    axes[0,0].hist(metrics_df['ssim'], bins=20, edgecolor='black', alpha=0.7, color='green')
    axes[0,0].axvline(metrics_df['ssim'].mean(), color='red', linestyle='--', 
                      linewidth=2, label=f'Mean: {metrics_df["ssim"].mean():.4f}')
    axes[0,0].axvline(metrics_df['ssim'].median(), color='blue', linestyle='--', 
                      linewidth=2, label=f'Median: {metrics_df["ssim"].median():.4f}')
    axes[0,0].set_xlabel('SSIM')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].set_title('SSIM Distribution')
    axes[0,0].legend()
    
    # 2. PSNR Distribution
    axes[0,1].hist(metrics_df['psnr'], bins=20, edgecolor='black', alpha=0.7, color='blue')
    axes[0,1].axvline(metrics_df['psnr'].mean(), color='red', linestyle='--', 
                      linewidth=2, label=f'Mean: {metrics_df["psnr"].mean():.2f} dB')
    axes[0,1].set_xlabel('PSNR (dB)')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].set_title('PSNR Distribution')
    axes[0,1].legend()
    
    # 3. MAE Distribution
    axes[0,2].hist(metrics_df['mae'], bins=20, edgecolor='black', alpha=0.7, color='orange')
    axes[0,2].axvline(metrics_df['mae'].mean(), color='red', linestyle='--', 
                      linewidth=2, label=f'Mean: {metrics_df["mae"].mean():.4f}')
    axes[0,2].set_xlabel('MAE')
    axes[0,2].set_ylabel('Frequency')
    axes[0,2].set_title('MAE Distribution')
    axes[0,2].legend()
    
    # 4. SSIM vs Missing Slice Index
    axes[1,0].scatter(metrics_df['missing_idx'], metrics_df['ssim'], alpha=0.6, s=20)
    axes[1,0].set_xlabel('Missing Slice Index')
    axes[1,0].set_ylabel('SSIM')
    axes[1,0].set_title('SSIM by Slice Position')
    # Add trend line
    z = np.polyfit(metrics_df['missing_idx'], metrics_df['ssim'], 1)
    p = np.poly1d(z)
    axes[1,0].plot(metrics_df['missing_idx'].sort_values(), 
                   p(metrics_df['missing_idx'].sort_values()), 
                   "r--", alpha=0.8, label=f'Trend (slope: {z[0]:.4f})')
    axes[1,0].legend()
    
    # 5. Metrics Correlation Heatmap
    corr = metrics_df[['ssim', 'psnr', 'mae', 'mse']].corr()
    im = axes[1,1].imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    axes[1,1].set_xticks(range(len(corr.columns)))
    axes[1,1].set_yticks(range(len(corr.columns)))
    axes[1,1].set_xticklabels(corr.columns)
    axes[1,1].set_yticklabels(corr.columns)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            axes[1,1].text(j, i, f'{corr.iloc[i, j]:.2f}', 
                          ha='center', va='center', color='white' if abs(corr.iloc[i, j]) > 0.5 else 'black')
    axes[1,1].set_title('Metrics Correlation')
    plt.colorbar(im, ax=axes[1,1])
    
    # 6. Box plots by slice position groups
    metrics_df['slice_group'] = pd.cut(metrics_df['missing_idx'], 
                                       bins=[0, 4, 8, 12, 16], 
                                       labels=['Early (0-4)', 'Mid-Early (4-8)', 'Mid-Late (8-12)', 'Late (12-16)'])
    metrics_df.boxplot(column='ssim', by='slice_group', ax=axes[1,2])
    axes[1,2].set_title('SSIM by Slice Position Group')
    axes[1,2].set_xlabel('Slice Group')
    axes[1,2].set_ylabel('SSIM')
    
    plt.suptitle('Validation Summary Statistics', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# -----------------------
# Main Validation Function
# -----------------------
def run_validation():
    """Main validation function"""
    print("\n" + "="*60)
    print("MODEL VALIDATION")
    print("="*60)
    
    # Load model
    model = load_model(CHECKPOINT_PATH, use_cbam=True, cbam_position='both')
    if model is None:
        return
    
    # Load validation dataset
    print("\n📂 Loading validation dataset...")
    val_dataset = MRI3DDataset(val_csv, root_dir, stack_depth=stack_depth)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"✅ Validation samples: {len(val_dataset)}")
    
    # Store all metrics
    all_metrics = []
    best_case = {'ssim': -1, 'data': None}
    worst_case = {'ssim': 2, 'data': None}
    
    # Validation loop
    print("\n🔍 Running validation...")
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(val_loader)):
            # Parse batch
            if isinstance(batch, (list, tuple)) and len(batch) >= 4:
                input_vol, target_vol, missing_idx, meta = batch[:4]
            else:
                input_vol, target_vol = batch[0], batch[1]
                missing_idx = batch[2] if len(batch) > 2 else 0
                meta = {"patient_id": f"val_{idx}"}
            
            # Get patient info
            patient_id = meta.get('patient_id', [f'val_{idx}'])
            if isinstance(patient_id, (list, tuple)):
                patient_id = patient_id[0]
            if isinstance(patient_id, torch.Tensor):
                patient_id = str(patient_id.item())
            
            missing_local = meta.get('missing_local_idx', missing_idx)
            if isinstance(missing_local, (list, tuple)):
                missing_local = missing_local[0]
            if isinstance(missing_local, torch.Tensor):
                missing_local = missing_local.item()
            
            # Move to device
            input_vol = input_vol.to(device)
            target_vol = target_vol.to(device)
            
            # Generate
            generated_volume = model(input_vol)
            
            # Extract the missing slice
            generated_slice = generated_volume[0, 0, missing_local, :, :].detach().cpu().numpy()
            real_slice = target_vol[0, 0, missing_local, :, :].detach().cpu().numpy()
            
            # Compute metrics
            metrics = compute_metrics(real_slice, generated_slice)
            metrics['patient_id'] = patient_id
            metrics['missing_idx'] = missing_local
            all_metrics.append(metrics)
            
            # Track best and worst cases
            if metrics['ssim'] > best_case['ssim']:
                best_case['ssim'] = metrics['ssim']
                best_case['data'] = (real_slice, generated_slice, missing_local, patient_id, metrics)
            
            if metrics['ssim'] < worst_case['ssim']:
                worst_case['ssim'] = metrics['ssim']
                worst_case['data'] = (real_slice, generated_slice, missing_local, patient_id, metrics)
            
            # Save comparison for all validation samples (optional - can limit to save space)
            if idx < 50:  # Save first 50 samples only
                save_path = f"validation_results/comparisons/{patient_id}_idx{missing_local}.png"
                visualize_validation_comparison(real_slice, generated_slice, 
                                               missing_local, patient_id, 
                                               metrics, save_path)
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame(all_metrics)
    
    # Save all metrics
    metrics_df.to_csv("validation_results/validation_metrics.csv", index=False)
    
    # Save best and worst cases
    if best_case['data']:
        real_slice, gen_slice, missing_idx, patient_id, metrics = best_case['data']
        visualize_validation_comparison(real_slice, gen_slice, missing_idx, 
                                       patient_id, metrics,
                                       "validation_results/best_worst/best_case.png")
    
    if worst_case['data']:
        real_slice, gen_slice, missing_idx, patient_id, metrics = worst_case['data']
        visualize_validation_comparison(real_slice, gen_slice, missing_idx,
                                       patient_id, metrics,
                                       "validation_results/best_worst/worst_case.png")
    
    # Create summary plots
    plot_validation_summary(metrics_df, "validation_results/validation_summary.png")
    
    # Print detailed results
    print("\n" + "="*60)
    print("VALIDATION RESULTS SUMMARY")
    print("="*60)
    print(f"Total samples validated: {len(all_metrics)}")
    print(f"\n📊 Overall Metrics (Mean ± Std):")
    print(f"   SSIM:  {metrics_df['ssim'].mean():.4f} ± {metrics_df['ssim'].std():.4f}")
    print(f"   PSNR:  {metrics_df['psnr'].mean():.2f} ± {metrics_df['psnr'].std():.2f} dB")
    print(f"   MAE:   {metrics_df['mae'].mean():.4f} ± {metrics_df['mae'].std():.4f}")
    print(f"   MSE:   {metrics_df['mse'].mean():.6f} ± {metrics_df['mse'].std():.6f}")
    
    print(f"\n📊 Percentiles:")
    for p in [10, 25, 50, 75, 90]:
        print(f"   SSIM at {p}th percentile: {metrics_df['ssim'].quantile(p/100):.4f}")
    
    print(f"\n🏆 Best Case (Highest SSIM):")
    print(f"   Patient: {best_case['data'][3]}, Index: {best_case['data'][2]}")
    print(f"   SSIM: {best_case['ssim']:.4f}, PSNR: {best_case['data'][4]['psnr']:.2f} dB")
    
    print(f"\n📉 Worst Case (Lowest SSIM):")
    print(f"   Patient: {worst_case['data'][3]}, Index: {worst_case['data'][2]}")
    print(f"   SSIM: {worst_case['ssim']:.4f}, PSNR: {worst_case['data'][4]['psnr']:.2f} dB")
    
    # Performance by slice position
    print(f"\n📊 Performance by Slice Position:")
    for pos in range(0, 16, 4):
        pos_metrics = metrics_df[metrics_df['missing_idx'].between(pos, pos+3)]
        if len(pos_metrics) > 0:
            print(f"   Slices {pos}-{pos+3}: SSIM = {pos_metrics['ssim'].mean():.4f} (n={len(pos_metrics)})")
    
    # Save summary to text file
    with open("validation_results/validation_summary.txt", "w") as f:
        f.write("="*60 + "\n")
        f.write("MODEL VALIDATION SUMMARY\n")
        f.write("="*60 + "\n\n")
        f.write(f"Checkpoint: {CHECKPOINT_PATH}\n")
        f.write(f"Validation samples: {len(all_metrics)}\n\n")
        f.write(f"Mean SSIM: {metrics_df['ssim'].mean():.4f} ± {metrics_df['ssim'].std():.4f}\n")
        f.write(f"Mean PSNR: {metrics_df['psnr'].mean():.2f} ± {metrics_df['psnr'].std():.2f} dB\n")
        f.write(f"Mean MAE: {metrics_df['mae'].mean():.4f} ± {metrics_df['mae'].std():.4f}\n")
        f.write(f"Mean MSE: {metrics_df['mse'].mean():.6f} ± {metrics_df['mse'].std():.6f}\n\n")
        f.write(f"Best SSIM: {best_case['ssim']:.4f} (Patient: {best_case['data'][3]})\n")
        f.write(f"Worst SSIM: {worst_case['ssim']:.4f} (Patient: {worst_case['data'][3]})\n")
    
    print("\n✅ Validation complete!")
    print(f"📁 Results saved to: validation_results/")
    print(f"   - validation_metrics.csv (all metrics)")
    print(f"   - validation_summary.txt (summary)")
    print(f"   - validation_summary.png (visualizations)")
    print(f"   - best_worst/ (best and worst cases)")
    print(f"   - comparisons/ (individual comparisons)")
    
    return metrics_df

def validate_checkpoint_ensemble(checkpoint_list):
    """Validate multiple checkpoints and compare"""
    print("\n" + "="*60)
    print("ENSEMBLE VALIDATION (Multiple Checkpoints)")
    print("="*60)
    
    results = {}
    
    for checkpoint_path in checkpoint_list:
        print(f"\n📦 Validating: {checkpoint_path}")
        CHECKPOINT_PATH = checkpoint_path
        metrics_df = run_validation()
        
        # Store key metrics
        epoch = checkpoint_path.split('_')[-1].split('.')[0]
        results[epoch] = {
            'ssim': metrics_df['ssim'].mean(),
            'psnr': metrics_df['psnr'].mean(),
            'mae': metrics_df['mae'].mean()
        }
    
    # Compare results
    print("\n" + "="*60)
    print("CHECKPOINT COMPARISON")
    print("="*60)
    comparison_df = pd.DataFrame(results).T
    print(comparison_df)
    comparison_df.to_csv("validation_results/checkpoint_comparison.csv")
    
    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].plot(comparison_df.index.astype(int), comparison_df['ssim'], 'bo-')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('SSIM')
    axes[0].set_title('SSIM by Epoch')
    axes[0].grid(True)
    
    axes[1].plot(comparison_df.index.astype(int), comparison_df['psnr'], 'ro-')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('PSNR (dB)')
    axes[1].set_title('PSNR by Epoch')
    axes[1].grid(True)
    
    axes[2].plot(comparison_df.index.astype(int), comparison_df['mae'], 'go-')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('MAE')
    axes[2].set_title('MAE by Epoch')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig("validation_results/checkpoint_comparison.png", dpi=150)
    plt.show()
    
    return comparison_df

# -----------------------
# Run Validation
# -----------------------
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate MRI Slice Generation Model')
    parser.add_argument('--checkpoint', type=str, 
                       default=r"checkpoints/generator_epoch_35.pth",
                       help='Path to model checkpoint')
    parser.add_argument('--ensemble', action='store_true',
                       help='Validate multiple checkpoints')
    parser.add_argument('--checkpoints', type=str, nargs='+',
                       help='List of checkpoint paths for ensemble validation')
    
    args = parser.parse_args()
    
    if args.ensemble:
        if args.checkpoints:
            checkpoint_list = args.checkpoints
        else:
            # Find all checkpoints
            import glob
            checkpoint_list = glob.glob("checkpoints/generator_epoch_*.pth")
            checkpoint_list.sort()
        validate_checkpoint_ensemble(checkpoint_list)
    else:
        CHECKPOINT_PATH = args.checkpoint
        run_validation()
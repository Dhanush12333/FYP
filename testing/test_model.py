# testing/test_model.py (FIXED VERSION)
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
test_csv = r"D:\FYP\MRI_GAN_Project\data\test_volumes.csv"

# Model checkpoint to test
CHECKPOINT_PATH = r"D:\FYP\MRI_GAN_Project\checkpoints\generator_epoch_35.pth"  # Change to your best checkpoint

# Test configuration
stack_depth = 16
batch_size = 1  # Keep 1 for testing

# Output directories
os.makedirs("test_results", exist_ok=True)
os.makedirs("test_results/comparisons", exist_ok=True)
os.makedirs("test_results/generated_slices", exist_ok=True)

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
# Testing Functions
# -----------------------
def generate_missing_slice(model, input_volume, missing_idx):
    """
    Generate the missing slice at specified index
    Args:
        model: Generator model
        input_volume: Input volume with missing slice (1, D, H, W)
        missing_idx: Index of missing slice (0-based)
    Returns:
        generated_slice: Generated 2D slice
    """
    with torch.no_grad():
        input_tensor = input_volume.to(device)
        output_volume = model(input_tensor)
        
        # FIX: Use .detach() and .cpu() properly
        generated_slice = output_volume[0, 0, missing_idx, :, :].detach().cpu().numpy()
        
    return generated_slice

def compute_metrics(real_slice, generated_slice):
    """Compute SSIM, PSNR, MSE, MAE between real and generated slices"""
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
    
    return {
        'ssim': ssim_value,
        'psnr': psnr_value,
        'mse': mse_value,
        'mae': mae_value
    }

def visualize_comparison(real_slice, generated_slice, missing_idx, patient_id, save_path):
    """Create side-by-side comparison visualization"""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Real slice
    axes[0].imshow(real_slice, cmap='gray')
    axes[0].set_title(f'Real Slice (Index: {missing_idx})', fontsize=12)
    axes[0].axis('off')
    
    # Generated slice
    axes[1].imshow(generated_slice, cmap='gray')
    axes[1].set_title(f'Generated Slice', fontsize=12)
    axes[1].axis('off')
    
    # Difference map
    diff = np.abs(real_slice - generated_slice)
    im = axes[2].imshow(diff, cmap='hot', vmin=0, vmax=0.2)
    axes[2].set_title(f'Difference Map (MAE: {np.mean(diff):.4f})', fontsize=12)
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.suptitle(f'Patient: {patient_id} | Missing Slice Index: {missing_idx}', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def visualize_full_volume(real_volume, generated_volume, patient_id, save_path):
    """Visualize all slices in a volume comparison"""
    depth = real_volume.shape[0]
    n_cols = 4
    n_rows = (depth * 2) // n_cols + 1
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 2 * n_rows))
    axes = axes.flatten()
    
    for i in range(depth):
        # Real slice
        axes[i*2].imshow(real_volume[i], cmap='gray')
        axes[i*2].set_title(f'Real {i}', fontsize=8)
        axes[i*2].axis('off')
        
        # Generated slice
        axes[i*2 + 1].imshow(generated_volume[i], cmap='gray')
        axes[i*2 + 1].set_title(f'Gen {i}', fontsize=8)
        axes[i*2 + 1].axis('off')
    
    # Hide unused subplots
    for j in range(depth*2, len(axes)):
        axes[j].axis('off')
    
    plt.suptitle(f'Patient: {patient_id} - Full Volume Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# -----------------------
# Main Test Function
# -----------------------
def run_tests():
    """Main testing function"""
    print("\n" + "="*60)
    print("STARTING MODEL TESTING")
    print("="*60)
    
    # Load model
    model = load_model(CHECKPOINT_PATH, use_cbam=True, cbam_position='both')
    if model is None:
        return
    
    # Load test dataset
    print("\n📂 Loading test dataset...")
    test_dataset = MRI3DDataset(test_csv, root_dir, stack_depth=stack_depth)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"✅ Test samples: {len(test_dataset)}")
    
    # Store all metrics
    all_metrics = []
    
    # Test each sample
    print("\n🧪 Running inference on test set...")
    for idx, batch in enumerate(tqdm(test_loader)):
        # Parse batch
        if isinstance(batch, (list, tuple)) and len(batch) >= 4:
            input_vol, target_vol, missing_idx, meta = batch[:4]
        else:
            input_vol, target_vol = batch[0], batch[1]
            missing_idx = batch[2] if len(batch) > 2 else 0
            meta = {"patient_id": f"test_{idx}"}
        
        # Get patient info
        patient_id = meta.get('patient_id', [f'test_{idx}'])
        if isinstance(patient_id, (list, tuple)):
            patient_id = patient_id[0]
        if isinstance(patient_id, torch.Tensor):
            patient_id = str(patient_id.item())
        
        missing_local = meta.get('missing_local_idx', missing_idx)
        if isinstance(missing_local, (list, tuple)):
            missing_local = missing_local[0]
        if isinstance(missing_local, torch.Tensor):
            missing_local = missing_local.item()
        
        # FIX: Ensure tensors are on correct device and properly detached
        input_vol = input_vol.to(device)
        target_vol = target_vol.to(device)
        
        # Generate missing slice with no gradients
        with torch.no_grad():
            generated_volume = model(input_vol)
            # FIX: Properly detach and convert to numpy
            generated_slice = generated_volume[0, 0, missing_local, :, :].detach().cpu().numpy()
            real_slice = target_vol[0, 0, missing_local, :, :].detach().cpu().numpy()
        
        # Compute metrics
        metrics = compute_metrics(real_slice, generated_slice)
        metrics['patient_id'] = patient_id
        metrics['missing_idx'] = missing_local
        all_metrics.append(metrics)
        
        # Save visualization
        save_path = f"test_results/comparisons/{patient_id}_idx{missing_local}.png"
        visualize_comparison(real_slice, generated_slice, missing_local, patient_id, save_path)
        
        # Save generated slice as separate image
        plt.imsave(f"test_results/generated_slices/{patient_id}_idx{missing_local}_generated.png", 
                   generated_slice, cmap='gray', vmin=0, vmax=1)
        plt.imsave(f"test_results/generated_slices/{patient_id}_idx{missing_local}_real.png", 
                   real_slice, cmap='gray', vmin=0, vmax=1)
        
        # Test full volume generation for first few samples
        if idx < 5:
            full_volume_path = f"test_results/{patient_id}_full_volume_comparison.png"
            real_vol = target_vol[0, 0].detach().cpu().numpy()
            gen_vol = generated_volume[0, 0].detach().cpu().numpy()
            visualize_full_volume(real_vol, gen_vol, patient_id, full_volume_path)
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv("test_results/test_metrics.csv", index=False)
    
    # Print summary statistics
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    print(f"Total samples tested: {len(all_metrics)}")
    print(f"\n📊 Average Metrics:")
    print(f"   SSIM:  {metrics_df['ssim'].mean():.4f} ± {metrics_df['ssim'].std():.4f}")
    print(f"   PSNR:  {metrics_df['psnr'].mean():.2f} ± {metrics_df['psnr'].std():.2f} dB")
    print(f"   MAE:   {metrics_df['mae'].mean():.4f} ± {metrics_df['mae'].std():.4f}")
    print(f"   MSE:   {metrics_df['mse'].mean():.6f} ± {metrics_df['mse'].std():.6f}")
    
    # Best and worst cases
    best_idx = metrics_df['ssim'].idxmax()
    worst_idx = metrics_df['ssim'].idxmin()
    print(f"\n🏆 Best SSIM: {metrics_df.iloc[best_idx]['ssim']:.4f} (Patient: {metrics_df.iloc[best_idx]['patient_id']})")
    print(f"📉 Worst SSIM: {metrics_df.iloc[worst_idx]['ssim']:.4f} (Patient: {metrics_df.iloc[worst_idx]['patient_id']})")
    
    # Save results to text file
    with open("test_results/test_summary.txt", "w") as f:
        f.write("="*60 + "\n")
        f.write("MODEL TESTING SUMMARY\n")
        f.write("="*60 + "\n\n")
        f.write(f"Checkpoint: {CHECKPOINT_PATH}\n")
        f.write(f"Test samples: {len(all_metrics)}\n\n")
        f.write(f"Average SSIM: {metrics_df['ssim'].mean():.4f} ± {metrics_df['ssim'].std():.4f}\n")
        f.write(f"Average PSNR: {metrics_df['psnr'].mean():.2f} ± {metrics_df['psnr'].std():.2f} dB\n")
        f.write(f"Average MAE: {metrics_df['mae'].mean():.4f} ± {metrics_df['mae'].std():.4f}\n")
        f.write(f"Average MSE: {metrics_df['mse'].mean():.6f} ± {metrics_df['mse'].std():.6f}\n")
    
    print("\n✅ Testing complete!")
    print(f"📁 Results saved to: test_results/")
    print(f"   - test_metrics.csv (all metrics)")
    print(f"   - test_summary.txt (summary)")
    print(f"   - comparisons/ (side-by-side images)")
    print(f"   - generated_slices/ (individual slices)")

def test_single_case(model_path, input_volume_path=None):
    """Test a single custom case"""
    print("\n🔍 Testing single case...")
    
    # Load model
    model = load_model(model_path, use_cbam=True, cbam_position='both')
    if model is None:
        return
    
    if input_volume_path:
        # Load custom volume here
        print(f"Loading custom volume from: {input_volume_path}")
        # Add custom loading logic
    else:
        # Use a sample from test set
        test_dataset = MRI3DDataset(test_csv, root_dir, stack_depth=stack_depth)
        input_vol, target_vol, missing_idx, meta = test_dataset[0]
        input_vol = input_vol.unsqueeze(0)  # Add batch dimension
        
        patient_id = meta.get('patient_id', 'sample')
        missing_local = meta.get('missing_local_idx', missing_idx)
        
        # Generate with no gradients
        with torch.no_grad():
            generated_vol = model(input_vol.to(device))
            generated_slice = generated_vol[0, 0, missing_local, :, :].detach().cpu().numpy()
            real_slice = target_vol[0, missing_local, :, :].detach().cpu().numpy()
        
        # Visualize
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(real_slice, cmap='gray')
        axes[0].set_title('Real Slice')
        axes[0].axis('off')
        axes[1].imshow(generated_slice, cmap='gray')
        axes[1].set_title('Generated Slice')
        axes[1].axis('off')
        diff = np.abs(real_slice - generated_slice)
        axes[2].imshow(diff, cmap='hot')
        axes[2].set_title(f'Difference (MAE: {diff.mean():.4f})')
        axes[2].axis('off')
        plt.suptitle(f'Patient: {patient_id} | Missing Index: {missing_local}')
        plt.tight_layout()
        plt.savefig("test_results/single_test_result.png", dpi=150)
        plt.show()
        
        print(f"✅ Single test complete! Results saved to test_results/single_test_result.png")

# -----------------------
# Run Tests
# -----------------------
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test MRI Slice Generation Model')
    parser.add_argument('--checkpoint', type=str, 
                       default=r"checkpoints/generator_epoch_35.pth",
                       help='Path to model checkpoint')
    parser.add_argument('--single', action='store_true',
                       help='Run single test case')
    parser.add_argument('--input', type=str, default=None,
                       help='Path to input volume for single test')
    
    args = parser.parse_args()
    
    # Update checkpoint path
    CHECKPOINT_PATH = args.checkpoint
    
    if args.single:
        test_single_case(CHECKPOINT_PATH, args.input)
    else:
        run_tests()
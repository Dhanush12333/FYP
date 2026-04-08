# training/train_3d_gan_enhanced.py
# Enhanced training script with CBAM, DICE Loss, and Perceptual Loss

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import random
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.cuda.amp import GradScaler, autocast

from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

# Project imports
from models.generator_3d_unet import Generator3D_UNet
from models.discriminator_3d import Discriminator3D
from models.losses import CombinedGeneratorLoss
from dataloaders.dataloader_3d import get_dataloaders_3d

# -----------------------
# Configuration
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# Paths (UPDATE THESE TO YOUR PATHS)
root_dir = r"D:\FYP\Processed_MRI"
train_csv = r"D:\FYP\MRI_GAN_Project\data\train_volumes.csv"
val_csv   = r"D:\FYP\MRI_GAN_Project\data\val_volumes.csv"
test_csv  = r"D:\FYP\MRI_GAN_Project\data\test_volumes.csv"

# Training hyperparameters
batch_size = 2
stack_depth = 16
num_epochs = 50

# ============= ENHANCED LOSS WEIGHTS =============
LAMBDA_ADV = 1.0        # Adversarial loss weight
LAMBDA_L1 = 100.0       # L1 reconstruction loss weight (increased from original 1)
LAMBDA_DICE = 0.5       # DICE loss weight (NEW)
LAMBDA_PERCEPTUAL = 0.1 # Perceptual loss weight (NEW)

# ============= CBAM CONFIGURATION =============
USE_CBAM = True          # Enable/disable CBAM attention
CBAM_POSITION = 'both'   # Options: 'bottleneck', 'skip', 'both'

# Learning rates
LR_G = 1e-4
LR_D = 2e-4

# Create directories
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# CSV log file
LOG_CSV = os.path.join("outputs", "output_log_enhanced.csv")
if not os.path.exists(LOG_CSV):
    with open(LOG_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "phase", "batch_idx", "sample_idx_in_batch",
            "patient_id", "plane", "modality", "missing_local_idx", "missing_global_idx",
            "total_slices", "window_start", "ssim", "psnr", "l1", "dice", "out_file"
        ])

print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    ENHANCED TRAINING CONFIGURATION                           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Device: {str(device):<68}║
║  Batch Size: {batch_size:<67}║
║  Stack Depth: {stack_depth:<67}║
║  Epochs: {num_epochs:<67}║
╠══════════════════════════════════════════════════════════════════════════════╣
║  CBAM Attention: {str(USE_CBAM):<14} | Position: {CBAM_POSITION:<51}║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Loss Weights:                                                               ║
║    - Adversarial (λ_adv):      {LAMBDA_ADV:<52}║
║    - L1 (λ_l1):                {LAMBDA_L1:<52}║
║    - DICE (λ_dice):            {LAMBDA_DICE:<52}║
║    - Perceptual (λ_perceptual): {LAMBDA_PERCEPTUAL:<52}║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Learning Rates: G: {LR_G:<6} | D: {LR_D:<57}║
╚══════════════════════════════════════════════════════════════════════════════╝
""")


# -----------------------
# Utility Functions (adapted from original)
# -----------------------
def safe_int(x, default=0):
    """Convert possible tensor / numpy scalar / int-like to int safely."""
    try:
        if isinstance(x, torch.Tensor):
            if x.numel() == 1:
                return int(x.item())
            else:
                return int(x.flatten()[0].item())
        if isinstance(x, (np.ndarray, list, tuple)):
            arr = np.array(x).ravel()
            if arr.size > 0:
                return int(arr[0])
            return default
        return int(x)
    except Exception:
        return default


def safe_str(x):
    """Convert identifiers to string safely (handles tensors, lists)."""
    try:
        if isinstance(x, torch.Tensor):
            if x.numel() == 1:
                return str(x.item())
            else:
                return str(x.flatten()[0].item())
        if isinstance(x, (list, tuple, np.ndarray)):
            if len(x) > 0:
                return str(x[0])
            return str(x)
        return str(x)
    except Exception:
        return str(x)


def infer_vertebral_level_by_index(global_idx, total_slices):
    """Coarse heuristic: upper/middle/lower vertebral levels."""
    try:
        if global_idx is None or total_slices is None or total_slices == 0:
            return "unknown"
        frac = float(global_idx) / float(max(1, total_slices))
        if frac < 0.33:
            return "upper"
        if frac < 0.66:
            return "middle"
        return "lower"
    except Exception:
        return "unknown"


def normalize_meta(meta, sample_idx=0):
    """
    Convert batched/list metadata into a single-sample plain python dict.
    Handles:
      - dict of tensors (default collate): extracts element sample_idx for each tensor
      - list of dicts: pick list[sample_idx]
      - single dict of scalars: return as-is (but convert tensors inside)
    """
    out = {}
    
    # list-of-dicts
    if isinstance(meta, (list, tuple)):
        if len(meta) == 0:
            return {}
        single = meta[sample_idx] if len(meta) > sample_idx else meta[0]
        return normalize_meta(single, sample_idx=0)

    if isinstance(meta, dict):
        for k, v in meta.items():
            # tensor case
            if isinstance(v, torch.Tensor):
                try:
                    if v.dim() >= 1 and v.shape[0] > sample_idx:
                        elem = v[sample_idx]
                    else:
                        elem = v
                except Exception:
                    elem = v
                if isinstance(elem, torch.Tensor):
                    if elem.numel() == 1:
                        out[k] = elem.item()
                    else:
                        try:
                            out[k] = elem.detach().cpu().numpy().tolist()
                        except Exception:
                            out[k] = elem
                else:
                    out[k] = elem
            else:
                # list/ndarray/scalar
                if isinstance(v, (list, tuple, np.ndarray)):
                    try:
                        out[k] = v[sample_idx] if len(v) > sample_idx else v[0]
                    except Exception:
                        out[k] = v
                else:
                    out[k] = v
        
        # normalize known fields
        if "patient_id" in out:
            out["patient_id"] = safe_str(out["patient_id"])
        if "missing_local_idx" in out:
            out["missing_local_idx"] = safe_int(out["missing_local_idx"], default=0)
        if "missing_global_idx" in out:
            mg = out["missing_global_idx"]
            out["missing_global_idx"] = None if mg in (None, "None") else safe_int(mg, default=None)
        if "total_slices" in out:
            out["total_slices"] = safe_int(out["total_slices"], default=0)
        if "window_start" in out:
            out["window_start"] = safe_int(out["window_start"], default=0)
        return out

    return {"patient_id": "unknown", "missing_local_idx": 0}


def save_sample_image(epoch, phase, batch_idx, sample_idx, fake_vol, real_vol, meta):
    """
    Save a side-by-side image and log metrics + metadata to CSV.
    Now includes DICE score in logging.
    """
    # convert tensors -> numpy
    if isinstance(fake_vol, torch.Tensor):
        fake_np = fake_vol.detach().cpu().numpy()
    else:
        fake_np = np.array(fake_vol)
    if isinstance(real_vol, torch.Tensor):
        real_np = real_vol.detach().cpu().numpy()
    else:
        real_np = np.array(real_vol)

    # squeeze to [D,H,W] for common shapes
    if fake_np.ndim == 4 and fake_np.shape[0] == 1:
        fake_np = fake_np.squeeze(0)
    if real_np.ndim == 4 and real_np.shape[0] == 1:
        real_np = real_np.squeeze(0)
    if fake_np.ndim == 5 and fake_np.shape[0] == 1 and fake_np.shape[1] == 1:
        fake_np = fake_np.squeeze(0).squeeze(0)
    if real_np.ndim == 5 and real_np.shape[0] == 1 and real_np.shape[1] == 1:
        real_np = real_np.squeeze(0).squeeze(0)

    # normalize meta
    if meta is None:
        meta = {}
    if not isinstance(meta, dict):
        try:
            meta = normalize_meta(meta, sample_idx=sample_idx)
        except Exception:
            meta = {"patient_id": "unknown", "missing_local_idx": 0}

    pid = meta.get("patient_id", "unknown")
    plane = meta.get("plane", "unknown")
    modality = meta.get("modality", "unknown")
    missing_local = meta.get("missing_local_idx", 0)
    missing_global = meta.get("missing_global_idx", None)
    total_slices = meta.get("total_slices", stack_depth)
    window_start = meta.get("window_start", 0)

    # ensure types
    try:
        missing_local = int(missing_local)
    except Exception:
        missing_local = safe_int(missing_local, default=0)
    try:
        if missing_global is not None:
            missing_global = int(missing_global)
    except Exception:
        missing_global = None
    try:
        total_slices = int(total_slices)
    except Exception:
        total_slices = stack_depth
    try:
        window_start = int(window_start)
    except Exception:
        window_start = 0

    level = infer_vertebral_level_by_index(missing_global, total_slices)
    gidx_str = str(missing_global) if missing_global is not None else f"local{missing_local}"
    out_fname = os.path.join("outputs", f"{pid}_g{gidx_str}_l{missing_local}_epoch{epoch}_{phase}.png")

    slice_idx = missing_local
    if slice_idx < 0 or slice_idx >= fake_np.shape[0]:
        slice_idx = max(0, min(fake_np.shape[0] - 1, slice_idx))

    fake_slice = fake_np[slice_idx]
    real_slice = real_np[slice_idx]

    # plot and save
    try:
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(real_slice, cmap='gray')
        axes[0].axis('off')
        axes[0].set_title(f"Real\npid:{pid} g:{gidx_str} local:{missing_local}")
        axes[1].imshow(fake_slice, cmap='gray')
        axes[1].axis('off')
        axes[1].set_title(f"Generated\nlevel:{level}")
        plt.suptitle(f"Patient: {pid} | Global: {gidx_str} | Local: {missing_local} | Epoch: {epoch}")
        plt.tight_layout()
        plt.savefig(out_fname, dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"Warning: failed to save sample image (plotting) - {e}")
        try:
            placeholder = np.zeros((256, 256), dtype=np.uint8)
            plt.imsave(os.path.join("outputs", f"placeholder_epoch{epoch}_{phase}.png"), placeholder, cmap="gray")
        except Exception:
            pass

    # metrics
    try:
        ssim_val = float(ssim(real_slice, fake_slice, data_range=1.0))
    except Exception:
        ssim_val = None
    try:
        psnr_val = float(psnr(real_slice, fake_slice, data_range=1.0))
    except Exception:
        psnr_val = None
    l1_val = float(np.mean(np.abs(real_slice - fake_slice)))
    
    # Compute DICE for the slice
    try:
        intersection = (fake_slice * real_slice).sum()
        dice_val = float((2. * intersection + 1e-6) / (fake_slice.sum() + real_slice.sum() + 1e-6))
    except Exception:
        dice_val = None

    with open(LOG_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch, phase, batch_idx, sample_idx,
            str(pid), plane, modality, missing_local, missing_global,
            total_slices, window_start, ssim_val, psnr_val, l1_val, dice_val, out_fname
        ])
    return out_fname


def safe_load_model(model, ckpt_path, map_location=None):
    """Load checkpoint tolerant to missing keys and 'module.' prefixes."""
    print(f"  -> loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=map_location)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']

    new_ckpt = {}
    for k, v in ckpt.items():
        new_k = k[len("module."):] if k.startswith("module.") else k
        new_ckpt[new_k] = v

    model_state = model.state_dict()
    loaded_keys = []
    skipped_keys = []
    for k, v in new_ckpt.items():
        if k in model_state and isinstance(v, torch.Tensor):
            try:
                if v.size() == model_state[k].size():
                    model_state[k] = v
                    loaded_keys.append(k)
                else:
                    skipped_keys.append(k)
            except Exception:
                skipped_keys.append(k)
        else:
            skipped_keys.append(k)

    model.load_state_dict(model_state)
    print(f"    loaded {len(loaded_keys)} keys, skipped {len(skipped_keys)} keys.")
    if len(skipped_keys) > 0:
        print(f"    sample skipped keys: {skipped_keys[:20]}")
    return loaded_keys, skipped_keys


# -----------------------
# Main training routine
# -----------------------
def main():
    print("Loading dataloaders...")
    train_loader, val_loader, _ = get_dataloaders_3d(
        train_csv, val_csv, test_csv,
        root_dir=root_dir,
        batch_size=batch_size,
        stack_depth=stack_depth
    )

    print("Initializing models with CBAM...")
    G = Generator3D_UNet(
        in_channels=1, 
        out_channels=1, 
        base_filters=32, 
        num_levels=4, 
        preserve_depth=True,
        use_cbam=USE_CBAM,
        cbam_position=CBAM_POSITION
    ).to(device)
    
    D = Discriminator3D(
        in_channels=1, 
        base_filters=64, 
        n_layers=4, 
        preserve_depth=True
    ).to(device)

    # Enhanced combined loss for generator
    criterion_combined = CombinedGeneratorLoss(
        lambda_adv=LAMBDA_ADV,
        lambda_l1=LAMBDA_L1,
        lambda_dice=LAMBDA_DICE,
        lambda_perceptual=LAMBDA_PERCEPTUAL
    )
    criterion_GAN = nn.BCEWithLogitsLoss()  # For discriminator

    opt_G = Adam(G.parameters(), lr=LR_G, betas=(0.5, 0.999))
    opt_D = Adam(D.parameters(), lr=LR_D, betas=(0.5, 0.999))
    scaler = GradScaler()

    # Resume from checkpoint if possible
    start_epoch = 0
    best_ssim = 0.0
    
    saved_epochs = [
        int(f.split('_')[-1].split('.')[0])
        for f in os.listdir("checkpoints")
        if f.startswith("generator_epoch_") and f.endswith(".pth")
    ]
    
    if saved_epochs:
        last_epoch = max(saved_epochs)
        print(f"🔁 Found checkpoints — attempting to resume from epoch {last_epoch}")
        gen_path = os.path.join("checkpoints", f"generator_epoch_{last_epoch}.pth")
        disc_path = os.path.join("checkpoints", f"discriminator_epoch_{last_epoch}.pth")
        if os.path.exists(gen_path):
            safe_load_model(G, gen_path, map_location=device)
        else:
            print(f"  ! generator checkpoint not found: {gen_path}")
        if os.path.exists(disc_path):
            safe_load_model(D, disc_path, map_location=device)
        else:
            print(f"  ! discriminator checkpoint not found: {disc_path}")
        start_epoch = last_epoch
    else:
        print("🆕 Starting fresh training")

    # training loop
    for epoch in range(start_epoch + 1, num_epochs + 1):
        G.train()
        D.train()
        total_loss_G = 0.0
        total_loss_D = 0.0
        
        # Track individual loss components for logging
        total_loss_adv = 0.0
        total_loss_l1 = 0.0
        total_loss_dice = 0.0
        total_loss_perceptual = 0.0

        loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}/{num_epochs}")
        
        for batch_idx, batch in loop:
            # Parse batch
            if isinstance(batch, (list, tuple)) and len(batch) >= 4:
                input_vol, target_vol, missing_idx, meta = batch[:4]
            elif isinstance(batch, (list, tuple)) and len(batch) == 3:
                input_vol, target_vol, missing_idx = batch
                meta = {"patient_id": "unknown", "missing_local_idx": missing_idx}
            else:
                input_vol, target_vol = batch[0], batch[1]
                missing_idx = batch[2] if len(batch) > 2 else 0
                meta = {"patient_id": "unknown", "missing_local_idx": missing_idx}

            input_vol = input_vol.to(device)
            target_vol = target_vol.to(device)

            # ---------------------
            # Train Discriminator
            # ---------------------
            with autocast():
                fake_vol = G(input_vol).detach()

                # add tiny gaussian noise to real images (stabilizes D)
                real_noisy = target_vol + 0.01 * torch.randn_like(target_vol, device=target_vol.device)
                torch.clamp(real_noisy, 0.0, 1.0, out=real_noisy)

                pred_real = D(real_noisy)
                pred_fake = D(fake_vol)

                # label smoothing for real labels
                real_labels = torch.full_like(pred_real, 0.9, device=pred_real.device)
                fake_labels = torch.zeros_like(pred_fake, device=pred_fake.device)

                loss_D = 0.5 * (criterion_GAN(pred_real, real_labels) + criterion_GAN(pred_fake, fake_labels))

            opt_D.zero_grad()
            scaler.scale(loss_D).backward()
            try:
                scaler.step(opt_D)
            except Exception as e:
                print("Warning: scaler.step(opt_D) failed:", e)
                try:
                    opt_D.step()
                except Exception:
                    pass
            scaler.update()

            # ---------------------
            # Train Generator
            # ---------------------
            with autocast():
                fake_vol = G(input_vol)
                pred_fake = D(fake_vol)
                
                # Use enhanced combined loss
                loss_G, loss_adv, loss_l1, loss_dice, loss_perceptual = criterion_combined(
                    fake_vol, target_vol, pred_fake
                )

            opt_G.zero_grad()
            scaler.scale(loss_G).backward()
            try:
                scaler.step(opt_G)
            except Exception as e:
                print("Warning: scaler.step(opt_G) failed:", e)
                try:
                    opt_G.step()
                except Exception:
                    pass
            scaler.update()

            total_loss_G += loss_G.item()
            total_loss_D += loss_D.item()
            total_loss_adv += loss_adv.item()
            total_loss_l1 += loss_l1.item()
            total_loss_dice += loss_dice.item()
            total_loss_perceptual += loss_perceptual.item()

            # save/log first sample of first batch each epoch
            if batch_idx == 0:
                try:
                    meta_sample = normalize_meta(meta, sample_idx=0)
                except Exception:
                    meta_sample = {"patient_id": "unknown", "missing_local_idx": 0}
                try:
                    save_sample_image(epoch, "train", batch_idx, 0,
                                      fake_vol[0].cpu(), target_vol[0].cpu(), meta_sample)
                except Exception as e:
                    print("Warning: failed to save sample image:", e)

            # Update progress bar every few batches
            if batch_idx % 10 == 0:
                loop.set_postfix(
                    G_total=f"{loss_G.item():.3f}",
                    G_adv=f"{loss_adv.item():.3f}",
                    G_l1=f"{loss_l1.item():.3f}",
                    G_dice=f"{loss_dice.item():.3f}",
                    D=f"{loss_D.item():.3f}"
                )

        # Epoch summary
        avg_loss_G = total_loss_G / len(train_loader)
        avg_loss_D = total_loss_D / len(train_loader)
        avg_loss_adv = total_loss_adv / len(train_loader)
        avg_loss_l1 = total_loss_l1 / len(train_loader)
        avg_loss_dice = total_loss_dice / len(train_loader)
        avg_loss_perceptual = total_loss_perceptual / len(train_loader)
        
        print(f"\n{'='*60}")
        print(f"Epoch [{epoch}/{num_epochs}] Summary:")
        print(f"  Generator Total Loss: {avg_loss_G:.4f}")
        print(f"    - Adversarial: {avg_loss_adv:.4f} (λ={LAMBDA_ADV})")
        print(f"    - L1: {avg_loss_l1:.4f} (λ={LAMBDA_L1})")
        print(f"    - DICE: {avg_loss_dice:.4f} (λ={LAMBDA_DICE})")
        print(f"    - Perceptual: {avg_loss_perceptual:.4f} (λ={LAMBDA_PERCEPTUAL})")
        print(f"  Discriminator Loss: {avg_loss_D:.4f}")
        print(f"{'='*60}")

        # ---------------------
        # Validation
        # ---------------------
        G.eval()
        ssim_total = 0.0
        psnr_total = 0.0
        l1_total = 0.0
        dice_total = 0.0
        n_val = 0
        
        with torch.no_grad():
            for v_batch_idx, v_batch in enumerate(val_loader):
                if v_batch_idx >= 10:  # Limit validation to 10 batches
                    break
                    
                if isinstance(v_batch, (list, tuple)) and len(v_batch) >= 4:
                    v_input, v_target, v_missing, v_meta = v_batch[:4]
                else:
                    v_input, v_target = v_batch[0], v_batch[1]
                    v_meta = {"patient_id": "unknown"}

                v_input = v_input.to(device)
                v_target = v_target.to(device)
                v_fake = G(v_input)

                fake_np = v_fake.squeeze().cpu().numpy()
                real_np = v_target.squeeze().cpu().numpy()

                try:
                    # Compute per-slice metrics
                    ssim_s = 0.0
                    psnr_s = 0.0
                    dice_s = 0.0
                    for s in range(fake_np.shape[0]):
                        ssim_s += ssim(real_np[s], fake_np[s], data_range=1.0)
                        psnr_s += psnr(real_np[s], fake_np[s], data_range=1.0)
                        intersection = (fake_np[s] * real_np[s]).sum()
                        dice_s += (2. * intersection + 1e-6) / (fake_np[s].sum() + real_np[s].sum() + 1e-6)
                    
                    ssim_total += (ssim_s / fake_np.shape[0])
                    psnr_total += (psnr_s / fake_np.shape[0])
                    dice_total += (dice_s / fake_np.shape[0])
                except Exception as e:
                    print(f"Validation metric error: {e}")
                    
                l1_total += np.mean(np.abs(fake_np - real_np))
                n_val += 1

        if n_val > 0:
            avg_ssim = ssim_total / n_val
            avg_psnr = psnr_total / n_val
            avg_l1 = l1_total / n_val
            avg_dice = dice_total / n_val
            
            print(f"\n📊 Validation Results (n={n_val}):")
            print(f"   SSIM:  {avg_ssim:.4f}")
            print(f"   PSNR:  {avg_psnr:.2f} dB")
            print(f"   DICE:  {avg_dice:.4f}")
            print(f"   L1:    {avg_l1:.4f}")
            
            # Save best model based on SSIM
            if avg_ssim > best_ssim:
                best_ssim = avg_ssim
                torch.save(G.state_dict(), "checkpoints/generator_best.pth")
                torch.save(D.state_dict(), "checkpoints/discriminator_best.pth")
                print(f"   ✨ New best model saved! (SSIM: {best_ssim:.4f})")
        else:
            print("Validation — skipped (no val samples computed)")

        # Save checkpoints each epoch
        torch.save(G.state_dict(), f"checkpoints/generator_epoch_{epoch}.pth")
        torch.save(D.state_dict(), f"checkpoints/discriminator_epoch_{epoch}.pth")
        
        # Also save optimizer states for full resume capability
        torch.save({
            'epoch': epoch,
            'generator_state_dict': G.state_dict(),
            'discriminator_state_dict': D.state_dict(),
            'optimizer_G_state_dict': opt_G.state_dict(),
            'optimizer_D_state_dict': opt_D.state_dict(),
            'best_ssim': best_ssim,
        }, f"checkpoints/checkpoint_epoch_{epoch}.pth")

    print("\n✅ Training complete!")
    print(f"🎯 Best SSIM achieved: {best_ssim:.4f}")


if __name__ == "__main__":
    # On Windows it's important to protect the entry point for multiprocessing DataLoader
    main()

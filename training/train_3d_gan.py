# training/train_3d_gan.py
## epoch 35 changed L1 to 5 from 10. Removed 0.9 from 1.0 and changed learning rate of both D and G at line 348
##epoch 40 changed L1 value to 1 from 5??
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

# project imports
from models.generator_3d_unet import Generator3D_UNet
from models.discriminator_3d import Discriminator3D
from dataloaders.dataloader_3d import get_dataloaders_3d


# -----------------------
# Config
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

root_dir = r"D:\FYP\Processed_MRI"
train_csv = r"D:\FYP\MRI_GAN_Project\data\train_volumes.csv"
val_csv   = r"D:\FYP\MRI_GAN_Project\data\val_volumes.csv"
test_csv  = r"D:\FYP\MRI_GAN_Project\data\test_volumes.csv"

batch_size = 2
stack_depth = 16
num_epochs = 50
lambda_L1 = 1
lr = 1e-4

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

LOG_CSV = os.path.join("outputs", "output_log.csv")
if not os.path.exists(LOG_CSV):
    with open(LOG_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch","phase","batch_idx","sample_idx_in_batch",
            "patient_id","plane","modality","missing_local_idx","missing_global_idx",
            "total_slices","window_start","ssim","psnr","l1","out_file"
        ])


# -----------------------
# Utilities
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
    # coarse heuristic: upper/middle/lower
    try:
        if global_idx is None or total_slices is None or total_slices == 0:
            return "unknown"
        frac = float(global_idx) / float(max(1, total_slices))
        if frac < 0.33: return "upper"
        if frac < 0.66: return "middle"
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
    Guarantees typical fields like patient_id, missing_local_idx become Python types.
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

    # fallback
    return {"patient_id": "unknown", "missing_local_idx": 0}


def save_sample_image(epoch, phase, batch_idx, sample_idx, fake_vol, real_vol, meta):
    """
    Save a side-by-side image and log metrics + metadata to CSV.
    fake_vol / real_vol expected as torch.Tensor (D,H,W) or numpy arrays.
    meta expected as normalized dict (use normalize_meta before calling).
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
        axes[0].imshow(real_slice, cmap='gray'); axes[0].axis('off')
        axes[0].set_title(f"Real\npid:{pid} g:{gidx_str} local:{missing_local}")
        axes[1].imshow(fake_slice, cmap='gray'); axes[1].axis('off')
        axes[1].set_title(f"Generated\nlevel:{level}")
        plt.suptitle(f"Patient: {pid} | Global: {gidx_str} | Local: {missing_local} | Epoch: {epoch}")
        plt.tight_layout()
        plt.savefig(out_fname, dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"Warning: failed to save sample image (plotting) - {e}")
        # fallback placeholder
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

    with open(LOG_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch, phase, batch_idx, sample_idx,
            str(pid), plane, modality, missing_local, missing_global,
            total_slices, window_start, ssim_val, psnr_val, l1_val, out_fname
        ])
    return out_fname


# -----------------------
# Safe checkpoint loader
# -----------------------
def safe_load_model(model, ckpt_path, map_location=None):
    """Load checkpoint tolerant to missing keys and 'module.' prefixes.
       Copies only size-compatible weights to the model."""
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

    # load the patched state dict (no strict check)
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

    print("Initializing models...")
    # preserve_depth=True in Discriminator/Generator if your model supports it
    G = Generator3D_UNet(in_channels=1, out_channels=1, base_filters=32, num_levels=4, preserve_depth=True).to(device)
    D = Discriminator3D(in_channels=1, base_filters=64, n_layers=4, preserve_depth=True).to(device)

    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_L1 = nn.L1Loss()

    opt_G = Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.999))  ##Change 1 
    opt_D = Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    scaler = GradScaler()

    # resume if possible
    start_epoch = 0
    saved_epochs = [
        int(f.split('_')[-1].split('.')[0])
        for f in os.listdir("checkpoints")
        if f.startswith("generator_epoch_")
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
        G.train(); D.train()
        total_loss_G = 0.0
        total_loss_D = 0.0

        loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}/{num_epochs}")
        for batch_idx, batch in loop:
            # dataset expected: (input, target, missing_idx, meta)
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
            # ---------------------
            # Train Discriminator
            # ---------------------
            with autocast():
                fake_vol = G(input_vol).detach()

                # add tiny gaussian noise to real images (stabilizes D)
                real_noisy = target_vol + 0.01 * torch.randn_like(target_vol, device=target_vol.device)
                # clamp to expected data range (uncomment correct range)
                torch.clamp(real_noisy, 0.0, 1.0, out=real_noisy)   # use if your input normalized to [0,1]
    # torch.clamp(real_noisy, -1.0, 1.0, out=real_noisy) # use if inputs in [-1,1]

                pred_real = D(real_noisy)
                pred_fake = D(fake_vol)

    # label smoothing
                real_labels = torch.full_like(pred_real, 1.0, device=pred_real.device)
                fake_labels = torch.zeros_like(pred_fake, device=pred_fake.device)

                loss_D = 0.5 * (criterion_GAN(pred_real, real_labels) + criterion_GAN(pred_fake, fake_labels))

            opt_D.zero_grad()
            scaler.scale(loss_D).backward()
            try:
                scaler.step(opt_D)
            except Exception as e:
                # tolerate optimizer step failures; report and continue
                print("Warning: scaler.step(opt_D) failed:", e)
                # fallback standard step (might error if gradients contain inf)
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
    # generator tries to make D predict "real" (smoothed)
                target_for_G = torch.full_like(pred_fake, 1.0, device=pred_fake.device)
                loss_G_GAN = criterion_GAN(pred_fake, target_for_G)
                loss_G_L1 = criterion_L1(fake_vol, target_vol)
                loss_G = loss_G_GAN + lambda_L1 * loss_G_L1

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

            # save/log first sample of first batch each epoch (customizable)
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
                    print("Meta sample debug:", repr(meta_sample))

            loop.set_postfix(loss_G=f"{loss_G.item():.4f}", loss_D=f"{loss_D.item():.4f}")

        avg_loss_G = total_loss_G / max(1, len(train_loader))
        avg_loss_D = total_loss_D / max(1, len(train_loader))
        print(f"\nEpoch [{epoch}/{num_epochs}] | Loss_D: {avg_loss_D:.4f} | Loss_G: {avg_loss_G:.4f}")

        # ---------------------
        # Validation (light)
        # ---------------------
        G.eval()
        ssim_total = 0.0
        psnr_total = 0.0
        l1_total = 0.0
        n_val = 0
        with torch.no_grad():
            for v_batch_idx, v_batch in enumerate(val_loader):
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
                    ssim_s = 0.0
                    psnr_s = 0.0
                    for s in range(fake_np.shape[0]):
                        ssim_s += ssim(real_np[s], fake_np[s], data_range=1.0)
                        psnr_s += psnr(real_np[s], fake_np[s], data_range=1.0)
                    ssim_total += (ssim_s / fake_np.shape[0])
                    psnr_total += (psnr_s / fake_np.shape[0])
                except Exception:
                    pass
                l1_total += np.mean(np.abs(fake_np - real_np))
                n_val += 1
                if n_val >= 10:
                    break

        if n_val > 0:
            print(f"Validation — SSIM: {ssim_total/n_val:.3f}, PSNR: {psnr_total/n_val:.2f} dB, L1: {l1_total/n_val:.4f}")
        else:
            print("Validation — skipped (no val samples computed)")

        # Save checkpoints each epoch
        torch.save(G.state_dict(), f"checkpoints/generator_epoch_{epoch}.pth")
        torch.save(D.state_dict(), f"checkpoints/discriminator_epoch_{epoch}.pth")

    print("\n✅ Training complete.")


if __name__ == "__main__":
    # On Windows it's important to protect the entry point for multiprocessing DataLoader
    main()

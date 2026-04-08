# training/train_3d_gan_fast.py
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import csv
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from torch.cuda.amp import GradScaler, autocast  # ⚡ mixed precision

# Import your models and dataloaders
from models.generator_3d_unet import Generator3D_UNet
from models.discriminator_3d import Discriminator3D
from dataloaders.dataloader_3d import get_dataloaders_3d


# ==============================
# ⚙️ CONFIG
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

root_dir = r"D:\FYP\Processed_MRI"
train_csv = r"D:\FYP\MRI_GAN_Project\data\train_volumes.csv"
val_csv   = r"D:\FYP\MRI_GAN_Project\data\val_volumes.csv"
test_csv  = r"D:\FYP\MRI_GAN_Project\data\test_volumes.csv"

# FAST MODE
batch_size = 2
stack_depth = 8
num_epochs = 5
lambda_L1 = 100
lr = 2e-4

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


def _unwrap_meta_for_sample(meta, sample_idx=0):
    """
    Convert a batched meta (common collate: dict of tensors) or list-of-dicts
    into a single-sample Python dict where values are scalars/lists/strings.

    Args:
        meta: could be:
            - dict of tensors (values shape [B, ...])
            - list of dicts (length B)
            - single dict (already per-sample)
        sample_idx: integer index to extract
    Returns:
        dict with Python scalars / lists / strings for sample sample_idx
    """
    # list of dicts case
    if isinstance(meta, list):
        single = meta[sample_idx]
        out = {}
        for k, v in single.items():
            if torch.is_tensor(v):
                if v.numel() == 1:
                    out[k] = v.item()
                else:
                    out[k] = v.cpu().numpy().tolist()
            else:
                out[k] = v
        return out

    # dict of tensors (default collate)
    if isinstance(meta, dict):
        out = {}
        for k, v in meta.items():
            # tensor -> pick sample element if possible
            if torch.is_tensor(v):
                try:
                    elem = v[sample_idx]
                except Exception:
                    # cannot index, convert whole tensor to list
                    try:
                        out[k] = v.cpu().numpy().tolist()
                    except Exception:
                        out[k] = v
                    continue

                if isinstance(elem, torch.Tensor):
                    if elem.numel() == 1:
                        out[k] = elem.item()
                    else:
                        out[k] = elem.cpu().numpy().tolist()
                else:
                    out[k] = elem
            else:
                # list/tuple or scalar
                if isinstance(v, (list, tuple)):
                    out[k] = v[sample_idx] if len(v) > sample_idx else v[0]
                else:
                    out[k] = v
        return out

    # already a single sample dict-like
    return meta


def infer_vertebral_level_by_index(global_idx, total_slices):
    # Very coarse heuristic — you can refine with labels later
    try:
        if global_idx is None or total_slices is None:
            return "unknown"
        frac = float(global_idx) / float(max(1, total_slices))
        if frac < 0.33: return "upper"
        if frac < 0.66: return "middle"
        return "lower"
    except Exception:
        return "unknown"


def save_and_log_sample(epoch, phase, batch_idx, sample_idx, fake_vol, real_vol, meta):
    """
    Saves one pair visualization and logs metrics + metadata to CSV.
    fake_vol, real_vol: either torch tensors or numpy arrays with shape [D,H,W] or [1,D,H,W]
    meta: dict produced by dataset (already unwrapped to scalars by helper)
    """
    if hasattr(fake_vol, "detach"):
        fake_np = fake_vol.detach().cpu().numpy()
    else:
        fake_np = np.array(fake_vol)
    if hasattr(real_vol, "detach"):
        real_np = real_vol.detach().cpu().numpy()
    else:
        real_np = np.array(real_vol)

    # ensure [D,H,W]
    if fake_np.ndim == 4:
        fake_np = fake_np.squeeze()
    if real_np.ndim == 4:
        real_np = real_np.squeeze()

    pid = str(meta.get("patient_id", "unknown"))
    plane = meta.get("plane", "unknown")
    modality = meta.get("modality", "unknown")

    # missing_local_idx may be tensor or int; ensure int
    try:
        missing_local = int(meta.get("missing_local_idx", 0))
    except Exception:
        # safe fallback
        try:
            missing_local = int(np.array(meta.get("missing_local_idx"))[0])
        except Exception:
            missing_local = 0

    missing_global = meta.get("missing_global_idx", None)
    try:
        total_slices = int(meta.get("total_slices", 0))
    except Exception:
        total_slices = 0
    try:
        window_start = int(meta.get("window_start", 0))
    except Exception:
        window_start = 0

    level = infer_vertebral_level_by_index(missing_global, total_slices)
    gidx_str = str(missing_global) if missing_global is not None else f"local{missing_local}"
    out_fname = os.path.join("outputs", f"{pid}_g{gidx_str}_l{missing_local}_epoch{epoch}_{level}.png")

    slice_idx = missing_local
    # bounds check
    if slice_idx < 0 or slice_idx >= fake_np.shape[0]:
        slice_idx = min(max(0, slice_idx), fake_np.shape[0] - 1)

    fake_slice = fake_np[slice_idx]
    real_slice = real_np[slice_idx]

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(real_slice, cmap='gray'); axes[0].axis('off')
    axes[0].set_title(f"Real\npid:{pid} g:{gidx_str} local:{missing_local}")
    axes[1].imshow(fake_slice, cmap='gray'); axes[1].axis('off')
    axes[1].set_title(f"Generated\nlevel:{level}")
    plt.suptitle(f"Patient: {pid} | Global: {gidx_str} | Local: {missing_local} | Epoch: {epoch}")
    plt.tight_layout()
    plt.savefig(out_fname, dpi=150)
    plt.close(fig)

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
            pid, plane, modality, missing_local, missing_global,
            total_slices, window_start, ssim_val, psnr_val, l1_val, out_fname
        ])


# ==============================
# DATA LOADERS
# ==============================
train_loader, val_loader, _ = get_dataloaders_3d(
    train_csv, val_csv, test_csv,
    root_dir=root_dir,
    batch_size=batch_size,
    stack_depth=stack_depth
)


# ==============================
# MODEL INIT
# ==============================
G = Generator3D_UNet(in_channels=1, out_channels=1, base_filters=16, num_levels=4).to(device)
D = Discriminator3D(in_channels=1, base_filters=32, n_layers=4, preserve_depth=True).to(device)

criterion_GAN = nn.BCEWithLogitsLoss()
criterion_L1 = nn.L1Loss()

opt_G = Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
opt_D = Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
scaler = GradScaler()

# ==============================
# RESUME CHECKPOINT (if any)
# ==============================
start_epoch = 0
saved_epochs = [
    int(f.split('_')[-1].split('.')[0])
    for f in os.listdir("checkpoints")
    if f.startswith("generator_epoch_")
]

if saved_epochs:
    last_epoch = max(saved_epochs)
    print(f"🔁 Resuming from epoch {last_epoch}")
    G.load_state_dict(torch.load(f"checkpoints/generator_epoch_{last_epoch}.pth", map_location=device))
    D.load_state_dict(torch.load(f"checkpoints/discriminator_epoch_{last_epoch}.pth", map_location=device))
    start_epoch = last_epoch
else:
    print("🆕 Starting fresh training")


# ==============================
# TRAIN LOOP (FAST)
# ==============================
for epoch in range(start_epoch + 1, num_epochs + 1):
    G.train(); D.train()
    total_loss_G, total_loss_D = 0, 0

    for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")):
        # batch is expected (input, target, missing_idx, meta) as per dataset
        if len(batch) == 4:
            input_vol, target_vol, missing_local_idx, meta = batch
        elif len(batch) == 3:
            input_vol, target_vol, missing_local_idx = batch
            meta = {"patient_id": "unknown", "missing_local_idx": missing_local_idx}
        else:
            # fallback
            input_vol, target_vol = batch[0], batch[1]
            missing_local_idx = batch[2] if len(batch) > 2 else 0
            meta = batch[3] if len(batch) > 3 else {"patient_id": "unknown", "missing_local_idx": missing_local_idx}

        input_vol = input_vol.to(device)
        target_vol = target_vol.to(device)

        # -----------------------------
        # Train Discriminator
        # -----------------------------
        with autocast():
            fake_vol = G(input_vol).detach()
            pred_real = D(target_vol)
            pred_fake = D(fake_vol)
            real_labels = torch.ones_like(pred_real)
            fake_labels = torch.zeros_like(pred_fake)
            loss_D = 0.5 * (criterion_GAN(pred_real, real_labels) + criterion_GAN(pred_fake, fake_labels))

        opt_D.zero_grad()
        scaler.scale(loss_D).backward()
        scaler.step(opt_D)
        scaler.update()

        # -----------------------------
        # Train Generator
        # -----------------------------
        with autocast():
            fake_vol = G(input_vol)
            pred_fake = D(fake_vol)
            loss_G_GAN = criterion_GAN(pred_fake, real_labels)
            loss_G_L1 = criterion_L1(fake_vol, target_vol)
            loss_G = loss_G_GAN + lambda_L1 * loss_G_L1

        opt_G.zero_grad()
        scaler.scale(loss_G).backward()
        scaler.step(opt_G)
        scaler.update()

        total_loss_G += loss_G.item()
        total_loss_D += loss_D.item()

        # Save/log the first sample of the first batch each epoch (adjust as desired)
        if batch_idx == 0:
            # Safely extract single-sample meta
            meta_sample = _unwrap_meta_for_sample(meta, sample_idx=0)
            save_and_log_sample(epoch, "train", batch_idx, 0,
                                fake_vol[0].cpu().squeeze(),
                                target_vol[0].cpu().squeeze(),
                                meta_sample)

    avg_loss_G = total_loss_G / len(train_loader)
    avg_loss_D = total_loss_D / len(train_loader)
    print(f"\nEpoch [{epoch}/{num_epochs}] | Loss_D: {avg_loss_D:.4f} | Loss_G: {avg_loss_G:.4f}")

    # ==============================
    # VALIDATION (save one val sample)
    # ==============================
    G.eval()
    with torch.no_grad():
        for v_batch_idx, v_batch in enumerate(val_loader):
            if len(v_batch) == 4:
                v_input, v_target, v_missing_idx, v_meta = v_batch
            else:
                v_input, v_target = v_batch[0], v_batch[1]
                v_meta = {"patient_id": "unknown"}

            v_input = v_input.to(device); v_target = v_target.to(device)
            v_fake = G(v_input)
            # save first sample of validation first batch
            if v_batch_idx == 0:
                meta_sample = _unwrap_meta_for_sample(v_meta, sample_idx=0)
                save_and_log_sample(epoch, "val", v_batch_idx, 0,
                                    v_fake[0].cpu().squeeze(),
                                    v_target[0].cpu().squeeze(),
                                    meta_sample)
            break  # only test first batch in fast mode

    # Save checkpoints
    torch.save(G.state_dict(), f"checkpoints/generator_epoch_{epoch}.pth")
    torch.save(D.state_dict(), f"checkpoints/discriminator_epoch_{epoch}.pth")

print("\n✅ Training Complete (Fast Mode) — ready for full-scale training later!")

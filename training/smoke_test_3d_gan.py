# smoke_test_3d_gan.py
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.optim import Adam
torch.autograd.set_detect_anomaly(True)

from models.generator_3d_unet import Generator3D_UNet
from models.discriminator_3d import Discriminator3D
from dataloaders.dataloader_3d import get_dataloaders_3d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 1
stack_depth = 16
lambda_l1 = 100
lr = 2e-4

root_dir = r"D:\FYP\Processed_MRI"
train_csv = r"D:\FYP\MRI_GAN_Project\data\train_volumes.csv"
val_csv   = r"D:\FYP\MRI_GAN_Project\data\val_volumes.csv"
test_csv  = r"D:\FYP\MRI_GAN_Project\data\test_volumes.csv"

train_loader, _, _ = get_dataloaders_3d(
    train_csv, val_csv, test_csv,
    root_dir=root_dir,
    batch_size=batch_size,
    stack_depth=stack_depth
)

G = Generator3D_UNet(in_channels=1, out_channels=1, base_filters=32, num_levels=4).to(device)
D = Discriminator3D(in_channels=1, base_filters=64, n_layers=4, preserve_depth=True).to(device)

criterion_GAN = nn.BCEWithLogitsLoss()
criterion_L1 = nn.L1Loss()
opt_G = Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
opt_D = Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

# fetch batch
batch = next(iter(train_loader))
input_vol, target_vol, missing_idx = batch
input_vol = input_vol.to(device)
target_vol = target_vol.to(device)

print(f"\nInput Volume: {input_vol.shape}")
print(f"Target Volume: {target_vol.shape}")
print(f"Masked Slice Index: {missing_idx}")

# G forward
fake_vol = G(input_vol)
print(f"Fake Volume (G output): {fake_vol.shape}")

# D forward (for updating D) - use detach()
pred_real = D(target_vol)
pred_fake_for_D = D(fake_vol.detach())

real_labels = torch.ones_like(pred_real, device=device)
fake_labels = torch.zeros_like(pred_fake_for_D, device=device)

loss_D_real = criterion_GAN(pred_real, real_labels)
loss_D_fake = criterion_GAN(pred_fake_for_D, fake_labels)
loss_D = 0.5 * (loss_D_real + loss_D_fake)

# update D
opt_D.zero_grad()
loss_D.backward(retain_graph=True)  # retain graph so G backward can use it if needed
opt_D.step()

# Recompute D(fake) for generator (fresh graph)
pred_fake_for_G = D(fake_vol)
loss_G_GAN = criterion_GAN(pred_fake_for_G, real_labels)
loss_G_L1 = criterion_L1(fake_vol, target_vol)
loss_G = loss_G_GAN + lambda_l1 * loss_G_L1

print(f"\nLoss_D: {loss_D.item():.4f}")
print(f"Loss_G: {loss_G.item():.4f} (GAN={loss_G_GAN.item():.4f}, L1={loss_G_L1.item():.4f})")

opt_G.zero_grad()
loss_G.backward()
opt_G.step()

print("\n✅ Backward and optimizer steps successful — full GAN pipeline verified!")

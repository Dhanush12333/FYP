# verify_3d_loader.py
import torch
from dataloader_3d import get_dataloaders_3d
import matplotlib.pyplot as plt

train_csv = r"D:\FYP\MRI_GAN_Project\data\train_volumes.csv"
val_csv   = r"D:\FYP\MRI_GAN_Project\data\val_volumes.csv"
test_csv  = r"D:\FYP\MRI_GAN_Project\data\test_volumes.csv"
root_dir  = r"D:\FYP\Processed_MRI"

train_loader, val_loader, test_loader = get_dataloaders_3d(
    train_csv, val_csv, test_csv, root_dir,
    batch_size=2, stack_depth=16
)

for batch_idx, (input_vol, target_vol, missing_idx) in enumerate(train_loader):
    print(f"\nBatch {batch_idx + 1}")
    print("Input volume shape :", tuple(input_vol.shape))
    print("Target volume shape:", tuple(target_vol.shape))
    print("Missing slice index:", missing_idx.tolist())

    slice_to_show = missing_idx[0].item()
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(input_vol[0, 0, slice_to_show].cpu(), cmap='gray')
    plt.title(f"Input Slice {slice_to_show} (Masked)")
    plt.subplot(1, 2, 2)
    plt.imshow(target_vol[0, 0, slice_to_show].cpu(), cmap='gray')
    plt.title(f"Target Slice {slice_to_show} (Ground Truth)")
    plt.show()

    if batch_idx == 1:
        break

import torch
import matplotlib.pyplot as plt
from dataloader import get_dataloaders

# Load DataLoaders
train_loader, val_loader, test_loader = get_dataloaders(batch_size=8)

# Get one batch
input_tensor, target_tensor = next(iter(train_loader))

print(f"Input batch shape: {input_tensor.shape}")   # (B, 2, 256, 256)
print(f"Target batch shape: {target_tensor.shape}") # (B, 1, 256, 256)

# Pick first example in the batch
prev_slice = input_tensor[0, 0].cpu().numpy()  # Channel 0 = previous slice
next_slice = input_tensor[0, 1].cpu().numpy()  # Channel 1 = next slice
target_slice = target_tensor[0, 0].cpu().numpy()  # Target slice

# Visualize them side-by-side
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
axs[0].imshow(prev_slice, cmap='gray')
axs[0].set_title("Previous Slice (L3)")

axs[1].imshow(target_slice, cmap='gray')
axs[1].set_title("Target Intermediate Slice (L4)")

axs[2].imshow(next_slice, cmap='gray')
axs[2].set_title("Next Slice (L5)")

for ax in axs:
    ax.axis('off')

plt.tight_layout()
plt.show()

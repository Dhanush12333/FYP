import torch
from generator_3d_unet import Generator3D_UNet  # ensure this file is named generator_3d_unet.py

# Initialize the model
model = Generator3D_UNet(in_channels=2, out_channels=1, base_filters=32, num_levels=4, preserve_depth=True)

# Create dummy input (batch_size=1, channels=2, depth=3, height=256, width=256)
dummy_input = torch.randn(1, 2, 3, 256, 256)

# Forward pass
output = model(dummy_input)

print(f"✅ Input shape:  {dummy_input.shape}")
print(f"✅ Output shape: {output.shape}")

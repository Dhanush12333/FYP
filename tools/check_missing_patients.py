import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from dataloaders.dataset_3d import MRI3DDataset

# === CONFIG ===
root_dir = r"D:\FYP\Processed_MRI"
csv_path = r"D:\FYP\MRI_GAN_Project\data\train_volumes.csv"

# === LOAD DATASET ===
dataset = MRI3DDataset(csv_path=csv_path, root_dir=root_dir, stack_depth=8)

valid_patients = 0
invalid_patients = 0
missing_ids = []

print("\n🔍 Checking each patient for valid slices...\n")

for i in range(len(dataset)):
    input_vol, target_vol, masked_idx = dataset[i]

    # Check if it's a dummy (no valid slices)
    if torch.sum(target_vol) == 0:
        invalid_patients += 1
        missing_ids.append(dataset.data.iloc[i]["patient_id"])
    else:
        valid_patients += 1

print("\n==============================")
print(f"✅ Total patients in CSV: {len(dataset)}")
print(f"✅ Patients with valid slices: {valid_patients}")
print(f"❌ Patients with NO valid slices: {invalid_patients}")
if missing_ids:
    print(f"🔎 Missing patient IDs (first 10): {missing_ids[:10]}")
print("==============================\n")

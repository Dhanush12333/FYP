# generate_volume_csvs.py
import os
import csv
import random
from tqdm import tqdm

# === CONFIG ===
DATA_ROOT = r"D:\FYP\Processed_MRI"                 # where your images actually are
OUT_DIR = r"D:\FYP\MRI_GAN_Project\data"            # where CSVs will be saved
PLANES = ["axial"]                                  # e.g., ["axial", "sagittal"] if both exist
MODALITIES = ["T2"]                                 # e.g., ["T1", "T2"]
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# === Step 1: Collect all patients ===
patients = sorted([p for p in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, p))])
print(f"Found {len(patients)} patient folders under {DATA_ROOT}")

all_records = []

# === Step 2: Traverse all patients, planes, modalities ===
for pid in tqdm(patients, desc="Scanning patients"):
    for plane in PLANES:
        for modality in MODALITIES:
            folder = os.path.join(DATA_ROOT, pid, plane, modality)
            if not os.path.exists(folder):
                continue
            slices = sorted([f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg"))])
            if len(slices) == 0:
                continue
            record = {
                "patient_id": pid,
                "plane": plane,
                "modality": modality,
                "slice_paths": str(slices)  # Save list as string
            }
            all_records.append(record)

print(f"✅ Collected {len(all_records)} total 3D volume records.")

# === Step 3: Shuffle and split into train/val/test ===
random.shuffle(all_records)
n_total = len(all_records)
n_train = int(n_total * TRAIN_SPLIT)
n_val = int(n_total * VAL_SPLIT)
n_test = n_total - n_train - n_val

splits = {
    "train_volumes.csv": all_records[:n_train],
    "val_volumes.csv": all_records[n_train:n_train + n_val],
    "test_volumes.csv": all_records[n_train + n_val:]
}

# === Step 4: Save each split to CSV ===
for fname, records in splits.items():
    out_path = os.path.join(OUT_DIR, fname)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["patient_id", "plane", "modality", "slice_paths"])
        writer.writeheader()
        for r in records:
            writer.writerow(r)
    print(f"✅ Saved {len(records)} entries to {out_path}")

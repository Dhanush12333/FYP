import os
import pydicom
import numpy as np
import cv2
from tqdm import tqdm
import re

# ==== PATHS ====
root_dir = r"D:\FYP\MRI_Data\01_MRI_Data"   # Input dataset
output_dir = r"D:\FYP\Processed_MRI"        # Output directory
os.makedirs(output_dir, exist_ok=True)

# ==== Helper function to extract slice number ====
def get_slice_number(filename):
    match = re.search(r'(\d+)\.ima$', filename)
    if match:
        return int(match.group(1))
    return 0

# ==== MAIN LOOP ====
for patient in tqdm(sorted(os.listdir(root_dir))):
    patient_path = os.path.join(root_dir, patient)
    if not os.path.isdir(patient_path):
        continue

    for root, dirs, files in os.walk(patient_path):
        if not files:
            continue

        # Identify modality (T1 or T2)
        modality = None
        if "T1" in root:
            modality = "T1"
        elif "T2" in root:
            modality = "T2"
        else:
            continue

        # Identify plane (Axial or Sagittal)
        plane = None
        if "TRA" in root:
            plane = "Axial"
        elif "SAG" in root:
            plane = "Sagittal"
        else:
            continue

        # Create folder structure: Patient -> Plane -> Modality
        save_folder = os.path.join(output_dir, patient, plane, modality)
        os.makedirs(save_folder, exist_ok=True)

        print(f"\nProcessing {modality} - {plane}: {root}")

        # Sort files numerically
        files_sorted = sorted([f for f in files if f.endswith(".ima")], key=get_slice_number)

        for file in files_sorted:
            file_path = os.path.join(root, file)
            try:
                dcm = pydicom.dcmread(file_path)
                img = dcm.pixel_array.astype(np.float32)

                # Normalize [0, 255]
                img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
                img = (img * 255).astype(np.uint8)

                # Resize for consistency
                img_resized = cv2.resize(img, (256, 256))

                # Save PNG
                out_name = os.path.splitext(file)[0] + ".png"
                save_path = os.path.join(save_folder, out_name)
                cv2.imwrite(save_path, img_resized)

            except Exception as e:
                print(f"Error reading {file_path}: {e}")

print("\n✅ All conversions done with Axial/Sagittal → T1/T2 structure!")

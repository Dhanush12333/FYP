import os
import pandas as pd
import ast

root_dir = r"D:\FYP\Processed_MRI"
csv_path = r"D:\FYP\MRI_GAN_Project\data\train_volumes.csv"

data = pd.read_csv(csv_path)

print(f"🔍 Total patients in CSV: {len(data)}")

# Check first few patients
for i in range(5):
    row = data.iloc[i]
    patient_id = str(row.get("patient_id", "Unknown"))
    plane = str(row.get("plane", "Axial"))
    modality = str(row.get("modality", "T2"))

    # zero-pad
    if patient_id.isdigit():
        patient_id = patient_id.zfill(4)

    slice_list_str = row["slice_paths"]
    try:
        slice_list = ast.literal_eval(slice_list_str)
    except Exception:
        slice_list = []

    if not isinstance(slice_list, list):
        slice_list = [slice_list]

    print(f"\n📁 Checking patient: {patient_id}, plane={plane}, modality={modality}")
    print(f"🧩 Example slice names: {slice_list[:2]}")

    found_any = False
    for fname in slice_list[:5]:  # only check first few slices
        fname = str(fname)
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            fname = fname + ".png"

        test_path_1 = os.path.join(root_dir, patient_id, plane, modality, fname)
        test_path_2 = os.path.join(root_dir, patient_id, modality, plane, fname)

        print(f"🔸Trying:\n  {test_path_1}\n  {test_path_2}")

        if os.path.exists(test_path_1):
            print("✅ FOUND at:", test_path_1)
            found_any = True
            break
        elif os.path.exists(test_path_2):
            print("✅ FOUND at:", test_path_2)
            found_any = True
            break

    if not found_any:
        print("❌ No valid slice found for this patient.")

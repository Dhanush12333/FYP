import os
import glob
import pandas as pd
import re

# ==== PATHS ====
base_path = r"D:\FYP\Processed_MRI"
output_csv = os.path.join(base_path, "intermediate_slice_triplets.csv")

triplets = []

def get_slice_number(filename):
    """Extract numeric slice index from filename."""
    match = re.search(r'(\d+)\.png$', filename)
    if match:
        return int(match.group(1))
    return 0

# ==== MAIN LOOP ====
for patient in sorted(os.listdir(base_path)):
    axial_t2_path = os.path.join(base_path, patient, "Axial", "T2")
    if not os.path.exists(axial_t2_path):
        continue

    # Get all png slices
    all_slices = glob.glob(os.path.join(axial_t2_path, "*.png"))
    if not all_slices:
        continue

    # Group slices by series
    series_dict = {}
    for slice_path in all_slices:
        filename = os.path.basename(slice_path)
        if filename.startswith("POSDISP"):
            series_name = "_".join(filename.split("_")[:-2])
        else:
            series_name = "_".join(filename.split("_")[:2])
        series_dict.setdefault(series_name, []).append(slice_path)

    # Create triplets (prev, next → target)
    for series, slices in series_dict.items():
        slices_sorted = sorted(slices, key=get_slice_number)

        # Must have at least 3 slices to form one triplet
        if len(slices_sorted) < 3:
            continue

        for i in range(1, len(slices_sorted) - 1):
            prev_slice = slices_sorted[i - 1]
            next_slice = slices_sorted[i + 1]
            target_slice = slices_sorted[i]

            triplets.append([patient, series, prev_slice, next_slice, target_slice])

# ==== SAVE CSV ====
df = pd.DataFrame(triplets, columns=["patient_id", "series", "prev_slice", "next_slice", "target_slice"])
df.to_csv(output_csv, index=False)

print(f"✅ Saved {len(df)} intermediate slice triplets to {output_csv}")

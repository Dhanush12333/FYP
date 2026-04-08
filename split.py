import pandas as pd
from sklearn.model_selection import train_test_split
import os

# ==== PATHS ====
csv_path = r"D:\FYP\Processed_MRI\intermediate_slice_triplets.csv"
output_dir = r"D:\FYP\Processed_MRI"
os.makedirs(output_dir, exist_ok=True)

# ==== LOAD DATA ====
df = pd.read_csv(csv_path)
print(f"Total samples in dataset: {len(df)}")

# ==== VERIFY COLUMNS ====
required_cols = ['patient_id', 'series', 'prev_slice', 'next_slice', 'target_slice']
assert all(col in df.columns for col in required_cols), \
       f"❌ Missing columns! Found: {df.columns.tolist()}"

# ==== SPLIT DATASET ====
# 70% train, 15% validation, 15% test
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, shuffle=True)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, shuffle=True)

# ==== SAVE SPLITS ====
train_csv = os.path.join(output_dir, "train_triplets.csv")
val_csv   = os.path.join(output_dir, "val_triplets.csv")
test_csv  = os.path.join(output_dir, "test_triplets.csv")

train_df.to_csv(train_csv, index=False)
val_df.to_csv(val_csv, index=False)
test_df.to_csv(test_csv, index=False)

# ==== STATS ====
print(f"✅ Dataset successfully split:")
print(f"  Train: {len(train_df)} samples → {train_csv}")
print(f"  Val:   {len(val_df)} samples → {val_csv}")
print(f"  Test:  {len(test_df)} samples → {test_csv}")

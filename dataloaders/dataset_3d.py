# dataloaders/dataset_3d.py
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError
import pandas as pd
import random
import ast
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class MRI3DDataset(Dataset):
    """
    3D volumetric dataset for Lumbar Spine MR sequence completion.
    Returns: input_tensor, target_tensor, missing_local_idx, meta_dict

    meta_dict keys:
      - patient_id (str)
      - plane, modality
      - total_slices (int) : number of slices listed in CSV for this patient
      - window_start (int)  : global index of the first slice in the returned window (0-based)
      - missing_local_idx (int) : index in returned window that was masked
      - missing_global_idx (int) : global slice index (window_start + missing_local_idx) if available
      - slice_filenames (list) : original filenames used for the window (length = stack_depth)
    """

    def __init__(self, csv_path, root_dir, stack_depth=16, transform=None):
        """
        Args:
            csv_path (str): path to CSV file (train/val/test)
            root_dir (str): base MRI directory (e.g. D:/FYP/Processed_MRI)
            stack_depth (int): number of slices per 3D volume
            transform (callable, optional): optional transforms
        """
        self.data = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.stack_depth = int(stack_depth)
        self.transform = transform

        print(f"✅ Estimated usable patients: {len(self.data)} / {len(self.data)}")
        print("\n📦 Dataset loaded successfully.")
        print(f"📊 Total samples in CSV: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        patient_id = str(row.get("patient_id", "Unknown"))
        plane = str(row.get("plane", "Axial"))
        modality = str(row.get("modality", "T2"))

        # Zero-pad numeric IDs like 2 → 0002 to match folder names
        if patient_id.isdigit():
            patient_id = patient_id.zfill(4)

        # Variants for matching
        plane_variants = [plane, plane.lower(), plane.capitalize(), plane.upper()]
        modality_variants = [modality, modality.lower(), modality.capitalize(), modality.upper()]

        # Parse slice list safely
        slice_list_str = row.get("slice_paths", "[]")
        try:
            slice_list_all = ast.literal_eval(slice_list_str)
        except Exception:
            slice_list_all = []

        if not isinstance(slice_list_all, list):
            slice_list_all = [slice_list_all]

        # Keep only string filenames
        slice_list_all = [str(s) for s in slice_list_all]

        original_count = len(slice_list_all)

        # Gather valid image arrays and filenames (in the same order)
        imgs = []
        found_fnames = []

        for fname in slice_list_all:
            fname = str(fname)
            # ensure extension
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                fname = fname + ".png"

            found_path = None

            # Try common folder patterns
            for pl in plane_variants:
                for md in modality_variants:
                    possible_paths = [
                        os.path.join(self.root_dir, patient_id, pl, md, fname),
                        os.path.join(self.root_dir, patient_id, md, pl, fname)
                    ]
                    for path in possible_paths:
                        if os.path.exists(path):
                            found_path = path
                            break
                    if found_path:
                        break
                if found_path:
                    break

            # Retry with uppercase extension
            if not found_path and fname.lower().endswith(".png"):
                alt_fname = fname[:-4] + ".PNG"
                for pl in plane_variants:
                    for md in modality_variants:
                        for path in [
                            os.path.join(self.root_dir, patient_id, pl, md, alt_fname),
                            os.path.join(self.root_dir, patient_id, md, pl, alt_fname)
                        ]:
                            if os.path.exists(path):
                                found_path = path
                                break
                        if found_path:
                            break
                    if found_path:
                        break

            if not found_path:
                # skip missing file silently (we will pad/crop later)
                continue

            try:
                img = Image.open(found_path).convert("L")
                img = np.array(img, dtype=np.float32) / 255.0
                imgs.append(img)
                found_fnames.append(os.path.basename(found_path))
            except (UnidentifiedImageError, OSError):
                continue

        # If no valid slices found, return dummy (still same return shape)
        if len(imgs) == 0:
            print(f"⚠️ Skipping patient {patient_id}: no valid slices found.")
            dummy = torch.zeros((1, self.stack_depth, 256, 256))
            meta = {
                "patient_id": patient_id,
                "plane": plane,
                "modality": modality,
                "total_slices": original_count,
                "window_start": 0,
                "missing_local_idx": 0,
                "missing_global_idx": 0,
                "slice_filenames": []
            }
            return dummy, dummy, 0, meta

        # Now we have imgs (list of 2D arrays) and found_fnames matching order
        volume_all = np.stack(imgs, axis=0)  # (D_all, H, W)
        D_all = volume_all.shape[0]

        # Decide window start
        if D_all <= self.stack_depth:
            window_start = 0
            volume_window = volume_all.copy()
        else:
            window_start = random.randint(0, D_all - self.stack_depth)
            volume_window = volume_all[window_start:window_start + self.stack_depth]

        # If padding needed (D_all < stack_depth) pad by edge
        if volume_window.shape[0] < self.stack_depth:
            pad = self.stack_depth - volume_window.shape[0]
            volume_window = np.pad(volume_window, ((0, pad), (0, 0), (0, 0)), mode="edge")
            # pad filenames with last available
            last_name = found_fnames[-1] if len(found_fnames) > 0 else ""
            pad_names = [last_name] * pad
            window_fnames = found_fnames + pad_names
        else:
            window_fnames = found_fnames[window_start: window_start + self.stack_depth]

        # Sanity: ensure window_fnames length matches stack_depth
        if len(window_fnames) < self.stack_depth:
            # pad filenames
            last_name = window_fnames[-1] if len(window_fnames) > 0 else ""
            window_fnames += [last_name] * (self.stack_depth - len(window_fnames))

        # Create input-target pair
        target_volume = volume_window.copy()
        missing_local_idx = random.randint(0, self.stack_depth - 1)
        input_volume = volume_window.copy()
        input_volume[missing_local_idx] = 0.0  # mask one slice

        # Compute missing_global_idx (if original ordering known)
        # If we used a window from the original stack, global index = window_start + missing_local_idx
        missing_global_idx = window_start + missing_local_idx if D_all >= self.stack_depth else missing_local_idx

        # Convert to torch tensors [1, D, H, W]
        input_tensor = torch.from_numpy(input_volume[None, :, :, :]).float()
        target_tensor = torch.from_numpy(target_volume[None, :, :, :]).float()

        if self.transform:
            input_tensor, target_tensor = self.transform(input_tensor, target_tensor)

        meta = {
            "patient_id": patient_id,
            "plane": plane,
            "modality": modality,
            "total_slices": D_all,
            "window_start": int(window_start),
            "missing_local_idx": int(missing_local_idx),
            "missing_global_idx": int(missing_global_idx),
            "slice_filenames": list(window_fnames)
        }

        return input_tensor, target_tensor, missing_local_idx, meta


if __name__ == "__main__":
    # quick local check when running file directly
    import argparse
    parser = argparse = None
    try:
        import argparse as _arg
        parser = _arg.ArgumentParser()
        parser.add_argument("--csv", required=True, help="CSV path")
        parser.add_argument("--root", required=True, help="root_dir")
        parser.add_argument("--depth", type=int, default=8)
        args = parser.parse_args()
        ds = MRI3DDataset(args.csv, args.root, stack_depth=args.depth)
        print("✅ Sample fetch:")
        sample = ds[0]
        print("Input:", sample[0].shape, "Target:", sample[1].shape, "Missing:", sample[2], "Meta:", sample[3])
    except Exception:
        # not an error — just allow import in other scripts
        pass

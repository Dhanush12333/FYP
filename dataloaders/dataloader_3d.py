# dataloaders/dataloader_3d.py
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataloaders.dataset_3d import MRI3DDataset
from torch.utils.data import DataLoader


def get_dataloaders_3d(train_csv, val_csv, test_csv, root_dir,
                       batch_size=2, stack_depth=16, num_workers=0):
    """
    Creates PyTorch DataLoaders for 3D MRI volumes.

    Args:
        train_csv, val_csv, test_csv: paths to dataset CSVs
        root_dir: base folder containing all MRI image folders
        batch_size: how many volumes per batch
        stack_depth: number of slices per volume
        num_workers: number of worker processes for data loading
    """
    train_ds = MRI3DDataset(train_csv, root_dir, stack_depth)
    val_ds   = MRI3DDataset(val_csv, root_dir, stack_depth)
    test_ds  = MRI3DDataset(test_csv, root_dir, stack_depth)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

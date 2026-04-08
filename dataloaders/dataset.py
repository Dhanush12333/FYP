import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd

class MRITripletDataset(Dataset):
    """
    Custom dataset for intermediate slice generation using previous and next slices.
    """
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform if transform else transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        prev_slice = Image.open(row['prev_slice']).convert('L')
        next_slice = Image.open(row['next_slice']).convert('L')
        target_slice = Image.open(row['target_slice']).convert('L')

        prev_tensor = self.transform(prev_slice)
        next_tensor = self.transform(next_slice)
        target_tensor = self.transform(target_slice)

        # Stack prev and next slices into 2-channel input
        input_tensor = torch.cat([prev_tensor, next_tensor], dim=0)

        return input_tensor, target_tensor

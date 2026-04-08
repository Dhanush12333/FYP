from torch.utils.data import DataLoader
from dataset import MRITripletDataset

def get_dataloaders(batch_size=8):
    train_csv = r"D:\FYP\MRI_GAN_Project\data\train_triplets.csv"
    val_csv   = r"D:\FYP\MRI_GAN_Project\data\val_triplets.csv"
    test_csv  = r"D:\FYP\MRI_GAN_Project\data\test_triplets.csv"

    train_dataset = MRITripletDataset(train_csv)
    val_dataset   = MRITripletDataset(val_csv)
    test_dataset  = MRITripletDataset(test_csv)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader

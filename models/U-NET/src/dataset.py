import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset


def fetch_loaders(npz_path, transform=None, target_transform=None, batch_size=32, shuffle=True):
    """ Function to fetch dataLoaders for the Training / Validation datasets
    Return:
        Returns train and val dataloaders
    """
    train_dataset = GlacierDataset(
        npz_path=os.path.join(npz_path, 'glacier_train.npz'),
        transform=transform,
        target_transform=target_transform
    )
    val_dataset = GlacierDataset(
        npz_path=os.path.join(npz_path, 'glacier_val.npz'),
        transform=transform,
        target_transform=target_transform
    )
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            num_workers=4, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                          num_workers=2, shuffle=False)

    return train_loader, val_loader


class GlacierDataset(Dataset):
    """Custom Dataset for Glacier Data

    Indexing the i^th element returns the underlying image and the associated
    binary mask

    """
    def __init__(self, npz_path, transform=None, target_transform=None):
        """
        Initialize dataset from a single .npz file.
        
        Args:
            npz_path: Path to the .npz file containing X, Y, means, stds
        """
        data = np.load(npz_path)
        self.images = data['X']  # (N, B, H, W)
        self.masks = data['Y']   # (N, H, W)

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        getitem method to retrieve a single instance of the dataset
        """
        image = self.images[index]  # (6, 32, 32)
        mask = self.masks[index]    # (32, 32)
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        
        return torch.from_numpy(image).float(), torch.from_numpy(mask).long()

    def __len__(self):
        """Function to return the length of the dataset"""
        return len(self.images)

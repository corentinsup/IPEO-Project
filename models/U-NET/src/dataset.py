import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from torchvision import tv_tensors

def fetch_loaders(npz_path, mode='train', val_size=0.2, transform=None, batch_size=32, train_shuffle=True):
    """ Function to fetch dataLoaders for the Training / Validation datasets
    Return:
        Returns train and val dataloaders (if mode='train') or test loader (if mode='test')
    """
    if mode == 'train':
        # Load full training dataset
        full_dataset = GlacierDataset(
            npz_path=os.path.join(npz_path, 'glacier_train.npz'),
            transform=transform
        )
        
        # Split indices for train and val
        indices = list(range(len(full_dataset)))
        train_indices, val_indices = train_test_split(
            indices, test_size=val_size, random_state=42
        )
        
        # Create subset datasets
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)
        
        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=train_shuffle)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=train_shuffle)
        
        return train_loader, val_loader
    
    else:
        # Load test dataset
        test_dataset = GlacierDataset(
            npz_path=os.path.join(npz_path, 'glacier_val.npz'),
            transform=None
        )
        
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return test_loader


class GlacierDataset(Dataset):
    """Custom Dataset for Glacier Data

    Indexing the i^th element returns the underlying image and the associated
    binary mask

    """
    def __init__(self, npz_path, transform=None):
        """
        Initialize dataset from a single .npz file.
        
        Args:
            npz_path: Path to the .npz file containing X, Y, means, stds
        """
        data = np.load(npz_path)
        self.images = data['X']  # (N, B, H, W)
        self.masks = data['Y']   # (N, H, W)

        self.transform = transform

    def __getitem__(self, index):
        """
        Get a single sample from the dataset.
        
        Args:
            index: Index of the sample to retrieve
            
        Returns:
            image: Tensor of shape (Bands, H, W) - Float32
            mask: Tensor of shape (H, W) - Long (for classification)
        """
        # Get numpy arrays
        image = self.images[index].copy()  # (Bands, H, W)
        mask = self.masks[index].copy()    # (H, W)

        # convert to tvorch tensors
        image = torch.from_numpy(image).float()
        mask  = torch.from_numpy(mask)

        image = tv_tensors.Image(image)
        mask  = tv_tensors.Mask(mask)

        # Apply transforms if provided
        if self.transform:
            image, mask = self.transform(image, mask)

        # return as tensors
        return image, mask

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.images)

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset


def fetch_loaders(x_train, y_train, x_val, y_val, batch_size=32, shuffle=True):
    """ Function to fetch dataLoaders for the Training / Validation

    Args:
        x_train(np.ndarray): Training images
        y_train(np.ndarray): Training labels
        x_val(np.ndarray): Validation images
        y_val(np.ndarray): Validation labels
        batch_size(int): The size of each batch during training. Defaults to 32.
        shuffle(bool): Whether to shuffle training data. Defaults to True.

    Return:
        Returns train and val dataloaders

    """
    # Convert numpy arrays to torch tensors
    x_train_t = torch.from_numpy(x_train).float()
    y_train_t = torch.from_numpy(y_train).long()
    x_val_t = torch.from_numpy(x_val).float()
    y_val_t = torch.from_numpy(y_val).long()
    
    # Create TensorDatasets
    train_dataset = TensorDataset(x_train_t, y_train_t)
    val_dataset = TensorDataset(x_val_t, y_val_t)
    
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

    def __init__(self, folder_path, transform=None, target_transform=None):
        """
        Initialize dataset.
        """

        self.img_files = sorted(glob.glob(os.path.join(folder_path, '*img*.npz')))
        self.mask_files = [s.replace("img", "mask") for s in self.img_files]
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):

        """ 
        getitem method to retrieve a single instance of the dataset
        """

        patch_path = self.img_files[index]
        mask_path = self.mask_files[index]
        data = np.load(patch_path)
        label = np.load(mask_path)

        return torch.from_numpy(data).float(), torch.from_numpy(label).float()

    def __len__(self):
        """ Function to return the length of the dataset
            Args:
                None
            Return:
                len(img_files)(int): The length of the dataset (img_files)

        """
        return len(self.img_files)
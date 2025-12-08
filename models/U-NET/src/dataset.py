import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class GlacierImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = decode_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    
def makeDataloader(csv_file, img_dir, batch_size, shuffle=True, transform=None, target_transform=None):
    dataset = GlacierImageDataset(annotations_file=csv_file, img_dir=img_dir, 
                                 transform=transform, target_transform=target_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


"""
Custom Dataset for Training
"""
import glob
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch

def fetch_loaders(processed_dir, batch_size=32,
                  train_folder='train', dev_folder='dev', test_folder='',
                  shuffle=True):
    """ Function to fetch dataLoaders for the Training / Validation

    Args:
        processed_dir(str): Directory with the processed data
        batch_size(int): The size of each batch during training. Defaults to 32.

    Return:
        Returns train and val dataloaders

    """
    train_dataset = GlacierDataset(processed_dir / train_folder)
    val_dataset = GlacierDataset(processed_dir / dev_folder)
    loader = {
        "train": DataLoader(train_dataset, batch_size=batch_size,
                            num_workers=8, shuffle=shuffle),
        "val": DataLoader(val_dataset, batch_size=batch_size,
                          num_workers=3, shuffle=False)}

    if test_folder:
        test_dataset = GlacierDataset(processed_dir / test_folder)
        loader["test"] = DataLoader(test_dataset, batch_size=batch_size,
                                    num_workers=3, shuffle=False)

    return loader


class GlacierDataset(Dataset):
    """Custom Dataset for Glacier Data

    Indexing the i^th element returns the underlying image and the associated
    binary mask

    """

    def __init__(self, folder_path):
        """Initialize dataset.

        Args:
            folder_path(str): A path to data directory

        """

        self.img_files = glob.glob(os.path.join(folder_path, '*img*'))
        self.mask_files = [s.replace("img", "mask") for s in self.img_files]

    def __getitem__(self, index):

        """ getitem method to retrieve a single instance of the dataset

        Args:
            index(int): Index identifier of the data instance

        Return:
            data(x) and corresponding label(y)
        """

        img_path = self.img_files[index]
        mask_path = self.mask_files[index]
        data = np.load(img_path)
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
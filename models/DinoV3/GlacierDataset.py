import os
import torch
from torch.utils.data import Dataset
import rasterio
import numpy as np
import random
import torchvision.transforms.functional as TF
from torchvision import transforms
from torchvision.transforms import InterpolationMode 

# CONSTANTS
SCALE_FACTOR = 10000.0 #  QUANTIFICATION_VALUE, see https://sentiwiki.copernicus.eu/web/s2-processing#:~:text=The%20reflectance,%20often%20between%200%20and%201,%20is%20converted%20into%20integer%20values,%20to%20preserve%20the%20dynamic%20range%20of%20the%20data%20by%20applying%20a%20fixed%20coefficient%20(10000%20by%20default)%20called
DINO_MEAN = [0.430, 0.411, 0.296] # see GitHub DINOv3
DINO_STD  = [0.213, 0.156, 0.143]
IMAGENET_MEAN = [0.485, 0.456, 0.406] # Standard ImageNet values
IMAGENET_STD  = [0.229, 0.224, 0.225]
TARGET_SIZE = (224, 224) # Strict size

class GlacierDataset(Dataset):
    
    def __init__(self, image_dir, mask_dir, model, mode="train"):
        """Initializes the Glacier Dataset.

        Args:
            image_dir (str): Path to the directory containing images.
            mask_dir (str): Path to the directory containing masks.
            model (str): Either 'DinoV3' or 'UNet'. (needed for normalization)
            mode (str, optional): Mode of the dataset, either "train" or "test" (disables data augmentation). Defaults to "train".
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.mode = mode
        self.images = [f for f in os.listdir(image_dir) if f.endswith('.tif')]
        
        assert model in ["DinoV3", "UNet"], "Model not supported, use either 'DinoV3' or 'UNet'."
        self.model = model
        if model == "UNet":
            self.normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        else: 
            self.normalize = transforms.Normalize(mean=DINO_MEAN, std=DINO_STD)
        
        self.groups = [img.split('_tile')[0] for img in self.images]
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        # load image and mask
        with rasterio.open(img_path) as src:
            image = src.read([3, 2, 1]).astype(np.float32)
        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.int64)
            
        # The test set has some missing info due to the swiss border, we will add it to the ignore index
        is_nodata = np.sum(image, axis=0) == 0
        mask[is_nodata] = 255

        # Physical scaling
        image = image / SCALE_FACTOR
        image = np.clip(image, 0, 1.0)

        img_tensor = torch.from_numpy(image)      # (3, H, W)
        mask_tensor = torch.from_numpy(mask)      # (H, W)

        # Image : Interpolation BILINEAR (lisse)
        img_tensor = TF.resize(img_tensor, TARGET_SIZE, interpolation=InterpolationMode.BILINEAR, antialias=True)
        
        # Mask : NEAREST -> we don't want partial classes
        # TF.resize expects (C, H, W) or (H, W), but to be sure we unsqueeze
        mask_tensor = mask_tensor.unsqueeze(0) 
        mask_tensor = TF.resize(mask_tensor, TARGET_SIZE, interpolation=InterpolationMode.NEAREST)
        mask_tensor = mask_tensor.squeeze(0)

        # 3. Augmentation 
        if self.mode == "train":
            # Flips
            if random.random() > 0.5:
                img_tensor = TF.hflip(img_tensor)
                mask_tensor = TF.hflip(mask_tensor)
            if random.random() > 0.5:
                img_tensor = TF.vflip(img_tensor)
                mask_tensor = TF.vflip(mask_tensor)
            
            # Rotations
            rotations = random.choice([0, 90, 180, 270])
            if rotations > 0:
                img_tensor = TF.rotate(img_tensor, rotations)
                
                # For the mask, we need to unsqueeze to have a channel dimension
                mask_float = mask_tensor.unsqueeze(0).float()
                mask_float = TF.rotate(mask_float, rotations, interpolation=InterpolationMode.NEAREST)
                mask_tensor = mask_float.squeeze(0).long()

        # end Normalization
        img_tensor = self.normalize(img_tensor)

        return img_tensor, mask_tensor
    
        
    
    

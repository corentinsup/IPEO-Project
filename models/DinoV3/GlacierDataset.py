import os
import torch
from torch.utils.data import Dataset
import rasterio
import numpy as np
import random
import torchvision.transforms.functional as TF
from torchvision import transforms
from torchvision.transforms import InterpolationMode # Import vital

# --- CONSTANTES ---
SCALE_FACTOR = 10000.0
DINO_MEAN = [0.430, 0.411, 0.296]
DINO_STD  = [0.213, 0.156, 0.143]
TARGET_SIZE = (224, 224) # La taille stricte attendue par le modèle

class GlacierDataset(Dataset):
    def __init__(self, image_dir, mask_dir, mode="train"):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.mode = mode
        self.images = [f for f in os.listdir(image_dir) if f.endswith('.tif')]
        self.normalize = transforms.Normalize(mean=DINO_MEAN, std=DINO_STD)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        # 1. Chargement
        with rasterio.open(img_path) as src:
            image = src.read([3, 2, 1]).astype(np.float32)
        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.int64)

        # 2. Scaling Physique
        image = image / SCALE_FACTOR
        image = np.clip(image, 0, 1.0)

        # Conversion Tensor
        img_tensor = torch.from_numpy(image)      # (3, H, W)
        mask_tensor = torch.from_numpy(mask)      # (H, W)

        # --- FIX #1 : FORCER LA TAILLE (RESIZE DE SÉCURITÉ) ---
        # Si l'image fait 228x228 ou 227x227, on la ramène à 224x224
        # Image : Interpolation BILINEAR (lisse)
        img_tensor = TF.resize(img_tensor, TARGET_SIZE, interpolation=InterpolationMode.BILINEAR, antialias=True)
        
        # Masque : Interpolation NEAREST (pas de mélange de classes !)
        # TF.resize attend (C, H, W) ou (H, W), mais pour être sûr on unsqueeze
        mask_tensor = mask_tensor.unsqueeze(0) 
        mask_tensor = TF.resize(mask_tensor, TARGET_SIZE, interpolation=InterpolationMode.NEAREST)
        mask_tensor = mask_tensor.squeeze(0)

        # 3. Augmentation (Train only)
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
                
                # --- FIX #2 : ROTATION DES MASQUES ---
                # On passe en float le temps de la rotation pour éviter le crash
                mask_float = mask_tensor.unsqueeze(0).float()
                mask_float = TF.rotate(mask_float, rotations, interpolation=InterpolationMode.NEAREST)
                mask_tensor = mask_float.squeeze(0).long()

        # 4. Normalisation
        img_tensor = self.normalize(img_tensor)

        return img_tensor, mask_tensor
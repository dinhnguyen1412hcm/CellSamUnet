import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple, Optional

class MoNuSegDataset(Dataset):
    def __init__(self, image_paths: list, mask_paths: list, transform: Optional[A.Compose] = None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        
        assert len(self.image_paths) == len(self.mask_paths), \
            f"Mismatch: Images ({len(self.image_paths)}) vs Masks ({len(self.mask_paths)})"

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            # Handle corrupt masks
            h, w, _ = image.shape
            mask = np.zeros((h, w), dtype=np.float32)

        # Resize mask if dimensions mismatch
        img_h, img_w, _ = image.shape
        if mask.shape[0] != img_h or mask.shape[1] != img_w:
            mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

        image = (image / 255.0).astype(np.float32)
        mask = (mask > 0).astype(np.float32) 
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image_tensor, mask_tensor = augmented['image'], augmented['mask']
        else:
            image_tensor = torch.from_numpy(image.transpose(2, 0, 1))
            mask_tensor = torch.from_numpy(mask)

        if len(mask_tensor.shape) == 2:
            mask_tensor = mask_tensor.unsqueeze(0)
        
        return {'image': image_tensor.float(), 'mask': mask_tensor.float()}

def get_train_augmentations(target_size: Tuple[int, int]):
    return A.Compose([
        A.Resize(height=target_size[0], width=target_size[1]), 
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5), 
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(p=0.7), 
        A.OneOf([A.RandomBrightnessContrast(p=0.5), A.HueSaturationValue(p=0.5)], p=0.7),
        A.GaussNoise(p=0.2), 
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=1.0), 
        ToTensorV2(),
    ])

def get_val_augmentations(target_size: Tuple[int, int]):
    return A.Compose([
        A.Resize(height=target_size[0], width=target_size[1]),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=1.0), 
        ToTensorV2()
    ])
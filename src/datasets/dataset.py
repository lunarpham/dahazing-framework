import torch
from torch.utils.data import Dataset
import random
import torchvision.transforms.functional as TF
from PIL import Image

from .pairing import get_image_pairs

class DehazeDataset(Dataset):
    """Dataset for loading hazy/clear image pairs with on-the-fly augmentation."""

    def __init__(self, hazy_dir, clear_dir, patch_size=128, mode='train',
                 augmentation=None):
        super().__init__()
        self.pairs = get_image_pairs(hazy_dir, clear_dir)
        self.patch_size = patch_size
        self.mode = mode

        if augmentation is None:
            self.augmentation = ['hflip', 'vflip'] if mode == 'train' else []
        else:
            self.augmentation = augmentation if mode == 'train' else []
        
        if len(self.pairs) == 0:
            print(f"Warning: No valid pairs found between {hazy_dir} and {clear_dir}.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        hazy_path, clear_path = self.pairs[idx]
        
        hazy_img = Image.open(hazy_path).convert('RGB')
        clear_img = Image.open(clear_path).convert('RGB')
        
        if self.mode == 'train':
            w, h = hazy_img.size
            
            # Pad if smaller than patch_size
            pad_w = max(0, self.patch_size - w)
            pad_h = max(0, self.patch_size - h)
            if pad_w > 0 or pad_h > 0:
                hazy_img = TF.pad(hazy_img, (0, 0, pad_w, pad_h))
                clear_img = TF.pad(clear_img, (0, 0, pad_w, pad_h))
                w, h = hazy_img.size
            
            # Random crop
            i = random.randint(0, h - self.patch_size)
            j = random.randint(0, w - self.patch_size)
            hazy_img = TF.crop(hazy_img, i, j, self.patch_size, self.patch_size)
            clear_img = TF.crop(clear_img, i, j, self.patch_size, self.patch_size)
            
            # Augmentations
            if 'hflip' in self.augmentation and random.random() > 0.5:
                hazy_img = TF.hflip(hazy_img)
                clear_img = TF.hflip(clear_img)
                
            if 'vflip' in self.augmentation and random.random() > 0.5:
                hazy_img = TF.vflip(hazy_img)
                clear_img = TF.vflip(clear_img)
                
        else:
            # Center crop for validation
            w, h = hazy_img.size
            pad_w = max(0, self.patch_size - w)
            pad_h = max(0, self.patch_size - h)
            if pad_w > 0 or pad_h > 0:
                hazy_img = TF.pad(hazy_img, (0, 0, pad_w, pad_h))
                clear_img = TF.pad(clear_img, (0, 0, pad_w, pad_h))
            
            hazy_img = TF.center_crop(hazy_img, self.patch_size)
            clear_img = TF.center_crop(clear_img, self.patch_size)

        hazy_tensor = TF.to_tensor(hazy_img)
        clear_tensor = TF.to_tensor(clear_img)
        
        return {
            'hazy': hazy_tensor,
            'clear': clear_tensor,
            'hazy_path': hazy_path
        }

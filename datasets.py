import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class BlocksDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.imps = [
            os.path.join(img_dir, file)
            for file in os.listdir(img_dir)
            if file.endswith('.png')
        ]
        self.transform = transform
        self.target_transform = target_transform
    def __len__(self):
        return len(self.imps)
    def __getitem__(self, idx):
        img_path = self.imps[idx]
        image = Image.open(img_path).convert("RGB")  # Ensure 3 channels
        image = np.array(image)[:16, :16, :3]
        image = Image.fromarray(image).resize((16, 16), Image.NEAREST)
        image = torch.tensor(np.array(image)).permute(2, 0, 1).float()[:3, :16, :16]*1/255
        if self.transform:
            image = self.transform(image)
        return (image, img_path)

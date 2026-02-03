import os
import torch
import numpy as np
import rasterio
from torch.utils.data import Dataset, DataLoader
from preprocessing import preprocess_gee_image

class Sentinel1Dataset(Dataset):
    def __init__(self, s1_dir, label_dir, patch_size=256):
        self.s1_dir = s1_dir
        self.label_dir = label_dir
        self.patch_size = patch_size
        self.filenames = sorted([f for f in os.listdir(s1_dir) if f.endswith('.tif')])
        
        self.patches = []
        for f_idx, fname in enumerate(self.filenames):
            s1_path = os.path.join(self.s1_dir, fname)
            with rasterio.open(s1_path) as src:
                h, w = src.height, src.width
            
            # Extract only valid 256x256 squares
            for i in range(0, h - patch_size + 1, patch_size):
                for j in range(0, w - patch_size + 1, patch_size):
                    self.patches.append((f_idx, i, j))
        
        print(f"✅ Dataset initialized: {len(self.patches)} valid patches from {len(self.filenames)} cities.")

    def remap_4_classes(self, label_mask):
        # 0:Blue(Water), 1:Green(Veg), 2:Red(Urban), 3:Yellow(Barren)
        remapped = np.zeros_like(label_mask)
        remapped[label_mask == 0] = 0 
        remapped[np.isin(label_mask, [1, 2, 3, 4, 5])] = 1 
        remapped[label_mask == 6] = 2 # Consolidated Urban
        remapped[np.isin(label_mask, [7, 8])] = 3 # Barren/Other
        return remapped

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        f_idx, y, x = self.patches[idx]
        s1_path = os.path.join(self.s1_dir, self.filenames[f_idx])
        label_path = os.path.join(self.label_dir, self.filenames[f_idx].replace('s1_', 'lc_'))

        with rasterio.open(s1_path) as src:
            window = rasterio.windows.Window(x, y, self.patch_size, self.patch_size)
            s1_raw = src.read(window=window).astype(np.float32)
        s1_patch = preprocess_gee_image(s1_raw)

        if os.path.exists(label_path):
            with rasterio.open(label_path) as src:
                label_raw = src.read(1, window=window).astype(np.int64)
                label_patch = self.remap_4_classes(label_raw)
        else:
            label_patch = np.zeros((self.patch_size, self.patch_size), dtype=np.int64)

        return torch.from_numpy(s1_patch).float(), torch.from_numpy(label_patch).long()

def get_dataloader(s1_path, lc_path, batch_size=8, shuffle=True):
    dataset = Sentinel1Dataset(s1_path, lc_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)
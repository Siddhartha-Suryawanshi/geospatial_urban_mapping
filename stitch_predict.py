import torch
import rasterio
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from model import DeepLabV3Plus
from preprocessing import preprocess_gee_image

def stitch_prediction(model_path, s1_path, lc_path, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepLabV3Plus(n_classes=4, n_channels=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    colors = ['#1f77b4', '#2ca02c', '#d62728', '#FFD700']
    cmap = ListedColormap(colors)
    patch_size = 256

    with rasterio.open(s1_path) as src_s1:
        h, w = src_s1.height, src_s1.width
        # Fix: Read the full radar data here so it is available for plotting later
        full_s1_data = src_s1.read().astype(np.float32)
        full_pred_mask = np.zeros((h, w), dtype=np.uint8)
        
        print(f"🧩 Stitching {os.path.basename(s1_path)}...")
        for y in range(0, h, patch_size):
            for x in range(0, w, patch_size):
                window_h = min(patch_size, h - y)
                window_w = min(patch_size, w - x)
                window = rasterio.windows.Window(x, y, window_w, window_h)
                
                s1_patch = src_s1.read(window=window).astype(np.float32)
                
                # Pad to 256x256 if at the edge
                pad_h = patch_size - window_h
                pad_w = patch_size - window_w
                if pad_h > 0 or pad_w > 0:
                    s1_patch = np.pad(s1_patch, ((0,0), (0, pad_h), (0, pad_w)), mode='reflect')

                processed = preprocess_gee_image(s1_patch)
                input_tensor = torch.from_numpy(processed).float().unsqueeze(0).to(device)
                
                with torch.no_grad():
                    output = model(input_tensor)
                    pred_patch = torch.argmax(output[0], dim=0).cpu().numpy()
                
                full_pred_mask[y:y+window_h, x:x+window_w] = pred_patch[:window_h, :window_w]

    # Load Ground Truth for comparison
    with rasterio.open(lc_path) as src_lc:
        full_lc_data = src_lc.read(1).astype(np.int64)

    # 4-Panel Visualization
    sar_img = full_s1_data[0] # VV Band
    sar_img = (sar_img - sar_img.min()) / (sar_img.max() - sar_img.min() + 1e-6)

    # Remap Ground Truth to 4 classes
    remapped_lc = np.zeros_like(full_lc_data)
    remapped_lc[full_lc_data == 0] = 0
    remapped_lc[np.isin(full_lc_data, [1, 2, 3, 4, 5])] = 1
    remapped_lc[full_lc_data == 6] = 2
    remapped_lc[np.isin(full_lc_data, [7, 8])] = 3

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    axes[0].imshow(sar_img, cmap='gray'); axes[0].set_title("Input Radar (VV)")
    axes[1].imshow(remapped_lc, cmap=cmap, vmin=0, vmax=3); axes[1].set_title("Ground Truth (Label)")
    axes[2].imshow(full_pred_mask, cmap=cmap, vmin=0, vmax=3); axes[2].set_title("Model Prediction")
    axes[3].imshow(sar_img, cmap='gray')
    axes[3].imshow(full_pred_mask, cmap=cmap, vmin=0, vmax=3, alpha=0.4); axes[3].set_title("Urban Overlay")
    
    for ax in axes: ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"✅ Saved full visualization to {output_path}")
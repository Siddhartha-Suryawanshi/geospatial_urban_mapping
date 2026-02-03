import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def save_visual_result(image, pred, mask, filename):
    sar_img = image[0].numpy()
    sar_img = (sar_img - sar_img.min()) / (sar_img.max() - sar_img.min() + 1e-6)
    
    # 0:Blue, 1:Green, 2:Red, 3:Yellow
    colors = ['#1f77b4', '#2ca02c', '#d62728', '#FFD700']
    cmap = ListedColormap(colors)

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    axes[0].imshow(sar_img, cmap='gray'); axes[0].set_title("Input Radar")
    axes[1].imshow(mask.numpy(), cmap=cmap, vmin=0, vmax=3); axes[1].set_title("Ground Truth")
    axes[2].imshow(pred.numpy(), cmap=cmap, vmin=0, vmax=3); axes[2].set_title("Prediction")
    axes[3].imshow(sar_img, cmap='gray')
    axes[3].imshow(pred.numpy(), cmap=cmap, vmin=0, vmax=3, alpha=0.4); axes[3].set_title("Urban Overlay")
    
    for ax in axes: ax.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close(fig)
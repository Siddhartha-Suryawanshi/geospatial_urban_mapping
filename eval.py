import torch
import numpy as np
import os
import sys
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from model import DeepLabV3Plus
from src.data_loader import get_dataloader

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = 'results/global_urban_final.pth'
    raw_data_path = './data/raw'
    n_classes = 4
    class_names = ['Water', 'Vegetation', 'Urban', 'Barren']

    # 1. Load Model
    model = DeepLabV3Plus(n_classes=n_classes, n_channels=2).to(device)
    if not os.path.exists(model_path):
        print("❌ Model not found. Train it first!")
        return
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 2. Setup Data
    val_loader = get_dataloader(os.path.join(raw_data_path, 'images'), 
                                os.path.join(raw_data_path, 'labels'), batch_size=8, shuffle=False)

    all_preds, all_labels = [], []

    print(f"📊 Evaluating on {len(val_loader.dataset)} patches across 25 global biomes...")
    with torch.no_grad():
        for images, masks in tqdm(val_loader):
            images, masks = images.to(device), masks.to(device).long()
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            all_preds.append(preds.flatten())
            all_labels.append(masks.cpu().numpy().flatten())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)

    # 3. Generate Metrics
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))
    
    # Calculate IoU
    intersection = np.diag(cm)
    union = cm.sum(axis=1) + cm.sum(axis=0) - intersection
    ious = intersection / (union + 1e-6)

    # 4. Final Professional Table for Report
    print("\n" + "="*75)
    print(f"{'LAND COVER CLASS':<16} | {'IoU (%)':<10} | {'Precision':<10} | {'Recall':<10}")
    print("-" * 75)
    for i, name in enumerate(class_names):
        p, r = report[name]['precision'], report[name]['recall']
        print(f"{name:<16} | {ious[i]*100:>8.2f}% | {p:>9.3f} | {r:>9.3f}")
    
    print("-" * 75)
    print(f"{'GLOBAL AVERAGE':<16} | mIoU: {np.mean(ious)*100:.2f}%  | Overall Acc: {report['accuracy']:>6.3f}")
    print("="*75)

if __name__ == "__main__":
    evaluate()
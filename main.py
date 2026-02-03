import torch
import torch.optim as optim
import os
import sys
import ee
import geemap

# 1. Authorize Earth Engine (Notebook Mode for remote servers)
try:
    ee.Initialize(project='ee-siddharthasuryawanshi02')
except Exception:
    print("🔑 Authenticating Earth Engine (Notebook Mode)...")
    ee.Authenticate(auth_mode='notebook')
    ee.Initialize(project='ee-siddharthasuryawanshi02')

# Path setup for local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from model import DeepLabV3Plus
from src.data_loader import get_dataloader
from stitch_predict import stitch_prediction

# Full 25-Region Dataset (19 Cities + 6 Diverse Biomes)
CITIES = {
    "london": [-0.15, 51.48, -0.05, 51.55], "new_york": [-74.02, 40.70, -73.92, 40.80],
    "cairo": [31.20, 30.00, 31.30, 30.10], "mumbai": [72.80, 18.90, 72.90, 19.10],
    "tokyo": [139.70, 35.65, 139.80, 35.75], "dubai": [55.15, 25.05, 55.25, 25.15],
    "sao_paulo": [-46.70, -23.60, -46.60, -23.50], "sydney": [151.15, -33.90, 151.25, -33.80],
    "nairobi": [36.78, -1.32, 36.85, -1.27], "paris": [2.32, 48.84, 2.37, 48.87],
    "singapore": [103.82, 1.28, 103.87, 1.32], "los_angeles": [-118.27, 34.03, -118.23, 34.07],
    "moscow": [37.59, 55.73, 37.63, 55.77], "johannesburg": [28.03, -26.21, 28.07, -26.17],
    "mexico_city": [-99.14, 19.42, -99.11, 19.44], "seoul": [126.96, 37.53, 127.01, 37.56],
    "bangkok": [100.48, 13.72, 100.53, 13.77], "madrid": [-3.71, 40.41, -3.68, 40.44],
    "toronto": [-79.39, 43.65, -79.37, 43.66],
    # Nature Biomes (Remote regions)
    "amazon_forest": [-62.50, -3.10, -62.40, -3.00], 
    "sahara_desert": [15.50, 25.50, 15.60, 25.60],
    "himalaya_hills": [85.30, 27.70, 85.40, 27.80],
    "botswana_delta": [22.50, -19.20, 22.60, -19.10],
    "australia_desert": [122.50, -21.50, 122.60, -21.40],
    "siberia_taiga": [105.30, 56.30, 105.40, 56.40]
}

def fetch_multi_city_data(output_dir, project_id):
    """Downloads radar and labels with flexible band searching."""
    img_dir, lbl_dir = os.path.join(output_dir, 'images'), os.path.join(output_dir, 'labels')
    os.makedirs(img_dir, exist_ok=True); os.makedirs(lbl_dir, exist_ok=True)

    for city, coords in CITIES.items():
        s1_path, lc_path = os.path.join(img_dir, f"s1_{city}.tif"), os.path.join(lbl_dir, f"lc_{city}.tif")
        if os.path.exists(s1_path) and os.path.exists(lc_path):
            continue

        print(f"🛰️ Fetching {city}...")
        roi = ee.Geometry.Rectangle(coords)
        
        # Robust S1 search: Wide date range and no mode restriction
        s1_coll = ee.ImageCollection('COPERNICUS/S1_GRD').filterBounds(roi).filterDate('2023-01-01', '2025-12-31')
        s1_img = s1_coll.median().clip(roi)
        
        dw_lbl = (ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1').filterBounds(roi)
                  .filterDate('2023-01-01', '2025-12-31').select('label').mode().clip(roi))

        try:
            bands = s1_img.bandNames().getInfo()
            if 'VV' in bands and 'VH' in bands:
                geemap.ee_export_image(s1_img.select(['VV', 'VH']), filename=s1_path, scale=10, region=roi)
                geemap.ee_export_image(dw_lbl, filename=lc_path, scale=10, region=roi)
            else:
                print(f"⚠️ {city} skipped: Required VV/VH bands not found in this region.")
        except Exception as e:
            print(f"❌ Error downloading {city}: {e}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_data_path, results_path = './data/raw', './results'
    os.makedirs(results_path, exist_ok=True)

    # 1. Sync Data (Fail-safe mode)
    fetch_multi_city_data(raw_data_path, 'ee-siddharthasuryawanshi02')

    # 2. Setup Training
    model = DeepLabV3Plus(n_classes=4, n_channels=2).to(device)
    train_loader = get_dataloader(os.path.join(raw_data_path, 'images'), 
                                  os.path.join(raw_data_path, 'labels'), batch_size=16)

    # Sweet Spot Weights & Optimizer
    weights = torch.tensor([0.8, 1.0, 1.5, 1.2]).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=7e-5)

    print(f"🚀 Training with {len(train_loader.dataset)} patches across 25 regions...")
    for epoch in range(60):
        model.train()
        epoch_loss = 0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device).long()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/60 | Avg Loss: {epoch_loss/len(train_loader):.4f}")
        torch.cuda.empty_cache()

    model_path = os.path.join(results_path, 'global_urban_final.pth')
    torch.save(model.state_dict(), model_path)

    # 3. Final Report Visuals (7 Cities + 3 Biomes)
    print("🎨 Generating 10 Final Report Visualizations...")
    visual_targets = ['london', 'paris', 'tokyo', 'new_york', 'mumbai', 'dubai', 'sydney', 
                      'amazon_forest', 'sahara_desert', 'himalaya_hills', 'botswana_delta', 'australia_desert','siberia_taiga']
    
    for area in visual_targets:
        s1 = os.path.join(raw_data_path, f'images/s1_{area}.tif')
        lc = os.path.join(raw_data_path, f'labels/lc_{area}.tif')
        out = os.path.join(results_path, f'REPORT_VISUAL_{area}.png')
        if os.path.exists(s1) and os.path.exists(lc):
            stitch_prediction(model_path, s1, lc, out)

    print("✅ Pipeline complete. Remember to run 'python eval.py' for final accuracy stats!")

if __name__ == "__main__":
    main()
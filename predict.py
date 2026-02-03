import torch
import rasterio
import numpy as np
import os
import sys
import ee
import geemap

# Ensure pathing is correct for your src folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from model import DeepLabV3Plus
from preprocessing import preprocess_gee_image
from utils import save_visual_result

def fetch_test_image(roi_coords, project_id, filename):
    """Downloads a fresh SAR image for testing with basic error handling."""
    try:
        ee.Initialize(project=project_id)
        roi = ee.Geometry.Rectangle(roi_coords)
        
        # Using a median reducer for 2025 data
        s1_img = (ee.ImageCollection('COPERNICUS/S1_GRD')
                  .filterBounds(roi)
                  .filterDate('2025-01-01', '2025-12-31')
                  .filter(ee.Filter.eq('instrumentMode', 'IW'))
                  .median().clip(roi).select(['VV', 'VH']))
        
        print(f"🛰️ Requesting download for {filename}...")
        geemap.ee_export_image(s1_img, filename=filename, scale=10, region=roi)
        
        if os.path.exists(filename):
            print(f"✅ Successfully downloaded {filename}")
            return True
        else:
            print(f"❌ Download failed: File was not created.")
            return False
    except Exception as e:
        print(f"❌ Earth Engine Error: {e}")
        return False

def run_prediction(model_path, tif_path, output_png):
    if not os.path.exists(tif_path):
        print(f"⚠️ Skipping prediction: {tif_path} does not exist.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load 4-Class Model (matching your updated 4-class trainer)
    model = DeepLabV3Plus(n_classes=4, n_channels=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(f"🧠 Loading {tif_path} into model...")
    with rasterio.open(tif_path) as src:
        s1_data = src.read().astype(np.float32)
    
    s1_processed = preprocess_gee_image(s1_data)
    input_tensor = torch.from_numpy(s1_processed).float().unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output[0], dim=0).cpu()

    # Save visualization (using pred as dummy mask)
    save_visual_result(torch.from_numpy(s1_processed), pred, pred, output_png)
    print(f"🎯 Prediction complete! Open {output_png} to see results.")

if __name__ == "__main__":
    MY_PROJECT = 'ee-siddharthasuryawanshi02'
    
    # PARIS TEST: Slightly smaller ROI for higher success rate on Uni servers
    # [min_lon, min_lat, max_lon, max_lat]
    paris_roi = [2.30, 48.83, 2.38, 48.88] 
    test_tif = 'test_paris.tif'
    
    # Step 1: Download
    download_success = fetch_test_image(paris_roi, MY_PROJECT, test_tif)
    
    # Step 2: Predict ONLY if download worked
    if download_success:
        run_prediction('results/global_urban_final.pth', test_tif, 'paris_prediction_result.png')
    else:
        print("💡 Suggestion: Try an even smaller ROI or check your internet connection.")
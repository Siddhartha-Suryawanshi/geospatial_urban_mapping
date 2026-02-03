import ee
import geemap
import os

def download_training_pair(roi_coords, date_start, date_end, output_dir):
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)
    
    roi = ee.Geometry.Rectangle(roi_coords)
    
    # 1. Fetch Sentinel-1 (The Input)
    s1_col = (ee.ImageCollection('COPERNICUS/S1_GRD')
              .filterBounds(roi)
              .filterDate(date_start, date_end)
              .filter(ee.Filter.eq('instrumentMode', 'IW')))
    s1_image = s1_col.median().clip(roi).select(['VV', 'VH'])

    # 2. Fetch Dynamic World (The Labels/Ground Truth)
    dw_col = (ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
              .filterBounds(roi)
              .filterDate(date_start, date_end))
    # We use 'mode' to get the most frequent label for that time period
    label_image = dw_col.select('label').mode().clip(roi)

    # 3. Export both to your local data folder
    geemap.ee_export_image(s1_image, filename=f"{output_dir}/images/s1_data.tif", scale=10, region=roi)
    geemap.ee_export_image(label_image, filename=f"{output_dir}/labels/lc_labels.tif", scale=10, region=roi)
    
    print("✅ Training pair (Image + Label) downloaded successfully.")
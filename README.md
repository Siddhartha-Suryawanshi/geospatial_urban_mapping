# Global Urban Mapping from Sentinel-1 Radar

## Project Overview
This project implements a **DeepLabV3+** architecture to perform global land cover classification using **Sentinel-1 SAR (Radar)** imagery. The model is trained on a 25-region dataset including 19 megacities and 6 diverse nature biomes, using labels derived from **Google Dynamic World V1**.

## Technical Pillars
1. **Architecture:** DeepLabV3+ with Atrous Spatial Pyramid Pooling (ASPP) for multi-scale feature extraction.
2. **Data:** Automated GEE pipeline for Sentinel-1 (VV/VH bands) and Dynamic World labels.
3. **Physics:** Log-scaling transforms and weighted cross-entropy loss to handle radar backscatter challenges.

## Execution Guide

### 1. Requirements
Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Data Acquisition & Training
Run the main script to download data from Earth Engine and start the training process:
```bash
python main.py
```

### 3. Evaluation
To generate global accuracy metrics (mIoU, Precision, Recall):
```bash
python eval.py
```

### 4. Results
Visualizations are automatically saved in the `results/` directory as 4-panel verification maps.

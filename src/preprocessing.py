import numpy as np

def preprocess_gee_image(image):
    """
    Enhanced SAR preprocessing to separate Water from low-intensity Urban shadows.
    """
    # 1. Convert to Log Scale to expand the dynamic range of low-intensity pixels
    # This makes the difference between a 'dark' road and 'dark' water visible.
    image = np.log1p(np.abs(image))
    
    # 2. Robust Scaling: Remove outliers by clipping to 1st and 99th percentiles
    for i in range(image.shape[0]):
        low, high = np.percentile(image[i], [1, 99])
        image[i] = np.clip(image[i], low, high)
        
        # 3. Min-Max Normalization to scale values between 0 and 1
        image[i] = (image[i] - low) / (high - low + 1e-6)
        
    return image
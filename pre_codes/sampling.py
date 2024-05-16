import os
import numpy as np
import rasterio
from rasterio.windows import Window
from skimage.util import view_as_windows
from PIL import Image

def normalize(data):
    """Normalize data to the range of 0-255."""
    min_val = np.min(data)
    max_val = np.max(data)
    return ((data - min_val) / (max_val - min_val) * 255).astype(np.uint8)

def duplicateto3bands(image):
    """Duplicate a single-band image into three bands."""
    return np.repeat(image[:, :, np.newaxis], 3, axis=2)

# Paths to rasters
path_A = "/shared/stormcenter/Qing_Y/GAN_ChangeDetection/MAE/Data/dem_3DEP_Ohio.tif"
path_B = "/shared/stormcenter/Qing_Y/GAN_ChangeDetection/MAE/Data/WOP_3dep_Ohio.tif"

# Output directory for patches
output_dir = "/shared/stormcenter/Qing_Y/GAN_ChangeDetection/MAE/samples/intial_pre"
os.makedirs(output_dir, exist_ok=True)

# Read raster B and create mask
with rasterio.open(path_B) as src_B:
    B = src_B.read(1)  # Assuming a single band raster
    mask = B >= 50

# Read raster A
with rasterio.open(path_A) as src_A:
    A = src_A.read(1)  # Assuming a single band raster
    A = normalize(A)
    
# Define patch size and overlap
patch_size = 224
overlap = 0.30

# Generate patches
patches = view_as_windows(mask, (patch_size, patch_size), step=int(patch_size * (1 - overlap)))
A_pathes = view_as_windows(A, (patch_size, patch_size), step=int(patch_size * (1 - overlap)))

# Save patches as .jpg files
for i in range(patches.shape[0]):
    for j in range(patches.shape[1]):
        patch = patches[i, j]
        A_ps = A_pathes[i, j]
        if not np.any(patch):
            A_ps_duplicated = duplicateto3bands(A_ps)
            patch_A_img = Image.fromarray(A_ps_duplicated, 'RGB')
            filename = f"patch_{i}_{j}.jpg"
            patch_A_img.save(os.path.join(output_dir, filename))
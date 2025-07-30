# 🧬 Cellpose Segmentation Pipeline 🧪

Welcome to the **Cellpose Segmentation** repository! This project provides a powerful and flexible Python-based pipeline for automated cell segmentation using the state-of-the-art Cellpose and Cellpose-SAM models. It is designed to work with microscopy TIFF image stacks, particularly for experiments involving DIC and fluorescence channels, and is optimized for GPU acceleration.  

---

## 🔬 Overview

This pipeline segments cells or bacteria in microscopy image stacks, quantifies shape and intensity features, and outputs detailed segmentation masks alongside comprehensive Excel reports. The main segmentation engine leverages the `cellpose` Python package (v4.0.6) with support for the latest SAM (Segment Anything Model) integration, allowing precise and scalable cell detection.

---

## 🧩 Detailed Description of the Python Scripts

### 1. **Main Segmentation Script**

This script orchestrates the full workflow from image loading, segmentation, feature extraction, to results saving:

- **GPU Check & Setup:**  
  It begins by detecting CUDA and GPU availability for hardware-accelerated segmentation on compatible NVIDIA GPUs (e.g., RTX 4500 Ada Generation). This significantly speeds up model inference.

- **Input/Output Folder Setup:**  
  You specify your input folder containing `.tif` or `.tiff` microscopy stacks and an output folder for saving segmentation masks and Excel result files. The output folder is created if it doesn't exist.

- **Model Loading:**  
  The script loads the Cellpose-SAM model using `models.CellposeModel` with the pretrained `cpsam` weights. This model is designed for high accuracy on diverse cell types and image modalities.

- **Segmentation Parameters:**  
  - `diameter`: Cell diameter for segmentation; if unknown, set to `None` for automatic detection.  
  - `batch_size`: Number of tiles processed simultaneously (default 4). Adjust based on your GPU VRAM.  
  - `tile_overlap`: Overlap fraction between tiles to avoid edge artifacts.  
  - `bsize`: Tile size for image splitting during segmentation (default 256).

- **Image Loading and Preprocessing:**  
  Each TIFF stack is expected to have two channels:  
  - Channel 0: Differential Interference Contrast (DIC) image used for segmentation.  
  - Channel 1: Fluorescence image used for measuring intensity within segmented cells.  

  Both images are normalized independently **except the fluorescence channel if you disable normalization**.

- **Padding:**  
  Images are padded via reflection to multiples of `bsize` for smooth tiling during segmentation and later cropped back to original size.

- **Cellpose-SAM Segmentation:**  
  The prepared DIC channel image is fed into the Cellpose-SAM model's `.eval()` function to produce segmentation masks. This returns:  
  - `masks`: Labeled segmentation mask array.  
  - `flows` and `styles`: Additional model outputs (not used downstream here).

- **Mask Cropping & Labeling:**  
  The padded masks are cropped back to original image size and labeled to identify individual segmented objects.

- **Feature Extraction:**  
  Using `skimage.measure.regionprops`, the script calculates per-object features, including:  
  - Area  
  - Length (major axis length)  
  - Width (minor axis length)  
  - Roundness (shape circularity)  
  - Aspect Ratio  
  - Mean fluorescence intensity inside the mask  
  - Centroid coordinates (X, Y)  

  Additionally, background fluorescence mean and standard deviation are computed from the non-cell regions.

- **GPU Memory Management & OOM Handling:**  
  The script monitors free GPU memory. If out-of-memory (OOM) errors occur, it automatically reduces the batch size and tile size and retries segmentation up to 3 times per stack.

- **Batch Processing:**  
  The script loops through all TIFF files in the input folder, processes them one by one with a progress bar, saves the segmentation masks (as 16-bit TIFFs), and exports per-object features to Excel files.

- **Error Handling:**  
  Any exception during processing a stack is caught and reported without stopping the entire batch.

---

## 🚀 How to Use

1. **Prepare your environment:**

   - Python 3.8+  
   - Install required packages (recommended in a virtual environment):  
     ```bash
     pip install cellpose==4.0.6 tifffile pandas scikit-image tqdm torch
     ```

2. **Organize your data:**

   - Input folder should contain TIFF stacks with exactly two channels:  
     - Channel 0: DIC or phase contrast image for segmentation  
     - Channel 1: Fluorescence image for intensity measurement  

3. **Adjust parameters:**

   - Set `input_folder` and `output_folder` paths in the script.  
   - Adjust `diameter` if you know the approximate cell size (pixels).  
   - Modify `batch_size` and `bsize` depending on GPU VRAM and performance.  
   - Optionally enable/disable fluorescence normalization by editing the code.

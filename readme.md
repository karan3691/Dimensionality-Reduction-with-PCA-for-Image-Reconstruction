# Dimensionality Reduction with PCA for Image Reconstruction

## Project Overview

This project demonstrates the application of Principal Component Analysis (PCA) for dimensionality reduction and image reconstruction. PCA is a powerful statistical technique used to reduce the number of features in a dataset while preserving as much variance (information) as possible. In this project, PCA is applied to a set of images to reduce their dimensionality and reconstruct them to visualize how well the technique retains information.

## Components

1. **Preprocessing (`preprocess.py`):**
   - Loads images from a specified folder.
   - Resizes and flattens images to prepare them for PCA.

2. **PCA Application (`pca.py`):**
   - Applies PCA to the image data to reduce dimensionality.
   - Transforms the original images into a lower-dimensional space and allows for their reconstruction.

3. **Visualization (`visualize.py`):**
   - Reconstructs images from the reduced dimensions.
   - Visualizes and compares the original and reconstructed images to evaluate PCA performance.

4. **Main Execution (`main.py`):**
   - Integrates preprocessing, PCA application, and visualization.
   - Prints the explained variance ratio to indicate how much information is preserved.

## Requirements

- Python 3.x
- `numpy`
- `scikit-learn`
- `pillow`
- `matplotlib`

Install the required libraries using:

```bash
pip install numpy scikit-learn pillow matplotlib
```

# Usage Instructions

## Prepare Images

1. Place your images in a folder.
2. Ensure the images are in a format supported by Pillow (e.g., PNG, JPEG).

## Update `main.py`

Modify the `image_folder` variable to point to the folder containing your images.

```python
image_folder = 'path_to_your_image_folder'
```

## Run The Project

1. Execute the script with:
```python
python3 main.py
```

## View Results

1. The console will display:
- Number of samples
- Number of features
- Explained variance ratio


## Key Notes

### Dimensionality and Components
- The **n_components** parameter in PCA determines the number of principal components to use. Adjust this parameter to balance between dimensionality reduction and reconstruction quality.
- Too few components may lead to blurred reconstructions.
- Too many components may not effectively reduce dimensionality.

### Image Quality
- Ensure that you have a sufficient number of images for meaningful PCA analysis. The quality of reconstructed images depends on the number of principal components used.

### Troubleshooting
- If the reconstructed images are blurred or not clear, consider increasing the number of components used in PCA.

### Contribution
- If you would like to contribute to this project, please fork the repository and submit a pull request.


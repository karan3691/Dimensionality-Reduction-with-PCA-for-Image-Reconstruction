import numpy as np
from pca import apply_pca
from visualize import reconstruct_images, visualize_images
from preprocess import preprocess_images

def main():
    image_folder = 'images'  # Update with your actual image folder path
    images = preprocess_images(image_folder)

    print(f"Number of samples: {len(images)}")
    print(f"Number of features: {images.shape[1]}")

    # Increase n_components to capture more details
    n_components = min(266, len(images) - 1)  # Example: Use up to 50 components or less depending on samples

    # Apply PCA for dimensionality reduction
    transformed_images, pca_model = apply_pca(images, n_components=n_components)
    
    # Reconstruct images to check PCA performance
    reconstructed_images = reconstruct_images(pca_model, transformed_images)
    
    # Visualize the first image (original vs. reconstructed)
    index = 0
    original_image = images[index].reshape(128, 128)  # Resize to 128x128 based on preprocess.py
    reconstructed_image = reconstructed_images[index].reshape(128, 128)  # Resize to 128x128
    visualize_images(original_image, reconstructed_image)
    print(f"Explained variance ratio: {np.sum(pca_model.explained_variance_ratio_)}")

if __name__ == "__main__":
    main()
import numpy as np
from pca import apply_pca
from visualize import reconstruct_images, visualize_images
from preprocess import preprocess_images
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim

def compute_metrics(original_image, reconstructed_image):
    mse_value = mean_squared_error(original_image, reconstructed_image)
    ssim_value = ssim(original_image, reconstructed_image, data_range=reconstructed_image.max() - reconstructed_image.min())
    
    return mse_value, ssim_value

def main():
    image_folder = 'flowers' 
    images = preprocess_images(image_folder)

    print(f"Number of samples: {len(images)}")
    print(f"Number of features: {images.shape[1]}")

    n_components = min(190, len(images) - 1)

    transformed_images, pca_model = apply_pca(images, n_components=n_components)

    reconstructed_images = reconstruct_images(pca_model, transformed_images)
    
    index = 50
    original_image = images[index].reshape(128, 128) 
    reconstructed_image = reconstructed_images[index].reshape(128, 128) 
    
    visualize_images(original_image, reconstructed_image)

    mse_value, ssim_value = compute_metrics(original_image, reconstructed_image)
    print(f"MSE: {mse_value}")
    print(f"SSIM: {ssim_value}")

    print(f"Explained variance ratio: {np.sum(pca_model.explained_variance_ratio_)}")

if __name__ == "__main__":
    main()

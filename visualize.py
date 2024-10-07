import matplotlib.pyplot as plt

def reconstruct_images(pca_model, transformed_images):
    # Inverse transform to get back to the original space
    reconstructed_images = pca_model.inverse_transform(transformed_images)
    
    return reconstructed_images

def visualize_images(original_image, reconstructed_image):
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_image, cmap='gray')
    
    plt.subplot(1, 2, 2)
    plt.title("Reconstructed Image")
    plt.imshow(reconstructed_image, cmap='gray')
    
    plt.show()
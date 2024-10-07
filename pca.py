import numpy as np
from sklearn.decomposition import PCA

def apply_pca(images, n_components):
    # Initialize PCA
    pca = PCA(n_components=n_components)
    
    # Fit and transform the images
    transformed_images = pca.fit_transform(images)
    
    # Print explained variance ratio
    print(f"Explained variance ratio: {np.sum(pca.explained_variance_ratio_)}")
    
    return transformed_images, pca
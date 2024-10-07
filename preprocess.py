from PIL import Image
import numpy as np
import os

def preprocess_images(image_folder):
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg') or f.endswith('.png')]
    images = []
    
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        with Image.open(image_path) as img:
            # Resize image to a larger size if needed (e.g., 128x128)
            img = img.resize((128, 128))
            # Convert image to grayscale
            img = img.convert('L')
            # Convert image to a numpy array and normalize pixel values
            image_array = np.array(img, dtype=np.float32) / 255.0  # Normalizing to range [0, 1]
            images.append(image_array.flatten())
    
    return np.array(images)
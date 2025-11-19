import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle
from PIL import UnidentifiedImageError

# Load the ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

# Add Global Max Pooling for feature extraction
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Function to extract features safely
def extract_features(img_path, model):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        result = model.predict(preprocessed_img).flatten()
        normalized_result = result / norm(result)
        return normalized_result
    except UnidentifiedImageError:
        print(f"Skipping unreadable or invalid image file: {img_path}")
        return None
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

# Collect only valid image files
image_folder = 'images'
filenames = [
    os.path.join(image_folder, file)
    for file in os.listdir(image_folder)
    if file.lower().endswith(('.jpg', '.jpeg', '.png'))
]

print(f"Found {len(filenames)} valid image files.")

# Extract features with progress bar
feature_list = []
for file in tqdm(filenames, desc="Extracting features"):
    features = extract_features(file, model)
    if features is not None:
        feature_list.append(features)

# Save features and filenames
pickle.dump(feature_list, open('embeddings.pkl', 'wb'))
pickle.dump(filenames, open('filenames.pkl', 'wb'))

print("\nFeature extraction completed successfully!")
print(f" Saved {len(feature_list)} feature vectors to embeddings.pkl")


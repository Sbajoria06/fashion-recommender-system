import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

# -----------------------------
# Load precomputed embeddings
# -----------------------------
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# -----------------------------
# Load Model
# -----------------------------
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# -----------------------------
# Streamlit App Title
# -----------------------------
st.title("👗 Fashion Recommender System")

# -----------------------------
# Ensure Upload Directory Exists
# -----------------------------
os.makedirs("uploads", exist_ok=True)

# -----------------------------
# Save Uploaded File
# -----------------------------
def save_uploaded_file(uploaded_file):
    try:
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        save_path = os.path.join(upload_dir, uploaded_file.name)

        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        return save_path  # ✅ Return the full path
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

# -----------------------------
# Feature Extraction
# -----------------------------
def feature_extraction(img_path, model):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        result = model.predict(preprocessed_img).flatten()
        normalized_result = result / norm(result)
        return normalized_result
    except Exception as e:
        st.error(f"Error during feature extraction: {e}")
        return None

# -----------------------------
# Recommendation Function
# -----------------------------
def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

# -----------------------------
# Streamlit File Upload
# -----------------------------
uploaded_file = st.file_uploader("Upload a fashion image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    save_path = save_uploaded_file(uploaded_file)

    if save_path is not None:
        # Display the uploaded image
        display_image = Image.open(uploaded_file)
        st.image(display_image, caption="Uploaded Image", use_column_width=True)

        # Extract features
        with st.spinner("Extracting features and finding similar styles..."):
            features = feature_extraction(save_path, model)

        if features is not None:
            # Get recommendations
            indices = recommend(features, feature_list)

            st.subheader("👜 Similar Fashion Items")
            cols = st.columns(5)
            for i, col in enumerate(cols):
                col.image(filenames[indices[0][i]])
    else:
        st.error("❌ Some error occurred while saving the uploaded file.")
else:
    st.info("📁 Please upload an image to get fashion recommendations.")


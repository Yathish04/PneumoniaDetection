import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('pneumonia_model.h5')

# Define image preprocessing function
def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
    img_resized = cv2.resize(img, (128, 128))         # Resize to 128x128
    img_normalized = img_resized / 255.0              # Normalize pixel values
    return np.expand_dims(img_normalized, axis=(0, -1))  # Add batch and channel dimensions

# Define image preprocessing function
#def preprocess_image(img_path):
 #   img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load image as grayscale
  #  img_resized = cv2.resize(img, (128, 128))         # Resize to 128x128
   # img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)  # Convert to 3 channels
   # img_normalized = img_rgb / 255.0                 # Normalize pixel values
    #return np.expand_dims(img_normalized, axis=0)    # Add batch dimension

# Streamlit app
st.title("Pneumonia Detection from Chest X-rays")
st.write("Upload a chest X-ray image to detect pneumonia.")

# File uploader
uploaded_file = st.file_uploader("Choose an X-ray image", type=["jpg", "jpeg", "png"])


if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.read())

    # Display the uploaded image
    st.image("temp_image.jpg", caption="Uploaded X-ray", use_column_width=True)

    # Preprocess the image
    preprocessed_img = preprocess_image("temp_image.jpg")

    # Make predictions
    predictions = model.predict(preprocessed_img)
    class_idx = np.argmax(predictions)
    confidence = predictions[0][class_idx] * 100

    # Display the result
    classes = ["Normal", "Pneumonia"]
    st.write(f"Prediction: **{classes[class_idx]}**")
    st.write(f"Confidence: **{confidence:.2f}%**")

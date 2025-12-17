import os
import urllib.request
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

# =========================================================
# STREAMLIT CONFIG
# =========================================================
st.set_page_config(
    page_title="Brain Tumor Detection",
    layout="centered"
)

st.title("🧠 Brain Tumor Detection App")
st.write("Upload an MRI image to detect the presence of a brain tumor.")

# =========================================================
# MODEL CONFIG
# =========================================================
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "brain_tumor_model.keras")

# ⚠️ IMPORTANT:
# This must be a *direct download* Google Drive link
# Format:
# https://drive.google.com/uc?id=FILE_ID
MODEL_URL = "https://drive.google.com/uc?id=YOUR_MODEL_FILE_ID"

CLASS_NAMES = ["No Tumor", "Tumor"]
IMAGE_SIZE = 224

# =========================================================
# LOAD MODEL (CACHED)
# =========================================================
@st.cache_resource
def load_trained_model():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        st.info("⬇️ Downloading trained model (first time only)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    return load_model(MODEL_PATH)

model = load_trained_model()

# =========================================================
# IMAGE PREPROCESSING
# =========================================================
def preprocess_image(img: Image.Image):
    img = img.convert("RGB")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# =========================================================
# FILE UPLOAD
# =========================================================
uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)

    if st.button("🔍 Predict"):
        with st.spinner("Analyzing image..."):
            processed_img = preprocess_image(image)
            prediction = model.predict(processed_img)[0]

            pred_index = np.argmax(prediction)
            confidence = prediction[pred_index] * 100

        st.success(f"🧪 Prediction: **{CLASS_NAMES[pred_index]}**")
        st.info(f"📊 Confidence: **{confidence:.2f}%**")

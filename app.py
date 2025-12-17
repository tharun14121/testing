import os
import requests
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="Brain Tumor Detection", layout="centered")
st.title("🧠 Brain Tumor Detection")
st.write("Upload an MRI image to detect tumor presence.")

IMAGE_SIZE = 224
CLASS_NAMES = ["No Tumor", "Tumor"]

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "brain_tumor_model.keras")

# 👇 YOUR FILE ID (FROM GOOGLE DRIVE)
FILE_ID = "1-gv3CuubtN9QxOO81o__j0z3y_yZimTZ"

# =========================================================
# GOOGLE DRIVE SAFE DOWNLOAD
# =========================================================
def download_from_gdrive(file_id, destination):
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={"id": file_id}, stream=True)
    token = None

    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value

    if token:
        params = {"id": file_id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

# =========================================================
# LOAD MODEL (CACHED)
# =========================================================
@st.cache_resource
def load_trained_model():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        st.info("⬇️ Downloading trained model (one-time)...")
        download_from_gdrive(FILE_ID, MODEL_PATH)

    return load_model(MODEL_PATH)

model = load_trained_model()

# =========================================================
# IMAGE PREPROCESSING
# =========================================================
def preprocess_image(img):
    img = img.convert("RGB")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# =========================================================
# UI
# =========================================================
uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI", use_column_width=True)

    if st.button("Predict"):
        with st.spinner("Analyzing..."):
            img = preprocess_image(image)
            pred = model.predict(img)[0]

        idx = np.argmax(pred)
        confidence = pred[idx] * 100

        st.success(f"Prediction: **{CLASS_NAMES[idx]}**")
        st.info(f"Confidence: **{confidence:.2f}%**")

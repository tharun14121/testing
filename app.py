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
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "brain_tumor_model.keras")

FILE_ID = "10gibKS6bEC5xIG6CquvuP_9vhk2dsp4p"
MIN_EXPECTED_SIZE_MB = 50  # sanity check

# =========================================================
# GOOGLE DRIVE DOWNLOAD (ROBUST)
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
        response = session.get(
            URL,
            params={"id": file_id, "confirm": token},
            stream=True
        )

    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

# =========================================================
# LOAD MODEL (CACHED + VERIFIED)
# =========================================================
@st.cache_resource
def load_trained_model():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.exists(MODEL_PATH):
        size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        if size_mb < MIN_EXPECTED_SIZE_MB:
            os.remove(MODEL_PATH)  # delete corrupted file

    if not os.path.exists(MODEL_PATH):
        st.info("⬇️ Downloading trained model (one-time)...")
        download_from_gdrive(FILE_ID, MODEL_PATH)

    size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    if size_mb < MIN_EXPECTED_SIZE_MB:
        st.error("❌ Model download failed (file too small).")
        st.stop()

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
            preds = model.predict(img)[0]

        idx = int(np.argmax(preds))
        confidence = preds[idx] * 100

        if CLASS_NAMES[idx] == "notumor":
            st.success(f"✅ No Tumor ({confidence:.2f}%)")
        else:
            st.error(f"⚠️ Tumor Detected: {CLASS_NAMES[idx]} ({confidence:.2f}%)")

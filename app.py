import os
import urllib.request
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

# ======================================================
# CONFIG
# ======================================================
st.set_page_config(page_title="Brain Tumor Detection", layout="centered")
st.title("🧠 Brain Tumor Detection")
st.write("Upload an MRI image to detect tumor presence.")

IMAGE_SIZE = 224
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "brain_tumor_model_streamlit.keras")

# Your NEW working model file ID
FILE_ID = "10gibKS6bEC5xIG6CquvuP_9vhk2dsp4p"

MODEL_URL = f"https://drive.google.com/uc?id={FILE_ID}"

# ======================================================
# LOAD MODEL (SIMPLE)
# ======================================================
@st.cache_resource
def load_trained_model():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        st.info("⬇️ Downloading model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    return load_model(MODEL_PATH)

model = load_trained_model()

# ======================================================
# IMAGE PREPROCESSING
# ======================================================
def preprocess_image(img):
    img = img.convert("RGB")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ======================================================
# UI
# ======================================================
uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI", use_column_width=True)

    if st.button("Predict"):
        with st.spinner("Analyzing image..."):
            img = preprocess_image(image)
            preds = model.predict(img)[0]

        idx = int(np.argmax(preds))
        confidence = preds[idx] * 100

        if CLASS_NAMES[idx] == "notumor":
            st.success(f"✅ No Tumor Detected ({confidence:.2f}%)")
        else:
            st.error(f"⚠️ Tumor Detected: {CLASS_NAMES[idx]} ({confidence:.2f}%)")

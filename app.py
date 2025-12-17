import os
import urllib.request
import numpy as np
import streamlit as st
from PIL import Image

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input

# =========================================================
# CONFIGURATION
# =========================================================
IMAGE_SIZE = 224

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "brain_tumor_model.keras")

# Google Drive direct download URL
MODEL_URL = (
    "https://drive.google.com/uc?export=download&id="
    "1-gv3CuubtN9QxOO81o__j0z3y_yZimTZ"
)

# IMPORTANT: must match training order exactly
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

# =========================================================
# STREAMLIT PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Brain Tumor Detection from MRI")
st.write("Upload an MRI brain image to detect the presence and type of tumor.")

# =========================================================
# DOWNLOAD MODEL (ONLY ONCE)
# =========================================================
@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_DIR, exist_ok=True)
        st.info("⬇️ Downloading trained model from Google Drive...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    return load_model(MODEL_PATH)

model = load_trained_model()

# =========================================================
# IMAGE PREPROCESSING
# =========================================================
def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image = img_to_array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image

# =========================================================
# FILE UPLOADER
# =========================================================
uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)

    img = preprocess_image(image)
    preds = model.predict(img)[0]

    pred_idx = int(np.argmax(preds))
    confidence = float(preds[pred_idx] * 100)
    label = CLASS_NAMES[pred_idx]

    st.subheader("Prediction Result")

    if label == "notumor":
        st.success(f"✅ No Tumor Detected ({confidence:.2f}%)")
    else:
        st.error(f"⚠️ Tumor Detected: **{label.upper()}** ({confidence:.2f}%)")

    st.progress(int(confidence))

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.caption("VGG16 Transfer Learning | MRI Brain Tumor Detection")

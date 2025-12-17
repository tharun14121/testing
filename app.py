import os
import gdown
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from utils import preprocess_image
from PIL import Image

MODEL_PATH = "models/brain_tumor_model.keras"
DRIVE_FILE_ID = "PASTE_YOUR_GOOGLE_DRIVE_FILE_ID"

CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

if not os.path.exists(MODEL_PATH):
    os.makedirs("models", exist_ok=True)
    gdown.download(
        f"https://drive.google.com/uc?id={DRIVE_FILE_ID}",
        MODEL_PATH,
        quiet=False
    )

model = load_model(MODEL_PATH)

st.title("🧠 Brain Tumor Detection")
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI", use_column_width=True)

    img = preprocess_image(uploaded_file)
    pred = model.predict(img)[0]

    idx = np.argmax(pred)
    confidence = pred[idx] * 100

    if CLASS_NAMES[idx] == "notumor":
        st.success(f"No Tumor Detected ({confidence:.2f}%)")
    else:
        st.error(f"Tumor Detected: {CLASS_NAMES[idx]} ({confidence:.2f}%)")

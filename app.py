import os
import urllib.request
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input

IMAGE_SIZE = 224
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "brain_tumor_model.keras")

FILE_ID = "1-gv3CuubtN9QxOO81o__j0z3y_yZimTZ"

def download_from_gdrive(file_id, destination):
    URL = "https://drive.google.com/uc?export=download"
    session = urllib.request.build_opener()
    response = session.open(f"{URL}&id={file_id}")
    content = response.read()

    # Write binary content
    with open(destination, "wb") as f:
        f.write(content)

@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_DIR, exist_ok=True)
        st.info("⬇️ Downloading trained model from Google Drive...")
        download_from_gdrive(FILE_ID, MODEL_PATH)

    return load_model(MODEL_PATH)

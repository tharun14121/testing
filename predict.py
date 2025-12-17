import os
import gdown
import numpy as np
from tensorflow.keras.models import load_model
from utils import preprocess_image

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

img_path = "sample.jpg"   # change image path
img = preprocess_image(img_path)
pred = model.predict(img)[0]

idx = np.argmax(pred)
print(f"Prediction: {CLASS_NAMES[idx]} ({pred[idx]*100:.2f}%)")

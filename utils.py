import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

IMAGE_SIZE = 224

def preprocess_image(path):
    img = load_img(path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img = img.convert("RGB")
    img = img_to_array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

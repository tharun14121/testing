import os
import random
import numpy as np
from PIL import Image, ImageEnhance
from sklearn.utils import shuffle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# =====================
# Dataset paths (Kaggle)
# =====================
DATASET_PATH = "/kaggle/input/brain-tumor-mri-dataset"
TRAIN_DIR = os.path.join(DATASET_PATH, "Training")
TEST_DIR = os.path.join(DATASET_PATH, "Testing")

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 25

CLASS_NAMES = sorted(os.listdir(TRAIN_DIR))
NUM_CLASSES = len(CLASS_NAMES)
class_to_index = {c: i for i, c in enumerate(CLASS_NAMES)}

# =====================
# Data loading
# =====================
def collect_paths(directory):
    paths, labels = [], []
    for cls in CLASS_NAMES:
        cls_path = os.path.join(directory, cls)
        for img in os.listdir(cls_path):
            paths.append(os.path.join(cls_path, img))
            labels.append(class_to_index[cls])
    return paths, labels

train_paths, train_labels = collect_paths(TRAIN_DIR)
test_paths, test_labels = collect_paths(TEST_DIR)
train_paths, train_labels = shuffle(train_paths, train_labels)

def augment(img):
    img = Image.fromarray(img.astype("uint8"))
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.9, 1.1))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.9, 1.1))
    return np.array(img)

def load_image(path, augment_image=False):
    img = load_img(path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img = img.convert("RGB")
    img = img_to_array(img)
    if augment_image:
        img = augment(img)
    img = preprocess_input(img)
    return img

def generator(paths, labels):
    while True:
        paths, labels = shuffle(paths, labels)
        for i in range(0, len(paths), BATCH_SIZE):
            batch_paths = paths[i:i+BATCH_SIZE]
            batch_labels = labels[i:i+BATCH_SIZE]
            images = [load_image(p, True) for p in batch_paths]
            yield np.array(images), np.array(batch_labels)

X_test = np.array([load_image(p) for p in test_paths])
y_test = np.array(test_labels)

# =====================
# Model
# =====================
base_model = VGG16(weights="imagenet", include_top=False,
                   input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

for layer in base_model.layers[:-4]:
    layer.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation="relu"),
    BatchNormalization(),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation="softmax")
])

model.compile(
    optimizer=Adam(1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ReduceLROnPlateau(patience=3, factor=0.2)
]

model.fit(
    generator(train_paths, train_labels),
    steps_per_epoch=len(train_paths)//BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_test, y_test),
    callbacks=callbacks
)

model.save("brain_tumor_model.keras")
print("✅ Model saved as brain_tumor_model.keras")

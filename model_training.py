import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

DATASET_ROOT = "fer2013"         # Rename if your root folder is named differently
IMG_SIZE = (48, 48)              # FER standard size

# List emotion folders inside train/
emotion_labels = sorted(os.listdir(os.path.join(DATASET_ROOT, "train")))
num_classes = len(emotion_labels)
print("Emotion classes:", emotion_labels)

def load_data(subset):
    images = []
    labels = []
    for idx, emotion in enumerate(emotion_labels):
        emotion_folder = os.path.join(DATASET_ROOT, subset, emotion)
        for img_file in os.listdir(emotion_folder):
            img_path = os.path.join(emotion_folder, img_file)
            img = load_img(img_path, color_mode="grayscale", target_size=IMG_SIZE)
            img_arr = img_to_array(img) / 255.0
            images.append(img_arr)
            labels.append(idx)
    images = np.array(images)
    labels = to_categorical(labels, num_classes=num_classes)
    return images, labels

print("Loading training data...")
x_train, y_train = load_data("train")
print(f"Loaded {x_train.shape[0]} training samples.")

print("Loading test data...")
x_test, y_test = load_data("test")
print(f"Loaded {x_test.shape[0]} test samples.")

# Build model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Training model...")
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5, batch_size=64)

print("Saving model...")
model.save("model.h5")
print("Model training and saving complete! Download 'model.h5' from your Render workspace.")

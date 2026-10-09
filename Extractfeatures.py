import os
import cv2
import numpy as np
import pickle
from skimage.feature import hog
import matplotlib.pyplot as plt

# CHANGE THIS TO YOUR FOLDER
DATASET_PATH = "/content/drive/MyDrive/images"

SETTINGS = {
    "A": {
        "orientations": 9,
        "pixels_per_cell": (8, 8),
        "cells_per_block": (2, 2)
    },
    "B": {
        "orientations": 9,
        "pixels_per_cell": (16, 16),
        "cells_per_block": (2, 2)
    }
}

def extract_hog_features(image, params, visualize=False):

    image = cv2.resize(image, (128, 128))

    if visualize:
        features, hog_image = hog(
            image,
            orientations=params["orientations"],
            pixels_per_cell=params["pixels_per_cell"],
            cells_per_block=params["cells_per_block"],
            visualize=True
        )
        return features, hog_image

    features = hog(
        image,
        orientations=params["orientations"],
        pixels_per_cell=params["pixels_per_cell"],
        cells_per_block=params["cells_per_block"]
    )

    return features


# -------------------------
# FEATURE EXTRACTION
# -------------------------

for setting_name, params in SETTINGS.items():

    X = []
    y = []

    # Dynamically list class names from the DATASET_PATH
    for class_name in os.listdir(DATASET_PATH):

        class_path = os.path.join(DATASET_PATH, class_name)

        if not os.path.isdir(class_path):
            print(f"Skipping non-directory: {class_path}")
            continue

        for img_name in os.listdir(class_path):

            img_path = os.path.join(class_path, img_name)

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            features = extract_hog_features(img, params)

            X.append(features)
            y.append(class_name)

    X = np.array(X)
    y = np.array(y)

    print(f"\nSetting {setting_name}")
    print("Number of Images:", len(X))
    if len(X) > 0:
        print("Feature Vector Length:", X.shape[1])
    else:
        print("No images found for feature extraction.")

    with open(f"features_{setting_name}.pkl", "wb") as f:
        pickle.dump((X, y), f)

print("\nFeatures saved successfully!")

# -------------------------
# HOG VISUALIZATION
# -------------------------

params = SETTINGS["A"]

# Dynamically list class names for visualization
for class_name in os.listdir(DATASET_PATH):

    class_path = os.path.join(DATASET_PATH, class_name)

    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)

    if len(images) == 0:
        continue

    # Pick the first image found in the folder for visualization
    img_path = os.path.join(class_path, images[0])

    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"Could not load image for visualization: {img_path}")
        continue

    image = cv2.resize(image, (128, 128))

    _, hog_img = extract_hog_features(
        image,
        params,
        visualize=True
    )

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap="gray")
    plt.title(f"{class_name} Original")

    plt.subplot(1, 2, 2)
    plt.imshow(hog_img, cmap="gray")
    plt.title(f"{class_name} HOG")

    plt.savefig(f"{class_name}_hog.png")
    plt.show()

print("HOG visualizations saved.")

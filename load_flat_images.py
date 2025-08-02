import os
from PIL import Image, ImageFile
import matplotlib.pyplot as plt
import numpy as np

# Allow loading truncated/broken images
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_SIZE = (224, 224)
DATASET_PATH = "dataset"

image_files = [os.path.join(DATASET_PATH, f) for f in os.listdir(DATASET_PATH)
               if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

print(f"Found {len(image_files)} image(s) in dataset folder.")

images = []
for path in image_files:
    try:
        img = Image.open(path).convert("RGB")
        img = img.resize(IMG_SIZE)
        img_array = np.array(img) / 255.0
        images.append(img_array)
    except Exception as e:
        print(f"Skipping {path}: {e}")

images = np.array(images)

if len(images) > 0:
    print("Displaying sample images...")
    plt.figure(figsize=(10, 10))
    for i in range(min(9, len(images))):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        plt.title("Meme")
        plt.axis("off")
    plt.tight_layout()
    plt.show()
else:
    print("No valid images could be loaded.")

import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Paths
real_path = "C:/Users/Administrator/ai-fake-image-detector/data/real"
fake_path = "C:/Users/Administrator/ai-fake-image-detector/data/fake"
processed_path = "C:/Users/Administrator/ai-fake-image-detector/data/processed"

# Create processed folder if not exists
os.makedirs(processed_path, exist_ok=True)

# Params
img_size = (128, 128)
X, y = [], []

# Load real images
for img_name in os.listdir(real_path):
    img = load_img(os.path.join(real_path, img_name), target_size=img_size)
    img = img_to_array(img) / 255.0
    X.append(img)
    y.append(0)  # real = 0

# Load fake images
for img_name in os.listdir(fake_path):
    img = load_img(os.path.join(fake_path, img_name), target_size=img_size)
    img = img_to_array(img) / 255.0
    X.append(img)
    y.append(1)  # fake = 1

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Save
np.save(os.path.join(processed_path, "X.npy"), X)
np.save(os.path.join(processed_path, "y.npy"), y)

print("✅ Saved processed dataset in", processed_path)

import os
from sklearn.datasets import fetch_olivetti_faces
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Path for saving real images
real_path = "data/real"

# ✅ Debug: show current working directory
print("📂 Current directory:", os.getcwd())

# ✅ Make sure folder exists
os.makedirs(real_path, exist_ok=True)

# Load dataset (Olivetti Faces)
dataset = fetch_olivetti_faces()
images = dataset.images  # shape = (400, 64, 64)

# Save first 20 images
for i, img in enumerate(images[:20]):
    img = (img * 255).astype(np.uint8)  # scale to [0,255]
    im = Image.fromarray(img)
    path = os.path.join(real_path, f"real_{i+1}.png")
    im.save(path)
    print("✅ Saved:", path)

print("🎉 All done!")

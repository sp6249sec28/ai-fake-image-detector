# verify_preprocessing.py

import numpy as np
import os
from collections import Counter

# Paths
processed_path = "data/processed"
report_path = "dataset_report.md"

# Load processed dataset
X = np.load(os.path.join(processed_path, "X.npy"))
y = np.load(os.path.join(processed_path, "y.npy"))

# Dataset stats
num_samples = len(X)
img_shape = X.shape[1:]  # (H, W, C)
dtype = X.dtype
val_min, val_max = X.min(), X.max()
class_counts = Counter(y)

# Create report text
report = f"""
# 📊 Dataset Report

### 🔹 General Info
- Dataset Name: Real vs Fake Faces (Processed)
- Total Images: {num_samples}
- Image Shape: {img_shape}
- Channels: {img_shape[-1]}
- Data Type: {dtype}
- Value Range: [{val_min}, {val_max}]

### 🔹 Class Distribution
- Real (0): {class_counts.get(0, 0)}
- Fake (1): {class_counts.get(1, 0)}

### 🔹 Preprocessing Notes
- All images resized to {img_shape[0]}×{img_shape[1]}.
- Normalized to [0,1] range.
- Stored in `.npy` format for fast loading.
"""

# Save to Markdown file
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report)

print(f"✅ Dataset report generated: {report_path}")

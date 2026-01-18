import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

images_dir = "dataset_annotation/original_images"
labels_dir = "dataset_annotation/labels"
out_path = "../datasets/grabbability_dataset.pt"

tf = transforms.Compose([
    transforms.Resize((480, 480)),
    transforms.ToTensor()
])

dataset = []

image_files = sorted([
    f for f in os.listdir(images_dir)
    if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
])

for fname in image_files:
    stem = os.path.splitext(fname)[0]
    lbl_path = os.path.join(labels_dir, stem + ".npy")

    if not os.path.exists(lbl_path):
        continue

    image = Image.open(os.path.join(images_dir, fname)).convert("RGB")
    image = tf(image)                     # (3, 480, 480)

    label = np.load(lbl_path)
    label = (label > 0).astype(np.float32)   
    label = torch.from_numpy(label).float().unsqueeze(0)  # (1, 480, 480)

    dataset.append((image, label))

torch.save(dataset, out_path)
print(f"[OK] Saved {len(dataset)} samples to {out_path}")
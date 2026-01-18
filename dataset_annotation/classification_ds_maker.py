import os
import torch
from PIL import Image
from torchvision import transforms

def generate_one_hot_encoding(label: str):
    label = label.lower()
    label_to_index = {
        "bronze": 0,
        "copper": 1,
        "iron": 2,
        "silver": 3,
        "disturb": 4,
        "uncertain": 5
    }
    one_hot = torch.zeros(6, dtype=torch.float32)
    one_hot[label_to_index[label]] = 1
    return one_hot

tf = transforms.Compose([
    transforms.ToTensor()
])

labels_directory = "classification_labels/train"
out_path = "../datasets/classification_dataset.pt"
classes = ["Bronze", "Copper", "Iron", "Silver", "Disturb", "Uncertain"]

dataset = []

for clas in classes:
    labels_path = os.path.join(labels_directory, clas)
    image_files = sorted([
        f for f in os.listdir(labels_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
    ])
    for fname in image_files:
        stem = os.path.splitext(fname)[0]
        
        image = Image.open(os.path.join(labels_path, fname)).convert("RGB")
        image = tf(image)  
        
        dataset.append((image, generate_one_hot_encoding(clas)))
        
torch.save(dataset, out_path)
print(f"[OK] Saved {len(dataset)} samples to {out_path}")
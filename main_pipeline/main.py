import torch
from alex_net import AlexNet
from metal_surface import MetalSurface

device = 'cuda'
ms = MetalSurface().to(device)
an = AlexNet(6).to(device)

ms.load_state_dict(torch.load("models/metal_surface_rgb_onehot_focalbce_globalcnn_maxpoolred.pth"))
ms.eval()

an.load_state_dict(torch.load("models/classification_model.pth"))
an.eval()

import cv2
import numpy as np
from scipy.ndimage import label
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tkinter import Tk, filedialog
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from scipy.ndimage import binary_opening, binary_closing

def choose_image():
    root = Tk()
    # root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[
            ("Image files", "*.png *.jpg")
        ]
    )
    return file_path

def convert_image_to_tensor(image):
    image_tensor = torch.from_numpy(image).float() / 255.0
    image_tensor = image_tensor.to(device)
    image_tensor = image_tensor.unsqueeze(0).permute(0, 3, 1, 2)
    return image_tensor

def color_to_rgb(color_name):
    rgb = mcolors.to_rgb(color_name)   # (0–1)
    return np.array(rgb) * 255          # (0–255)

def main():
    with torch.no_grad():
        print("Choosing image...")
        img_path = choose_image()
        image = None
        if not img_path:
            print("Closing application as no image was chosen...")
            return
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (480, 480), interpolation=cv2.INTER_LINEAR)
        image_tensor = convert_image_to_tensor(image)
        
        heatmap = torch.sigmoid(ms(image_tensor))
        heatmap = F.avg_pool2d(heatmap, kernel_size=5, stride=1, padding=2)
        heatmap = heatmap.reshape(480, 480)
        binary = (heatmap >= 0.35)
        binary_np = binary.detach().cpu().numpy()
        
        binary_np = binary_opening(binary_np, structure=np.ones((3,3)))
        binary_np = binary_closing(binary_np, structure=np.ones((5,5)))
        
        # plt.imshow(binary_np, cmap="hot")
        # plt.title("Generated Heatmap")
        # plt.show()
        
        labels_cc, num_components = label(binary_np)
        labels_cc_reshaped = labels_cc.reshape(480, 480, 1)
        index_to_label = {
            0: "bronze",
            1: "copper",
            2: "iron",
            3: "silver",
            4: "disturb",
            5: "uncertain"
        }
        
        label_to_color = {
            "bronze": "yellow",
            "copper": "orange",
            "iron": "grey",
            "silver": "white",
            "disturb": "purple",
            "uncertain": "red"
        }
        
        k_to_color = {}
        
        for k in range(1, num_components + 1):
            num_pixels = np.sum(labels_cc == k)
            if num_pixels < 400:
                k_to_color[k] = "black"
                continue
            
            ys, xs = np.where(labels_cc == k)
            y_min, y_max = ys.min(), ys.max()
            x_min, x_max = xs.min(), xs.max()
            kth_image = image * (labels_cc_reshaped == k)
            
            kth_crop = kth_image[y_min: y_max + 1, x_min: x_max + 1, :]
            kth_crop = cv2.resize(kth_crop, (224, 224), interpolation=cv2.INTER_LINEAR)
            kth_crop_tensor = torch.from_numpy(kth_crop).unsqueeze(0).permute(0, 3, 1, 2).float().to(device) / 255.0
            
            predicted_one_hot = an(kth_crop_tensor)[0]
            class_idx = torch.argmax(predicted_one_hot)
            class_label = index_to_label[class_idx.item()]
            kth_color = label_to_color[class_label]
            k_to_color[k] = kth_color
        
        overlay = np.zeros_like(image, dtype=np.float32)

        for k, color_name in k_to_color.items():
            if color_name == "black":
                continue

            mask = (labels_cc == k)
            overlay[mask] = color_to_rgb(color_name)
            
        alpha = 0.5  # transparency
        blended = (
            image.astype(np.float32) * (1 - alpha) +
            overlay * alpha
        ).astype(np.uint8)
        
        plt.figure(figsize=(8, 8))
        plt.imshow(blended)
        plt.axis("off")
        
        legend_items = {}
        for k, color in k_to_color.items():
            if color != "black":
                kth_label = [l for l, c in label_to_color.items() if c == color][0]
                legend_items[kth_label] = color
                
        handles = [
            Patch(facecolor=color, label=kth_label)
            for kth_label, color in legend_items.items()
        ]

        plt.legend(
            handles=handles,
            loc="upper right",
            fontsize=10
        )

        plt.title("Metal Classification Overlay")
        plt.show()

main()
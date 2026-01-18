import os
import helper 
import numpy as np
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import matplotlib.cm as cm
import matplotlib.colors as mcolors

image_size = 480  # pixels
images_directory = r'dataset_annotation/original_images'
label_directory = r'dataset_annotation/labels'

class AnnotatorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Grabbability Annotator")
        self.stddev = 5
        self.label = np.zeros((480, 480))
        self.canvas = tk.Canvas(root, width=image_size, height=image_size)
        self.canvas.pack()
        self.image_list = [f for f in os.listdir(images_directory) if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp'))]
        self.total_images = len(self.image_list)
        self.image_number_text = self.canvas.create_text(
            0, 0, 
            text=f'0 / {self.total_images - 1}', 
            fill='white', 
            anchor='nw', 
            font=('Arial', 14), 
            tags=('image_number',)
        )
        self.current_image_index = 0
        self.original_image = None
        self.resized_image = None
        self.displayed_image = None
        self.load_image_to_canvas()
        self.draw_overlay()
        
        self.undo_stack = []
        
        self.canvas.bind("<Button-1>", self.apply_mask)
        self.canvas.bind("<B1-Motion>", self.apply_mask)
        self.root.bind_all("<Return>", self.load_next_image_to_canvas)
        self.root.bind_all("<Motion>", self.draw_square)
        self.root.bind_all("<MouseWheel>", self.change_stddev)
        self.root.bind_all("<Control-z>", self.undo)
    
    def undo(self, event):
        if self.undo_stack:
            x, y, stddev = self.undo_stack.pop()
            gk = helper.generate_grabbability_kernel(stddev)
            helper.subtract_grabbability_kernel_to_image_inplace(self.label, gk, (y, x))
            self.draw_overlay()
    
    def apply_mask(self, event):
        self.undo_stack.append((event.x, event.y, self.stddev))
        gk = helper.generate_grabbability_kernel(self.stddev)
        helper.add_grabbability_kernel_to_image_inplace(self.label, gk, (event.y, event.x))
        self.draw_overlay()
    
    def draw_square(self, event):
        self.canvas.delete("preview_square")
        size = 2 * self.stddev + 1
        self.canvas.create_rectangle(
            event.x - (size // 2), event.y - (size // 2),
            event.x + (size // 2), event.y + (size // 2),
            outline="red",
            width=2,
            tags=("preview_square",)
        )
    
    def load_image_to_canvas(self):
        if self.current_image_index >= len(self.image_list):
            messagebox.showinfo("All Images Annotated", "Closing Application")
            print("All images annotated... closing application.")
            self.root.destroy()
            return
        
        image_path = os.path.join(images_directory, self.image_list[self.current_image_index])
        print("Loading Image:", image_path)
        self.original_image = Image.open(image_path).convert("RGB")
        self.resized_image = self.original_image.resize((image_size, image_size), Image.BILINEAR)
        self.displayed_image = ImageTk.PhotoImage(self.resized_image)
        self.canvas.delete('image')
        self.canvas.create_image(0, 0, anchor='nw', image=self.displayed_image, tags=('image',))
        
        self.draw_overlay()
        
        self.canvas.tag_lower('image')
        self.canvas.tag_raise('overlay')
        self.canvas.tag_raise('image_number')
        
        if self.current_image_index == 0:
            messagebox.showinfo(
                "Controls", 
                "Left Click -> Apply Mask\nScroll -> Change Mask Size\nControl + Z -> Undo Last Mask"
            )
        
        self.root.focus_force()
        self.root.focus_set()

    def load_next_image_to_canvas(self, event):
        np.clip(self.label, 0, 0, out=self.label)
        self.save_label()
        np.clip(self.label, 0, 0, out=self.label)
        self.undo_stack = []
        self.current_image_index += 1
        self.canvas.itemconfig(self.image_number_text, text=f'{self.current_image_index} / {self.total_images - 1}')
        self.load_image_to_canvas()
        self.draw_overlay()

    def change_stddev(self, event):
        if event.delta > 0:
            self.stddev = min(50, self.stddev + 1)
        else:
            self.stddev = max(1, self.stddev - 1)
        self.draw_square(event)

    def create_overlay_image(self):
        image_overlay_label = np.clip(self.label, 0, 1)
        cmap = cm.get_cmap('hot')
        colored = cmap(image_overlay_label)
        colored[..., 3] = 0.3
        overlay_uint8 = (colored * 255).astype(np.uint8)
        overlay_img = Image.fromarray(overlay_uint8, mode="RGBA")
        return ImageTk.PhotoImage(overlay_img)
    
    def draw_overlay(self):
        self.canvas.delete('overlay')
        overlay = self.create_overlay_image()
        if overlay is not None:
            self.overlay_image = overlay
            self.canvas.create_image(
                0, 0, 
                anchor='nw', 
                image=self.overlay_image, 
                tags=('overlay',)
            )
            
    def save_label(self):
        name = os.path.splitext(self.image_list[self.current_image_index])[0]
        path = os.path.join(label_directory, name + ".npy")
        np.save(path, self.label)


root = tk.Tk()
viewer = AnnotatorApp(root)
root.mainloop()
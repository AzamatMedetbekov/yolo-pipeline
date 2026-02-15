"""Simple manual crop-and-classify tool for 6 refrigerator classes.

Usage:
1) Install dependency (once):
   pip install pillow
2) Set SRC_DIR and OUT_DIR below.
3) Run: python scripts/crop_and_classify.py
4) Draw a box (left-click drag) around a device, then click a class button to save the crop.
   - You can make multiple crops per page before moving to next image.
   - Next/Prev buttons (or Right/Left keys) change the source image without saving a crop.

Classes:
1) vertical_closed   (upright with doors)
2) horizontal_closed (chest/lidded)
3) vertical_open     (tall open multi-deck)
4) horizontal_open   (island/bunker open-top)
5) combination       (hybrids e.g., upright + bunker)
6) coldroom          (walk-in/panelized room)
"""

import argparse
import glob
import os
from pathlib import Path
from PIL import Image, ImageTk
import tkinter as tk

# --- CONFIG ---
DEFAULT_SRC = Path("data/pdf_images")
DEFAULT_OUT = Path("data/dataset")
CLASSES = [
    "vertical_closed",
    "vertical_open",
    "horizontal_closed",
    "horizontal_open",
    "combination",
    "coldroom",
]
CANVAS_W, CANVAS_H = 1200, 900
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Manual crop-and-classify GUI")
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC, help="Folder with page images")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output root folder for class subfolders")
    return parser.parse_args()


args = parse_args()
SRC_DIR = args.src
OUT_DIR = args.out

# ensure output folders exist
OUT_DIR.mkdir(parents=True, exist_ok=True)
for c in CLASSES:
    (OUT_DIR / c).mkdir(parents=True, exist_ok=True)

# gather images
images = [
    p for p in sorted(SRC_DIR.glob("*"))
    if p.suffix.lower() in ALLOWED_EXTS
]

idx = 0
im = None
scale = 1.0
start = None
rect = None
tk_im = None

root = tk.Tk()
root.title("Crop & Classify")

# Layout frames so buttons stay visible even on small screens
top_bar = tk.Frame(root)
top_bar.grid(row=0, column=0, columnspan=8, sticky="ew", padx=5, pady=5)

canvas = tk.Canvas(root, width=CANVAS_W, height=CANVAS_H, bg="gray")
canvas.grid(row=1, column=0, columnspan=8, padx=5, pady=5)

status = tk.Label(root, text="", anchor="w")
status.grid(row=2, column=0, columnspan=8, sticky="ew", padx=5)


def load_image():
    global im, tk_im, scale
    canvas.delete("all")
    if not images:
        canvas.create_text(CANVAS_W // 2, CANVAS_H // 2, text="No images found", fill="white", font=("Arial", 24))
        status.config(text="Set SRC_DIR to your images")
        return
    if idx >= len(images):
        canvas.create_text(CANVAS_W // 2, CANVAS_H // 2, text="All done", fill="white", font=("Arial", 24))
        status.config(text="Reached end")
        return
    path = images[idx]
    im = Image.open(path).convert("RGB")
    scale = min(CANVAS_W / im.width, CANVAS_H / im.height, 1.0)
    disp = im.resize((int(im.width * scale), int(im.height * scale)), Image.LANCZOS)
    tk_im = ImageTk.PhotoImage(disp)
    canvas.create_image(0, 0, anchor="nw", image=tk_im)
    canvas.config(scrollregion=(0, 0, disp.width, disp.height))
    root.title(f"{idx + 1}/{len(images)}  {os.path.basename(path)}")
    status.config(text="Draw box, then click a class button or press 1-6 to save. Left/Right arrows to navigate.")


def on_down(event):
    global start, rect
    start = (event.x, event.y)
    if rect:
        canvas.delete(rect)
        rect = None


def on_drag(event):
    global rect
    if not start:
        return
    if rect:
        canvas.delete(rect)
    rect = canvas.create_rectangle(start[0], start[1], event.x, event.y, outline="red", width=2)


def save_crop(class_name):
    global rect, start
    if not images or im is None or rect is None or start is None:
        status.config(text="No crop selected")
        return
    x0, y0, x1, y1 = canvas.coords(rect)
    x0i, y0i = int(x0 / scale), int(y0 / scale)
    x1i, y1i = int(x1 / scale), int(y1 / scale)
    box = (min(x0i, x1i), min(y0i, y1i), max(x0i, x1i), max(y0i, y1i))
    if box[2] - box[0] < 5 or box[3] - box[1] < 5:
        status.config(text="Box too small; not saved")
        return
    crop = im.crop(box)
    out_dir = os.path.join(OUT_DIR, class_name)
    base = os.path.splitext(os.path.basename(images[idx]))[0]
    seq = len([f for f in os.listdir(out_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) + 1
    out_path = os.path.join(out_dir, f"{base}_{seq:04d}.jpg")
    crop.save(out_path, quality=95)
    status.config(text=f"Saved: {out_path}")
    # keep same image to allow multiple crops
    start = None
    if rect:
        canvas.delete(rect)
        rect = None


def go_next(event=None):
    global idx, start, rect
    if not images:
        return
    idx = min(idx + 1, len(images))
    start = None
    if rect:
        canvas.delete(rect)
        rect = None
    load_image()


def go_prev(event=None):
    global idx, start, rect
    if not images:
        return
    idx = max(idx - 1, 0)
    start = None
    if rect:
        canvas.delete(rect)
        rect = None
    load_image()


def on_digit(event):
    key = event.keysym
    # Handle top-row numbers and keypad numbers
    digit_map = {
        "1": CLASSES[0], "KP_1": CLASSES[0],
        "2": CLASSES[1], "KP_2": CLASSES[1],
        "3": CLASSES[2], "KP_3": CLASSES[2],
        "4": CLASSES[3], "KP_4": CLASSES[3],
        "5": CLASSES[4], "KP_5": CLASSES[4],
        "6": CLASSES[5], "KP_6": CLASSES[5],
    }
    cls = digit_map.get(key)
    if cls:
        save_crop(cls)


for i, cls in enumerate(CLASSES):
    btn = tk.Button(top_bar, text=cls, width=18, command=lambda c=cls: save_crop(c))
    btn.grid(row=0, column=i, sticky="ew", padx=2, pady=2)

nav_prev = tk.Button(top_bar, text="Prev", command=go_prev)
nav_prev.grid(row=0, column=6, padx=2, pady=2, sticky="ew")
nav_next = tk.Button(top_bar, text="Next", command=go_next)
nav_next.grid(row=0, column=7, padx=2, pady=2, sticky="ew")

canvas.bind("<ButtonPress-1>", on_down)
canvas.bind("<B1-Motion>", on_drag)
root.bind("<Left>", go_prev)
root.bind("<Right>", go_next)
root.bind("<Key>", on_digit)

load_image()
root.mainloop()

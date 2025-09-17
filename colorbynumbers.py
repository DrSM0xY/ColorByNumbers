#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Color-by-numbers generator:
- Uses Crayola 48 colors (unique per cluster where possible).
- Edge simplification
"""

import io
import math
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
from PIL import Image, ImageTk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patheffects as PathEffects
import matplotlib.patches as patches
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from skimage.measure import label as sk_label, regionprops, find_contours
from skimage.morphology import disk, closing, opening
from scipy.ndimage import distance_transform_edt
from sklearn.cluster import KMeans

# ---------- Constants ----------
A4_MM = (210.0, 297.0)
MARGIN_MM = 12.0
LEGEND_HEIGHT_MM = 32.0
FONT_PT_MIN = 5
FONT_PT_MAX = 7
LEGEND_SWATCH_SIZE_MM = 8.0
LEGEND_GAP_MM = 4.0
LEGEND_TEXT_PT = 4
PREVIEW_DPI = 150
MAX_PIXELS = 800 * 800

# Register a font (Helvetica is built in, but we can still set explicitly)
pdfmetrics.registerFont(TTFont("Helvetica", "Helvetica.ttf"))

# Crayola 48 colors (RGB normalized to [0,1])
CRAYOLA_48 = {
    "Blue": (0.11, 0.47, 0.85), "Black": (0.0, 0.0, 0.0), "Brown": (0.69, 0.35, 0.20),
    "Green": (0.18, 0.65, 0.41), "Orange": (1.0, 0.65, 0.0), "Red": (0.93, 0.17, 0.21),
    "Violet": (0.58, 0.33, 0.65), "Yellow": (1.0, 0.94, 0.26), "Carnation Pink": (1.0, 0.67, 0.81),
    "Blue Green": (0.05, 0.55, 0.67), "Blue Violet": (0.45, 0.40, 0.74), "Red Orange": (1.0, 0.43, 0.29),
    "Red Violet": (0.72, 0.25, 0.48), "White": (1.0, 1.0, 1.0), "Yellow Green": (0.77, 0.90, 0.52),
    "Yellow Orange": (1.0, 0.77, 0.33), "Apricot": (0.99, 0.85, 0.68), "Bluetiful": (0.20, 0.40, 0.87),
    "Cerulean": (0.11, 0.67, 0.84), "Gray": (0.58, 0.58, 0.58), "Green Yellow": (0.94, 0.91, 0.55),
    "Indigo": (0.36, 0.46, 0.67), "Scarlet": (0.99, 0.15, 0.28), "Violet Red": (0.97, 0.33, 0.58),
    "Cadet Blue": (0.69, 0.76, 0.80), "Chestnut": (0.74, 0.36, 0.35), "Melon": (0.99, 0.74, 0.71),
    "Peach": (1.0, 0.80, 0.65), "Sky Blue": (0.46, 0.85, 0.92), "Tan": (0.85, 0.64, 0.42),
    "Timberwolf": (0.86, 0.84, 0.80), "Wisteria": (0.80, 0.65, 0.80), "Burnt Sienna": (0.91, 0.45, 0.32),
    "Cornflower": (0.60, 0.81, 0.93), "Goldenrod": (0.98, 0.85, 0.37), "Granny Smith Apple": (0.66, 0.89, 0.63),
    "Lavender": (0.96, 0.76, 0.90), "Macaroni and Cheese": (1.0, 0.74, 0.47), "Mahogany": (0.80, 0.29, 0.25),
    "Mauvelous": (0.94, 0.60, 0.67), "Olive Green": (0.73, 0.72, 0.42), "Purple Mountains Majesty": (0.62, 0.51, 0.74),
    "Raw Sienna": (0.84, 0.54, 0.35), "Salmon": (1.0, 0.65, 0.65), "Sea Green": (0.62, 0.87, 0.75),
    "Sepia": (0.65, 0.41, 0.31), "Spring Green": (0.93, 0.96, 0.73), "Tumbleweed": (0.87, 0.67, 0.53)
}
CRAYOLA_COLORS = np.array(list(CRAYOLA_48.values()), dtype=np.float32)
CRAYOLA_NAMES = list(CRAYOLA_48.keys())

# ---------- Helpers ----------
def mm_to_inches(mm: float) -> float:
    return mm / 25.4

# (functions quantize_image, mm_per_pixel, smooth_mask, extract_regions, render_preview remain unchanged)
# ... [UNCHANGED CODE OMITTED FOR BREVITY, same as before] ...

# ---------- Rendering ----------
# (render_preview unchanged)

def generate_pdf(idx, regions, pdf_path, print_size_mm, palette, color_indices, orig_size):
    img_h, img_w = orig_size  # 🔑 use original image size for aspect ratio
    pw, ph = A4_MM
    cw = pw - 2 * MARGIN_MM
    ch = ph - 2 * MARGIN_MM - LEGEND_HEIGHT_MM
    img_aspect = img_w / img_h
    cont_aspect = cw / ch
    if img_aspect >= cont_aspect:
        ow, oh = cw, cw / img_aspect
    else:
        oh, ow = ch, ch * img_aspect

    # Create PDF canvas
    c = canvas.Canvas(pdf_path, pagesize=(pw * mm, ph * mm))

    # Scale drawing to fit page
    x0 = MARGIN_MM * mm
    y0 = (MARGIN_MM + LEGEND_HEIGHT_MM) * mm
    scale_x = (ow * mm) / idx.shape[1]  # scale according to processed image
    scale_y = (oh * mm) / idx.shape[0]

    # Draw contours + numbers
    for reg in regions:
        mask = reg["mask"]
        for contour in find_contours(mask, 0.5):
            path = c.beginPath()
            path.moveTo(x0 + contour[0, 1] * scale_x, y0 + (idx.shape[0] - contour[0, 0]) * scale_y)
            for pt in contour[1:]:
                path.lineTo(x0 + pt[1] * scale_x, y0 + (idx.shape[0] - pt[0]) * scale_y)
            path.close()
            c.setLineWidth(0.8)
            c.setStrokeColorRGB(0, 0, 0)
            c.drawPath(path, stroke=1, fill=0)

        r, cx = reg["best_point"]
        tx = x0 + cx * scale_x
        ty = y0 + (idx.shape[0] - r) * scale_y
        c.setFont("Helvetica", 6)
        c.drawCentredString(tx, ty, str(reg["color"] + 1))

    # Legend (only used colors)
    used_colors = sorted(set(reg["color"] for reg in regions))
    sw = LEGEND_SWATCH_SIZE_MM * mm
    gap = LEGEND_GAP_MM * mm
    cellw = sw + gap * 3
    perrow = max(1, int((cw * mm) // cellw))
    for legend_idx, color_id in enumerate(used_colors):
        row, col = divmod(legend_idx, perrow)
        x = MARGIN_MM * mm + col * cellw + gap
        y = MARGIN_MM * mm + LEGEND_HEIGHT_MM * mm - (row + 1) * (sw + gap)
        face = palette[color_id] if color_id < len(palette) else (1.0, 1.0, 1.0)
        c.setFillColorRGB(*face)
        c.rect(x, y, sw, sw, fill=1, stroke=1)
        cray_idx = color_indices[color_id] if color_id < len(color_indices) else None
        color_name = CRAYOLA_NAMES[cray_idx] if cray_idx is not None and cray_idx < len(CRAYOLA_NAMES) else "Color"
        c.setFillColorRGB(0, 0, 0)
        c.setFont("Helvetica", 6)
        c.drawString(x + sw + 2, y + sw / 4, f"{color_id+1} ({color_name})")

    c.showPage()
    c.save()

# ---------- GUI ----------
class App:
    def __init__(self, root):
        self.root = root
        root.title("Color by Numbers — Iteration 6")
        self.image_path = None
        self.original_img = None
        self.idx = None
        self.palette = None
        self.color_indices = None
        self.regions = None
        self.orig_size = None  # 🔑 store original size

        frm = tk.Frame(root, padx=10, pady=10)
        frm.pack(fill=tk.BOTH, expand=True)

        file_row = tk.Frame(frm)
        file_row.pack(fill=tk.X, pady=(0, 8))
        tk.Label(file_row, text="Input PNG:").pack(side=tk.LEFT)
        self.file_entry = tk.Entry(file_row, width=50)
        self.file_entry.pack(side=tk.LEFT, padx=6)
        tk.Button(file_row, text="Browse…", command=self.browse_file).pack(side=tk.LEFT)

        params = tk.Frame(frm)
        params.pack(fill=tk.X, pady=(0, 8))
        tk.Label(params, text="Colors:").grid(row=0, column=0)
        self.colors_var = tk.IntVar(value=8)
        tk.Spinbox(params, from_=2, to=64, textvariable=self.colors_var, width=6).grid(row=0, column=1, padx=(6, 18))
        tk.Label(params, text="Min area mm²:").grid(row=0, column=2)
        self.min_area_var = tk.DoubleVar(value=40.0)
        tk.Entry(params, textvariable=self.min_area_var, width=8).grid(row=0, column=3, padx=(6, 18))
        tk.Label(params, text="Max area mm²:").grid(row=0, column=4)
        self.max_area_var = tk.DoubleVar(value=5000.0)
        tk.Entry(params, textvariable=self.max_area_var, width=8).grid(row=0, column=5, padx=(6, 18))
        tk.Label(params, text="Edge simplification:").grid(row=1, column=0)
        self.smooth_var = tk.IntVar(value=2)
        tk.Scale(params, from_=0, to=10, orient=tk.HORIZONTAL, variable=self.smooth_var, length=200).grid(row=1, column=1, columnspan=3, sticky="w")

        btns = tk.Frame(frm)
        btns.pack(fill=tk.X, pady=(0, 8))
        tk.Button(btns, text="Preview", command=self.on_preview).pack(side=tk.LEFT)
        tk.Button(btns, text="Export PDF…", command=self.on_export).pack(side=tk.LEFT, padx=6)

        self.preview_lbl = tk.Label(frm, text="Preview will appear here.", bd=1, relief=tk.SUNKEN)
        self.preview_lbl.pack(fill=tk.BOTH, expand=True)

    def browse_file(self):
        path = filedialog.askopenfilename(filetypes=[("PNG", "*.png"), ("All", "*.*")])
        if path:
            self.image_path = path
            self.original_img = None
            self.idx = None
            self.palette = None
            self.color_indices = None
            self.regions = None
            self.orig_size = None
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, path)
            self.preview_lbl.configure(image="", text="Preview will appear here.")

    def load_image(self):
        if not self.image_path:
            raise FileNotFoundError("No image selected.")
        img = Image.open(self.image_path)
        if img.mode in ("RGBA", "LA"):
            bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
            bg.paste(img, mask=img.split()[-1])
            img = bg.convert("RGB")
        else:
            img = img.convert("RGB")

        self.orig_size = img.size[::-1]  # (height, width)

        # Downscale to speed up processing
        if img.width * img.height > MAX_PIXELS:
            scale = math.sqrt(MAX_PIXELS / (img.width * img.height))
            new_size = (int(img.width * scale), int(img.height * scale))
            img = img.resize(new_size, Image.LANCZOS)

        self.original_img = img

    def process(self):
        if self.original_img is None:
            self.load_image()
        n = int(self.colors_var.get())
        idx, pal, color_indices = quantize_image(self.original_img, n)
        ps = (A4_MM[0] - 2 * MARGIN_MM, A4_MM[1] - 2 * MARGIN_MM - LEGEND_HEIGHT_MM)
        regs = extract_regions(idx, float(self.min_area_var.get()), float(self.max_area_var.get()), ps, int(self.smooth_var.get()))
        self.idx, self.palette, self.color_indices, self.regions = idx, pal, color_indices, regs

    def on_preview(self):
        try:
            self.process()
            img = render_preview(self.idx, self.regions, self.palette)
            scale = min(900 / img.width, 600 / img.height, 1.0)
            if scale < 1.0:
                img = img.resize((int(img.width * scale), int(img.height * scale)))
            tkimg = ImageTk.PhotoImage(img)
            self.preview_lbl.configure(image=tkimg, text="")
            self.preview_lbl.image = tkimg
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def on_export(self):
        try:
            if self.idx is None:
                self.process()
            pdf_path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF", "*.pdf")])
            if not pdf_path:
                return
            ps = (A4_MM[0] - 2 * MARGIN_MM, A4_MM[1] - 2 * MARGIN_MM - LEGEND_HEIGHT_MM)
            generate_pdf(self.idx, self.regions, pdf_path, ps, self.palette, self.color_indices, self.orig_size)
            messagebox.showinfo("Saved", f"PDF saved:\n{pdf_path}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

# ---------- Main ----------
def main():
    root = tk.Tk()
    App(root)
    root.minsize(720, 480)
    root.mainloop()

if __name__ == "__main__":
    main()

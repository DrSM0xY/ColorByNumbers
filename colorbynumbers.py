#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Color-by-numbers generator (iteration 5):
- Uses Crayola 48 colors (unique per cluster where possible).
- Reduced smoothing for outer (border) contours compared to inner ones.
- PDF contains outlines + numbers, no fill.
- GUI slider for edge simplification (morphology radius).
- Legend only shows colors that are actually used.
- Thicker contour lines for better printability.
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

def quantize_image(img, n_colors):
    """
    Quantize image into n_colors clusters and map each cluster to a Crayola color.
    Ensure each cluster is assigned a unique Crayola color where possible. If
    n_colors > number of Crayola colors, reuse closest colors for the extras.
    Returns: idx (h,w): cluster indices, palette (n_clusters x 3): RGB, color_indices: list of Crayola indices
    """
    img_rgb = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
    h, w, _ = img_rgb.shape
    pixels = img_rgb.reshape(-1, 3)

    # Bound clusters to a reasonable amount to avoid KMeans failing (but we still run with requested clusters)
    k = max(1, int(n_colors))
    kmeans = KMeans(n_clusters=k, n_init=5, random_state=0).fit(pixels)
    labels = kmeans.predict(pixels)
    centers = kmeans.cluster_centers_

    # Map cluster centers to nearest Crayola colors (try to avoid duplicates)
    palette = np.zeros_like(centers)
    color_indices = []
    used = set()
    n_cray = len(CRAYOLA_COLORS)
    for i, center in enumerate(centers):
        distances = np.linalg.norm(CRAYOLA_COLORS - center, axis=1)
        sorted_idxs = np.argsort(distances)
        chosen = None
        # Take the closest unused; if none available, fall back to closest regardless of used set
        for idx in sorted_idxs:
            if idx not in used:
                chosen = idx
                break
        if chosen is None:
            # all used, reuse closest
            chosen = sorted_idxs[0]
        palette[i] = CRAYOLA_COLORS[chosen]
        color_indices.append(int(chosen))
        used.add(chosen)

    idx = labels.reshape(h, w)
    return idx, palette, color_indices

def mm_per_pixel(idx_shape, print_size_mm):
    h, w = idx_shape
    pw, ph = print_size_mm
    img_aspect, page_aspect = w / h, pw / ph
    if img_aspect >= page_aspect:
        out_w, out_h = pw, pw / img_aspect
    else:
        out_h, out_w = ph, ph * img_aspect
    # out_w/out_h are mm; divide by px to get mm per pixel
    return out_w / w, out_h / h

def smooth_mask(mask, radius, is_border=False):
    if radius <= 0:
        return mask
    effective_radius = radius // 2 if is_border else radius
    if effective_radius <= 0:
        return mask
    se = disk(effective_radius)
    return opening(closing(mask, se), se)

def extract_regions(idx, min_area_mm2, max_area_mm2, print_size_mm, smooth_radius):
    regions = []
    h, w = idx.shape
    mmppx, mmppy = mm_per_pixel(idx.shape, print_size_mm)
    px_area_to_mm2 = mmppx * mmppy

    for color in np.unique(idx):
        mask = (idx == color)
        if not np.any(mask):
            continue
        # if region touches the image border, treat smoothing differently
        is_border = np.any(mask[0, :]) or np.any(mask[-1, :]) or np.any(mask[:, 0]) or np.any(mask[:, -1])
        mask = smooth_mask(mask, smooth_radius, is_border)
        lbl = sk_label(mask, connectivity=2)
        for p in regionprops(lbl):
            coords = np.array(p.coords)
            area_mm2 = p.area * px_area_to_mm2
            if area_mm2 < min_area_mm2:
                continue
            if area_mm2 > max_area_mm2:
                # split large region using clustering of coordinates
                k = math.ceil(area_mm2 / max_area_mm2)
                k = min(k, len(coords))
                if k <= 1:
                    # fallback, treat as single region
                    reg_mask = (lbl == p.label)
                    reg_mask = smooth_mask(reg_mask, smooth_radius, is_border)
                    dist = distance_transform_edt(reg_mask)
                    rr, cc = np.unravel_index(np.argmax(dist), dist.shape)
                    regions.append({
                        "color": int(color),
                        "area_px": int(p.area),
                        "best_point": (int(rr), int(cc)),
                        "mask": reg_mask
                    })
                    continue
                km = KMeans(n_clusters=k, n_init=3, random_state=0).fit(coords)
                for cl in range(k):
                    subcoords = coords[km.labels_ == cl]
                    if len(subcoords) == 0:
                        continue
                    submask = np.zeros_like(mask)
                    submask[tuple(subcoords.T)] = True
                    is_border_sub = np.any(submask[0, :]) or np.any(submask[-1, :]) or \
                                   np.any(submask[:, 0]) or np.any(submask[:, -1])
                    submask = smooth_mask(submask, smooth_radius, is_border_sub)
                    if not np.any(submask):
                        continue
                    dist = distance_transform_edt(submask)
                    rr, cc = np.unravel_index(np.argmax(dist), dist.shape)
                    regions.append({
                        "color": int(color),
                        "area_px": int(len(subcoords)),
                        "best_point": (int(rr), int(cc)),
                        "mask": submask
                    })
                continue
            reg_mask = (lbl == p.label)
            reg_mask = smooth_mask(reg_mask, smooth_radius, is_border)
            if not np.any(reg_mask):
                continue
            dist = distance_transform_edt(reg_mask)
            rr, cc = np.unravel_index(np.argmax(dist), dist.shape)
            regions.append({
                "color": int(color),
                "area_px": int(p.area),
                "best_point": (int(rr), int(cc)),
                "mask": reg_mask
            })
    return regions

# ---------- Rendering ----------
def render_preview(idx, regions, palette):
    h, w = idx.shape
    overlay = np.ones((h, w, 3), dtype=np.uint8) * 255
    fig = plt.figure(figsize=(w / PREVIEW_DPI, h / PREVIEW_DPI), dpi=PREVIEW_DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(overlay, origin="upper")
    ax.axis("off")
    for reg in regions:
        mask = reg["mask"]
        for contour in find_contours(mask, 0.5):
            ax.plot(contour[:, 1], contour[:, 0], color="black", linewidth=1.0)
        r, cx = reg["best_point"]
        ax.text(cx, r, f"{reg['color']+1}", ha="center", va="center",
                fontsize=8, color="black",
                path_effects=[PathEffects.Stroke(linewidth=1, foreground="white"),
                              PathEffects.Normal()])
    buf = io.BytesIO()
    fig.canvas.print_png(buf)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

def generate_pdf(idx, regions, pdf_path, print_size_mm, palette, color_indices):
    img_h, img_w = idx.shape
    pw, ph = A4_MM
    cw = pw - 2 * MARGIN_MM
    ch = ph - 2 * MARGIN_MM - LEGEND_HEIGHT_MM
    img_aspect = img_w / img_h
    cont_aspect = cw / ch
    if img_aspect >= cont_aspect:
        ow, oh = cw, cw / img_aspect
    else:
        oh, ow = ch, ch * img_aspect

    fig = plt.figure(figsize=(mm_to_inches(pw), mm_to_inches(ph)), dpi=300)
    left = MARGIN_MM / pw
    bottom = (MARGIN_MM + LEGEND_HEIGHT_MM + (ch - oh) / 2) / ph
    ax = fig.add_axes([left, bottom, ow / pw, oh / ph])
    ax.set_xlim(0, img_w)
    ax.set_ylim(img_h, 0)
    ax.axis("off")

    mmppx, mmppy = mm_per_pixel(idx.shape, print_size_mm)
    px2mm2 = mmppx * mmppy

    for reg in regions:
        mask = reg["mask"]
        for contour in find_contours(mask, 0.5):
            ax.plot(contour[:, 1], contour[:, 0], color="black", linewidth=1.2)  # thicker for print
        r, cx = reg["best_point"]
        area_mm2 = reg["area_px"] * px2mm2
        size_pt = min(FONT_PT_MAX, max(FONT_PT_MIN, (math.sqrt(area_mm2) / 10) * 6 + FONT_PT_MIN))
        ax.text(cx, r, f"{reg['color']+1}", ha="center", va="center", fontsize=size_pt, color="black",
                path_effects=[PathEffects.Stroke(linewidth=max(1.0, size_pt / 10.0), foreground="white"),
                              PathEffects.Normal()])

    # Legend: only used colors
    used_colors = sorted(set(reg["color"] for reg in regions))
    lax = fig.add_axes([MARGIN_MM / pw, MARGIN_MM / ph, cw / pw, LEGEND_HEIGHT_MM / ph])
    lax.axis('off')
    sw = LEGEND_SWATCH_SIZE_MM
    gap = LEGEND_GAP_MM
    cellw = sw + gap * 3
    perrow = max(1, int(cw // cellw))
    for legend_idx, color_id in enumerate(used_colors):
        row, col = divmod(legend_idx, perrow)
        x_mm = col * cellw + gap
        y_mm = LEGEND_HEIGHT_MM - (row + 1) * (sw + gap)
        x, y = x_mm / cw, y_mm / LEGEND_HEIGHT_MM
        wn, hn = sw / cw, sw / LEGEND_HEIGHT_MM
        # palette index matches cluster index (color_id)
        face = tuple(palette[color_id]) if color_id < len(palette) else (1.0, 1.0, 1.0)
        rect = patches.Rectangle((x, y), wn, hn, transform=lax.transAxes,
                                 facecolor=face, edgecolor='black')
        lax.add_patch(rect)
        # color_indices maps cluster -> Crayola index
        cray_idx = color_indices[color_id] if color_id < len(color_indices) else None
        color_name = CRAYOLA_NAMES[cray_idx] if cray_idx is not None and cray_idx < len(CRAYOLA_NAMES) else "Color"
        lax.text(x + wn / 2, y + hn / 2, f"{color_id+1} ({color_name})", transform=lax.transAxes,
                 ha='center', va='center', fontsize=LEGEND_TEXT_PT, color="black",
                 path_effects=[PathEffects.Stroke(linewidth=1.2, foreground='white'), PathEffects.Normal()])

    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

# ---------- GUI ----------
class App:
    def __init__(self, root):
        self.root = root
        root.title("Color by Numbers — Iteration 5")
        self.image_path = None
        self.original_img = None
        self.idx = None
        self.palette = None
        self.color_indices = None
        self.regions = None

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
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, path)

    def load_image(self):
        if not self.image_path:
            raise FileNotFoundError("No image selected.")
        img = Image.open(self.image_path)
        # Flatten alpha channel to white if present
        if img.mode in ("RGBA", "LA"):
            bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
            bg.paste(img, mask=img.split()[-1])
            img = bg.convert("RGB")
        else:
            img = img.convert("RGB")
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
            generate_pdf(self.idx, self.regions, pdf_path, ps, self.palette, self.color_indices)
            messagebox.showinfo("Saved", f"PDF saved:\n{pdf_path}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

def main():
    root = tk.Tk()
    App(root)
    root.minsize(720, 480)
    root.mainloop()

if __name__ == "__main__":
    main()

from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pathlib import Path
from random import choices,uniform

import argparse
import sys
import matplotlib.pyplot as plt
import numpy as np

def get_kmeans_colors(img, nmin=8, nmax=16):
    
    img = np.array(img);
    
    # Get all colors in the image
    height, width, _ = img.shape
    img_reshaped = img.reshape(height * width, 3).astype(float)
    img_normalized = img_reshaped / 255.0

    # Pick colors from the named colors as starting points
    colors = get_named_colors()
    ncolors = colors.shape[0]
    colorscores = np.zeros(ncolors)
    for i in range(ncolors):
      colorscores[i] = np.linalg.norm(img_normalized - colors[i])
    cdx = colorscores.argsort()
    
    subimg = choices(img_normalized,k=5000)
    ss = np.zeros(nmax-nmin)
    for i in range(nmin,nmax):
      if i > ncolors:
          icolors = colors;
          for _ in range(i-ncolors):
            icolors = np.vstack((icolors,np.random.uniform(0,1,3)))
      else: 
          icolors = colors[cdx[0:i]];
      km = KMeans(n_clusters=i,init=icolors)
      km.fit(img_normalized)
      labels = km.fit_predict(subimg);
      ss[i-nmin] = silhouette_score(subimg,labels)
      
    # use the best number of colors
    nn = max(np.where(ss > max(ss)*0.95)[0])+nmin;
    if nn > ncolors:
        icolors = colors;
        for _ in range(nn-ncolors):
          icolors = np.vstack((icolors,np.random.uniform(0,1,3)))
    else: 
        icolors = colors[cdx[0:nn]];
    km = KMeans(n_clusters=nn,init=icolors)
    km.fit(img_normalized)

    return km.cluster_centers_


def get_named_colors():
    named_colors = np.array([
        [0, 0, 0],         # Black
        [255, 255, 255],   # White
        [128, 128, 128],   # Gray
        [192, 192, 192],   # Silver
        [255, 0, 0],       # Red
        [128, 0, 0],       # Maroon
        [255, 255, 0],     # Yellow
        [128, 128, 0],     # Olive
        [0, 255, 0],       # Lime
        [0, 128, 0],       # Green
        [0, 255, 255],     # Aqua
        [0, 128, 128],     # Teal
        [0, 0, 255],       # Blue
        [0, 0, 128],       # Navy
        [255, 0, 255],     # Fuchsia
        [128, 0, 128]      # Purple
    ]) / 255.0  # Scale to [0, 1]
    return named_colors
    
def get_crayola_colors():
    named_colors = np.array([
        [35, 35, 35],      # Black
        [252, 232, 131],   # Yellow
        [31, 117, 254],    # Blue
        [180, 103, 77],    # Brown
        [255, 117, 56],    # Orange
        [28, 172, 120],    # Green
        [146, 110, 174],   # Violet
        [192, 68, 143],    # Red Violet
        [255, 83, 73],     # Red Orange
        [197, 227, 132],   # Yellow Green
        [115, 102, 189],   # Blue Violet
        [255, 170, 204],   # Carnation Pink
        [255, 182, 83],    # Yellow Orange
        [25, 158, 189],    # Blue Green
        [237, 237, 237]    # Whilte
    ]) / 255.0  # Scale to [0, 1]
    return named_colors

def downsample_figure(img, npixels=None, colors=None):
    
    img = np.array(img);

    if colors is None:
        colors = get_named_colors()

    if img.ndim == 2:  # Grayscale
        img = np.stack([img] * 3, axis=-1)

    if npixels is None:
        nr, nc = img.shape[:2]
        while nr * nc > 5000:
            nr //= 2
            nc //= 2
    else:
        rows, cols = img.shape[:2]
        total_pixels = rows * cols
        scale_factor = np.sqrt(npixels / total_pixels)
        nr = int(rows * scale_factor)
        nc = int(cols * scale_factor)

    resized_img = np.array(Image.fromarray(img).resize((nc, nr)))

    # Map pixels to nearest named color
    nr, nc, _ = resized_img.shape
    out_img = np.zeros_like(resized_img, dtype=float)
    for i in range(nr):
        for j in range(nc):
            pixel = resized_img[i, j, :].astype(float) / 255
            distances = np.linalg.norm(colors - pixel, axis=1)
            idx = np.argmin(distances)
            out_img[i, j, :] = colors[idx]

    out_img = Image.fromarray((out_img * 255).astype(np.uint8));
    return out_img

def get_unique_colors(img):
    img = np.array(img);
    unique_colors = np.unique(img.reshape(-1, img.shape[2]), axis=0)
    return unique_colors.astype(float) / 255.0


def ascolorsheet(img):
    
    img = np.array(img)
    rows, cols, _ = img.shape
    colors = get_unique_colors(img)

    cidxs = np.zeros((rows, cols), dtype=int)
    for i in range(rows):
        for j in range(cols):
            pixel = img[i, j, :].astype(float) / 255
            distances = np.linalg.norm(colors - pixel, axis=1)
            cidxs[i, j] = np.argmin(distances)

    # Add visualization logic (similar to MATLAB logic, omitted for brevity)
    margin = 5; # Size of the margin (pixel)
    sqsz = 40;  # Size of each square (pixel)
    padding = 10; # Padding for the legend
    lhgt = sqsz * 2; # Space for the legend

    ohgt = margin + rows * sqsz + lhgt + padding + margin;
    owdt = margin + cols * sqsz + padding + margin;

    outimg = np.ones([ohgt,owdt,3]) # white background
    numberimages = []
    for i in range(colors.shape[0]):
        numberimages.append(number_image(i+1));

    # Draw the squares with numbers
    for i in range(rows-1):
        for j in range(cols-1):
            
            ijnr = cidxs[i, j];
            top = margin + i * sqsz + 1;
            left = margin + j * sqsz + 1;
            bottom = top + sqsz;
            right = left + sqsz;

            # Get the number in the middle
            ni = np.array(numberimages[ijnr])/255;
            outimg[top:bottom,left:right,:] = 1-(1-ni)*0.3;

            # Draw the colored square
            outimg[top:bottom, left, :] = 0;
            outimg[top:bottom, right, :] = 0;
            outimg[top, left:right, :] = 0;
            outimg[bottom, left:right, :] = 0;

    # Draw the legend
    for i in range(colors.shape[0]):

        ltop = margin + rows * sqsz + padding;
        lleft = margin + i * sqsz;
        lbottom = ltop + sqsz;
        lright = lleft + sqsz;
        
        r, g, b = colors[i]
        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
        if luminance > 0.5:
          # Darken the color for contrast
          contrasting_color = np.array([r * 0.5, g * 0.5, b * 0.5])
        else:
          # Lighten the color for contrast
          contrasting_color = np.array([r + (1 - r) * 0.5, g + (1 - g) * 0.5, b + (1 - b) * 0.5]);

        # Color patch
        ni = np.array(numberimages[i])/255;
        nir,nig,nib = ni.T;
        iswhite = (nir >= 0.5) & (nig >= 0.5) & (nib >= 0.5)
        isblack = ~iswhite
        ni[iswhite.T,:] = colors[i];
        ni[isblack.T,:] = contrasting_color;

        # Draw the colored square
        outimg[ltop:lbottom, lleft:lright, :] = ni;
        outimg[ltop:lbottom, lleft, :] = 0;
        outimg[ltop:lbottom, lright, :] = 0;
        outimg[ltop, lleft:lright, :] = 0;
        outimg[lbottom, lleft:lright, :] = 0;
        
    outimg = Image.fromarray((outimg * 255).astype(np.uint8));
    return outimg
    
# Helper function to generate binary images of numbers (as placeholders for numberimage)
def number_image(number, size=40):
    
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

    fig, ax = plt.subplots(figsize=(size / 10, size / 10), dpi=10)
    ax.text(0.5, 0.5, str(number), fontsize=200, ha='center', va='center')
    ax.axis('off')
    canvas = FigureCanvas(fig)
    canvas.draw()
    img = np.array(canvas.renderer.buffer_rgba())
    plt.close(fig)
    img = Image.fromarray(img[:, :, :3])
    return img
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
                    prog='Color by numbers',
                    description='create a color-by-numbers page from png',
                    epilog='Ask Kristof Smolders for more info')

    parser.add_argument('filename')
    parser.add_argument('-c', '--crayola_colors', action='store_true')  # on/off flag
    parser.add_argument('-n', action="store", dest="minimum", default=3, type=int)
    parser.add_argument('-x', action="store", dest="maximum", default=16, type=int)

    args = parser.parse_args()

    filename = args.filename;
    nmin = args.minimum;
    nmax = args.maximum;
    outfile = filename.replace(".","_colorsheet.");
    img = Image.open(filename);
    if args.crayola_colors:
      colors = get_crayola_colors();
    else:
      colors = get_kmeans_colors(img,nmin,nmax);
    img = downsample_figure(img,colors=colors)
    img = ascolorsheet(img)
    img.save(outfile);

# 🎨 Color-by-Numbers Generator

Convert any image into a **color-by-numbers** template, complete with Crayola colors and automatic region labeling.  

🪟 **[👉 Download Windows Program](https://github.com/DrSM0xY/ColorByNumbers/releases/download/windows/CBNCreator.exe)**

---

## ✨ Features

- Automatically converts any **PNG** image into a printable **color-by-numbers PDF**.  
- Uses real **Crayola 48-color palette** for beautiful, recognizable colors.  
- Adjustable:
  - Number of colors (2–64)
  - Minimum/maximum region area (mm²)
  - Edge smoothing
- Built-in **live preview** before exporting.
- Exports to **A4-sized PDFs** with a labeled legend and color swatches.

---

## Installation (Python users)

If you prefer to run it from source, install dependencies and launch manually:

git clone https://github.com/yourusername/color-by-numbers.git
cd color-by-numbers
pip install numpy pillow matplotlib scikit-image scipy scikit-learn
python color_by_numbers.py

### Building the .exe Yourself (Optional)
If you want to create your own Windows binary:

pip install pyinstaller
pyinstaller --onefile --windowed color_by_numbers.py --name ColorByNumbers

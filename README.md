# 🎨 Color-by-Numbers Generator

Convert any image into a **paint-by-numbers** or **color-by-numbers** template, complete with Crayola colors and automatic region labeling.  

🪟 **[👉 Download Windows Executable (no Python required)](https://github.com/DrSM0xY/ColorByNumbers/ColorByNumbers.exe)**

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

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/color-by-numbers.git
cd color-by-numbers

### 2. Install dependencies
```bash
pip install numpy pillow matplotlib scikit-image scipy scikit-learn

### 3. Run the app
python color_by_numbers.py

### Building the .exe Yourself (Optional)
If you want to create your own Windows binary:

pip install pyinstaller
pyinstaller --onefile --windowed color_by_numbers.py --name ColorByNumbers

import os

base = "Air Pollution Image Dataset"
for c in ["clean", "moderate", "polluted"]:
    folder = os.path.join(base, c)
    files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'))]
    print(f"{c}: {len(files)} valid image files")

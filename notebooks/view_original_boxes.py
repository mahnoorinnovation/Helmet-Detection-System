from PIL import Image, ImageDraw
from pathlib import Path

name = "BikesHelmets247_png_jpg.rf.aebfad26df01a84a3f7d664016ce748f"
image_path = Path("dataset/images/train") / f"{name}.jpg"
label_path = Path("dataset/labels/train") / f"{name}.txt"

image = Image.open(image_path).convert("RGB")
draw = ImageDraw.Draw(image)
w, h = image.size

with open(label_path) as f:
    for line in f:
        cls, x, y, bw, bh = map(float, line.split())
        x1, y1 = int((x-bw/2)*w), int((y-bh/2)*h)
        x2, y2 = int((x+bw/2)*w), int((y+bh/2)*h)
        draw.rectangle((x1, y1, x2, y2), outline="green", width=3)

out = Path("outputs/dataset_check/original_bikes247.jpg")
image.save(out)
print(f"Saved: {out}")

from PIL import Image, ImageDraw
from pathlib import Path

name = "BikesHelmets389_png_jpg.rf.0d462adc796da0346e097fb546862361"
base = Path("Bike Helmet Detection.v1i.yolov8/train")

image = Image.open(base / "images" / f"{name}.jpg").convert("RGB")
draw = ImageDraw.Draw(image)
w, h = image.size

with open(base / "labels" / f"{name}.txt") as f:
    for line in f:
        cls, x, y, bw, bh = map(float, line.split())
        x1 = int((x - bw/2) * w)
        y1 = int((y - bh/2) * h)
        x2 = int((x + bw/2) * w)
        y2 = int((y + bh/2) * h)
        color = "green" if int(cls) == 0 else "red"
        draw.rectangle((x1, y1, x2, y2), outline=color, width=2)

out = Path("outputs/dataset_check/annotated_bikes389.jpg")
out.parent.mkdir(parents=True, exist_ok=True)
image.save(out)
print(f"Saved: {out}")

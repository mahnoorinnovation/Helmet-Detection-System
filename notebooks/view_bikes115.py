from PIL import Image, ImageDraw
from pathlib import Path

name = "BikesHelmets115_png_jpg.rf.e2d44812a061abe2376fea5b3f9dcdc6"
base = Path("Bike Helmet Detection.v1i.yolov8/train")
image = Image.open(base / "images" / f"{name}.jpg").convert("RGB")
draw = ImageDraw.Draw(image)
w, h = image.size

with open(base / "labels" / f"{name}.txt") as f:
    for line in f:
        cls, x, y, bw, bh = map(float, line.split())
        box = ((x-bw/2)*w, (y-bh/2)*h, (x+bw/2)*w, (y+bh/2)*h)
        draw.rectangle(box, outline="green" if int(cls)==0 else "red", width=3)

out = Path("outputs/dataset_check/annotated_bikes115.jpg")
image.save(out)
print("Saved:", out)

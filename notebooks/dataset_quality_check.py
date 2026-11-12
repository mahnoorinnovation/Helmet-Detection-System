from pathlib import Path
from PIL import Image
import csv

base = Path("Bike Helmet Detection.v1i.yolov8/train")
images_dir = base / "images"
labels_dir = base / "labels"

image_extensions = {".jpg", ".jpeg", ".png"}
images = sorted(p for p in images_dir.iterdir()
                if p.suffix.lower() in image_extensions)

issues = []
total_boxes = 0
empty_labels = 0
valid_boxes = 0

for image_path in images:
    label_path = labels_dir / (image_path.stem + ".txt")

    if not label_path.exists():
        issues.append([image_path.name, "Missing label file"])
        continue

    if label_path.stat().st_size == 0:
        empty_labels += 1
        issues.append([image_path.name, "Empty label file"])
        continue

    try:
        with Image.open(image_path) as img:
            img.verify()
    except Exception:
        issues.append([image_path.name, "Invalid image file"])
        continue

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line_no, line in enumerate(lines, start=1):
        total_boxes += 1
        parts = line.split()

        if len(parts) != 5:
            issues.append([image_path.name, f"Line {line_no}: Invalid format"])
            continue

        try:
            cls = int(parts[0])
            x, y, w, h = map(float, parts[1:])
        except ValueError:
            issues.append([image_path.name, f"Line {line_no}: Non-numeric values"])
            continue

        if cls not in (0, 1):
            issues.append([image_path.name, f"Line {line_no}: Invalid class ID {cls}"])
        elif not all(0 <= v <= 1 for v in (x, y, w, h)):
            issues.append([image_path.name, f"Line {line_no}: Coordinates outside 0-1"])
        elif w <= 0 or h <= 0 or x-w/2 < 0 or y-h/2 < 0 or x+w/2 > 1 or y+h/2 > 1:
            issues.append([image_path.name, f"Line {line_no}: Invalid bounding box"])
        else:
            valid_boxes += 1

for label_path in labels_dir.glob("*.txt"):
    matching = any(
        (images_dir / (label_path.stem + ext)).exists()
        for ext in image_extensions
    )
    if not matching:
        issues.append([label_path.name, "Missing matching image"])

report = Path("outputs/dataset_check/quality_check_report.csv")
report.parent.mkdir(parents=True, exist_ok=True)

with open(report, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["File", "Issue"])
    writer.writerows(issues)

print("\n===== DATASET QUALITY REPORT =====")
print("Total images:", len(images))
print("Total boxes:", total_boxes)
print("Valid boxes:", valid_boxes)
print("Empty labels:", empty_labels)
print("Total issues:", len(issues))
print("Report saved:", report)

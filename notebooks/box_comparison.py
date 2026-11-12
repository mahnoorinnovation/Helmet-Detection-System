from pathlib import Path
import csv

report = Path("outputs/dataset_check/perceptual_matching_report.csv")
original_dir = Path("dataset/labels/train")
manual_dir = Path("Bike Helmet Detection.v1i.yolov8/train/labels")


def read_boxes(path):
    boxes = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) == 5:
            cls, x, y, w, h = parts
            boxes.append((int(cls), float(x), float(y), float(w), float(h)))
    return boxes


def iou(a, b):
    _, ax, ay, aw, ah = a
    _, bx, by, bw, bh = b

    left = max(ax - aw / 2, bx - bw / 2)
    top = max(ay - ah / 2, by - bh / 2)
    right = min(ax + aw / 2, bx + bw / 2)
    bottom = min(ay + ah / 2, by + bh / 2)

    intersection = max(0, right - left) * max(0, bottom - top)
    union = aw * ah + bw * bh - intersection

    return intersection / union if union else 0


results = []

with report.open(encoding="utf-8") as f:
    for row in csv.DictReader(f):
        original_name = Path(row["best_original_match"]).stem + ".txt"
        manual_name = Path(row["exported_image"]).stem + ".txt"

        original_boxes = read_boxes(original_dir / original_name)
        manual_boxes = read_boxes(manual_dir / manual_name)

        matched = 0

        for box in manual_boxes:
            candidates = [
                iou(box, old_box)
                for old_box in original_boxes
                if box[0] == old_box[0]
            ]

            if candidates and max(candidates) >= 0.5:
                matched += 1

        results.append({
            "image": row["exported_image"],
            "original_boxes": len(original_boxes),
            "manual_boxes": len(manual_boxes),
            "matched_boxes": matched,
            "manual_match_percent": round(
                matched / len(manual_boxes) * 100, 2
            ) if manual_boxes else 0
        })

output = Path("outputs/dataset_check/box_comparison.csv")

with output.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("Images compared:", len(results))
print("Average manual box match:", round(
    sum(r["manual_match_percent"] for r in results) / len(results), 2
), "%")
print("Detailed report:", output)
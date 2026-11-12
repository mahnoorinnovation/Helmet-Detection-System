from pathlib import Path
import csv

report = Path("outputs/dataset_check/perceptual_matching_report.csv")
original_labels = Path("dataset/labels/train")
manual_labels = Path("Bike Helmet Detection.v1i.yolov8/train/labels")

results = []

with report.open(encoding="utf-8") as f:
    for row in csv.DictReader(f):
        original_name = Path(row["best_original_match"]).stem + ".txt"
        manual_name = Path(row["exported_image"]).stem + ".txt"

        original_path = original_labels / original_name
        manual_path = manual_labels / manual_name

        original_lines = [
            x.strip() for x in original_path.read_text().splitlines() if x.strip()
        ]
        manual_lines = [
            x.strip() for x in manual_path.read_text().splitlines() if x.strip()
        ]

        results.append({
            "image": row["exported_image"],
            "original_objects": len(original_lines),
            "manual_objects": len(manual_lines),
            "original_classes": sorted(set(x.split()[0] for x in original_lines)),
            "manual_classes": sorted(set(x.split()[0] for x in manual_lines))
        })

output = Path("outputs/dataset_check/label_comparison.csv")
output.parent.mkdir(parents=True, exist_ok=True)

with output.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("Images compared:", len(results))
print("Comparison saved:", output)
print("First 5 comparisons:")

for item in results[:5]:
    print(item)
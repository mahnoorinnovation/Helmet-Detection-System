from pathlib import Path
from PIL import Image
import csv

report = Path("outputs/dataset_check/perceptual_matching_report.csv")
original_dir = Path("annotation_work/images")
exported_dir = Path("Bike Helmet Detection.v1i.yolov8/train/images")

results = []

with report.open(encoding="utf-8") as f:
    for row in csv.DictReader(f):
        original = original_dir / row["best_original_match"]
        exported = exported_dir / row["exported_image"]

        with Image.open(original) as a, Image.open(exported) as b:
            ow, oh = a.size
            ew, eh = b.size

        same_ratio = abs((ow / oh) - (ew / eh)) < 0.001

        results.append({
            "original": original.name,
            "exported": exported.name,
            "original_size": f"{ow}x{oh}",
            "exported_size": f"{ew}x{eh}",
            "same_ratio": same_ratio
        })

print("Total checked:", len(results))
print("Same aspect ratio:", sum(r["same_ratio"] for r in results))
print("Different aspect ratio:", sum(not r["same_ratio"] for r in results))
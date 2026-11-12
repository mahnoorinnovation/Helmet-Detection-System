from pathlib import Path
from PIL import Image
import csv

original_dir = Path("annotation_work/images")
exported_dir = Path("Bike Helmet Detection.v1i.yolov8/train/images")

def dhash(path):
    with Image.open(path) as img:
        img = img.convert("L").resize((9, 8))
        pixels = list(img.getdata())

    bits = []
    for y in range(8):
        for x in range(8):
            left = pixels[y * 9 + x]
            right = pixels[y * 9 + x + 1]
            bits.append(left > right)

    value = 0
    for bit in bits:
        value = (value << 1) | bit
    return value

originals = []
for path in original_dir.iterdir():
    if path.is_file():
        try:
            originals.append((path.name, dhash(path)))
        except Exception:
            pass

results = []

for path in exported_dir.iterdir():
    if path.is_file():
        try:
            h = dhash(path)
            comparisons = []

            for name, original_hash in originals:
                distance = (h ^ original_hash).bit_count()
                similarity = (64 - distance) / 64 * 100
                comparisons.append((similarity, name))

            comparisons.sort(reverse=True)
            best_score, best_name = comparisons[0]

            results.append({
                "exported_image": path.name,
                "best_original_match": best_name,
                "similarity_percent": round(best_score, 2)
            })
        except Exception:
            pass

output = Path("outputs/dataset_check/perceptual_matching_report.csv")
output.parent.mkdir(parents=True, exist_ok=True)

with output.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "exported_image",
            "best_original_match",
            "similarity_percent"
        ]
    )
    writer.writeheader()
    writer.writerows(results)

print("Exported images checked:", len(results))
print("Report saved:", output)
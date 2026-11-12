
from pathlib import Path
from collections import defaultdict, Counter
import csv
import random
import shutil

ROOT = Path(__file__).resolve().parent.parent

# Source folders
MANUAL_DIR = ROOT / "Bike Helmet Detection.v1i.yolov8"
ORIGINAL_IMAGES = ROOT / "dataset" / "images" / "train"
ORIGINAL_LABELS = ROOT / "dataset" / "labels" / "train"

SELECTION_CSV = ROOT / "outputs" / "dataset_check" / "additional_100_selection.csv"
MATCH_REPORT = ROOT / "outputs" / "dataset_check" / "perceptual_matching_report.csv"

# Final destination
FINAL = ROOT / "final_dataset"

SPLITS = ["train", "val", "test"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

random.seed(42)


def get_category(label_path):
    """Identify image category from YOLO class IDs."""

    if not label_path.exists():
        raise FileNotFoundError(f"Missing label: {label_path}")

    classes = set()

    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        parts = line.split()

        if len(parts) != 5:
            raise ValueError(f"Invalid YOLO label line: {label_path}")

        class_id = int(parts[0])

        if class_id not in (0, 1):
            raise ValueError(f"Unexpected class ID {class_id}: {label_path}")

        values = [float(x) for x in parts[1:]]

        if not all(0 <= x <= 1 for x in values):
            raise ValueError(f"Invalid normalized coordinates: {label_path}")

        if values[2] <= 0 or values[3] <= 0:
            raise ValueError(f"Invalid box size: {label_path}")

        classes.add(class_id)

    if classes == {0, 1}:
        return "Both classes"
    elif classes == {0}:
        return "Only With Helmet"
    elif classes == {1}:
        return "Only Without Helmet"

    raise ValueError(f"Empty label file: {label_path}")


# Do not overwrite an existing final dataset
for split in SPLITS:
    for folder in ["images", "labels"]:
        destination = FINAL / folder / split

        if not destination.exists():
            raise FileNotFoundError(f"Missing destination folder: {destination}")

        if any(destination.iterdir()):
            raise RuntimeError(
                f"{destination} is not empty. Stopping to protect existing files."
            )


# Load matched original filenames
with MATCH_REPORT.open("r", newline="", encoding="utf-8") as file:
    matched_originals = {
        row["best_original_match"]
        for row in csv.DictReader(file)
    }


# Store records by source and category
groups = defaultdict(list)

# 1. Read 100 manually annotated images
manual_images = [
    p for p in (MANUAL_DIR / "train" / "images").iterdir()
    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
]

if len(manual_images) != 100:
    raise ValueError(f"Expected 100 manual images, found {len(manual_images)}")

for image_path in manual_images:
    label_path = MANUAL_DIR / "train" / "labels" / (image_path.stem + ".txt")
    category = get_category(label_path)

    groups[("manual", category)].append({
        "image": image_path,
        "label": label_path,
        "source": "manual",
        "category": category
    })


# 2. Read 100 additional original images
with SELECTION_CSV.open("r", newline="", encoding="utf-8-sig") as file:
    selected_rows = list(csv.DictReader(file))

if len(selected_rows) != 100:
    raise ValueError(f"Expected 100 additional images, found {len(selected_rows)}")

for row in selected_rows:
    filename = row["Image"]

    # Prevent overlap with the 100 manually annotated source images
    if filename in matched_originals:
        raise ValueError(f"Duplicate original counterpart selected: {filename}")

    image_path = ORIGINAL_IMAGES / filename
    label_path = ORIGINAL_LABELS / (Path(filename).stem + ".txt")

    if not image_path.exists():
        raise FileNotFoundError(image_path)

    category = get_category(label_path)

    if category != row["Category"]:
        raise ValueError(f"Category mismatch for {filename}")

    groups[("original", category)].append({
        "image": image_path,
        "label": label_path,
        "source": "original",
        "category": category
    })


# Verify source counts
manual_total = sum(
    len(items) for (source, _), items in groups.items()
    if source == "manual"
)

original_total = sum(
    len(items) for (source, _), items in groups.items()
    if source == "original"
)

if manual_total != 100 or original_total != 100:
    raise ValueError("Expected exactly 100 manual and 100 original images.")


# 3. Stratified 70/20/10 split
split_records = defaultdict(list)

for (source, category), items in groups.items():
    random.shuffle(items)

    train_count = round(len(items) * 0.70)
    val_count = round(len(items) * 0.20)

    split_records["train"].extend(items[:train_count])
    split_records["val"].extend(
        items[train_count:train_count + val_count]
    )
    split_records["test"].extend(
        items[train_count + val_count:]
    )


# Verify exact split sizes before copying
expected = {"train": 140, "val": 40, "test": 20}

for split in SPLITS:
    if len(split_records[split]) != expected[split]:
        raise ValueError(f"Unexpected {split} count: {len(split_records[split])}")


# 4. Copy images and labels
manifest = []

for split in SPLITS:
    for record in split_records[split]:

        prefix = "manual__" if record["source"] == "manual" else "original__"
        new_name = prefix + record["image"].name
        new_label_name = Path(new_name).stem + ".txt"

        image_destination = FINAL / "images" / split / new_name
        label_destination = FINAL / "labels" / split / new_label_name

        shutil.copy2(record["image"], image_destination)
        shutil.copy2(record["label"], label_destination)

        manifest.append({
            "image": new_name,
            "label": new_label_name,
            "source": record["source"],
            "category": record["category"],
            "split": split
        })


# 5. Create data.yaml
yaml_content = """path: .
train: images/train
val: images/val
test: images/test

nc: 2
names:
  0: 'With Helmet'
  1: 'Without Helmet'
"""

(FINAL / "data.yaml").write_text(yaml_content, encoding="utf-8")

# 6. Save selection manifest
with (FINAL / "dataset_manifest.csv").open(
    "w", newline="", encoding="utf-8"
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=["image", "label", "source", "category", "split"]
    )
    writer.writeheader()
    writer.writerows(manifest)


# 7. Print final report
print("\n===== FINAL DATASET CREATED =====")
print("Manual images:", manual_total)
print("Additional original images:", original_total)
print("Total images:", len(manifest))

for split in SPLITS:
    records = split_records[split]
    print(f"\n{split.upper()}: {len(records)} images")

    print("Sources:", dict(Counter(r["source"] for r in records)))
    print("Categories:", dict(Counter(r["category"] for r in records)))

print("\nData YAML:", FINAL / "data.yaml")
print("Manifest:", FINAL / "dataset_manifest.csv")
print("\nOriginal dataset was not modified.")


from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "final_dataset"

SPLITS = ["train", "val", "test"]
EXPECTED = {"train": 140, "val": 40, "test": 20}

issues = []
total_images = 0
total_boxes = 0

for split in SPLITS:
    image_dir = DATASET / "images" / split
    label_dir = DATASET / "labels" / split

    images = [
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]

    labels = list(label_dir.glob("*.txt"))

    print(f"\nChecking {split.upper()}...")

    if len(images) != EXPECTED[split]:
        issues.append(
            f"{split}: expected {EXPECTED[split]} images, found {len(images)}"
        )

    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}

    for missing in image_stems - label_stems:
        issues.append(f"{split}: missing label for {missing}")

    for extra in label_stems - image_stems:
        issues.append(f"{split}: label without image: {extra}")

    for image_path in images:
        label_path = label_dir / f"{image_path.stem}.txt"

        try:
            with Image.open(image_path) as img:
                width, height = img.size
                img.verify()
        except Exception as exc:
            issues.append(f"Invalid image {image_path.name}: {exc}")
            continue

        total_images += 1

        if not label_path.exists():
            continue

        try:
            lines = label_path.read_text(encoding="utf-8").splitlines()
        except Exception as exc:
            issues.append(f"Cannot read label {label_path.name}: {exc}")
            continue

        if not lines:
            issues.append(f"Empty label: {label_path.name}")
            continue

        for line_number, line in enumerate(lines, start=1):
            parts = line.split()

            if len(parts) != 5:
                issues.append(
                    f"{label_path.name}, line {line_number}: expected 5 values"
                )
                continue

            try:
                class_id = int(parts[0])
                x_center, y_center, box_width, box_height = map(float, parts[1:])
            except ValueError:
                issues.append(
                    f"{label_path.name}, line {line_number}: non-numeric value"
                )
                continue

            if class_id not in (0, 1):
                issues.append(
                    f"{label_path.name}, line {line_number}: invalid class {class_id}"
                )
                continue

            if not all(
                0 <= value <= 1
                for value in (x_center, y_center, box_width, box_height)
            ):
                issues.append(
                    f"{label_path.name}, line {line_number}: coordinate outside 0-1"
                )
                continue

            if box_width <= 0 or box_height <= 0:
                issues.append(
                    f"{label_path.name}, line {line_number}: invalid box size"
                )
                continue

            x1 = x_center - box_width / 2
            y1 = y_center - box_height / 2
            x2 = x_center + box_width / 2
            y2 = y_center + box_height / 2

            if x1 < 0 or y1 < 0 or x2 > 1 or y2 > 1:
                issues.append(
                    f"{label_path.name}, line {line_number}: box outside image"
                )
                continue

            total_boxes += 1

    print(f"Images: {len(images)}")
    print(f"Labels: {len(labels)}")

print("\n===== FINAL DATASET VERIFICATION =====")
print("Total images checked:", total_images)
print("Valid bounding boxes:", total_boxes)
print("Total issues:", len(issues))

if issues:
    print("\nIssues found:")
    for issue in issues:
        print("-", issue)
else:
    print("\nAll structural and label checks passed!")

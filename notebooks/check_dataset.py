
from pathlib import Path
import cv2
import random

# Dataset folders
images_dir = Path("dataset/images/train")
labels_dir = Path("dataset/labels/train")
output_dir = Path("outputs/dataset_check")

output_dir.mkdir(parents=True, exist_ok=True)

class_names = {
    0: "With Helmet",
    1: "Without Helmet"
}

# Select 5 random training images
images = list(images_dir.glob("*.*"))
samples = random.sample(images, min(5, len(images)))

for image_path in samples:

    label_path = labels_dir / (image_path.stem + ".txt")
    image = cv2.imread(str(image_path))

    if image is None or not label_path.exists():
        print("Skipped:", image_path.name)
        continue

    height, width = image.shape[:2]

    with open(label_path, "r") as file:
        for line in file:
            values = line.strip().split()

            if len(values) != 5:
                continue

            class_id = int(values[0])
            x_center, y_center, box_width, box_height = map(float, values[1:])

            x1 = int((x_center - box_width / 2) * width)
            y1 = int((y_center - box_height / 2) * height)
            x2 = int((x_center + box_width / 2) * width)
            y2 = int((y_center + box_height / 2) * height)

            label = class_names.get(class_id, "Unknown")

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image, label, (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0), 2
            )

    output_path = output_dir / image_path.name
    cv2.imwrite(str(output_path), image)
    print("Saved:", output_path)

print("\nAnnotation checking images generated!")

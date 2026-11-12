
import cv2
from pathlib import Path

# Project paths
ROOT = Path(__file__).resolve().parent.parent

image_path = ROOT / "Bike Helmet Detection.v1i.yolov8" / "train" / "images" / "BikesHelmets247_png_jpg.rf.b73f78e884bed2cafef5f3e2454df24d.jpg"
label_path = ROOT / "Bike Helmet Detection.v1i.yolov8" / "train" / "labels" / "BikesHelmets247_png_jpg.rf.b73f78e884bed2cafef5f3e2454df24d.txt"

image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError(f"Image not found: {image_path}")

height, width = image.shape[:2]

# Class IDs: 0 = With Helmet, 1 = Without Helmet
boxes = []
drawing = False
start_x = start_y = 0
display = image.copy()

scale = min(1100 / width, 750 / height, 1.0)
shown = cv2.resize(image, (int(width * scale), int(height * scale)))

def mouse_callback(event, x, y, flags, param):
    global drawing, start_x, start_y, display

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_x, start_y = x, y

    elif event == cv2.EVENT_LBUTTONUP and drawing:
        drawing = False
        x1, y1 = sorted([start_x, x])
        y1, y2 = sorted([start_y, y])

        # Convert displayed coordinates to original image coordinates
        X1, Y1 = int(x1 / scale), int(y1 / scale)
        X2, Y2 = int(x / scale), int(y / scale)

        X1, X2 = sorted([X1, X2])
        Y1, Y2 = sorted([Y1, Y2])

        if X2 > X1 and Y2 > Y1:
            boxes.append((X1, Y1, X2, Y2, 1))
            print(f"Box {len(boxes)} added: Without Helmet (Class 1)")

        redraw()

def redraw():
    global display
    display = shown.copy()

    for i, (x1, y1, x2, y2, class_id) in enumerate(boxes):
        sx1, sy1 = int(x1 * scale), int(y1 * scale)
        sx2, sy2 = int(x2 * scale), int(y2 * scale)

        cv2.rectangle(display, (sx1, sy1), (sx2, sy2), (0, 0, 255), 2)
        cv2.putText(display, f"Without Helmet {i+1}",
                    (sx1, max(20, sy1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow("Helmet Annotation - Draw head boxes | U=Undo | S=Save | Q=Quit", display)

cv2.namedWindow("Helmet Annotation - Draw head boxes | U=Undo | S=Save | Q=Quit")
cv2.setMouseCallback("Helmet Annotation - Draw head boxes | U=Undo | S=Save | Q=Quit", mouse_callback)

redraw()

while True:
    key = cv2.waitKey(20) & 0xFF

    if key == ord("u") and boxes:
        boxes.pop()
        redraw()

    elif key == ord("s"):
        lines = []

        for x1, y1, x2, y2, class_id in boxes:
            xc = ((x1 + x2) / 2) / width
            yc = ((y1 + y2) / 2) / height
            bw = (x2 - x1) / width
            bh = (y2 - y1) / height

            lines.append(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        print(f"Saved {len(boxes)} boxes to: {label_path}")
        break

    elif key == ord("q"):
        print("Exited without saving.")
        break

cv2.destroyAllWindows()

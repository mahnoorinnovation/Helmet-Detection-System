# Motorcycle Helmet Detection System

An object detection project built using **YOLOv8** to identify motorcycle riders wearing helmets and riders without helmets.

## 1. Project Overview

Road safety is an important concern for motorcycle riders. This project uses computer vision and deep learning to detect helmet usage in images.

The model detects two classes:

- **With Helmet**
- **Without Helmet**

## 2. Technologies Used

- Python
- YOLOv8 (Ultralytics)
- PyTorch
- OpenCV
- NumPy
- Pandas
- Matplotlib
- PyYAML
- Google Colab (GPU training)

## 3. Dataset

The final training dataset contains:

| Detail | Value |
|---|---:|
| Total images | 200 |
| Training images | 140 |
| Validation images | 40 |
| Testing images | 20 |
| Classes | 2 |
| Valid bounding boxes | 857 |
| Train/Validation/Test split | 70/20/10 |

### Classes

- Class 0: With Helmet
- Class 1: Without Helmet

The final dataset was organized in YOLO format with separate image and label folders.

**Dataset location:** Google Drive `cv_project/dataset/final_dataset/`

Dataset sharing link: Add your accessible Google Drive link here.

## 4. Model Training

The project uses the YOLOv8n pretrained model.

| Parameter | Configuration |
|---|---|
| Model | YOLOv8n |
| Epochs | 30 |
| Image size | 640 |
| Batch size | 8 |
| GPU | NVIDIA Tesla T4 (15 GB) |
| Training time | 4.27 minutes |
| Ultralytics | 8.4.211 |

The best trained model is saved as `best.pt`.

## 5. Model Evaluation

### Validation Results

| Metric | Score |
|---|---:|
| Precision | 0.549 |
| Recall | 0.528 |
| mAP@0.5 | 0.513 |
| mAP@0.5:0.95 | 0.219 |

### Test Results

| Metric | Score |
|---|---:|
| Precision | 0.698 |
| Recall | 0.596 |
| mAP@0.5 | 0.650 |
| mAP@0.5:0.95 | 0.277 |

The test evaluation used 20 images containing 89 ground-truth instances.

## 6. Inference

The trained model was tested on 20 test images. Predictions include:

- Bounding boxes
- Predicted class labels
- Confidence scores

Prediction images are available in `outputs/predictions/`.

## 7. Error Analysis

The project includes an error-analysis process to examine:

- False positives
- False negatives
- Wrong-class predictions
- Weak or incorrect detections

A selected review table is available at:

`outputs/error_analysis/manual_review_table.csv`

The report documents the selected review records and their limitations.

## 8. Project Structure

```text
Helmet-Detection-System/
│
├── notebooks/
│   └── training_notebook.ipynb
│
├── models/
│   └── best.pt
│
├── outputs/
│   ├── predictions/
│   └── error_analysis/
│
├── report/
│   └── final_report.pdf
│
├── dataset/
│   ├── images/
│   ├── labels/
│   └── data.yaml
│
├── requirements.txt
└── README.md
```

The dataset folder is excluded from Git tracking. Use the shared Google Drive dataset link for access.

## 9. Installation

Clone the repository:

```bash
git clone https://github.com/mahnoorinnovation/Helmet-Detection-System.git
cd Helmet-Detection-System
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 10. Future Improvements

- Increase dataset size and diversity.
- Improve detection under poor lighting and blurry conditions.
- Improve performance on small or distant riders.
- Evaluate additional model configurations.
- Develop a real-time video detection demo.

## 11. Final Report

The complete project report is available at:

`report/final_report.pdf`

## 12. Repository

GitHub: https://github.com/mahnoorinnovation/Helmet-Detection-System

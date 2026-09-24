# Traffic Sign Recognition using YOLOv8

Real-time traffic sign detection using **YOLOv8s, Python, and OpenCV** with a custom 18-class traffic sign dataset.

## Overview

This project implements a real-time traffic sign detection system using the **YOLOv8s object detection model**.

The trained model detects 18 different traffic signs and traffic-light classes from camera input. The final system was developed and tested on an **Apple Silicon M2 MacBook Air** using Python, PyTorch, Ultralytics YOLO, and OpenCV.

The project focuses on training a YOLO object detection model and using the trained model for real-time webcam detection.

---

## Detected Classes

The final trained model contains the following 18 classes:

| ID | Class                    |
| -: | ------------------------ |
|  0 | `trespass_sign`          |
|  1 | `no_entry`               |
|  2 | `straight_or_left_only`  |
|  3 | `straight_or_right_only` |
|  4 | `left_only_ahead`        |
|  5 | `20_speed_limit_end`     |
|  6 | `30_speed_limit`         |
|  7 | `20_speed_limit`         |
|  8 | `right_only_ahead`       |
|  9 | `no_right_turn`          |
| 10 | `no_left_turn`           |
| 11 | `stop`                   |
| 12 | `no_parking`             |
| 13 | `park`                   |
| 14 | `bus_stop`               |
| 15 | `red_light`              |
| 16 | `yellow_light`           |
| 17 | `green_light`            |

---

## Technologies Used

* Python
* YOLOv8s
* Ultralytics
* PyTorch
* OpenCV
* NumPy
* Apple MPS acceleration

---

## Project Structure

```text
traffic-sign-recognition-yolo/
│
├── README.md
├── requirements.txt
├── detect.py
├── run_detection.py
├── generate_dataset.py
├── obj.names
│
└── dataset/
    └── data.yaml
```

The dataset images, labels, training outputs, virtual environment, and trained model weights are intentionally excluded from the GitHub repository.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/traffic-sign-recognition-yolo.git
cd traffic-sign-recognition-yolo
```

Create and activate a virtual environment:

### macOS / Linux

```bash
python3 -m venv yolo_env
source yolo_env/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Running Real-Time Detection

The final webcam detection program is:

```bash
python3 run_detection.py
```

The program:

1. Opens the computer's webcam.
2. Captures video frames.
3. Runs the trained YOLOv8s model on each frame.
4. Detects traffic signs and traffic lights.
5. Displays the detected class names and confidence scores.

Press:

```text
Q
```

or

```text
Esc
```

to close the detection window.

---

## Detection Configuration

The final detection script uses:

```text
Model: best_named.pt
Confidence threshold: 0.40
Camera resolution: 1280 × 720
```

The model path used by `run_detection.py` is:

```text
dataset/runs/detect/train-2/weights/best_named.pt
```

The trained model weights are not included in this repository.

---

## YOLO Detection Pipeline

The detection workflow is:

```text
Webcam Frame
     │
     ▼
Image Preprocessing
     │
     ▼
YOLOv8s Model
     │
     ▼
Object Detection
     │
     ├── Bounding Box
     ├── Class
     └── Confidence
     │
     ▼
Confidence Filtering
     │
     ▼
Detection Results
     │
     ▼
Display on Webcam
```

For each frame, YOLO predicts the location, class, and confidence of detected objects. Predictions below the configured confidence threshold are filtered out before the results are displayed.

---

## Model Training

The project uses **YOLOv8s** as the base object detection model.

The training configuration and dataset preparation were developed separately from the final real-time detection workflow.

The repository does not include:

* Training images
* Training labels
* Training run outputs
* Model checkpoints
* Virtual environment files

This keeps the repository lightweight while preserving the source code and configuration used for the project.

---

## Dataset

The project uses a custom traffic sign dataset in **YOLO format**.

The dataset contains the images and corresponding YOLO-format annotation files required for model training.

The complete dataset is not included in this repository.

The dataset configuration is provided in:

```text
dataset/data.yaml
```

The dataset can therefore be kept separately from the source-code repository.

---

## Hardware

The project was developed and tested on:

```text
MacBook Air
Apple M2
```

Apple's **Metal Performance Shaders (MPS)** backend was used for compatible PyTorch model inference.

---

## Future Improvements

Possible improvements include:

* Increasing the size and diversity of the training dataset
* Adding additional traffic-sign classes
* Improving detection under different lighting conditions
* Testing the model on a standardized traffic-sign benchmark
* Evaluating precision, recall, and mAP
* Testing larger YOLO model variants
* Optimizing inference speed
* Deploying the detector on an embedded or edge device
* Adding image/video file detection in addition to webcam detection

---

## Notes

The dataset and trained model weights are excluded from this repository because of their size.

The repository contains the source code and configuration used for the project. The exact trained model and dataset are maintained separately.

---

## License

No license has been specified for this project yet.

# Face Detection

We are going to implement Face Detection using facexlib library. This will help us to detect faces in the images.

## Installation

CPU installation:
```bash
pip install facexlib
```

GPU CUDA 12.8 Installation (~10x faster):
```bash
pip install facexlib
pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/test/cu128
```

## Usage

There is a button in the UI in ImagesManagement tab to run face detection on the images of a character.
When clicked it will detect the position of the faces bboxes in the images and store them in the character.yaml file.
These bboxes can later be visualized in the UI to see where the faces are located in the images.

## Implementation

New class `FaceDetection` is created in `src\detections\face_detection.py` file.
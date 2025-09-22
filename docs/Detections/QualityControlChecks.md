# Quality Control Checks

This documents describes the quality control checks implemented in the image final stage pipeline 7, or manually launched on any stage via button in the UI in images management tab.

## Overview

Quality checks are recorded in the `quality_checks.yaml` file located in the character's main directory (e.g., `characters/{characterName}/quality_checks.yaml`). This file contains a list of checks performed on the images, along with their results.
Each check can assign tags to the images, which can be used for filtering and management.

## Available Quality Checks

- **Duplicate Check**: Identifies duplicate images based on file hashes. Tags duplicates with `duplicate`.
- **Similarity Check**: Compares images to find visually similar ones using perceptual hashing. Tags similar images with `similar`.
- **Resolution Check**: 
  - On any stage checks if the image is at least 230x230 pixels. Tags images below this threshold with `too_small`.
  - After stage 4 including stage 4 each image should have a resolution of exactly 1024x1024. Tags images below this threshold with `wrong_resolution`.
- **Aspect Ratio Check**: Ensures images have a 1:1 aspect ratio. Tags images that do not meet this criterion with `wrong_aspect_ratio`.
- **Corruption Check**: Verifies that images are not corrupted and can be opened successfully. Tags corrupted images with `corrupted`.
- **Blurriness Check**: Detects blurry images using the variance of the Laplacian method. Tags blurry images with `blurry`.
- **Single Color Check**: Identifies images that are predominantly a single color. Tags such images with `single_color`.
- **Border Check**: Detects images with borders or frames. Tags such images with `bordered`.
- **Image Count Check**: This will check only on stage 7 if the images are ideally 166. If they are less than 30 it will tag them with `too_few_images`, if they are more than 260 it will tag them with `too_many_images`.
- **Face Similarity Check**: Compares faces in images to ensure they are similar with character face image. If the similarity is below a certain threshold, tags the image with `face_dissimilar`.
- **Person detection**: Uses a combination of YOLO and facedetection to detect if there is a person in the image. If no person is detected, tags the image with `no_person_detected`.
- **Proper Format Check**: Ensures images are in JPEG or PNG format. Tags images in other formats with `wrong_format`. This check can be performed after and including Stage 3.
- **Watermark Check**: Detects watermarks in images using a pre-trained model. Tags images with detected watermarks with `watermarked`. This check can be performed after and including Stage 4.
- **Text Overlay Check**: Identifies images with text overlays using OCR. Tags such images with `text_overlay`. This check can be performed after and including Stage 4.

## Filter & Order by Tags
In the image management tab, you can filter images based on the tags assigned by the quality checks. This allows you to easily identify and manage images that may require attention.
You can also order images by the number of tags they have, which can help prioritize images that may need the most attention.
You can group the results by tags, so you can see all images with a specific tag together.

## Code Implementation
The quality checks are implemented in the `src/quality_checks` directory. Each check is defined in its own module, and the main logic for running the checks and recording results is in `quality_checks_manager.py`.
This manager handles loading images, executing each check, and updating the `quality_checks.yaml` file with the results. It can do partial run for specific checks or a full run for all checks. Depends on user choice in the UI.
Batch processing is implemented to handle large sets of images efficiently. Each check is executed by predefined thread numbers to optimize performance. And each check waits for the previous check to finish before starting.
You can also disable Quality Control Checks for each character individually via Character Tab in the UI.

## How are tags displayed in the UI
In the Images Management Tab, each image displays its associated tags inside the image thumbnail in small font in a round rectangular box pill. Tags are shown as colored labels, making it easy to identify the quality issues at a glance.
Tags are color-coded based on their severity:
- **Red**: Critical issues (e.g., `corrupted`, `too_small`, `wrong_resolution`, `too_few_images`, `too_many_images`, `no_person_detected`)
- **Orange**: Major issues (e.g., `blurry`, `single_color`, `bordered`, `face_dissimilar`, `watermarked`, `text_overlay`)
- **Yellow**: Minor issues (e.g., `duplicate`, `similar`, `wrong_aspect_ratio`, `wrong_format`)
- **Green**: Informational tags (e.g., `high_quality`)

## Quality Checks Configuration
We have a configuration file located at `config/quality_checks_config.yaml` where you can set parameters for each quality check, such as thresholds for blurriness, similarity, and face detection. You can also enable or disable specific checks globally or per character.
Here is an example configuration:

```yaml
duplicate_check:
    enabled: true
    hash_algorithm: md5
    threads: 4
similarity_check:
    enabled: true
    threshold: 10
    threads: 4
resolution_check:
    enabled: true
    min_width: 230
    min_height: 230
    exact_width: 1024
    exact_height: 1024
    threads: 4
aspect_ratio_check:
    enabled: true
    ratio: 1.0
    threads: 4
corruption_check:
    enabled: true
    threads: 4
blurriness_check:
    enabled: true
    threshold: 100.0
    threads: 4
single_color_check:
    enabled: true
    color_threshold: 0.9
    threads: 4
border_check:
    enabled: true
    border_threshold: 0.1
    threads: 4
image_count_check:
    enabled: true
    min_count: 30
    max_count: 260
    threads: 4
face_similarity_check:
    enabled: true
    threshold: 0.6
    threads: 4
person_detection_check:
    enabled: true
    yolo_model: yolov5s
    face_detection_model: mtcnn
    threads: 4
proper_format_check:
    enabled: true
    allowed_formats:
        - jpg
        - jpeg
        - png
    threads: 4
watermark_check:
    enabled: true
    model_path: models/watermark_detector.pth
    threshold: 0.5
    threads: 4
text_overlay_check:
    enabled: true
    ocr_engine: tesseract
    threshold: 0.5
    threads: 4
```

We have a tool to work with this configuration file in the UI in the Settings Tab under Quality Checks section. You can modify the parameters and save the configuration directly from the UI. If the configuration file is empty create this one with default values.
git stat
## Running Quality Checks
You can run quality checks manually from the UI by clicking the "Run Quality Checks" button in the Images Management Tab. You can choose to run all checks or select specific checks to run.
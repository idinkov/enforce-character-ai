# Border Detection

We are going to implement Border Detection using OpenCV morphological operations. This will help us automatically detect and crop unnecessary borders from images.

## Installation

The border detection functionality uses OpenCV and NumPy which are already included in the requirements:

```bash
pip install opencv-python>=4.5.0
pip install numpy>=1.21.0
pip install pillow>=9.0.0
```

## Usage

There is a button in the UI in ImagesManagement tab to run border detection on the images of a character.
When clicked it will automatically detect image borders using Otsu thresholding and morphological operations, then crop the images to remove unnecessary borders.
The processed images will have borders removed while preserving the main content.

## Features

- **Automatic Border Detection**: Uses Otsu thresholding to automatically detect borders
- **Smart Cropping**: Compares normal and inverted image processing to choose the most aggressive cropping
- **Morphological Operations**: Uses opening and closing operations to refine border detection
- **Format Support**: Works with PIL Images, NumPy arrays, and image file paths
- **Batch Processing**: Can process multiple images in a directory
- **Size Optimization**: Only saves images that were actually cropped (size changed)

## Implementation

New class `BorderDetection` is created in `src\detections\border_detection.py` file.

### Key Methods

- `detect_and_crop_borders()`: Main method for detecting and cropping borders from images
- `process_image_file()`: Process a single image file with border detection
- `_detect_borders()`: Internal method using Otsu thresholding and morphological operations
- `_crop_to_content()`: Internal method for cropping to detected content area

### Algorithm Details

1. **Preprocessing**: Converts input to OpenCV BGR format
2. **Border Detection**: Uses Otsu thresholding to separate foreground from background
3. **Morphological Refinement**: Applies closing and opening operations with 9x9 kernel
4. **Dual Processing**: Processes both normal and inverted images
5. **Smart Selection**: Chooses the result with smaller size (more aggressive cropping)
6. **Content Preservation**: Ensures main content is preserved while removing borders

### Example Usage

```python
from src.detections.border_detection import BorderDetection

# Initialize border detection
border_detector = BorderDetection(log_callback=print)

# Process single image
success = border_detector.process_image_file("input.jpg", "output.jpg")

# Or get cropped array directly
cropped_img = border_detector.detect_and_crop_borders("input.jpg")
```

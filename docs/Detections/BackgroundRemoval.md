# Background Removal

We are going to implement Background Removal using U2Net model with ONNX runtime. This will help us automatically remove backgrounds from images and extract the main subject.

## Installation

CPU installation:
```bash
pip install onnxruntime>=1.12.0
pip install opencv-python>=4.5.0
pip install numpy>=1.21.0
pip install pillow>=9.0.0
```

GPU CUDA Installation (~10x faster):
```bash
pip install onnxruntime-gpu>=1.12.0
pip install opencv-python>=4.5.0
pip install numpy>=1.21.0
pip install pillow>=9.0.0
```

## Model Requirements

The background removal requires the U2Net model file:
- **File**: `u2net_human_seg.onnx`
- **Location**: `models/u2net_human_seg.onnx`
- **Purpose**: Pre-trained human segmentation model for accurate background removal

## Usage

There is a button in the UI in ImagesManagement tab to run background removal on the images of a character.
When clicked it will use the U2Net model to detect the main subject (person) in the images and remove the background.
The result can be saved as RGBA images with transparent backgrounds or cropped images focusing on the detected subject.

## Features

- **AI-Powered Segmentation**: Uses U2Net neural network for accurate human detection
- **GPU Acceleration**: Automatic GPU detection with CUDA support for faster processing
- **Dual Output Modes**: 
  - Background removal (RGBA with transparency)
  - Smart cropping (original image cropped to subject)
- **Adjustable Thresholds**: Customizable foreground/background detection sensitivity
- **Post-Processing**: Morphological operations for mask refinement
- **Bounding Box Detection**: Automatic calculation of subject boundaries
- **Batch Processing**: Can process multiple images efficiently

## Implementation

New class `BackgroundRemoval` is created in `src\detections\background_removal.py` file.

### Key Methods

- `remove_background()`: Main method for background removal with transparency
- `crop_image()`: Crop original image to detected subject area
- `process_image_file_with_crop()`: Process single file with both removal and cropping
- `_preprocess_image()`: Prepare image for U2Net model (320x320 input)
- `_postprocess_mask()`: Refine segmentation mask with thresholding and morphology

### Algorithm Details

1. **Model Loading**: Loads U2Net ONNX model with GPU acceleration if available
2. **Preprocessing**: Resizes input to 320x320 pixels and normalizes values
3. **Inference**: Runs U2Net model to generate segmentation mask
4. **Postprocessing**: Applies thresholding and morphological operations
5. **Mask Refinement**: Uses foreground/background thresholds for binary classification
6. **Result Generation**: Creates RGBA output or crops original image

### Parameters

- **foreground_threshold** (240): Pixels above this value are considered foreground
- **background_threshold** (15): Pixels below this value are considered background  
- **apply_post_processing** (True): Whether to apply morphological refinement
- **padding** (10): Pixels to add around detected area when cropping

### Example Usage



### Performance Notes

- **GPU vs CPU**: GPU acceleration provides ~10x speed improvement
- **Memory Usage**: 2GB GPU memory limit configured for stability
- **Batch Processing**: More efficient for multiple images due to model loading overhead

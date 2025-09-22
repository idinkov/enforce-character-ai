"""
Image viewer GUI component for displaying and manipulating individual images.
"""
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from PIL import Image, ImageTk, ImageDraw, ImageFont
import os
import threading
from ..detections.face_detection import FaceDetection


class ImageViewerGui:
    """Standalone image viewer window for displaying individual images."""

    # Class variable to track open viewers for smart positioning
    _open_viewers = []
    _next_offset = 0

    def __init__(self, parent, image_path: Path, navigation_images: list[Path] = None):
        self.image_path = Path(image_path)
        self.parent = parent
        self.is_fullscreen = False

        # Navigation support for arrow keys
        self.navigation_images = navigation_images or []
        self.current_index = 0
        if self.navigation_images:
            try:
                self.current_index = self.navigation_images.index(self.image_path)
            except ValueError:
                # If current image is not in navigation list, add it and use it as current
                self.navigation_images.append(self.image_path)
                self.current_index = len(self.navigation_images) - 1

        # Face detection related attributes
        self.face_detection = None
        self.detected_faces = []
        self.show_face_boxes = False
        self.current_character_path = None
        self.face_similarity_scores = []

        # YOLOv8n-seg segmentation related attributes
        self.detected_segments = []
        self.show_segmentation_masks = False
        self.segmentation_opacity = 0.5
        self.selected_person_idx = -1

        # Create the viewer window
        self.window = tk.Toplevel(parent)

        # Set initial title with navigation info if available
        if self.navigation_images and len(self.navigation_images) > 1:
            current_num = self.current_index + 1
            total_num = len(self.navigation_images)
            self.window.title(f"Image Viewer - {self.image_path.name} ({current_num}/{total_num})")
        else:
            self.window.title(f"Image Viewer - {self.image_path.name}")

        self.window.geometry("1000x750")  # Made slightly larger for extra controls

        # Make window appear as separate taskbar entry
        self.window.wm_attributes("-toolwindow", False)  # Show in taskbar
        self.window.transient()  # Remove parent dependency for taskbar

        # Set window icon (inherit from parent if available)
        try:
            if hasattr(parent, 'iconbitmap'):
                self.window.iconbitmap(parent.iconbitmap())
        except:
            pass

        # Smart positioning to avoid overlapping windows
        self._position_window()

        # Track this viewer
        ImageViewerGui._open_viewers.append(self)

        self._create_widgets()
        self._load_image()

        # Handle window close
        self.window.protocol("WM_DELETE_WINDOW", self._on_close)

        # Make window resizable
        self.window.minsize(500, 400)

        # Bind fullscreen toggle
        self.window.bind("<F11>", self._toggle_fullscreen)
        self.window.bind("<Escape>", self._exit_fullscreen)

        # Bind window resize event to re-center image
        self.window.bind("<Configure>", self._on_window_resize)

        # Focus the new window
        self.window.focus_set()
        self.window.lift()

    def _position_window(self):
        """Position the window intelligently to avoid overlapping with other viewers."""
        self.window.update_idletasks()

        # Base position (centered)
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        window_width = 1000
        window_height = 750

        base_x = (screen_width // 2) - (window_width // 2)
        base_y = (screen_height // 2) - (window_height // 2)

        # Apply offset for multiple windows
        offset = ImageViewerGui._next_offset * 30
        x = base_x + offset
        y = base_y + offset

        # Wrap around if we go off screen
        if x + window_width > screen_width - 50:
            ImageViewerGui._next_offset = 0
            offset = 0
            x = base_x
            y = base_y

        if y + window_height > screen_height - 50:
            ImageViewerGui._next_offset = 0
            offset = 0
            x = base_x
            y = base_y

        self.window.geometry(f"{window_width}x{window_height}+{x}+{y}")
        ImageViewerGui._next_offset += 1

    def _create_widgets(self):
        """Create the UI widgets for the image viewer."""
        # Top frame for image info and controls
        top_frame = ttk.Frame(self.window)
        top_frame.pack(fill="x", padx=10, pady=5)

        # Image path label (make it selectable)
        path_frame = ttk.Frame(top_frame)
        path_frame.pack(side="left", fill="x", expand=True)

        ttk.Label(path_frame, text="Path:", font=("Arial", 9, "bold")).pack(side="left")
        self.path_var = tk.StringVar(value=str(self.image_path))
        path_entry = ttk.Entry(path_frame, textvariable=self.path_var, state="readonly", font=("Arial", 9))
        path_entry.pack(side="left", fill="x", expand=True, padx=(5, 0))

        # Control buttons frame
        btn_frame = ttk.Frame(top_frame)
        btn_frame.pack(side="right", padx=(10, 0))

        ttk.Button(btn_frame, text="Fit to Window", command=self._fit_to_window).pack(side="left", padx=2)
        ttk.Button(btn_frame, text="Actual Size", command=self._actual_size).pack(side="left", padx=2)
        ttk.Button(btn_frame, text="Fullscreen (F11)", command=self._toggle_fullscreen).pack(side="left", padx=2)
        ttk.Button(btn_frame, text="Open in Explorer", command=self._open_in_explorer).pack(side="left", padx=2)
        ttk.Button(btn_frame, text="Close", command=self._on_close).pack(side="left", padx=2)

        # Face detection controls frame
        face_frame = ttk.LabelFrame(self.window, text="Face Detection & Similarity", padding=5)
        face_frame.pack(fill="x", padx=10, pady=5)

        # Character selection
        char_row = ttk.Frame(face_frame)
        char_row.pack(fill="x", pady=2)

        ttk.Label(char_row, text="Character:").pack(side="left")
        self.character_var = tk.StringVar()
        self.character_combo = ttk.Combobox(char_row, textvariable=self.character_var, state="readonly", width=25)
        self.character_combo.pack(side="left", padx=(5, 10))
        self._populate_characters()

        # Control buttons
        ttk.Button(char_row, text="Run Face Detection", command=self._run_face_detection).pack(side="left", padx=2)

        # Toggle face boxes
        self.show_boxes_var = tk.BooleanVar()
        ttk.Checkbutton(char_row, text="Show Face Boxes", variable=self.show_boxes_var,
                       command=self._toggle_face_boxes).pack(side="left", padx=(10, 2))

        # Status label
        self.face_status_label = ttk.Label(face_frame, text="Select a character and click 'Run Face Detection' to start",
                                         font=("Arial", 9))
        self.face_status_label.pack(anchor="w", pady=2)

        # YOLOv8n-seg Person Segmentation controls frame
        seg_frame = ttk.LabelFrame(self.window, text="YOLOv8n-seg Person Segmentation", padding=5)
        seg_frame.pack(fill="x", padx=10, pady=5)

        # Segmentation controls row
        seg_row = ttk.Frame(seg_frame)
        seg_row.pack(fill="x", pady=2)

        # Run segmentation button
        ttk.Button(seg_row, text="Run Person Segmentation", command=self._run_person_segmentation).pack(side="left", padx=2)

        # Toggle segmentation masks
        self.show_masks_var = tk.BooleanVar()
        ttk.Checkbutton(seg_row, text="Show Segmentation Masks", variable=self.show_masks_var,
                       command=self._toggle_segmentation_masks).pack(side="left", padx=(10, 2))

        # Opacity control
        ttk.Label(seg_row, text="Opacity:").pack(side="left", padx=(20, 5))
        self.opacity_var = tk.DoubleVar(value=0.5)
        opacity_scale = ttk.Scale(seg_row, from_=0.1, to=1.0, variable=self.opacity_var, orient="horizontal", length=100)
        opacity_scale.pack(side="left", padx=2)
        opacity_scale.bind("<Motion>", self._on_opacity_change)
        opacity_scale.bind("<ButtonRelease-1>", self._on_opacity_change)

        # Person selection dropdown
        ttk.Label(seg_row, text="Select Person:").pack(side="left", padx=(20, 5))
        self.person_var = tk.StringVar(value="All")
        self.person_combo = ttk.Combobox(seg_row, textvariable=self.person_var, state="readonly", width=15)
        self.person_combo['values'] = ["All"]
        self.person_combo.bind("<<ComboboxSelected>>", self._on_person_selection_change)
        self.person_combo.pack(side="left", padx=2)

        # Status label for segmentation
        self.seg_status_label = ttk.Label(seg_frame, text="Click 'Run Person Segmentation' to detect people in the image",
                                        font=("Arial", 9))
        self.seg_status_label.pack(anchor="w", pady=2)

        # Main frame for image display
        main_frame = ttk.Frame(self.window)
        main_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # Canvas with scrollbars for image display
        self.canvas = tk.Canvas(main_frame, bg="gray90", highlightthickness=0)

        # Scrollbars
        v_scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=self.canvas.yview)
        h_scrollbar = ttk.Scrollbar(main_frame, orient="horizontal", command=self.canvas.xview)

        self.canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

        # Pack scrollbars and canvas
        v_scrollbar.pack(side="right", fill="y")
        h_scrollbar.pack(side="bottom", fill="x")
        self.canvas.pack(side="left", fill="both", expand=True)

        # Bind mouse events for panning and zooming
        self.canvas.bind("<ButtonPress-1>", self._start_pan)
        self.canvas.bind("<B1-Motion>", self._pan)
        self.canvas.bind("<MouseWheel>", self._zoom)
        self.canvas.bind("<Button-4>", self._zoom)  # Linux scroll up
        self.canvas.bind("<Button-5>", self._zoom)  # Linux scroll down

        # Keyboard shortcuts
        self.window.bind("<Control-0>", lambda e: self._actual_size())
        self.window.bind("<Control-equal>", lambda e: self._zoom_in())
        self.window.bind("<Control-minus>", lambda e: self._zoom_out())
        self.window.bind("<Control-f>", lambda e: self._fit_to_window())

        # Arrow key navigation
        self.window.bind("<Left>", self._navigate_previous)
        self.window.bind("<Right>", self._navigate_next)
        self.window.bind("<KeyPress-Left>", self._navigate_previous)
        self.window.bind("<KeyPress-Right>", self._navigate_next)

        self.window.focus_set()  # Enable keyboard events

        # Bottom frame for image info
        bottom_frame = ttk.Frame(self.window)
        bottom_frame.pack(fill="x", padx=10, pady=5)

        self.info_label = ttk.Label(bottom_frame, text="", font=("Arial", 9))
        self.info_label.pack(side="left")

        # Scale factor display
        self.scale_label = ttk.Label(bottom_frame, text="Scale: 100%", font=("Arial", 9))
        self.scale_label.pack(side="right")

        # Initialize pan variables
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.scale_factor = 1.0

    def _populate_characters(self):
        """Populate the character selection combobox."""
        try:
            # Look for characters directory relative to the project root
            project_root = Path(__file__).parent.parent.parent
            characters_dir = project_root / "characters"

            if not characters_dir.exists():
                return

            characters = []
            for char_dir in characters_dir.iterdir():
                if char_dir.is_dir() and (char_dir / "character.yaml").exists():
                    characters.append(char_dir.name)

            characters.sort()
            self.character_combo['values'] = characters

            if characters:
                self.character_combo.set(characters[0])

        except Exception as e:
            print(f"Error populating characters: {e}")

    def _run_face_detection(self):
        """Run face detection and similarity comparison in a separate thread."""
        if not self.character_var.get():
            messagebox.showwarning("No Character", "Please select a character first.")
            return

        # Update status
        self.face_status_label.config(text="Running face detection...")
        self.window.update()

        # Run in separate thread to avoid blocking UI
        thread = threading.Thread(target=self._detect_faces_thread)
        thread.daemon = True
        thread.start()

    def _detect_faces_thread(self):
        """Thread function for face detection and similarity comparison."""
        try:
            # Get character path
            project_root = Path(__file__).parent.parent.parent
            character_path = project_root / "characters" / self.character_var.get()

            # Initialize face detection
            face_detector = FaceDetection(character_path, log_callback=self._log_face_detection)

            # Detect faces in the image
            self.detected_faces = face_detector.detect_faces_in_image(self.image_path)

            # Calculate similarity scores for each detected face
            self.face_similarity_scores = []
            for i, face in enumerate(self.detected_faces):
                similarity = face_detector.compare_face_with_reference(self.image_path, face)
                self.face_similarity_scores.append(similarity)

            # Update UI on main thread
            self.window.after(0, self._face_detection_completed)

        except Exception as e:
            self.window.after(0, lambda: self._face_detection_error(str(e)))

    def _log_face_detection(self, message):
        """Log callback for face detection."""
        self.window.after(0, lambda: self.face_status_label.config(text=message))

    def _face_detection_completed(self):
        """Called when face detection is completed."""
        if self.detected_faces:
            num_faces = len(self.detected_faces)
            if self.face_similarity_scores:
                max_similarity = max(self.face_similarity_scores)
                status_text = f"Found {num_faces} face(s). Highest similarity: {max_similarity:.3f}"
            else:
                status_text = f"Found {num_faces} face(s). No similarity scores available."

            self.face_status_label.config(text=status_text)

            # Automatically show face boxes
            self.show_boxes_var.set(True)
            self.show_face_boxes = True
            self._update_display()
        else:
            self.face_status_label.config(text="No faces detected in the image.")

    def _face_detection_error(self, error_msg):
        """Called when face detection encounters an error."""
        self.face_status_label.config(text=f"Error: {error_msg}")
        messagebox.showerror("Face Detection Error", f"Failed to detect faces: {error_msg}")

    def _toggle_face_boxes(self):
        """Toggle the display of face bounding boxes."""
        self.show_face_boxes = self.show_boxes_var.get()
        self._update_display()

    def _run_person_segmentation(self):
        """Run YOLOv8n-seg person segmentation in a separate thread."""
        # Update status
        self.seg_status_label.config(text="Running YOLOv8n-seg person segmentation...")
        self.window.update()

        # Run in separate thread to avoid blocking UI
        thread = threading.Thread(target=self._segment_persons_thread)
        thread.daemon = True
        thread.start()

    def _segment_persons_thread(self):
        """Thread function for YOLOv8n-seg person segmentation."""
        try:
            from ultralytics import YOLO
            from ..models import get_model_manager
            from ..services.gpu_service import get_gpu_service
            import cv2
            import numpy as np

            # Convert PIL to numpy array (RGB format)
            image_np = np.array(self.original_image)

            # Use the model manager to get the YOLOv8n-seg model path
            model_manager = get_model_manager()
            model_path = model_manager.get_model_path('yolov8n_seg')

            if not model_path or not model_path.exists():
                # Try to download the model if it's missing
                if not model_manager.is_model_available('yolov8n_seg'):
                    self.window.after(0, lambda: self.seg_status_label.config(text="Downloading YOLOv8n-seg model..."))
                    success = model_manager.download_model('yolov8n_seg')
                    if success:
                        model_path = model_manager.get_model_path('yolov8n_seg')
                    else:
                        self.window.after(0, lambda: self._segmentation_error("Failed to download YOLOv8n-seg model"))
                        return
                else:
                    self.window.after(0, lambda: self._segmentation_error("YOLOv8n-seg model not found"))
                    return

            # Load YOLOv8n-seg model and force CPU usage to avoid CUDA NMS issues
            model = YOLO(str(model_path))

            # Force CPU usage to avoid CUDA compatibility issues
            device = 'cpu'
            self.window.after(0, lambda: self.seg_status_label.config(text="Using CPU for YOLOv8n-seg (avoiding CUDA issues)"))
            model.to(device)

            # Perform inference
            self.window.after(0, lambda: self.seg_status_label.config(text="Running inference..."))
            results = model(image_np, verbose=False, device=device)

            # Extract person segmentations (class 0 = person in COCO dataset)
            self.detected_segments = []
            for result in results:
                if result.masks is not None and result.boxes is not None:
                    for mask, box in zip(result.masks.data, result.boxes):
                        # Check if detection is a person (class 0)
                        if int(box.cls[0]) == 0:  # Person class
                            confidence = float(box.conf[0])

                            # Only consider detections with reasonable confidence
                            if confidence > 0.3:
                                # Get segmentation mask as numpy array
                                mask_np = mask.cpu().numpy()

                                # Get bounding box coordinates
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                                # Resize mask to match original image dimensions if needed
                                if mask_np.shape != image_np.shape[:2]:
                                    mask_resized = cv2.resize(mask_np, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_NEAREST)
                                else:
                                    mask_resized = mask_np

                                self.detected_segments.append({
                                    'mask': mask_resized,
                                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                                    'confidence': confidence
                                })

            # Update UI on main thread
            self.window.after(0, self._segmentation_completed)

        except Exception as e:
            # Capture the error message in a variable to avoid scoping issues
            error_msg = str(e)
            self.window.after(0, lambda: self._segmentation_error(error_msg))

    def _segmentation_completed(self):
        """Called when person segmentation is completed."""
        if self.detected_segments:
            num_persons = len(self.detected_segments)
            avg_confidence = sum(seg['confidence'] for seg in self.detected_segments) / num_persons
            status_text = f"Found {num_persons} person(s). Average confidence: {avg_confidence:.3f}"
            self.seg_status_label.config(text=status_text)

            # Update person selection dropdown
            person_options = ["All"] + [f"Person {i+1} ({seg['confidence']:.3f})" for i, seg in enumerate(self.detected_segments)]
            self.person_combo['values'] = person_options
            self.person_combo.set("All")

            # Automatically show segmentation masks
            self.show_masks_var.set(True)
            self.show_segmentation_masks = True
            self._update_display()
        else:
            self.seg_status_label.config(text="No persons detected in the image.")

    def _segmentation_error(self, error_msg):
        """Called when person segmentation encounters an error."""
        self.seg_status_label.config(text=f"Error: {error_msg}")
        messagebox.showerror("Person Segmentation Error", f"Failed to segment persons: {error_msg}")

    def _toggle_segmentation_masks(self):
        """Toggle the display of segmentation masks."""
        self.show_segmentation_masks = self.show_masks_var.get()
        self._update_display()

    def _on_opacity_change(self, event=None):
        """Handle opacity slider change."""
        self.segmentation_opacity = self.opacity_var.get()
        if self.show_segmentation_masks:
            self._update_display()

    def _on_person_selection_change(self, event=None):
        """Handle person selection change."""
        selection = self.person_var.get()
        if selection == "All":
            self.selected_person_idx = -1
        else:
            # Extract person index from selection like "Person 1 (0.950)"
            try:
                self.selected_person_idx = int(selection.split()[1]) - 1
            except:
                self.selected_person_idx = -1

        if self.show_segmentation_masks:
            self._update_display()

    def _load_image(self):
        """Load and display the image."""
        try:
            if not self.image_path.exists():
                messagebox.showerror("Error", f"Image file not found: {self.image_path}")
                self._on_close()
                return

            # Load the image
            self.original_image = Image.open(self.image_path)
            self.current_image = self.original_image.copy()

            # Get image info
            width, height = self.original_image.size
            file_size = self.image_path.stat().st_size
            file_size_mb = file_size / (1024 * 1024)

            info_text = f"Size: {width}x{height} | File size: {file_size_mb:.2f} MB | Format: {self.original_image.format}"
            self.info_label.config(text=info_text)

            # Display the image initially at actual size
            self._update_display()

            # After the canvas is properly sized, fit the image to window
            self.window.after_idle(self._fit_to_window)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
            self._on_close()

    def _update_display(self):
        """Update the image display with current scale, face boxes, and segmentation masks."""
        try:
            # Start with the original image
            display_image = self.original_image.copy()

            # Draw segmentation masks if enabled (draw before face boxes so boxes appear on top)
            if self.show_segmentation_masks and self.detected_segments:
                display_image = self._draw_segmentation_masks_on_image(display_image)

            # Draw face boxes if enabled
            if self.show_face_boxes and self.detected_faces:
                display_image = self._draw_face_boxes_on_image(display_image)

            # Calculate new size based on scale factor
            original_width, original_height = display_image.size
            new_width = int(original_width * self.scale_factor)
            new_height = int(original_height * self.scale_factor)

            # Resize image if needed
            if self.scale_factor != 1.0:
                resample = Image.Resampling.LANCZOS
                self.current_image = display_image.resize((new_width, new_height), resample)
            else:
                self.current_image = display_image

            # Convert to PhotoImage
            self.photo = ImageTk.PhotoImage(self.current_image)

            # Clear canvas and add image
            self.canvas.delete("all")
            self.image_item = self.canvas.create_image(0, 0, anchor="nw", image=self.photo)

            # Always center the image after updating display
            self._center_image()

            # Update scale label
            scale_percent = int(self.scale_factor * 100)
            self.scale_label.config(text=f"Scale: {scale_percent}%")

        except Exception as e:
            print(f"Error updating display: {e}")

    def _draw_segmentation_masks_on_image(self, image):
        """Draw YOLOv8n-seg segmentation masks on the image with adjustable opacity."""
        try:
            import numpy as np

            # Create a copy to draw on
            img_with_masks = image.copy()
            img_array = np.array(img_with_masks)

            # Define colors for different persons (bright, distinguishable colors)
            colors = [
                (255, 0, 0),    # Red
                (0, 255, 0),    # Green
                (0, 0, 255),    # Blue
                (255, 255, 0),  # Yellow
                (255, 0, 255),  # Magenta
                (0, 255, 255),  # Cyan
                (255, 128, 0),  # Orange
                (128, 0, 255),  # Purple
                (255, 192, 203), # Pink
                (0, 128, 0),    # Dark Green
            ]

            # Determine which segments to draw
            segments_to_draw = []
            if self.selected_person_idx == -1:
                # Draw all segments
                segments_to_draw = list(enumerate(self.detected_segments))
            else:
                # Draw only selected person
                if 0 <= self.selected_person_idx < len(self.detected_segments):
                    segments_to_draw = [(self.selected_person_idx, self.detected_segments[self.selected_person_idx])]

            # Draw each segment
            for i, segment in segments_to_draw:
                mask = segment['mask']
                confidence = segment['confidence']

                # Get color for this person
                color = colors[i % len(colors)]

                # Create colored overlay where mask is positive
                mask_bool = mask > 0.5

                # Apply color with opacity
                overlay = img_array.copy().astype(np.float32)
                overlay[mask_bool] = overlay[mask_bool] * (1 - self.segmentation_opacity) + np.array(color) * self.segmentation_opacity

                img_array = overlay.astype(np.uint8)

            # Draw bounding boxes and confidence scores for segments
            img_with_overlays = Image.fromarray(img_array)
            draw = ImageDraw.Draw(img_with_overlays)

            # Try to load a font, fall back to default if not available
            try:
                font = ImageFont.truetype("arial.ttf", 14)
            except:
                try:
                    font = ImageFont.load_default()
                except:
                    font = None

            # Draw bounding boxes and labels for displayed segments
            for i, segment in segments_to_draw:
                bbox = segment['bbox']
                confidence = segment['confidence']
                x1, y1, x2, y2 = bbox

                # Get color for this person
                color = colors[i % len(colors)]
                color_str = f"rgb({color[0]},{color[1]},{color[2]})"

                # Draw bounding box
                draw.rectangle([x1, y1, x2, y2], outline=color_str, width=2)

                # Draw label with person number and confidence
                label_text = f"Person {i+1}: {confidence:.3f}"
                text_y = max(0, y1 - 20)

                # Draw text background
                if font:
                    bbox_text = draw.textbbox((x1, text_y), label_text, font=font)
                    draw.rectangle(bbox_text, fill=color_str, outline=color_str)
                    draw.text((x1, text_y), label_text, fill="white", font=font)
                else:
                    draw.text((x1, text_y), label_text, fill=color_str)

            return img_with_overlays

        except Exception as e:
            print(f"Error drawing segmentation masks: {e}")
            return image

    def _draw_face_boxes_on_image(self, image):
        """Draw bounding boxes and similarity scores on the image."""
        try:
            # Create a copy to draw on
            img_with_boxes = image.copy()
            draw = ImageDraw.Draw(img_with_boxes)

            # Try to load a font, fall back to default if not available
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                try:
                    font = ImageFont.load_default()
                except:
                    font = None

            # Find the face with highest similarity score
            highest_similarity_idx = -1
            if self.face_similarity_scores:
                highest_similarity_idx = self.face_similarity_scores.index(max(self.face_similarity_scores))

            # Draw each detected face
            for i, face in enumerate(self.detected_faces):
                x, y, width, height = face['x'], face['y'], face['width'], face['height']

                # Choose color based on similarity score
                if i == highest_similarity_idx and len(self.face_similarity_scores) > 1:
                    # Highest similarity face in green
                    box_color = "lime"
                    text_color = "lime"
                else:
                    # Other faces in red
                    box_color = "red"
                    text_color = "red"

                # Draw bounding box
                draw.rectangle([x, y, x + width, y + height], outline=box_color, width=3)

                # Draw similarity score if available
                if i < len(self.face_similarity_scores):
                    similarity = self.face_similarity_scores[i]
                    score_text = f"{similarity:.3f}"

                    # Position text above the box
                    text_y = max(0, y - 25)
                    if font:
                        draw.text((x, text_y), score_text, fill=text_color, font=font)
                    else:
                        draw.text((x, text_y), score_text, fill=text_color)

            return img_with_boxes

        except Exception as e:
            print(f"Error drawing face boxes: {e}")
            return image

    def _start_pan(self, event):
        """Start panning operation."""
        self.canvas.scan_mark(event.x, event.y)
        self.pan_start_x = event.x
        self.pan_start_y = event.y

    def _pan(self, event):
        """Handle panning motion."""
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def _fit_to_window(self):
        """Fit the image to the current window size."""
        if not hasattr(self, 'original_image'):
            return

        # Get canvas size
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            return

        # Get image size
        img_width, img_height = self.original_image.size

        # Calculate scale to fit
        scale_x = canvas_width / img_width
        scale_y = canvas_height / img_height
        self.scale_factor = min(scale_x, scale_y, 1.0)  # Don't scale up

        self._update_display()

    def _actual_size(self):
        """Show image at actual size (100%)."""
        self.scale_factor = 1.0
        self._update_display()

    def _zoom_in(self):
        """Zoom in programmatically."""
        self.scale_factor *= 1.2
        self.scale_factor = min(self.scale_factor, 5.0)
        self._update_display()

    def _zoom_out(self):
        """Zoom out programmatically."""
        self.scale_factor /= 1.2
        self.scale_factor = max(self.scale_factor, 0.1)
        self._update_display()

    def _center_image(self):
        """Center the image in the canvas."""
        self.canvas.update_idletasks()

        # Get canvas and image dimensions
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if hasattr(self, 'current_image'):
            img_width, img_height = self.current_image.size

            # Calculate center position
            x = max(0, (canvas_width - img_width) // 2)
            y = max(0, (canvas_height - img_height) // 2)

            # Update scroll region to center the image
            self.canvas.configure(scrollregion=(-x, -y, img_width + x, img_height + y))

            # Center the view
            if img_width > canvas_width:
                self.canvas.xview_moveto(0.5)
            if img_height > canvas_height:
                self.canvas.yview_moveto(0.5)

    def _zoom(self, event):
        """Handle mouse wheel zoom."""
        # Determine zoom direction
        if event.delta > 0 or event.num == 4:
            # Zoom in
            zoom_factor = 1.1
        else:
            # Zoom out
            zoom_factor = 0.9

        # Apply zoom
        old_scale = self.scale_factor
        self.scale_factor *= zoom_factor
        self.scale_factor = max(0.1, min(self.scale_factor, 5.0))

        if old_scale != self.scale_factor:
            # Update display (this will automatically center the image)
            self._update_display()

    def _open_in_explorer(self):
        """Open the image location in file explorer."""
        try:
            # Open the parent directory and select the file
            os.startfile(self.image_path.parent)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open explorer: {str(e)}")

    def _on_close(self):
        """Handle window close."""
        try:
            # Remove from tracking list
            if self in ImageViewerGui._open_viewers:
                ImageViewerGui._open_viewers.remove(self)

            # Reset offset counter if no windows are open
            if not ImageViewerGui._open_viewers:
                ImageViewerGui._next_offset = 0

            # Clean up image references
            if hasattr(self, 'original_image'):
                del self.original_image
            if hasattr(self, 'current_image'):
                del self.current_image
            if hasattr(self, 'photo'):
                del self.photo

            # Destroy the window
            self.window.destroy()
        except Exception as e:
            print(f"Error closing image viewer: {e}")

    def _toggle_fullscreen(self, event=None):
        """Toggle fullscreen mode."""
        self.is_fullscreen = not self.is_fullscreen
        self.window.attributes("-fullscreen", self.is_fullscreen)

    def _exit_fullscreen(self, event=None):
        """Exit fullscreen mode."""
        self.is_fullscreen = False
        self.window.attributes("-fullscreen", False)

    def _on_window_resize(self, event):
        """Handle window resize events to automatically refit the image."""
        # Only handle resize events for the main window, not child widgets
        if event.widget == self.window:
            # Add a small delay to avoid excessive calls during dragging
            if hasattr(self, '_resize_job'):
                self.window.after_cancel(self._resize_job)
            # Use after_idle for immediate scheduling after current event processing
            self._resize_job = self.window.after_idle(self._delayed_refit)

    def _delayed_refit(self):
        """Delayed refitting of the image to window size (for resize events)."""
        self._fit_to_window()

    def _navigate_previous(self, event=None):
        """Navigate to the previous image in the navigation list."""
        if not self.navigation_images or len(self.navigation_images) <= 1:
            return

        if self.current_index > 0:
            self.current_index -= 1
            self._navigate_to_current_image()

    def _navigate_next(self, event=None):
        """Navigate to the next image in the navigation list."""
        if not self.navigation_images or len(self.navigation_images) <= 1:
            return

        if self.current_index < len(self.navigation_images) - 1:
            self.current_index += 1
            self._navigate_to_current_image()

    def _navigate_to_current_image(self):
        """Load and display the current image in the navigation list."""
        try:
            self.image_path = self.navigation_images[self.current_index]
            self.path_var.set(str(self.image_path))

            # Update window title to include navigation info
            current_num = self.current_index + 1
            total_num = len(self.navigation_images)
            self.window.title(f"Image Viewer - {self.image_path.name} ({current_num}/{total_num})")

            # Clear any existing detection results since we're switching images
            self.detected_faces = []
            self.face_similarity_scores = []
            self.detected_segments = []
            self.show_face_boxes = False
            self.show_segmentation_masks = False
            self.show_boxes_var.set(False)
            self.show_masks_var.set(False)

            # Reset status labels
            self.face_status_label.config(text="Select a character and click 'Run Face Detection' to start")
            self.seg_status_label.config(text="Click 'Run Person Segmentation' to detect people in the image")

            # Load the new image
            self._load_image()

        except Exception as e:
            print(f"Error navigating to image {self.current_index}: {e}")
            messagebox.showerror("Navigation Error", f"Failed to load image: {str(e)}")

    @classmethod
    def get_open_viewers_count(cls):
        """Get the number of currently open image viewers."""
        return len(cls._open_viewers)

    @classmethod
    def close_all_viewers(cls):
        """Close all open image viewers."""
        viewers_to_close = cls._open_viewers.copy()
        for viewer in viewers_to_close:
            try:
                viewer._on_close()
            except:
                pass


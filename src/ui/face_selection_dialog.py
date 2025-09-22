"""
Face selection dialog for choosing a face from multiple detected faces.
"""
import tkinter as tk
from tkinter import ttk, messagebox
from typing import List, Dict, Optional, Callable
from pathlib import Path
from PIL import Image, ImageTk, ImageDraw
import math


class FaceSelectionDialog:
    """Dialog for selecting a face when multiple faces are detected in an image."""

    def __init__(self, parent, image_path: Path, faces: List[Dict],
                 on_face_selected: Optional[Callable[[Dict], None]] = None,
                 status_callback: Optional[Callable[[str], None]] = None):
        """
        Initialize the face selection dialog.

        Args:
            parent: Parent window
            image_path: Path to the image containing faces
            faces: List of face dictionaries with bounding box info
            on_face_selected: Callback when a face is selected
            status_callback: Optional callback for status updates
        """
        self.parent = parent
        self.image_path = image_path
        self.faces = faces
        self.on_face_selected = on_face_selected
        self.status_callback = status_callback
        self.selected_face = None
        self.selected_face_index = None

        if self.status_callback:
            self.status_callback("Opening face selection dialog...")

        # Dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Select Face")
        self.dialog.geometry("800x600")
        self.dialog.resizable(True, True)
        self.dialog.transient(parent)
        self.dialog.grab_set()

        # Center the dialog
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (self.dialog.winfo_width() // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (self.dialog.winfo_height() // 2)
        self.dialog.geometry(f"+{x}+{y}")

        # Load and prepare the image
        self.original_image = None
        self.display_image = None
        self.photo = None
        self.scale_factor = 1.0

        self._load_image()
        self._create_widgets()
        self._draw_face_boxes()

        # Bind close event
        self.dialog.protocol("WM_DELETE_WINDOW", self._on_cancel)

        if self.status_callback:
            self.status_callback("Face selection dialog ready - click on a face to select it")

    def _load_image(self):
        """Load and scale the image for display."""
        try:
            if self.status_callback:
                self.status_callback("Loading image for face selection...")

            self.original_image = Image.open(self.image_path)

            # Calculate scale factor to fit image in dialog (max 700x500)
            max_width = 700
            max_height = 500

            img_width, img_height = self.original_image.size
            scale_width = max_width / img_width
            scale_height = max_height / img_height
            self.scale_factor = min(scale_width, scale_height, 1.0)  # Don't upscale

            # Resize image
            new_width = int(img_width * self.scale_factor)
            new_height = int(img_height * self.scale_factor)
            self.display_image = self.original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

            if self.status_callback:
                self.status_callback("Image loaded for face selection")

        except Exception as e:
            error_msg = f"Failed to load image: {str(e)}"
            if self.status_callback:
                self.status_callback(error_msg)
            messagebox.showerror("Error", error_msg)
            self.dialog.destroy()

    def _create_widgets(self):
        """Create all widgets for the dialog."""
        # Title
        title_label = ttk.Label(self.dialog, text="Multiple faces detected. Please select the face you want to use:",
                               font=("Arial", 12))
        title_label.pack(pady=10)

        # Main frame
        main_frame = ttk.Frame(self.dialog)
        main_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # Image frame with scrollbars
        image_frame = ttk.Frame(main_frame)
        image_frame.pack(fill="both", expand=True)

        # Create canvas with scrollbars
        self.canvas = tk.Canvas(image_frame, bg="white")
        v_scrollbar = ttk.Scrollbar(image_frame, orient="vertical", command=self.canvas.yview)
        h_scrollbar = ttk.Scrollbar(image_frame, orient="horizontal", command=self.canvas.xview)

        self.canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        v_scrollbar.pack(side="right", fill="y")
        h_scrollbar.pack(side="bottom", fill="x")

        # Bind canvas click
        self.canvas.bind("<Button-1>", self._on_canvas_click)

        # Info label
        self.info_label = ttk.Label(main_frame, text=f"Found {len(self.faces)} faces. Click on a face to select it.",
                                   font=("Arial", 10))
        self.info_label.pack(pady=5)

        # Button frame
        button_frame = ttk.Frame(self.dialog)
        button_frame.pack(pady=10)

        ttk.Button(button_frame, text="Confirm Selection", command=self._on_confirm,
                  style='Accent.TButton').pack(side="left", padx=5)
        ttk.Button(button_frame, text="Cancel", command=self._on_cancel).pack(side="left", padx=5)

    def _draw_face_boxes(self):
        """Draw the image with face bounding boxes."""
        if not self.display_image:
            return

        # Create a copy of the image to draw on
        img_with_boxes = self.display_image.copy()
        draw = ImageDraw.Draw(img_with_boxes)

        # Draw face boxes
        for i, face in enumerate(self.faces):
            # Scale the face coordinates
            x = int(face['x'] * self.scale_factor)
            y = int(face['y'] * self.scale_factor)
            width = int(face['width'] * self.scale_factor)
            height = int(face['height'] * self.scale_factor)

            # Draw bounding box
            color = "red" if i == self.selected_face_index else "yellow"
            thickness = 3 if i == self.selected_face_index else 2

            for offset in range(thickness):
                draw.rectangle([x - offset, y - offset, x + width + offset, y + height + offset],
                             outline=color, fill=None)

            # Draw face number
            text = str(i + 1)
            text_bbox = draw.textbbox((0, 0), text)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            # Text background
            text_bg_x1 = x
            text_bg_y1 = y - text_height - 4
            text_bg_x2 = x + text_width + 4
            text_bg_y2 = y

            draw.rectangle([text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2],
                         fill=color, outline=color)
            draw.text((x + 2, y - text_height - 2), text, fill="white")

        # Convert to PhotoImage
        self.photo = ImageTk.PhotoImage(img_with_boxes)

        # Clear canvas and add image
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)

        # Update scroll region
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_click(self, event):
        """Handle canvas click to select a face."""
        # Get canvas coordinates
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)

        # Find which face was clicked
        for i, face in enumerate(self.faces):
            # Scale the face coordinates
            x = int(face['x'] * self.scale_factor)
            y = int(face['y'] * self.scale_factor)
            width = int(face['width'] * self.scale_factor)
            height = int(face['height'] * self.scale_factor)

            # Check if click is within this face box
            if x <= canvas_x <= x + width and y <= canvas_y <= y + height:
                self.selected_face_index = i
                self.selected_face = face
                self._draw_face_boxes()  # Redraw with selection
                self.info_label.config(text=f"Selected face {i + 1}. Click 'Confirm Selection' to proceed.")

                if self.status_callback:
                    self.status_callback(f"Face {i + 1} selected")
                break

    def _on_confirm(self):
        """Handle confirm button click."""
        if self.selected_face is None:
            messagebox.showwarning("No Selection", "Please select a face first.")
            return

        if self.status_callback:
            self.status_callback("Face selection confirmed")

        if self.on_face_selected:
            self.on_face_selected(self.selected_face)

        self.dialog.destroy()

    def _on_cancel(self):
        """Handle cancel button click."""
        if self.status_callback:
            self.status_callback("Face selection cancelled")
        self.dialog.destroy()


def show_face_selection_dialog(parent, image_path: Path, faces: List[Dict],
                              status_callback: Optional[Callable[[str], None]] = None) -> Optional[Dict]:
    """
    Show face selection dialog and return selected face.

    Args:
        parent: Parent window
        image_path: Path to image containing faces
        faces: List of detected faces
        status_callback: Optional callback for status updates

    Returns:
        Selected face dictionary or None if cancelled
    """
    selected_face = None

    def on_face_selected(face):
        nonlocal selected_face
        selected_face = face

    dialog = FaceSelectionDialog(parent, image_path, faces, on_face_selected, status_callback)
    parent.wait_window(dialog.dialog)

    return selected_face

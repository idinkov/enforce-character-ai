"""
Images management tab UI component.
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional, Callable, Set
from pathlib import Path

from src.services.image_service import ImageService, StageProgressTracker
from src.config.app_config import config
from src.ui.image_grid_widget import SimpleImageGridWidget
from src.ui.training_config_dialog import train_model
from src.ui.processing_status_dialog import ProcessingStatusDialog
from src.ui.create_images_dialog import open_txt2img_dialog
from src.models.model_manager import get_trainable_models


class ImagesTab:
    """Images management tab component."""

    def __init__(self, parent_notebook: ttk.Notebook, image_service: ImageService,
                 progress_tracker: StageProgressTracker, character_repo, main_app=None):
        self.image_service = image_service
        self.progress_tracker = progress_tracker
        self.character_repo = character_repo
        self.main_app = main_app
        self.current_character: Optional[str] = None
        self.current_stage = tk.StringVar(value="1_raw")

        # Callbacks
        self.on_face_image_set_requested: Optional[Callable[[str, Path], None]] = None

        # Create the tab
        self.frame = ttk.Frame(parent_notebook)
        parent_notebook.add(self.frame, text="Images")

        # Initialize processing status dialog
        self.processing_dialog = ProcessingStatusDialog(
            parent_frame=self.frame,
            current_character_getter=lambda: self.current_character,
            current_stage_getter=lambda: self.current_stage.get(),
            progress_tracker=self.progress_tracker,
            update_progress_display_callback=self._update_progress_display,
            refresh_images_callback=self._refresh_images,
            main_app_reference=self.main_app
        )

        self._create_widgets()
        self._update_stage_button_styles()

    def _create_widgets(self):
        """Create all widgets for the images tab."""
        # Top panel - character info and progress
        self._create_top_panel()

        # Stage selection panel
        self._create_stage_panel()

        # Image operations panel
        self._create_operations_panel()

        # Image grid panel
        self._create_image_grid_panel()

    @staticmethod
    def get_character_trained_model_ids(models_dir: Path, character_name: str) -> set:
        """Return a set of model_id suffixes for all .safetensors files in the character's models directory."""
        # Replace " " with "_" in character_name to match filename format
        character_name = character_name.replace(" ", "_")
        model_ids = set()
        if not models_dir.exists() or not models_dir.is_dir():
            return model_ids
        for file in models_dir.glob(f"{character_name}_*.safetensors"):
            # Extract model_id from filename: {character_name}_{model_id}.safetensors
            stem = file.stem
            parts = stem.split("_")
            # Get last part, it can be a lot of _ so get the last only
            if len(parts) >= 2:
                model_id = parts[-1]
                model_ids.add(model_id)


        return model_ids

    def _update_models_button_style(self):
        """Update the Models button style to green if all trainable models are present for the character."""
        if not self.current_character:
            self.models_btn.configure(style='Accent.TButton')
            return
        all_trainable = get_trainable_models()
        required_model_ids = {model.train_suffix for model in all_trainable.values() if hasattr(model, 'train_suffix')}
        models_dir = self.image_service.characters_path / self.current_character / "models"
        present_model_ids = self.get_character_trained_model_ids(models_dir, self.current_character)
        if required_model_ids and required_model_ids.issubset(present_model_ids):
            style = ttk.Style()
            # Define a green style if not already defined
            if 'Green.TButton' not in style.theme_names():
                style.configure('Green.TButton', background='#4CAF50')
            self.models_btn.configure(style='Green.TButton')
        else:
            self.models_btn.configure(style='Accent.TButton')

    def _setup_keyboard_shortcuts(self):
        """Setup keyboard shortcuts for the images tab."""
        # Get the root window to bind global keyboard shortcuts
        root = self.frame.winfo_toplevel()

        # Make the frame focusable and bind keyboard events
        self.frame.config(takefocus=True)
        self.frame.bind("<FocusIn>", self._on_focus_in)

        # Bind keyboard events to multiple widgets
        widgets_to_bind = [
            self.frame,
            self.img_canvas,
            self.img_scrollable_frame,
            root  # Also bind to root window
        ]

        for widget in widgets_to_bind:
            widget.bind("<Delete>", self._on_delete_key)
            widget.bind("<Shift-Delete>", self._on_shift_delete_key)
            widget.bind("<KeyPress-Delete>", self._on_delete_key)
            widget.bind("<Shift-KeyPress-Delete>", self._on_shift_delete_key)

        # Make widgets focusable
        self.frame.config(takefocus=True)
        self.img_canvas.config(takefocus=True)

        # Set up click handlers to manage focus
        self.frame.bind("<Button-1>", self._on_frame_click)
        self.img_canvas.bind("<Button-1>", self._on_canvas_click)

        # Set initial focus after a short delay
        self.frame.after(100, self._set_initial_focus)

        print("Enhanced keyboard shortcuts setup completed for Images tab")

    def _on_focus_in(self, event=None):
        """Handle focus in event."""
        print("Images tab gained focus")

    def _set_initial_focus(self):
        """Set initial focus to enable keyboard shortcuts."""
        try:
            self.frame.focus_set()
            print("Initial focus set to Images tab frame")
        except Exception as e:
            print(f"Error setting focus: {e}")

    def _on_frame_click(self, event=None):
        """Handle frame click to set focus."""
        self.frame.focus_set()
        print("Focus set to frame via click")

    def _on_canvas_click(self, event=None):
        """Handle canvas click to set focus."""
        self.img_canvas.focus_set()
        print("Focus set to canvas via click")

    def _on_delete_key(self, event=None):
        """Handle Delete key press - delete with confirmation dialog."""
        print("Delete key handler called")
        selected = self.image_grid.selected_images
        if not selected:
            print("No images selected")
            return "break"

        print(f"Deleting {len(selected)} selected images with confirmation")
        self._delete_selected_images()
        return "break"

    def _on_shift_delete_key(self, event=None):
        """Handle Shift+Delete key press - force delete without dialog."""
        print("Shift+Delete key handler called")
        selected = self.image_grid.selected_images
        if not selected:
            print("No images selected for force delete")
            return "break"

        print(f"Force deleting {len(selected)} selected images")
        self._force_delete_selected_images()
        return "break"

    def _force_delete_selected_images(self):
        """Force delete selected images without confirmation dialog."""
        selected = self.image_grid.selected_images
        if not selected:
            messagebox.showwarning("Warning", "Please select images to delete")
            return

        # Delete without confirmation
        errors = self.image_service.delete_images(selected)
        if errors:
            messagebox.showwarning("Warning", f"Some errors occurred:\n" + "\n".join(errors[:5]))
        else:
            # Show brief success message
            messagebox.showinfo("Deleted", f"Successfully deleted {len(selected)} images")
        self._refresh_images()

    def _create_top_panel(self):
        """Create the top panel with character info and progress."""
        top_panel = ttk.Frame(self.frame)
        top_panel.pack(fill="x", padx=10, pady=5)

        # Character info frame
        char_info_frame = ttk.Frame(top_panel)
        char_info_frame.pack(side="left")

        # Face image placeholder
        self.images_face_label = tk.Label(char_info_frame, text="No\nFace", bg="lightgray",
                                         relief="sunken", font=("Arial", 8))
        self.images_face_label.pack(side="left", padx=(0, 10))

        # Character name
        char_name_frame = ttk.Frame(char_info_frame)
        char_name_frame.pack(side="left")

        ttk.Label(char_name_frame, text="Selected Character:").pack(anchor="w")
        self.selected_char_label = ttk.Label(char_name_frame, text="None", font=("Arial", 20, "bold"))
        self.selected_char_label.pack(anchor="w")

        # Action buttons
        char_action_frame = ttk.Frame(char_info_frame)
        char_action_frame.pack(side="left", padx=(10, 0))

        self.process_stage_button = ttk.Button(char_action_frame, text="▶", width=3,
                                               command=self.processing_dialog.process_current_stage)
        self.process_stage_button.pack(side="left", padx=(0, 6))

        self.process_all_button = ttk.Button(char_action_frame, text="▶ All Stages",
                                             command=self.processing_dialog.process_all_stages)
        self.process_all_button.pack(side="left")

        # Progress section
        progress_frame = ttk.Frame(top_panel)
        progress_frame.pack(side="right", padx=(20, 0))

        ttk.Label(progress_frame, text="Overall Progress:").pack(side="left")
        self.overall_progress_var = tk.DoubleVar()
        self.overall_progress_bar = ttk.Progressbar(progress_frame, variable=self.overall_progress_var,
                                                   maximum=len(config.STAGES), length=config.PROGRESS_BAR_LENGTH)
        self.overall_progress_bar.pack(side="left", padx=5)
        self.progress_label = ttk.Label(progress_frame, text="0/7")
        self.progress_label.pack(side="left", padx=5)

        # Cache stats (optional debug info)
        cache_frame = ttk.Frame(top_panel)
        cache_frame.pack(side="right", padx=(10, 0))

        self.cache_stats_label = ttk.Label(cache_frame, text="Cache: 0/0", font=("Arial", 8))
        self.cache_stats_label.pack()

        # Update cache stats periodically
        self._update_cache_stats()

    def _update_cache_stats(self):
        """Update cache statistics display."""
        try:
            stats = self.image_service.get_cache_stats()
            thumb_stats = stats.get('thumbnail_cache', {})
            cache_text = f"Cache: {thumb_stats.get('size', 0)}/{thumb_stats.get('max_size', 0)} ({thumb_stats.get('memory_mb', 0)}MB)"
            self.cache_stats_label.config(text=cache_text)
        except:
            pass

        # Schedule next update
        self.frame.after(5000, self._update_cache_stats)

    def _create_stage_panel(self):
        """Create the stage selection panel."""
        stage_panel = ttk.Frame(self.frame)
        stage_panel.pack(fill="x", padx=10, pady=5)

        stage_frame = ttk.Frame(stage_panel)
        stage_frame.pack(side="left")

        ttk.Label(stage_frame, text="Image Stage:").pack(side="top", anchor="w")

        # Create stage buttons
        self.stage_buttons = {}
        button_frame = ttk.Frame(stage_frame)
        button_frame.pack(side="top", pady=5)

        for stage_id, stage_num, stage_name in config.STAGES:
            btn = ttk.Button(button_frame, text=stage_num, width=4,
                           command=lambda s=stage_id: self._select_stage(s))
            btn.pack(side="left", padx=2)
            self.stage_buttons[stage_id] = btn
            self._create_tooltip(btn, stage_name)
            # Insert Train Model button after stage 7
            if stage_id == "7_final_dataset":
                self.train_model_button = ttk.Button(button_frame, text="🚀 Train Model",
                                                    command=self._train_model,
                                                    style='Accent.TButton')
                self.train_model_button.pack(side="left", padx=(5, 2))
                self.train_model_button.config(state='enabled')  # Default to disabled
                self._create_tooltip(self.train_model_button, "Train model on stage 7 images")

        # Add Models button after stage buttons and Train Model button
        self.models_btn = ttk.Button(button_frame, text="Models", width=8,
                                   command=self._open_models_folder,
                                   style='Accent.TButton')
        self.models_btn.pack(side="left", padx=(10, 2))
        self._create_tooltip(self.models_btn, "Open character models directory")

        # Add Create button after Models
        self.create_btn = ttk.Button(button_frame, text="Create", width=8,
                                   command=self._open_create_dialog,
                                   style='Accent.TButton')
        self.create_btn.pack(side="left", padx=(2, 2))
        self._create_tooltip(self.create_btn, "Create images with SDXL txt2img")

    def _create_operations_panel(self):
        """Create the image operations panel."""
        mid_panel = ttk.Frame(self.frame)
        mid_panel.pack(fill="x", padx=10, pady=5)

        # Left side - main operations
        left_ops = ttk.Frame(mid_panel)
        left_ops.pack(side="left")

        ttk.Button(left_ops, text="Upload Images", command=self._upload_images).pack(side="left", padx=5)
        ttk.Button(left_ops, text="Open Folder", command=self._open_current_folder).pack(side="left", padx=5)
        ttk.Button(left_ops, text="Refresh", command=self._refresh_images).pack(side="left", padx=5)

        self.set_face_button = ttk.Button(left_ops, text="Set as Face",
                                         command=self._set_selected_as_face, state='disabled')
        self.set_face_button.pack(side="left", padx=5)

        ttk.Button(left_ops, text="Select All", command=self._select_all_images).pack(side="left", padx=5)
        ttk.Button(left_ops, text="Select None", command=self._select_none_images).pack(side="left", padx=5)

        self.open_selected_button = ttk.Button(left_ops, text="Open Selected",
                                              command=self._open_selected_images, state='disabled')
        self.open_selected_button.pack(side="left", padx=5)

        ttk.Button(left_ops, text="Delete Selected", command=self._delete_selected_images).pack(side="left", padx=5)

        # OneTrainer section - Train Model button (only visible for stage 7)
        self.train_model_button = ttk.Button(left_ops, text="🚀 Train Model",
                                           command=self._train_model,
                                           style='Accent.TButton')
        # Initially hidden, will be shown when appropriate
        self.train_model_button.pack_forget()

        # Center - pagination controls
        self.pagination_frame = ttk.Frame(mid_panel)
        self.pagination_frame.pack(side="left", padx=20)

        self.prev_button = ttk.Button(self.pagination_frame, text="◀ Prev", width=8,
                                     command=self._previous_page, state='disabled')
        self.prev_button.pack(side="left", padx=2)

        self.page_label = ttk.Label(self.pagination_frame, text="Page: 0/0")
        self.page_label.pack(side="left", padx=10)

        self.next_button = ttk.Button(self.pagination_frame, text="Next ▶", width=8,
                                     command=self._next_page, state='disabled')
        self.next_button.pack(side="left", padx=2)

        # Right side - image count display
        self.image_count_label = ttk.Label(mid_panel, text="Images: 0")
        self.image_count_label.pack(side="right")

    def _create_image_grid_panel(self):
        """Create the optimized image grid panel."""
        bottom_panel = ttk.Frame(self.frame)
        bottom_panel.pack(fill="both", expand=True, padx=10, pady=5)

        # Create scrollable canvas for images
        self.img_canvas = tk.Canvas(bottom_panel)
        img_scrollbar_v = ttk.Scrollbar(bottom_panel, orient="vertical", command=self.img_canvas.yview)

        self.img_scrollable_frame = ttk.Frame(self.img_canvas)

        self.img_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.img_canvas.configure(scrollregion=self.img_canvas.bbox("all"))
        )

        self.img_canvas.create_window((0, 0), window=self.img_scrollable_frame, anchor="nw")
        self.img_canvas.configure(yscrollcommand=img_scrollbar_v.set)

        self.img_canvas.pack(side="left", fill="both", expand=True)
        img_scrollbar_v.pack(side="right", fill="y")

        # Create optimized image grid widget
        self.image_grid = SimpleImageGridWidget(self.img_canvas, self.img_scrollable_frame, self.image_service)
        self.image_grid.on_selection_changed = self._on_image_selection_changed
        self.image_grid.on_image_viewer_requested = self._on_image_viewer_requested
        self.image_grid.on_pagination_changed = self._on_pagination_changed

        # Set up drag & drop callbacks
        self.image_grid.set_callbacks(
            get_current_stage_callback=lambda: self.current_stage.get(),
            refresh_images_callback=self._refresh_images
        )

    def set_current_character(self, character_name: Optional[str]):
        """Set the current character and update the display."""
        self.current_character = character_name
        self.selected_char_label.config(text=character_name or "None")

        if character_name:
            self._update_progress_display()
            self._refresh_images()
            self._update_face_image_display()
        else:
            self.image_grid.clear_images()
            self.image_count_label.config(text="Images: 0")
            self._clear_face_image_display()

    def _update_face_image_display(self):
        """Update the face image display for current character."""
        if not self.current_character:
            return

        try:
            # Load character data to get face image path
            character = self.character_repo.load_character(self.current_character)
            if character and character.face_image:
                face_path = Path(character.face_image)

                # Check if it's an absolute path or relative to character directory
                if not face_path.is_absolute():
                    char_dir = self.character_repo.characters_path / self.current_character
                    face_path = char_dir / character.face_image

                if face_path.exists():
                    # Create face thumbnail
                    face_photo = self.image_service.create_face_thumbnail(face_path)
                    if face_photo:
                        self.images_face_label.config(image=str(face_photo), text="")
                        self.images_face_label.image = face_photo  # Keep reference
                        return

            # No face image or failed to load
            self._clear_face_image_display()

        except Exception as e:
            print(f"Error updating face image display: {e}")
            self._clear_face_image_display()

    def _clear_face_image_display(self):
        """Clear the face image display."""
        self.images_face_label.config(image="", text="No\nFace")
        self.images_face_label.image = None

    def refresh_face_image_display(self):
        """Refresh the face image display (called after face image is updated)."""
        self._update_face_image_display()

    def _select_stage(self, stage_id: str):
        """Select image stage for viewing."""
        self.current_stage.set(stage_id)
        self._update_stage_button_styles()
        self._refresh_images()

    def _update_stage_button_styles(self):
        """Update stage button styles based on current selection."""
        current_stage = self.current_stage.get()

        for stage_id, button in self.stage_buttons.items():
            if stage_id == current_stage:
                button.state(['pressed'])
            else:
                button.state(['!pressed'])

    def _refresh_images(self):
        """Refresh the image display for current character and stage."""
        if not self.current_character:
            self.image_grid.clear_images()
            self.image_count_label.config(text="Images: 0")
            return

        self._update_models_button_style()

        stage = self.current_stage.get()
        # Pass reset_page=False to preserve current page position during refresh
        self.image_grid.load_images(self.current_character, stage, reset_page=False)

        # Update image count
        count = len(self.image_grid.all_image_files)
        self.image_count_label.config(text=f"Images: {count}")

        # Show or hide the Train Model button based on stage and image availability
        if stage == "7_final_dataset" and count > 0:
            self.train_model_button.pack(side="left", padx=5)
        else:
            self.train_model_button.pack_forget()

    def _upload_images(self):
        """Upload images to current stage."""
        if not self.current_character:
            messagebox.showwarning("Warning", "Please select a character first")
            return

        files = filedialog.askopenfilenames(
            title="Select images to upload",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")]
        )

        if files:
            stage = self.current_stage.get()
            # Convert tuple to list for proper type compatibility
            files_list = list(files)
            count = self.image_service.upload_images(self.current_character, stage, files_list)
            messagebox.showinfo("Success", f"Uploaded {count} images to {stage}")
            self._refresh_images()

    def _open_current_folder(self):
        """Open current image folder in file explorer."""
        if not self.current_character:
            messagebox.showwarning("Warning", "Please select a character first")
            return

        stage = self.current_stage.get()
        folder_path = self.image_service.characters_path / self.current_character / "images" / stage

        if folder_path.exists():
            import os
            os.startfile(folder_path)
        else:
            messagebox.showerror("Error", f"Folder does not exist: {folder_path}")

    def _open_models_folder(self):
        """Open character models folder in file explorer."""
        if not self.current_character:
            messagebox.showwarning("Warning", "Please select a character first")
            return

        models_folder_path = self.image_service.characters_path / self.current_character / "models"

        # Create the models folder if it doesn't exist
        if not models_folder_path.exists():
            try:
                models_folder_path.mkdir(parents=True, exist_ok=True)
                print(f"Created models directory: {models_folder_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not create models folder: {str(e)}")
                return

        # Open the folder in file explorer
        try:
            import os
            os.startfile(models_folder_path)
            print(f"Opened models folder: {models_folder_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not open models folder: {str(e)}")

    def _select_all_images(self):
        """Select all images."""
        self.image_grid.select_all_images()

    def _select_none_images(self):
        """Deselect all images."""
        self.image_grid.select_none_images()

    def _open_selected_images(self):
        """Open the image viewer with the currently selected images."""
        selected = self.image_grid.selected_images
        if not selected:
            messagebox.showwarning("Warning", "Please select images to open")
            return

        try:
            # Import here to avoid circular imports
            from src.ui.image_viewer import ImageViewerGui
            
            # Sort selected images for consistent order
            navigation_images = sorted(list(selected))
            
            # Open the first selected image with all selected images for navigation
            first_image = navigation_images[0]
            ImageViewerGui(self.frame, first_image, navigation_images)
            
        except Exception as e:
            print(f"Error opening selected images: {e}")
            messagebox.showerror("Error", f"Failed to open image viewer: {str(e)}")

    def _delete_selected_images(self):
        """Delete selected images."""
        selected = self.image_grid.selected_images
        if not selected:
            messagebox.showwarning("Warning", "Please select images to delete")
            return

        if messagebox.askyesno("Confirm Delete", f"Delete {len(selected)} selected images?"):
            errors = self.image_service.delete_images(selected)
            if errors:
                messagebox.showwarning("Warning", f"Some errors occurred:\n" + "\n".join(errors[:5]))
            self._refresh_images()

    def _set_selected_as_face(self):
        """Set selected image as face image with face detection and cropping."""
        selected = self.image_grid.selected_images
        if len(selected) != 1:
            messagebox.showwarning("Warning", "Please select exactly one image")
            return

        if not self.current_character:
            messagebox.showwarning("Warning", "No character selected")
            return

        selected_image_path = list(selected)[0]

        # Status callback function
        def status_callback(message):
            if self.main_app and hasattr(self.main_app, 'set_status_bar_text'):
                self.main_app.set_status_bar_text(message)
            print(message)  # Fallback to console

        try:
            # Import required modules
            from src.utils.face_processor import FaceProcessor
            from src.ui.face_selection_dialog import show_face_selection_dialog

            # Update status and show processing message
            status_callback("Detecting faces in the image...")

            # Get character path for face detection
            character_path = self.image_service.characters_path / self.current_character

            # Detect faces in the selected image
            faces = FaceProcessor.detect_faces_in_image(selected_image_path, character_path, status_callback)

            if not faces:
                status_callback("No faces detected in the selected image")
                messagebox.showwarning("No Faces", "No faces detected in the selected image. Please choose a different image.")
                return

            selected_face = None

            if len(faces) == 1:
                # Only one face detected, use it directly
                selected_face = faces[0]
                status_callback("One face detected. Processing...")
            else:
                # Multiple faces detected, show selection dialog
                status_callback(f"Found {len(faces)} faces. Please select the face you want to use.")
                selected_face = show_face_selection_dialog(self.frame, selected_image_path, faces, status_callback)

                if not selected_face:
                    # User cancelled the selection
                    status_callback("Face selection cancelled")
                    return
                else:
                    status_callback("Face selected. Processing...")

            # Crop and save the face
            face_output_path = character_path / "face.png"
            status_callback("Cropping and saving face image...")

            # Extract landmarks if available
            landmarks = selected_face.get('landmarks', None)

            success = FaceProcessor.crop_and_save_face(
                image_path=selected_image_path,
                face=selected_face,
                output_path=face_output_path,
                max_size=1024,
                landmarks=landmarks,
                status_callback=status_callback
            )

            if not success:
                status_callback("Failed to crop and save the face image")
                messagebox.showerror("Error", "Failed to crop and save the face image.")
                return

            # Update character's face_image field
            status_callback("Updating character configuration...")
            success = FaceProcessor.update_character_face_image(
                character_repo=self.character_repo,
                character_name=self.current_character,
                face_image_path=face_output_path,
                status_callback=status_callback
            )

            if not success:
                status_callback("Failed to update character face image reference")
                messagebox.showerror("Error", "Failed to update character face image reference.")
                return

            # Refresh the face image display
            self.refresh_face_image_display()

            # Notify main app if callback is set
            if self.on_face_image_set_requested is not None:
                self.on_face_image_set_requested(self.current_character, face_output_path)

            status_callback("Face image saved successfully as face.png")
            # Removed redundant messagebox - status bar message is sufficient

        except Exception as e:
            error_msg = f"Failed to process face image: {str(e)}"
            status_callback(error_msg)
            messagebox.showerror("Error", error_msg)
            print(f"Error in _set_selected_as_face: {e}")
            import traceback
            traceback.print_exc()

    def _update_progress_display(self):
        """Update the progress display for current character."""
        if not self.current_character:
            return

        status = self.progress_tracker.get_completion_status(self.current_character)
        completed = status['completed_count']
        total = status['total_stages']

        self.overall_progress_var.set(completed)
        self.progress_label.config(text=f"{completed}/{total}")

        # Update stage button styles based on completion
        for stage_id, button in self.stage_buttons.items():
            if status.get(stage_id, False):
                button.configure(style='Completed.TButton')
            else:
                button.configure(style='TButton')

    def _on_image_selection_changed(self, selected_images: Set[Path]):
        """Handle image selection changes."""
        # Update Set as Face Image button state
        self.set_face_button.config(state='normal' if len(selected_images) == 1 else 'disabled')

        # Update Open Selected button state
        self.open_selected_button.config(state='normal' if len(selected_images) > 0 else 'disabled')

    def _on_image_viewer_requested(self, img_path: Path):
        """Handle image viewer requests."""
        try:
            # Import here to avoid circular imports
            from src.ui.image_viewer import ImageViewerGui

            # Determine navigation images based on selection state
            navigation_images = []

            # Check if there are multiple selected images
            if len(self.image_grid.selected_images) > 1:
                # Use selected images for navigation, sorted by name for consistent order
                navigation_images = sorted(list(self.image_grid.selected_images))
            else:
                # Use all images in the current view for navigation (either single selection or no selection)
                navigation_images = sorted(self.image_grid.all_image_files)

            # Launch image viewer with navigation support
            ImageViewerGui(self.frame, img_path, navigation_images)

        except Exception as e:
            print(f"Error opening image viewer for {img_path}: {e}")
            messagebox.showerror("Error", f"Failed to open image viewer: {str(e)}")

    def _create_tooltip(self, widget, text):
        """Create a tooltip for a widget."""
        def show_tooltip(event):
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")

            label = ttk.Label(tooltip, text=text, background="lightyellow",
                            relief="solid", borderwidth=1, font=("Arial", 9))
            label.pack()

            widget.tooltip = tooltip

        def hide_tooltip(event):
            if hasattr(widget, 'tooltip'):
                widget.tooltip.destroy()
                del widget.tooltip

        widget.bind("<Enter>", show_tooltip)
        widget.bind("<Leave>", hide_tooltip)

    def _on_pagination_changed(self, current_page: int, total_pages: int):
        """Handle pagination changes from the image grid."""
        # Update page label
        self.page_label.config(text=f"Page: {current_page}/{total_pages}")

        # Update button states
        self.prev_button.config(state='normal' if current_page > 1 else 'disabled')
        self.next_button.config(state='normal' if current_page < total_pages else 'disabled')

    def _previous_page(self):
        """Go to the previous page."""
        self.image_grid.previous_page()

    def _next_page(self):
        """Go to the next page."""
        self.image_grid.next_page()

    def _get_main_app(self):
        """Get reference to the main application instance."""
        # Navigate up the widget hierarchy to find the main app
        widget = self.frame
        while widget:
            if hasattr(widget, 'master') and hasattr(widget.master, 'app'):
                return widget.master.app
            widget = getattr(widget, 'master', None)

        # Alternative approach: check if the root has the app reference
        root = self.frame.winfo_toplevel()
        if hasattr(root, 'app'):
            return root.app

        return None

    def _train_model(self):
        """Train the model using OneTrainer with stage 7 images."""
        train_model(self.frame, self.current_character, self.image_service, self.character_repo)

    def _open_create_dialog(self):
        """Open a dialog to generate an image from text using SDXL txt2img."""
        open_txt2img_dialog(
            parent_frame=self.frame,
            current_character=self.current_character,
            image_service=self.image_service,
            refresh_images_callback=self._refresh_images
        )

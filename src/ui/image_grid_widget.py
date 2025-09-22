"""
Image grid widget for displaying images with pagination and lazy loading.
"""
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional, Callable, Set, List, Dict, Union, Any
from pathlib import Path
import subprocess
import platform

# Add drag & drop support
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    DND_AVAILABLE = True
except ImportError:
    DND_AVAILABLE = False
    print("Warning: tkinterdnd2 not available. Drag & drop functionality will be disabled.")

from src.services.image_service import ImageService
from src.config.app_config import config


class ImageFrame(ttk.Frame):
    """Extended Frame class with image-specific attributes."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.checkbox_var: Optional[tk.BooleanVar] = None
        self.img_path: Optional[Path] = None
        self.checkbox: Optional[tk.Checkbutton] = None
        self.canvas: Optional[tk.Canvas] = None
        self.loading_text: Optional[int] = None  # Canvas text item ID
        self.filename_label: Optional[tk.Label] = None
        self.loaded: bool = False
        self.image_item: Optional[int] = None  # Canvas image item ID
        self.selected: bool = False


class SimpleImageGridWidget:
    """Simple image grid widget with standard scrolling and async loading."""

    def __init__(self, parent_canvas: tk.Canvas, scrollable_frame: ttk.Frame, image_service: ImageService):
        self.parent_canvas = parent_canvas
        self.scrollable_frame = scrollable_frame
        self.image_service = image_service

        # Track current character for deleted images checking
        self.current_character_name: Optional[str] = None

        # Simple state management
        self.all_image_files: List[Path] = []
        self.image_widgets: List[ImageFrame] = []
        self.selected_images: Set[Path] = set()
        self.loading_placeholders: Dict[Path, Union[tk.Label, tk.Canvas]] = {}

        # Pagination settings
        self.images_per_page = 100
        self.current_page = 0
        self.total_pages = 0
        self.current_page_images: List[Path] = []

        # Grid layout parameters
        self.cols_per_row = 6

        # Lazy loading state
        self.loaded_thumbnails: Set[int] = set()
        self.viewport_buffer = 200  # Load thumbnails 200px before they enter viewport
        self.last_scroll_position = 0
        self.loading_batch_size = 20  # Load thumbnails in batches
        self.scroll_check_pending = False  # Debounce scroll checks

        # Callbacks
        self.on_selection_changed: Optional[Callable[[Set[Path]], None]] = None
        self.on_image_viewer_requested: Optional[Callable[[Path], None]] = None
        self.on_pagination_changed: Optional[Callable[[int, int], None]] = None  # current_page, total_pages

        # Bind canvas resize to reorganize grid
        self.parent_canvas.bind("<Configure>", self._on_canvas_configure)

        # Setup comprehensive scroll event binding
        self._setup_scroll_binding()

        # Periodically check viewport for lazy loading
        self._schedule_viewport_check()

        # Enable drag & drop if available
        if DND_AVAILABLE:
            self._enable_drag_and_drop()

    def _setup_scroll_binding(self):
        """Setup comprehensive scroll event binding for the canvas and all child widgets."""
        # Bind mouse wheel events to canvas and scrollable frame
        self._bind_mouse_wheel(self.parent_canvas)
        self._bind_mouse_wheel(self.scrollable_frame)

        # Also bind to scrollbar changes if available
        if hasattr(self.parent_canvas, 'master') and hasattr(self.parent_canvas.master, 'v_scrollbar'):
            self.parent_canvas.master.v_scrollbar.bind("<ButtonRelease-1>", self._on_scrollbar_release)

    def _bind_mouse_wheel(self, widget):
        """Bind mouse wheel events to a widget and recursively to all its children."""
        # Mouse wheel events for Windows and most systems
        widget.bind("<MouseWheel>", self._on_scroll)

        # Linux scroll events
        widget.bind("<Button-4>", self._on_scroll)
        widget.bind("<Button-5>", self._on_scroll)

        # Recursively bind to all existing child widgets
        for child in widget.winfo_children():
            self._bind_mouse_wheel(child)

    def _bind_mouse_wheel_to_widget_tree(self, widget):
        """Recursively bind mouse wheel events to a widget and all its children."""
        self._bind_mouse_wheel(widget)

    def load_images(self, character_name: str, stage: str, reset_page: bool = True):
        """Load images with simple scrolling."""
        self.clear_images()

        # Store character name for deleted images checking
        self.current_character_name = character_name

        # Store the current page before getting new images (for refresh scenarios)
        previous_page = self.current_page if hasattr(self, 'current_page') else 0

        # Get all images
        self.all_image_files = self.image_service.get_stage_images(character_name, stage)

        if not self.all_image_files:
            self.current_page = 0
            self.total_pages = 0
            self.current_page_images = []
            return

        # Calculate pagination
        self.total_pages = (len(self.all_image_files) + self.images_per_page - 1) // self.images_per_page

        # Only reset to first page if explicitly requested, otherwise preserve current page
        if reset_page:
            self.current_page = 0
        else:
            # Preserve current page, but ensure it's within bounds
            self.current_page = min(previous_page, self.total_pages - 1) if self.total_pages > 0 else 0

        self._update_current_page_images()

        # Calculate grid dimensions
        self._update_grid_dimensions()

        # Create all image widgets at once
        self._create_image_grid()

        # Reset lazy loading state
        self.loaded_thumbnails.clear()
        self.last_scroll_position = 0

        # Load initial visible thumbnails only (first few rows)
        self._load_initial_thumbnails()

        # Notify pagination change
        if self.on_pagination_changed:
            self.on_pagination_changed(self.current_page + 1, self.total_pages)

    def _update_current_page_images(self):
        """Update the current page images based on current page."""
        start_idx = self.current_page * self.images_per_page
        end_idx = start_idx + self.images_per_page
        self.current_page_images = self.all_image_files[start_idx:end_idx]

    def _update_grid_dimensions(self):
        """Update grid dimensions based on canvas size."""
        self.parent_canvas.update_idletasks()
        canvas_width = max(self.parent_canvas.winfo_width(), 400)

        # Calculate columns based on available width
        self.cols_per_row = max(config.MIN_GRID_COLUMNS, canvas_width // config.IMAGE_WIDGET_WIDTH)

    def _create_image_grid(self):
        """Create the complete image grid."""
        for i, img_path in enumerate(self.current_page_images):
            row = i // self.cols_per_row
            col = i % self.cols_per_row
            self._create_image_widget(img_path, row, col)

    def _create_image_widget(self, img_path: Path, row: int, col: int):
        """Create an image widget in the grid."""
        frame = ImageFrame(self.scrollable_frame)
        frame.grid(row=row, column=col, padx=8, pady=8, sticky="n")

        # Create invisible checkbox for state tracking only
        var = tk.BooleanVar()
        checkbox = tk.Checkbutton(frame, variable=var,
                                command=lambda: self._toggle_image_selection(img_path, var))
        # Hide the checkbox completely
        checkbox.pack_forget()

        # Create canvas for image with exact pixel dimensions and selection styling
        canvas = tk.Canvas(frame,
                          width=config.THUMBNAIL_SIZE[0],
                          height=config.THUMBNAIL_SIZE[1],
                          bg="lightgray",
                          relief="flat",
                          borderwidth=0,
                          highlightthickness=0)
        canvas.pack()

        # Add loading text to canvas
        loading_text = canvas.create_text(
            config.THUMBNAIL_SIZE[0]//2,
            config.THUMBNAIL_SIZE[1]//2,
            text="Loading...",
            font=("Arial", 10),
            fill="black"
        )

        # Filename label
        filename_label = tk.Label(frame, text=img_path.name, wraplength=200, font=("Arial", 9))
        filename_label.pack()

        # Bind mouse wheel events to all child widgets for proper scrolling
        self._bind_mouse_wheel_to_widget_tree(frame)

        # Store references
        frame.checkbox_var = var
        frame.img_path = img_path
        frame.checkbox = checkbox
        frame.canvas = canvas
        frame.loading_text = loading_text
        frame.filename_label = filename_label
        frame.loaded = False
        frame.image_item = None  # Will store canvas image item
        frame.selected = False  # Track selection state

        self.image_widgets.append(frame)
        self.loading_placeholders[img_path] = canvas

    def _load_initial_thumbnails(self):
        """Load all thumbnails immediately when page opens, prioritized from top to bottom."""
        if not self.current_page_images:
            return

        # Load all images immediately but with priority based on position
        # Top images get highest priority, bottom images get lowest priority
        total_images = len(self.current_page_images)

        for i, img_path in enumerate(self.current_page_images):
            if i in self.loaded_thumbnails:
                continue  # Skip already loaded thumbnails

            # Calculate priority: higher number = higher priority
            # Top images (smaller index) get higher priority
            priority = total_images - i

            # Load thumbnail immediately
            self.image_service.create_thumbnail_async(
                img_path,
                self._create_thumbnail_callback(i),
                priority=priority
            )
            self.loaded_thumbnails.add(i)  # Mark as loaded

    def _create_thumbnail_callback(self, index: int):
        """Create a callback function for thumbnail loading."""
        def callback(path: Path, thumbnail: Optional[tk.PhotoImage]):
            self._on_thumbnail_loaded(index, path, thumbnail)
        return callback

    def _on_thumbnail_loaded(self, index: int, image_path: Path, thumbnail: Optional[tk.PhotoImage]):
        """Handle thumbnail loading completion."""
        if index >= len(self.image_widgets):
            return

        widget = self.image_widgets[index]

        # Check if this widget still corresponds to the expected image path
        if not hasattr(widget, 'img_path') or widget.img_path != image_path:
            return

        if thumbnail:
            # Clear the loading text
            widget.canvas.delete(widget.loading_text)

            # Add the thumbnail image to the canvas centered
            widget.image_item = widget.canvas.create_image(
                config.THUMBNAIL_SIZE[0]//2,
                config.THUMBNAIL_SIZE[1]//2,
                image=thumbnail,
                anchor=tk.CENTER
            )

            # Keep reference to prevent garbage collection
            widget.canvas.image = thumbnail

            # Ensure canvas has proper dimensions after image load
            widget.canvas.config(
                width=config.THUMBNAIL_SIZE[0],
                height=config.THUMBNAIL_SIZE[1]
            )

            # Add click handlers to canvas
            widget.canvas.bind("<Button-1>",
                              lambda e, path=image_path: self._on_image_clicked(path))
            widget.canvas.bind("<Double-Button-1>",
                              lambda e, path=image_path: self._open_image_viewer(path))
            widget.canvas.bind("<Button-3>",
                              lambda e, path=image_path: self._show_context_menu(e, path))
            widget.canvas.config(cursor="hand2")

            # Apply initial visual state
            self._update_visual_selection(widget)
        else:
            # Show error text but maintain canvas size
            widget.canvas.delete(widget.loading_text)
            widget.canvas.create_text(
                config.THUMBNAIL_SIZE[0]//2,
                config.THUMBNAIL_SIZE[1]//2,
                text="Error\nLoading",
                font=("Arial", 10),
                fill="red"
            )

        widget.loaded = True

        # Remove from loading placeholders
        if image_path in self.loading_placeholders:
            del self.loading_placeholders[image_path]

    def _on_canvas_configure(self, event):
        """Handle canvas resize events."""
        if hasattr(self, '_resize_after_id'):
            self.scrollable_frame.after_cancel(self._resize_after_id)
        self._resize_after_id = self.scrollable_frame.after(100, lambda: self._handle_resize())

    def _handle_resize(self):
        """Handle canvas resize after debounce period."""
        if self.all_image_files:
            old_cols = self.cols_per_row
            self._update_grid_dimensions()

            # If column count changed significantly, rebuild grid
            if abs(old_cols - self.cols_per_row) > 1:
                self._recreate_grid_layout()

    def _recreate_grid_layout(self):
        """Recreate the grid layout with new column count."""
        # Re-grid all existing widgets
        for i, widget in enumerate(self.image_widgets):
            row = i // self.cols_per_row
            col = i % self.cols_per_row
            widget.grid(row=row, column=col, padx=8, pady=8, sticky="n")

    def clear_images(self):
        """Clear all image widgets."""
        for widget in self.image_widgets:
            widget.destroy()

        self.image_widgets.clear()
        self.selected_images.clear()
        self.all_image_files.clear()
        self.loading_placeholders.clear()

        if self.on_selection_changed:
            self.on_selection_changed(self.selected_images)

    def select_all_images(self):
        """Select all images."""
        self.selected_images = set(self.all_image_files)

        for widget in self.image_widgets:
            if hasattr(widget, 'checkbox_var'):
                widget.checkbox_var.set(True)
                widget.selected = True

        # Update all visual selections
        self._update_all_visual_selections()

        if self.on_selection_changed:
            self.on_selection_changed(self.selected_images)

    def select_none_images(self):
        """Deselect all images."""
        self.selected_images.clear()

        for widget in self.image_widgets:
            if hasattr(widget, 'checkbox_var'):
                widget.checkbox_var.set(False)
                widget.selected = False

        # Update all visual selections
        self._update_all_visual_selections()

        if self.on_selection_changed:
            self.on_selection_changed(self.selected_images)

    def _toggle_image_selection(self, img_path: Path, var: tk.BooleanVar):
        """Toggle image selection."""
        if var.get():
            self.selected_images.add(img_path)
        else:
            self.selected_images.discard(img_path)

        if self.on_selection_changed:
            self.on_selection_changed(self.selected_images)

    def _open_image_viewer(self, img_path: Path):
        """Open image viewer."""
        if self.on_image_viewer_requested:
            self.on_image_viewer_requested(img_path)

    def _show_context_menu(self, event, img_path: Path):
        """Show context menu for image."""
        context_menu = tk.Menu(self.scrollable_frame, tearoff=0)
        context_menu.add_command(label="Open",
                                command=lambda: self._open_image_viewer(img_path))
        context_menu.add_command(label="Open with Photos",
                                command=lambda: self._open_with_photos(img_path))
        context_menu.add_separator()

        # Check if image is deleted to show appropriate options
        is_deleted = False
        if self.current_character_name:
            is_deleted = self.image_service._is_image_deleted(self.current_character_name, img_path.name)

        if is_deleted:
            context_menu.add_command(label="Undelete Image",
                                    command=lambda: self._undelete_image(img_path))

        context_menu.add_command(label="Delete from List",
                                command=lambda: self._delete_from_list(img_path))

        try:
            context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            context_menu.grab_release()

    def _undelete_image(self, img_path: Path):
        """Remove image from deleted.yaml to undelete it."""
        try:
            if not self.current_character_name:
                messagebox.showerror("Error", "No character selected")
                return

            # Remove from deleted list using the image service method
            self.image_service._remove_from_deleted_list(self.current_character_name, img_path.name)

            # Update the visual appearance of all widgets to reflect the change
            self._update_all_visual_selections()

            messagebox.showinfo("Success", f"Image '{img_path.name}' has been undeleted")

        except Exception as e:
            print(f"Error undeleting image: {e}")
            messagebox.showerror("Error", f"Failed to undelete image: {str(e)}")

    def _delete_from_list(self, img_path: Path):
        """Delete image from the current list (not from disk)."""
        try:
            # Remove from all image files list
            if img_path in self.all_image_files:
                self.all_image_files.remove(img_path)

            # Remove from selected images if it was selected
            if img_path in self.selected_images:
                self.selected_images.remove(img_path)

            # Recalculate pagination
            self.total_pages = (len(self.all_image_files) + self.images_per_page - 1) // self.images_per_page
            if self.total_pages == 0:
                self.total_pages = 1

            # Adjust current page if needed
            if self.current_page >= self.total_pages:
                self.current_page = max(0, self.total_pages - 1)

            # Clear current widgets and rebuild the grid
            for widget in self.image_widgets:
                widget.destroy()
            self.image_widgets.clear()
            self.loading_placeholders.clear()

            # Update current page images and recreate grid
            self._update_current_page_images()
            self._create_image_grid()

            # Schedule lazy loading check
            self._schedule_viewport_check()

            # Notify about selection change
            if self.on_selection_changed:
                self.on_selection_changed(self.selected_images)

            # Notify about pagination change
            if self.on_pagination_changed:
                self.on_pagination_changed(self.current_page + 1, self.total_pages)

        except Exception as e:
            print(f"Error deleting image from list: {e}")
            messagebox.showerror("Error", f"Failed to delete image from list: {str(e)}")

    def _open_with_photos(self, img_path: Path):
        """Open image with system photo viewer."""
        try:
            if platform.system() == "Windows":
                subprocess.run(["start", "ms-photos:", str(img_path)], shell=True, check=True)
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", str(img_path)], check=True)
            else:  # Linux
                subprocess.run(["xdg-open", str(img_path)], check=True)
        except Exception as e:
            print(f"Error opening with Photos: {e}")
            messagebox.showerror("Error", f"Failed to open with Photos: {str(e)}")

    def _on_image_clicked(self, img_path: Path):
        """Handle single click on image to toggle selection."""
        # Find the corresponding widget
        widget = None
        for w in self.image_widgets:
            if hasattr(w, 'img_path') and w.img_path == img_path:
                widget = w
                break

        if widget:
            # Toggle selection state
            current_state = widget.checkbox_var.get()
            new_state = not current_state
            widget.checkbox_var.set(new_state)
            widget.selected = new_state

            # Update selected images set
            if new_state:
                self.selected_images.add(img_path)
            else:
                self.selected_images.discard(img_path)

            # Update visual appearance
            self._update_visual_selection(widget)

            # Notify callback
            if self.on_selection_changed:
                self.on_selection_changed(self.selected_images)

    def _update_visual_selection(self, widget):
        """Update the visual appearance of a widget based on selection state and deleted status."""
        if not hasattr(widget, 'canvas') or not hasattr(widget, 'img_path'):
            return

        is_selected = widget.img_path in self.selected_images
        widget.selected = is_selected

        # Check if image is deleted
        is_deleted = False
        if self.current_character_name:
            is_deleted = self.image_service._is_image_deleted(self.current_character_name, widget.img_path.name)

        if is_deleted:
            # Deleted image style: red background to indicate it's deleted
            if is_selected:
                # Selected and deleted: darker red with blue highlight
                widget.canvas.config(
                    bg="#FFCDD2",  # Light red background
                    highlightbackground="#F44336",  # Red highlight
                    highlightthickness=3,
                    relief="flat"
                )
                widget.filename_label.config(font=("Arial", 9, "bold"), fg="#D32F2F", text=f"[DELETED] {widget.img_path.name}")
            else:
                # Just deleted: light red background
                widget.canvas.config(
                    bg="#FFEBEE",  # Very light red background
                    highlightbackground="#FFCDD2",  # Light red highlight
                    highlightthickness=2,
                    relief="flat"
                )
                widget.filename_label.config(font=("Arial", 9), fg="#D32F2F", text=f"[DELETED] {widget.img_path.name}")
        elif is_selected:
            # Selected style: light blue background with minimal border styling
            widget.canvas.config(
                bg="#E3F2FD",  # Light blue background
                highlightbackground="#2196F3",  # Blue highlight
                highlightthickness=2,
                relief="flat"
            )
            # Make filename label bold
            widget.filename_label.config(font=("Arial", 9, "bold"), fg="#1976D2", text=widget.img_path.name)
        else:
            # Unselected style: minimal styling for performance
            widget.canvas.config(
                bg="lightgray",
                highlightbackground="lightgray",  # No visible highlight
                highlightthickness=0,  # No highlight thickness for unselected
                relief="flat"
            )
            # Make filename label normal
            widget.filename_label.config(font=("Arial", 9), fg="black", text=widget.img_path.name)

    def _update_all_visual_selections(self):
        """Update visual selection for all widgets."""
        for widget in self.image_widgets:
            self._update_visual_selection(widget)

    def _update_pagination(self):
        """Update pagination state and recreate grid."""
        # Clear existing widgets
        for widget in self.image_widgets:
            widget.destroy()
        self.image_widgets.clear()
        self.loading_placeholders.clear()

        # Reset lazy loading state for new page
        self.loaded_thumbnails.clear()
        self.last_scroll_position = 0

        # Update current page images
        self._update_current_page_images()

        # Recreate grid with new page images
        self._create_image_grid()

        # Load thumbnails for new page
        self._load_initial_thumbnails()

        # Notify pagination change
        if self.on_pagination_changed:
            self.on_pagination_changed(self.current_page + 1, self.total_pages)

    def next_page(self):
        """Go to the next page."""
        if self.current_page < self.total_pages - 1:
            self.current_page += 1
            self._update_pagination()

    def previous_page(self):
        """Go to the previous page."""
        if self.current_page > 0:
            self.current_page -= 1
            self._update_pagination()

    def _on_scroll(self, event):
        """Handle scroll events and delegate to canvas scrolling."""
        # Get the canvas widget to scroll
        canvas = self.parent_canvas

        # Determine scroll direction and amount
        if event.num == 4 or event.delta > 0:
            # Scroll up
            canvas.yview_scroll(-1, "units")
        elif event.num == 5 or event.delta < 0:
            # Scroll down
            canvas.yview_scroll(1, "units")

        # Handle lazy loading after scroll
        self._handle_scroll_event_delayed()

        # Return "break" to prevent event from propagating further
        return "break"

    def _handle_scroll_event_delayed(self):
        """Handle scroll event with debouncing for lazy loading."""
        # Debounce scroll events to avoid excessive calls
        if self.scroll_check_pending:
            return

        self.scroll_check_pending = True
        self.parent_canvas.after(50, self._handle_scroll_event)

    def _handle_scroll_event(self):
        """Handle scroll event after debounce."""
        self.scroll_check_pending = False
        self._load_thumbnails_in_viewport()

    def _load_thumbnails_in_viewport(self):
        """Load thumbnails for images that are about to enter the viewport."""
        if not self.all_image_files or not self.current_page_images or not self.image_widgets:
            return

        try:
            # Get viewport information
            canvas_top = self.parent_canvas.canvasy(0)
            canvas_height = self.parent_canvas.winfo_height()
            canvas_bottom = canvas_top + canvas_height

            # Add buffer zones
            load_top = canvas_top - self.viewport_buffer
            load_bottom = canvas_bottom + self.viewport_buffer

            # Check each widget to see if it should be loaded
            loaded_this_cycle = 0
            for i, widget in enumerate(self.image_widgets):
                if i in self.loaded_thumbnails:
                    continue  # Skip already loaded thumbnails

                # Get widget position relative to canvas
                try:
                    widget.update_idletasks()  # Ensure geometry is updated
                    widget_top = self.parent_canvas.canvasy(widget.winfo_y())
                    widget_bottom = widget_top + widget.winfo_height()

                    # Check if widget is in the loading zone
                    if widget_bottom >= load_top and widget_top <= load_bottom:
                        # Load thumbnail
                        img_path = self.current_page_images[i]
                        self.image_service.create_thumbnail_async(
                            img_path,
                            self._create_thumbnail_callback(i),
                            priority=len(self.current_page_images) - i
                        )
                        self.loaded_thumbnails.add(i)
                        loaded_this_cycle += 1

                        # Limit how many we load per cycle to avoid blocking
                        if loaded_this_cycle >= 10:
                            break

                except Exception as e:
                    print(f"Error checking widget {i} position: {e}")
                    continue

        except Exception as e:
            print(f"Error in viewport loading: {e}")

    def _schedule_viewport_check(self):
        """Schedule periodic checks for viewport updates."""
        # Start the periodic viewport checking after a short delay
        self.parent_canvas.after(200, self._check_viewport)

    def _check_viewport(self):
        """Check the viewport and load thumbnails as needed."""
        if not self.all_image_files or not self.current_page_images:
            # Reschedule the check for when content is available
            self.parent_canvas.after(500, self._check_viewport)
            return

        # Load thumbnails in viewport
        self._load_thumbnails_in_viewport()

        # Reschedule the check
        self.parent_canvas.after(200, self._check_viewport)

    def _on_scrollbar_release(self, event):
        """Handle scrollbar release events."""
        # Trigger thumbnail loading when user releases scrollbar
        self.parent_canvas.after(100, self._load_thumbnails_in_viewport)

    def _enable_drag_and_drop(self):
        """Enable drag and drop functionality for the image grid."""
        try:
            # Check if drag & drop is available at the application level
            root = self.parent_canvas.winfo_toplevel()

            # More comprehensive check for TkinterDnD support
            tkdnd_available = False
            if hasattr(root, '_tkdnd'):
                tkdnd_available = True
                print("TkinterDnD detected via _tkdnd attribute")
            elif hasattr(root, 'tk') and hasattr(root.tk, 'call'):
                # Try to check if tkdnd commands are available
                try:
                    root.tk.call('package', 'require', 'tkdnd')
                    tkdnd_available = True
                    print("TkinterDnD detected via tkdnd package")
                except:
                    pass

            if not tkdnd_available:
                print("Warning: TkinterDnD support not detected, attempting to proceed anyway")

            # Try to register drop targets regardless of the check
            self.scrollable_frame.drop_target_register(DND_FILES)
            self.parent_canvas.drop_target_register(DND_FILES)

            # Bind the drop events
            self.scrollable_frame.dnd_bind('<<Drop>>', self._on_file_drop)
            self.parent_canvas.dnd_bind('<<Drop>>', self._on_file_drop)

            # Bind drag enter/leave events for visual feedback
            self.scrollable_frame.dnd_bind('<<DragEnter>>', self._on_drag_enter)
            self.scrollable_frame.dnd_bind('<<DragLeave>>', self._on_drag_leave)
            self.parent_canvas.dnd_bind('<<DragEnter>>', self._on_drag_enter)
            self.parent_canvas.dnd_bind('<<DragLeave>>', self._on_drag_leave)

            print("Drag & drop enabled successfully")

        except AttributeError as e:
            if "tkdnd::drop_target" in str(e):
                print("Warning: TkinterDnD not properly initialized. Drag & drop disabled.")
            else:
                print(f"AttributeError enabling drag & drop: {e}")
            # Note: Cannot modify global DND_AVAILABLE from here, but it's not critical
        except Exception as e:
            print(f"Error enabling drag & drop: {e}")
            # Note: Cannot modify global DND_AVAILABLE from here, but it's not critical

    def _on_drag_enter(self, event):
        """Handle drag enter events for visual feedback."""
        # Change background color to indicate drop zone - only for Canvas
        try:
            self.parent_canvas.config(bg="#E8F5E8")  # Light green
        except:
            pass  # Ignore if canvas doesn't support bg option

    def _on_drag_leave(self, event):
        """Handle drag leave events to restore normal appearance."""
        # Restore normal background colors - only for Canvas
        try:
            self.parent_canvas.config(bg="white")
        except:
            pass  # Ignore if canvas doesn't support bg option

    def _on_file_drop(self, event):
        """Handle file drop events."""
        try:
            # Restore normal appearance first
            self._on_drag_leave(event)

            # Check if we have a current character selected
            if not self.current_character_name:
                messagebox.showwarning("Warning", "Please select a character first before uploading images")
                return

            # Get the list of file paths from the drop event
            file_paths = self._get_file_paths_from_drop_event(event)

            if not file_paths:
                return

            # Filter for image files only
            image_files = self._filter_image_files(file_paths)

            if not image_files:
                messagebox.showwarning("Warning", "No valid image files found in the dropped items")
                return

            # Process the dropped image files
            self._process_dropped_images(image_files)

        except Exception as e:
            print(f"Error handling file drop: {e}")
            messagebox.showerror("Error", f"Failed to handle file drop: {str(e)}")

    def _get_file_paths_from_drop_event(self, event) -> List[Path]:
        """Extract file paths from the drop event."""
        file_paths = []

        try:
            if hasattr(event, 'data') and event.data:
                # Handle the data which might be a string with multiple paths
                data_string = event.data

                # Split by newlines to handle multiple files
                paths = data_string.strip().split('\n')

                for path in paths:
                    # Clean up the path (remove curly braces and extra whitespace)
                    cleaned_path = path.strip().strip('{}')
                    if cleaned_path:
                        file_paths.append(Path(cleaned_path))

        except Exception as e:
            print(f"Error parsing drop data: {e}")

        return file_paths

    def _filter_image_files(self, file_paths: List[Path]) -> List[str]:
        """Filter the file paths to only include valid image files."""
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif', '.webp'}
        image_files = []

        for file_path in file_paths:
            try:
                if file_path.exists() and file_path.is_file():
                    if file_path.suffix.lower() in valid_extensions:
                        image_files.append(str(file_path))

            except Exception as e:
                print(f"Error checking file {file_path}: {e}")
                continue

        return image_files

    def _process_dropped_images(self, image_files: List[str]):
        """Process the dropped image files by uploading them using the existing image service."""
        try:
            # Get the current stage from the parent (this will need to be passed down)
            current_stage = "1_raw"  # Default stage

            # Try to get the current stage from the callback if available
            if hasattr(self, 'get_current_stage_callback') and self.get_current_stage_callback:
                current_stage = self.get_current_stage_callback()

            # Upload the images using the existing image service
            uploaded_count = self.image_service.upload_images(
                self.current_character_name,
                current_stage,
                image_files
            )

            # Show success message
            messagebox.showinfo(
                "Upload Complete",
                f"Successfully uploaded {uploaded_count} images to {current_stage}"
            )

            # Refresh the images display if callback is available
            if hasattr(self, 'refresh_images_callback') and self.refresh_images_callback:
                self.refresh_images_callback()

        except Exception as e:
            print(f"Error processing dropped images: {e}")
            messagebox.showerror("Error", f"Failed to upload images: {str(e)}")

    def set_callbacks(self, get_current_stage_callback=None, refresh_images_callback=None):
        """Set callbacks for getting current stage and refreshing images."""
        self.get_current_stage_callback = get_current_stage_callback
        self.refresh_images_callback = refresh_images_callback

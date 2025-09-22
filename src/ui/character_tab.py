"""
Character management tab UI component.
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional, Callable
from pathlib import Path
from PIL import Image, ImageTk
import threading
import time

from src.models.character import CharacterData, CharacterRepository
from src.config.app_config import config
from src.utils.onetrainer_util import get_onetrainer_manager


class CharacterTab:
    """Character management tab component."""

    def __init__(self, parent_notebook: ttk.Notebook, character_repo: CharacterRepository, image_service):
        self.character_repo = character_repo
        self.image_service = image_service
        self.current_character: Optional[CharacterData] = None
        self.form_vars = {}

        # Flag to prevent redundant character loading
        self._loading_character = False

        # Background loading state
        self._character_info_cache = {}  # Cache for character info to avoid repeated calls
        self._background_loading = False
        self._update_queue = []  # Queue of characters needing info updates

        # Callbacks for external communication
        self.on_character_selected: Optional[Callable[[str], None]] = None
        self.on_character_updated: Optional[Callable[[], None]] = None

        # Create the tab
        self.frame = ttk.Frame(parent_notebook)
        parent_notebook.add(self.frame, text="Characters")

        self._create_widgets()
        self._load_characters()

    def _create_widgets(self):
        """Create all widgets for the character tab."""
        # Left panel - Character list
        left_panel = ttk.Frame(self.frame)
        left_panel.pack(side="left", fill="both", expand=False, padx=5)

        # Header with Characters label and refresh button
        header_frame = ttk.Frame(left_panel)
        header_frame.pack(fill="x", pady=5)

        ttk.Label(header_frame, text="Characters", font=("Arial", 12, "bold")).pack(side="left")

        # Square refresh button with reload symbol
        refresh_btn = ttk.Button(header_frame, text="⟲", command=self.refresh_character_list, width=3)
        refresh_btn.pack(side="right")

        # Character listbox with scrollbar
        list_frame = ttk.Frame(left_panel)
        list_frame.pack(fill="both", expand=True)

        self.char_listbox = tk.Listbox(list_frame, width=35)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.char_listbox.yview)
        self.char_listbox.config(yscrollcommand=scrollbar.set)

        self.char_listbox.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.char_listbox.bind('<<ListboxSelect>>', self._on_character_select)

        # Character action buttons
        btn_frame = ttk.Frame(left_panel)
        btn_frame.pack(fill="x", pady=5)

        # Create a frame for the three action buttons in one row
        action_buttons_frame = ttk.Frame(btn_frame)
        action_buttons_frame.pack(fill="x", pady=2)

        ttk.Button(action_buttons_frame, text="New", command=self._new_character).pack(side="left", fill="x", expand=True, padx=(0, 2))
        ttk.Button(action_buttons_frame, text="Delete", command=self._delete_character).pack(side="left", fill="x", expand=True, padx=(2, 2))
        ttk.Button(action_buttons_frame, text="Archive", command=self._archive_character).pack(side="left", fill="x", expand=True, padx=(2, 0))

        # Restore archived characters button
        ttk.Button(btn_frame, text="Restore archived characters", command=self._restore_archived_characters).pack(fill="x", pady=2)

        # Separator for OneTrainer section
        ttk.Separator(btn_frame, orient="horizontal").pack(fill="x", pady=5)

        # OneTrainer button
        ttk.Button(btn_frame, text="Launch OneTrainer", command=self._launch_onetrainer,
                  style='Accent.TButton').pack(fill="x", pady=2)

        # Right panel - Character details
        right_panel = ttk.Frame(self.frame)
        right_panel.pack(side="right", fill="both", expand=True, padx=5)

        ttk.Label(right_panel, text="Character Details", font=("Arial", 12, "bold")).pack(pady=5)

        # Character form in scrollable frame
        canvas = tk.Canvas(right_panel)
        scrollbar_right = ttk.Scrollbar(right_panel, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar_right.set)

        self._create_character_form(scrollable_frame)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar_right.pack(side="right", fill="y")

        # Save button
        ttk.Button(right_panel, text="Save Character", command=self._save_character).pack(pady=10)

    def _create_character_form(self, parent):
        """Create the character form fields."""
        self.form_vars = {}

        # Basic character info
        fields = [
            ("name", "Name*", "entry"),
            ("aliases", "Aliases (comma-separated)", "entry"),
            ("description", "Description", "text"),
            ("personality", "Personality", "text"),
            ("hair_color", "Hair Color", "entry"),
            ("eye_color", "Eye Color", "entry"),
            ("height_cm", "Height (cm)", "entry"),
            ("weight_kg", "Weight (kg)", "entry"),
            ("age", "Age", "entry"),
            ("birthdate", "Birthdate (YYYY-MM-DD)", "entry"),
            ("gender", "Gender", "entry"),
            ("training_prompt", "Training Prompt", "entry"),
        ]

        row = 0
        for field, label, widget_type in fields:
            ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)

            if widget_type == "entry":
                var = tk.StringVar()
                entry = ttk.Entry(parent, textvariable=var, width=40)
                entry.grid(row=row, column=1, sticky=tk.W+tk.E, padx=5, pady=2)
                self.form_vars[field] = var
            elif widget_type == "text":
                text = tk.Text(parent, height=3, width=40)
                text.grid(row=row, column=1, sticky=tk.W+tk.E, padx=5, pady=2)
                self.form_vars[field] = text

            row += 1

        # Image file selection
        self._create_image_selection_fields(parent, row)

    def _create_image_selection_fields(self, parent, start_row):
        """Create image selection fields."""
        row = start_row

        # Character image
        ttk.Label(parent, text="Character Image").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        img_frame = ttk.Frame(parent)
        img_frame.grid(row=row, column=1, sticky=tk.W+tk.E, padx=5, pady=2)

        self.form_vars["image"] = tk.StringVar()
        ttk.Entry(img_frame, textvariable=self.form_vars["image"], width=30).pack(side="left", fill="x", expand=True)
        ttk.Button(img_frame, text="Browse", command=lambda: self._browse_file("image")).pack(side="right")

        row += 1

        # Face image
        ttk.Label(parent, text="Face Image").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        face_frame = ttk.Frame(parent)
        face_frame.grid(row=row, column=1, sticky=tk.W+tk.E, padx=5, pady=2)

        self.form_vars["face_image"] = tk.StringVar()
        face_entry = ttk.Entry(face_frame, textvariable=self.form_vars["face_image"], width=30)
        face_entry.pack(side="left", fill="x", expand=True)
        ttk.Button(face_frame, text="Browse", command=lambda: self._browse_file("face_image")).pack(side="right")

        # Bind face image path changes to update display
        self.form_vars["face_image"].trace_add('write', self._update_face_image_display)

        # Face image display
        row += 1
        self.face_image_label = tk.Label(parent, text="No face image", bg="lightgray",
                                        relief="sunken", justify="center")
        self.face_image_label.grid(row=row, column=1, sticky=tk.W+tk.E, padx=5, pady=10)

    def set_preloaded_character_cache(self, character_cache: dict):
        """Set the pre-loaded character cache from splash screen initialization."""
        if character_cache:
            self._character_info_cache.update(character_cache)
            print(f"Received pre-loaded character cache with {len(character_cache)} characters")

            # If characters are already displayed, update them with cached info
            if self.char_listbox.size() > 0:
                self._update_display_with_cache()

    def _update_display_with_cache(self):
        """Update the character display using cached information."""
        for i in range(self.char_listbox.size()):
            item_text = self.char_listbox.get(i)
            char_name = item_text.split(' (')[0] if ' (' in item_text else item_text

            if char_name in self._character_info_cache:
                cached_info = self._character_info_cache[char_name]
                photo_count = cached_info.get('photo_count', 0)
                current_stage = cached_info.get('current_stage', '0')
                is_completed = cached_info.get('is_completed', False)

                # Update the display immediately
                self._update_character_display(char_name, photo_count, current_stage, is_completed)

    def _load_characters(self):
        """Load all characters from the repository using pre-loaded cache when available."""
        self.char_listbox.delete(0, tk.END)
        chars = self.character_repo.get_all_character_names()

        # Filter out the "archived" folder from the character list
        chars = [char for char in chars if char != "archived"]
        chars.sort()

        # Use pre-loaded cache when available for instant display
        for char in chars:
            # Check cache first (including splash screen pre-loaded cache)
            if char in self._character_info_cache:
                cached_info = self._character_info_cache[char]
                photo_count = cached_info.get('photo_count', 0)
                current_stage = cached_info.get('current_stage', '0')
                is_completed = cached_info.get('is_completed', False)

                # Use cached data immediately - no loading needed!
                print(f"Using pre-loaded data for character: {char}")
            else:
                # Show loading placeholder for characters not in cache
                photo_count = '...'
                current_stage = '...'
                is_completed = False
                # Add to update queue for background loading
                self._update_queue.append(char)

            # Format: {name} ({number_of_photos_stage_1}) [{current_stage}]
            display_text = f"{char} ({photo_count}) [{current_stage}]"

            # Insert into listbox
            self.char_listbox.insert(tk.END, display_text)

            # Set color for completed characters (green)
            if is_completed:
                index = self.char_listbox.size() - 1
                self.char_listbox.itemconfig(index, {'bg': 'lightgreen', 'fg': 'darkgreen'})

        # Only start background loading for characters not in cache
        if not self._background_loading and self._update_queue:
            print(f"Starting background loading for {len(self._update_queue)} characters not in cache")
            self._start_background_info_loading()
        else:
            print("All characters loaded from cache - no background loading needed!")

    def _start_background_info_loading(self):
        """Start background thread to load character info."""
        if self._background_loading:
            return

        self._background_loading = True

        def background_loader():
            """Background thread function to load character info."""
            try:
                while self._update_queue:
                    char_name = self._update_queue.pop(0)

                    try:
                        # Load character info in background
                        photo_count = self.image_service.count_stage_images(char_name, "1_raw")
                        current_stage = self.image_service.get_current_stage(char_name)
                        is_completed = self.image_service.is_character_completed(char_name)

                        # Cache the results
                        self._character_info_cache[char_name] = {
                            'photo_count': photo_count,
                            'current_stage': current_stage,
                            'is_completed': is_completed
                        }

                        # Schedule UI update on main thread
                        self.frame.after(0, self._update_character_display, char_name, photo_count, current_stage, is_completed)

                        # Small delay to prevent overwhelming the system
                        time.sleep(0.01)

                    except Exception as e:
                        print(f"Error loading info for character {char_name}: {e}")
                        # Remove from cache if there was an error
                        self._character_info_cache.pop(char_name, None)

            except Exception as e:
                print(f"Background loader error: {e}")
            finally:
                self._background_loading = False

        # Start the background thread
        thread = threading.Thread(target=background_loader, daemon=True)
        thread.start()

    def _update_character_display(self, char_name: str, photo_count: int, current_stage: str, is_completed: bool):
        """Update character display in the listbox (called from main thread)."""
        try:
            # Find the character in the listbox
            for i in range(self.char_listbox.size()):
                item_text = self.char_listbox.get(i)
                # Extract character name from display text
                displayed_char = item_text.split(' (')[0] if ' (' in item_text else item_text

                if displayed_char == char_name:
                    # Update the display text
                    new_display_text = f"{char_name} ({photo_count}) [{current_stage}]"

                    # Remember selection state
                    was_selected = i in self.char_listbox.curselection()

                    # Update the item
                    self.char_listbox.delete(i)
                    self.char_listbox.insert(i, new_display_text)

                    # Restore selection if it was selected
                    if was_selected:
                        self.char_listbox.selection_set(i)

                    # Set color for completed characters
                    if is_completed:
                        self.char_listbox.itemconfig(i, {'bg': 'lightgreen', 'fg': 'darkgreen'})
                    else:
                        self.char_listbox.itemconfig(i, {'bg': 'white', 'fg': 'black'})

                    break

        except Exception as e:
            print(f"Error updating character display for {char_name}: {e}")

    def _on_character_select(self, event):
        """Handle character selection from the list."""
        # If already loading a character, ignore new selections
        if self._loading_character:
            return

        selection = event.widget.curselection()
        if not selection:
            return

        try:
            # Get the selected index and validate it's within bounds
            selected_index = selection[0]
            if selected_index < 0 or selected_index >= self.char_listbox.size():
                return

            display_text = event.widget.get(selected_index)
            if not display_text or not display_text.strip():
                return

            # Extract character name from format: "Name (count) [stage]"
            char_name = display_text.split(' (')[0] if ' (' in display_text else display_text
            char_name = char_name.strip()

            # Validate that this character actually exists
            if char_name not in self.character_repo.get_all_character_names():
                return

            # Only proceed if this is actually a different character
            current_char_name = self.current_character.name if self.current_character else None
            if char_name == current_char_name:
                return

            # Disable the listbox during loading
            self._disable_character_list()

            self._load_character_data(char_name)

            # Notify external listeners
            if self.on_character_selected:
                self.on_character_selected(char_name)

        except (IndexError, AttributeError, TypeError) as e:
            # Silently handle any selection errors to prevent crashes
            print(f"Character selection error: {e}")
            # Re-enable the listbox if there was an error
            self._enable_character_list()
            return

    def _disable_character_list(self):
        """Disable the character listbox to prevent clicks during loading."""
        self.char_listbox.config(state='disabled')
        # Change cursor to indicate loading
        self.char_listbox.config(cursor='wait')

    def _enable_character_list(self):
        """Re-enable the character listbox after loading is complete."""
        self.char_listbox.config(state='normal')
        # Restore normal cursor
        self.char_listbox.config(cursor='')

    def _load_character_data(self, char_name: str):
        """Load character data and populate form."""
        # Prevent redundant loading if already loading this character
        if self._loading_character:
            return

        self._loading_character = True  # Set flag to prevent redundant loads
        try:
            self.current_character = self.character_repo.load_character(char_name)

            # Temporarily disable face image trace to prevent slow image loading during form population
            face_image_var = self.form_vars.get("face_image")
            if face_image_var and isinstance(face_image_var, tk.StringVar):
                # Remove existing trace - use a simpler approach
                try:
                    # Get all traces and remove the ones for 'write' mode
                    traces = face_image_var.trace_info()
                    for trace_id, mode, callback in traces:
                        if mode == 'write':
                            face_image_var.trace_remove('write', trace_id)
                except:
                    # If trace removal fails, continue anyway
                    pass

            # Clear form
            for field, var in self.form_vars.items():
                if isinstance(var, tk.StringVar):
                    var.set("")
                elif isinstance(var, tk.Text):
                    var.delete(1.0, tk.END)

            if self.current_character:
                # Populate form
                for field, var in self.form_vars.items():
                    value = getattr(self.current_character, field, "")
                    if isinstance(var, tk.StringVar):
                        if field == "aliases" and isinstance(value, list):
                            var.set(", ".join(value))
                        else:
                            var.set(str(value) if value else "")
                    elif isinstance(var, tk.Text):
                        var.insert(1.0, str(value) if value else "")

            # Re-enable face image trace after form is populated
            if face_image_var and isinstance(face_image_var, tk.StringVar):
                face_image_var.trace_add('write', self._update_face_image_display)
                # Manually trigger face image update once
                self._update_face_image_display()
        finally:
            self._loading_character = False  # Reset flag after loading
            # Re-enable the character list
            self._enable_character_list()

    def _new_character(self):
        """Create a new character."""
        dialog = tk.Toplevel(self.frame)
        dialog.title("New Character")
        dialog.geometry("300x150")
        dialog.grab_set()

        ttk.Label(dialog, text="Character Name:").pack(pady=10)
        name_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=name_var, width=30).pack(pady=5)

        def create_char():
            name = name_var.get().strip()
            if not name:
                messagebox.showerror("Error", "Please enter a character name")
                return

            if name in self.character_repo.get_all_character_names():
                messagebox.showerror("Error", "Character already exists")
                return

            success = self.character_repo.create_character_structure(name, config.STAGE_FOLDERS)
            if success:
                self._load_characters()
                dialog.destroy()

                # Select the new character
                items = list(self.char_listbox.get(0, tk.END))
                if name in items:
                    index = items.index(name)
                    self.char_listbox.selection_set(index)
                    self._load_character_data(name)

                if self.on_character_updated:
                    self.on_character_updated()
            else:
                messagebox.showerror("Error", "Failed to create character")

        ttk.Button(dialog, text="Create", command=create_char).pack(pady=10)
        ttk.Button(dialog, text="Cancel", command=dialog.destroy).pack()

    def _archive_character(self):
        """Archive selected character."""
        selection = self.char_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a character to archive")
            return

        display_text = self.char_listbox.get(selection[0])
        # Extract character name from format: "Name (count) [stage]"
        char_name = display_text.split(' (')[0] if ' (' in display_text else display_text
        char_name = char_name.strip()

        if messagebox.askyesno("Confirm Archive", f"Are you sure you want to archive '{char_name}'?\n\nThis will move the character to an archived state."):
            try:
                # For now, we'll implement a simple archive by creating an archived folder
                char_path = self.character_repo.characters_path / char_name
                archived_path = self.character_repo.characters_path / "archived"
                archived_path.mkdir(exist_ok=True)

                import shutil
                archived_char_path = archived_path / char_name

                if archived_char_path.exists():
                    messagebox.showerror("Error", f"Character '{char_name}' is already archived")
                    return

                # Move character to archived folder
                shutil.move(str(char_path), str(archived_char_path))

                # Refresh the character list
                self._load_characters()
                self.current_character = None

                if self.on_character_updated:
                    self.on_character_updated()

                messagebox.showinfo("Success", f"Character '{char_name}' has been archived successfully!")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to archive character: {str(e)}")

    def _delete_character(self):
        """Delete selected character."""
        selection = self.char_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a character to delete")
            return

        display_text = self.char_listbox.get(selection[0])
        # Extract character name from format: "Name (count) [stage]"
        char_name = display_text.split(' (')[0] if ' (' in display_text else display_text
        char_name = char_name.strip()

        if messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete '{char_name}' and all its data?"):
            success = self.character_repo.delete_character(char_name)
            if success:
                self._load_characters()
                self.current_character = None
                if self.on_character_updated:
                    self.on_character_updated()
            else:
                messagebox.showerror("Error", "Failed to delete character")

    def _save_character(self):
        """Save current character data."""
        if not self.current_character:
            messagebox.showwarning("Warning", "Please select a character first")
            return

        # Collect form data
        for field, var in self.form_vars.items():
            if isinstance(var, tk.StringVar):
                value = var.get().strip()
                if field == "aliases" and value:
                    setattr(self.current_character, field, [alias.strip() for alias in value.split(",")])
                else:
                    setattr(self.current_character, field, value)
            elif isinstance(var, tk.Text):
                value = var.get(1.0, tk.END).strip()
                setattr(self.current_character, field, value)

        success = self.character_repo.save_character(self.current_character)
        if success:
            messagebox.showinfo("Success", "Character saved successfully!")
            if self.on_character_updated:
                self.on_character_updated()
        else:
            messagebox.showerror("Error", "Failed to save character")

    def _browse_file(self, field: str):
        """Browse for image files."""
        filename = filedialog.askopenfilename(
            title=f"Select {field}",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")]
        )
        if filename:
            self.form_vars[field].set(filename)

    def _update_face_image_display(self, *args):
        """Update the face image display when path changes."""
        face_path = self.form_vars["face_image"].get()
        if face_path and Path(face_path).exists():
            try:
                img = Image.open(face_path)
                # Calculate display size to maintain aspect ratio with a larger maximum size
                # Use a more reasonable size for full image display (400x400 max)
                display_width, display_height = 400, 400

                # Calculate scaling to fit within display area while maintaining aspect ratio
                img_width, img_height = img.size
                scale = min(display_width / img_width, display_height / img_height)

                # Only scale down if image is larger than display area
                if scale < 1.0:
                    new_width = int(img_width * scale)
                    new_height = int(img_height * scale)
                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                photo = ImageTk.PhotoImage(img)
                self.face_image_label.config(image=str(photo), text="", width=display_width, height=display_height)
                self.face_image_label.image = photo  # Keep a reference
            except Exception as e:
                self.face_image_label.config(image="", text=f"Error: {e}", width=400, height=400)
                self.face_image_label.image = None
        else:
            self.face_image_label.config(image="", text="No face image", width=400, height=400)
            self.face_image_label.image = None

    def get_current_character_name(self) -> Optional[str]:
        """Get the name of the currently selected character."""
        return self.current_character.name if self.current_character else None

    def refresh_character_list(self):
        """Refresh the character list display."""
        # Clear cache to force fresh data
        self._character_info_cache.clear()
        self._update_queue.clear()
        self._load_characters()

    def _launch_onetrainer(self):
        """Launch OneTrainer application using the OneTrainer utility."""
        try:
            # Get OneTrainer manager
            ot_manager = get_onetrainer_manager()

            # Check installation status first
            status = ot_manager.get_installation_status()

            if not status['is_installed']:
                messagebox.showerror(
                    "OneTrainer Not Found",
                    f"OneTrainer is not installed.\n\n"
                    f"Error: {status.get('error_message', 'Unknown error')}\n\n"
                    "Please restart the application to install OneTrainer automatically."
                )
                return

            if not status['is_functional']:
                messagebox.showerror(
                    "OneTrainer Not Functional",
                    f"OneTrainer is installed but not functional.\n\n"
                    f"Error: {status.get('error_message', 'Unknown error')}\n\n"
                    "Please check the installation or restart the application."
                )
                return

            # Show launching message
            messagebox.showinfo(
                "Launching OneTrainer",
                "OneTrainer is starting up. This may take a moment...\n\n"
                "A new window will open with the OneTrainer interface."
            )

            # Launch OneTrainer
            success, error_message, process = ot_manager.launch()

            if success:
                print(f"OneTrainer launched successfully from: {ot_manager.onetrainer_dir}")
            else:
                print(f"Failed to launch OneTrainer: {error_message}")
                messagebox.showerror(
                    "Launch Failed",
                    f"Failed to launch OneTrainer:\n\n{error_message}"
                )

        except Exception as e:
            print(f"Error in OneTrainer launcher: {e}")
            messagebox.showerror(
                "OneTrainer Error",
                f"An error occurred while trying to launch OneTrainer:\n{str(e)}"
            )

    def _restore_archived_characters(self):
        """Show popup to restore individual archived characters."""
        archived_path = self.character_repo.characters_path / "archived"

        if not archived_path.exists():
            messagebox.showinfo("Restore Characters", "No archived characters found")
            return

        # Get list of archived characters
        archived_chars = [char.name for char in archived_path.iterdir() if char.is_dir()]

        if not archived_chars:
            messagebox.showinfo("Restore Characters", "No archived characters found")
            return

        # Create restore dialog
        dialog = tk.Toplevel(self.frame)
        dialog.title("Restore Archived Characters")
        dialog.geometry("400x200")
        dialog.grab_set()

        # Center the dialog relative to parent window
        dialog.transient(self.frame.winfo_toplevel())

        # Update the dialog to get accurate dimensions
        dialog.update_idletasks()

        # Get parent window position and size
        parent = self.frame.winfo_toplevel()
        parent_x = parent.winfo_x()
        parent_y = parent.winfo_y()
        parent_width = parent.winfo_width()
        parent_height = parent.winfo_height()

        # Get dialog dimensions
        dialog_width = dialog.winfo_width()
        dialog_height = dialog.winfo_height()

        # Calculate center position
        x = parent_x + (parent_width // 2) - (dialog_width // 2)
        y = parent_y + (parent_height // 2) - (dialog_height // 2)

        # Set the dialog position
        dialog.geometry(f"{dialog_width}x{dialog_height}+{x}+{y}")

        ttk.Label(dialog, text="Select a character to restore:", font=("Arial", 10, "bold")).pack(pady=10)

        # Dropdown for character selection
        selected_char = tk.StringVar()
        char_dropdown = ttk.Combobox(dialog, textvariable=selected_char, values=archived_chars,
                                    state="readonly", width=40)
        char_dropdown.pack(pady=10)

        if archived_chars:
            char_dropdown.current(0)  # Select first character by default

        # Buttons frame
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=20)

        def restore_character():
            char_name = selected_char.get()
            if not char_name:
                messagebox.showwarning("Warning", "Please select a character to restore")
                return

            if messagebox.askyesno("Confirm Restore", f"Are you sure you want to restore '{char_name}'?"):
                try:
                    import shutil

                    # Move character from archived to main characters folder
                    archived_char_path = archived_path / char_name
                    main_char_path = self.character_repo.characters_path / char_name

                    if main_char_path.exists():
                        messagebox.showerror("Error", f"Character '{char_name}' already exists in main characters list")
                        return

                    shutil.move(str(archived_char_path), str(main_char_path))

                    messagebox.showinfo("Success", f"Character '{char_name}' has been restored successfully!")

                    # Refresh the character list
                    self._load_characters()

                    if self.on_character_updated:
                        self.on_character_updated()

                    dialog.destroy()

                except Exception as e:
                    messagebox.showerror("Error", f"Failed to restore character: {str(e)}")

        ttk.Button(btn_frame, text="Restore", command=restore_character).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Cancel", command=dialog.destroy).pack(side="left", padx=5)

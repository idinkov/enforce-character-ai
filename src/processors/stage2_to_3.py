"""
Stage 2 to 3: Upscale Processor
Monitors upscaling progress by tracking file counts
"""
import threading
import subprocess
import os
from .base_processor import BaseProcessor
from ..config.app_config import config
from ..models.model_manager import ModelManager
from ..utils.gfpgan_upscaler import GFPGANUpscaler


class UpscaleProcessor(BaseProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.monitoring_thread = None
        self.stop_monitoring = False
        self.source_dir = None
        self.target_dir = None
        self.completion_event = threading.Event()
        self.process_completed = False
        # New variables for enhanced functionality
        self.no_progress_checks = 0
        self.last_processed_count = 0
        self.remaining_dir = None
        self.has_remaining_files = False
        self.process_skipped = False
        # Dialog reference for integration
        self.progress_dialog = None
        # Store character name for deletion filtering
        self.current_char_name = None

    def set_progress_dialog(self, dialog):
        """Set the progress dialog reference for UI integration"""
        self.progress_dialog = dialog

    def get_source_stage(self):
        return "2_raw_filtered"

    def get_target_stage(self):
        return "3_raw_upscaled"

    def process_character(self, char_name):
        """Process character with automatic or manual upscaling based on configuration"""
        try:
            # Store character name for use in other methods
            self.current_char_name = char_name
            self.log(f"Starting upscaling for {char_name}...")

            self.source_dir = self.characters_path / char_name / "images" / self.get_source_stage()
            self.target_dir = self.characters_path / char_name / "images" / self.get_target_stage()
            self.remaining_dir = self.source_dir / "remaining"

            if not self.source_dir.exists():
                self.log(f"Error: Source directory {self.get_source_stage()} does not exist")
                return False

            # Physically delete images marked as deleted in deleted.yaml
            deleted_count = self.physically_delete_marked_images(char_name, self.source_dir)
            if deleted_count > 0:
                self.log(f"Physically deleted {deleted_count} images marked as deleted")

            # Ensure target directory exists
            self.target_dir.mkdir(parents=True, exist_ok=True)

            # First, remove any prefixes from target files to match source filenames
            self._normalize_target_filenames(char_name, self.source_dir, self.target_dir)

            # Count source files once, excluding deleted images
            source_files = self.get_image_files(self.source_dir, char_name)
            total_source_files = len(source_files)

            if total_source_files == 0:
                self.log(f"No images found in {self.get_source_stage()}")
                return False

            # Check for pre-existing files and handle remaining files
            self._handle_pre_existing_files(char_name, source_files)

            # Refresh source files list after handling pre-existing files
            source_files = self.get_image_files(self.source_dir, char_name)
            remaining_source_files = len(source_files)

            if remaining_source_files == 0:
                self.log("All images are already processed. No additional processing needed.")
                self.process_completed = True
                self._handle_completion_cleanup()
                return True

            self.log(f"Found {remaining_source_files} images to process")

            # Check if automatic upscaling is enabled
            if config.AUTOMATIC_UPSCALING:
                return self._process_automatic_upscaling(char_name, source_files)
            else:
                return self._process_manual_upscaling(char_name, remaining_source_files)

        except Exception as e:
            self.log(f"Error during upscaling setup: {e}")
            return False

    def _process_automatic_upscaling(self, char_name, source_files):
        """Process automatic upscaling using GFPGAN"""
        try:
            self.log("Automatic upscaling enabled - using GFPGAN")

            # Initialize model manager and GFPGAN upscaler
            model_manager = ModelManager()

            # Ensure GFPGAN model is available
            if not model_manager.is_model_available("gfpgan"):
                self.log("GFPGAN model not found. Attempting to download...")
                if self.progress_dialog:
                    self.progress_dialog.update_stage("Downloading GFPGAN model...", 0)

                success = model_manager.download_model("gfpgan", self._download_progress_callback)
                if not success:
                    self.log("Failed to download GFPGAN model. Falling back to manual upscaling.")
                    return self._process_manual_upscaling(char_name, len(source_files))

                self.log("GFPGAN model downloaded successfully")

            # Initialize GFPGAN upscaler
            upscaler = GFPGANUpscaler(
                model_manager=model_manager,
                scale_factor=config.UPSCALING_SCALE_FACTOR,
                quality_preset=config.UPSCALING_QUALITY_PRESET
            )

            if self.progress_dialog:
                self.progress_dialog.update_stage("Initializing GFPGAN...", 0)
                # Add automatic upscaling buttons
                self._add_automatic_upscaling_buttons(upscaler)

            self.log(f"Starting automatic upscaling of {len(source_files)} images")
            self.log(f"Scale factor: {config.UPSCALING_SCALE_FACTOR}x, Quality: {config.UPSCALING_QUALITY_PRESET}")

            # Process images with GFPGAN
            successful_count, total_count = upscaler.upscale_batch(
                input_dir=self.source_dir,
                output_dir=self.target_dir,
                progress_callback=self._upscaling_progress_callback
            )

            # Clean up GFPGAN resources
            upscaler.cleanup()

            if successful_count == total_count:
                self.log(f"Automatic upscaling completed successfully: {successful_count}/{total_count} images")
                self.process_completed = True
                if self.progress_dialog:
                    self.progress_dialog.update_stage("✓ Automatic upscaling completed!", 100)
                    self.progress_dialog.update_stage_progress(100)
                self._handle_completion_cleanup()
                return True
            else:
                self.log(f"Automatic upscaling partially completed: {successful_count}/{total_count} images")
                if self.progress_dialog:
                    self.progress_dialog.update_stage(f"Partial completion: {successful_count}/{total_count}",
                                                    (successful_count / total_count) * 100)

                # If some files failed, offer option to continue manually
                if successful_count > 0:
                    return self._handle_partial_completion(successful_count, total_count)
                else:
                    self.log("Automatic upscaling failed completely. Falling back to manual mode.")
                    return self._process_manual_upscaling(char_name, len(source_files))

        except Exception as e:
            self.log(f"Error during automatic upscaling: {e}")
            self.log("Falling back to manual upscaling mode")
            return self._process_manual_upscaling(char_name, len(source_files))

    def _process_manual_upscaling(self, char_name, remaining_source_files):
        """Process manual upscaling (original monitoring behavior)"""
        try:
            self.log("Manual upscaling mode - monitoring progress")

            # Add upscaling-specific buttons to the dialog if available
            if self.progress_dialog:
                self._add_upscaling_buttons()

            # Update dialog with upscaling instructions
            if self.progress_dialog:
                self.progress_dialog.update_stage("Ready for Manual Upscaling", 0)
                self.log("Please use external software to apply upscaling here. We are tracking your progress.")
                self.log(f"Input folder: {self.source_dir}")
                self.log(f"Target folder: {self.target_dir}")

            # Reset completion tracking
            self.completion_event.clear()
            self.process_completed = False
            self.process_skipped = False
            self.no_progress_checks = 0
            self.last_processed_count = 0

            # Start monitoring in a separate thread
            self.stop_monitoring = False
            self.monitoring_thread = threading.Thread(
                target=self._monitor_progress,
                args=(self.target_dir, remaining_source_files, char_name),
                daemon=True
            )
            self.monitoring_thread.start()

            # Wait for completion before returning
            self.log("Waiting for upscaling to complete...")
            self.completion_event.wait()

            # Handle post-completion cleanup
            if self.process_completed and not self.process_skipped:
                self._handle_completion_cleanup()

            self.log("Manual upscaling process completed")
            return self.process_completed

        except Exception as e:
            self.log(f"Error during manual upscaling: {e}")
            return False

    def _download_progress_callback(self, downloaded_bytes, total_bytes, model_name):
        """Callback for model download progress"""
        if total_bytes > 0:
            progress = (downloaded_bytes / total_bytes) * 100
            self.log(f"Downloading {model_name}: {progress:.1f}%")
            if self.progress_dialog:
                self.progress_dialog.update_stage_progress(progress)

    def _upscaling_progress_callback(self, current, total, filename):
        """Callback for upscaling progress"""
        progress = (current / total) * 100 if total > 0 else 0
        self.log(f"Upscaling progress: {current}/{total} files ({progress:.1f}%) - {filename}")

        if self.progress_dialog:
            self.progress_dialog.update_stage_progress(progress)
            self.progress_dialog.update_stage(f"Upscaling: {current}/{total} files", progress)

        # Update main progress callback
        self.update_progress(progress)

    def _handle_partial_completion(self, successful_count, total_count):
        """Handle partial completion of automatic upscaling"""
        try:
            import tkinter.messagebox as messagebox

            parent_window = self.progress_dialog.popup if self.progress_dialog else None

            message = (
                f"Automatic upscaling completed {successful_count} out of {total_count} images.\n\n"
                f"Do you want to:\n"
                f"• Continue with {successful_count} successfully upscaled images, or\n"
                f"• Switch to manual mode to process the remaining {total_count - successful_count} images?"
            )

            result = messagebox.askyesno(
                "Partial Upscaling Completion",
                message,
                parent=parent_window
            )

            if result:  # Continue with partial results
                self.process_completed = True
                self.log(f"Continuing with {successful_count} successfully upscaled images")
                if self.progress_dialog:
                    self.progress_dialog.update_stage("✓ Continuing with partial results", 100)
                self._handle_completion_cleanup()
                return True
            else:  # Switch to manual mode
                self.log("Switching to manual upscaling mode for remaining files")
                return self._process_manual_upscaling(None, total_count - successful_count)

        except Exception as e:
            self.log(f"Error handling partial completion: {e}")
            # Default to completing with partial results
            self.process_completed = True
            return True

    def _add_automatic_upscaling_buttons(self, upscaler):
        """Add automatic upscaling specific buttons to the progress dialog"""
        if not self.progress_dialog or not hasattr(self.progress_dialog, 'popup'):
            return

        try:
            import tkinter as tk
            from tkinter import ttk

            # Create additional button frame for automatic upscaling controls
            dialog_popup = self.progress_dialog.popup

            # Add automatic upscaling buttons frame
            self.auto_upscaling_frame = ttk.LabelFrame(dialog_popup, text="Automatic Upscaling")
            self.auto_upscaling_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

            # Info frame
            info_frame = ttk.Frame(self.auto_upscaling_frame)
            info_frame.pack(fill=tk.X, padx=10, pady=5)

            info_label = ttk.Label(
                info_frame,
                text=f"GFPGAN automatic upscaling enabled ({config.UPSCALING_SCALE_FACTOR}x, {config.UPSCALING_QUALITY_PRESET} quality)"
            )
            info_label.pack()

            # Button frame
            button_frame = ttk.Frame(self.auto_upscaling_frame)
            button_frame.pack(fill=tk.X, padx=10, pady=5)

            # Stop automatic upscaling button
            self.stop_auto_button = ttk.Button(
                button_frame,
                text="Stop & Switch to Manual",
                command=lambda: self._stop_automatic_upscaling(upscaler)
            )
            self.stop_auto_button.pack(side="left", padx=(0, 10))

            # Open folders buttons
            self.auto_input_button = ttk.Button(
                button_frame,
                text="Open Input Folder",
                command=self._open_input_folder
            )
            self.auto_input_button.pack(side="left", padx=(0, 10))

            self.auto_target_button = ttk.Button(
                button_frame,
                text="Open Target Folder",
                command=self._open_target_folder
            )
            self.auto_target_button.pack(side="left")

        except Exception as e:
            self.log(f"Error adding automatic upscaling buttons: {e}")

    def _stop_automatic_upscaling(self, upscaler):
        """Stop automatic upscaling and switch to manual mode"""
        try:
            import tkinter.messagebox as messagebox

            parent_window = self.progress_dialog.popup if self.progress_dialog else None

            result = messagebox.askyesno(
                "Stop Automatic Upscaling",
                "Are you sure you want to stop automatic upscaling and switch to manual mode?",
                parent=parent_window
            )

            if result:
                self.log("Stopping automatic upscaling...")
                upscaler.cleanup()

                # Count already processed files
                processed_files = []
                if self.target_dir.exists():
                    for file_path in self.target_dir.iterdir():
                        if file_path.is_file() and file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']:
                            processed_files.append(file_path)

                remaining_count = len(self.get_image_files(self.source_dir, self.current_char_name)) - len(processed_files)

                if remaining_count > 0:
                    self.log(f"Switching to manual mode with {remaining_count} remaining files")
                    return self._process_manual_upscaling(None, remaining_count)
                else:
                    self.log("All files have been processed")
                    self.process_completed = True
                    self.completion_event.set()
                    return True

        except Exception as e:
            self.log(f"Error stopping automatic upscaling: {e}")

    def _add_upscaling_buttons(self):
        """Add upscaling-specific buttons to the progress dialog"""
        if not self.progress_dialog or not hasattr(self.progress_dialog, 'popup'):
            return

        try:
            import tkinter as tk
            from tkinter import ttk

            # Create additional button frame for upscaling controls
            dialog_popup = self.progress_dialog.popup

            # Unpack the log frame and button frame to reorder them
            self.progress_dialog.log_frame.pack_forget()
            self.progress_dialog.button_frame.pack_forget()

            # Add upscaling buttons frame after Current Stage and before log frame
            self.upscaling_frame = ttk.LabelFrame(dialog_popup, text="Upscaling Controls")
            self.upscaling_frame.pack(fill=tk.X, padx=10, pady=10)

            # Single row for all buttons
            buttons_frame = ttk.Frame(self.upscaling_frame)
            buttons_frame.pack(fill=tk.X, padx=10, pady=5)

            # Open Input Folder button
            self.input_folder_button = ttk.Button(
                buttons_frame,
                text="Open Input Folder",
                command=self._open_input_folder
            )
            self.input_folder_button.pack(side="left", padx=(0, 5))

            # Open Target Folder button
            self.target_folder_button = ttk.Button(
                buttons_frame,
                text="Open Target Folder",
                command=self._open_target_folder
            )
            self.target_folder_button.pack(side="left", padx=(0, 5))

            # Skip button
            self.skip_button = ttk.Button(
                buttons_frame,
                text="Skip Upscaling",
                command=self._skip_upscaling
            )
            self.skip_button.pack(side="left", padx=(0, 5))

            # Move to remaining button (initially hidden)
            self.remaining_button = ttk.Button(
                buttons_frame,
                text="Move Remaining Files",
                command=self._move_remaining_files
            )

            # Re-pack the log frame and button frame
            self.progress_dialog.log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            self.progress_dialog.button_frame.pack(fill=tk.X, padx=10, pady=10)

        except Exception as e:
            self.log(f"Error adding upscaling buttons: {e}")

    def _open_input_folder(self):
        """Open the input folder in file explorer"""
        try:
            if self.source_dir and self.source_dir.exists():
                if os.name == 'nt':  # Windows
                    subprocess.run(['explorer', str(self.source_dir)], check=True)
                elif os.name == 'posix':  # macOS and Linux
                    subprocess.run(['open' if os.uname().sysname == 'Darwin' else 'xdg-open', str(self.source_dir)], check=True)
                self.log(f"Opened input folder: {self.source_dir}")
        except Exception as e:
            self.log(f"Error opening input folder: {e}")

    def _open_target_folder(self):
        """Open the target folder in file explorer"""
        try:
            folder_to_open = self.remaining_dir if self.has_remaining_files and self.remaining_dir.exists() else self.target_dir
            if folder_to_open and folder_to_open.exists():
                if os.name == 'nt':  # Windows
                    subprocess.run(['explorer', str(folder_to_open)], check=True)
                elif os.name == 'posix':  # macOS and Linux
                    subprocess.run(['open' if os.uname().sysname == 'Darwin' else 'xdg-open', str(folder_to_open)], check=True)
                self.log(f"Opened folder: {folder_to_open}")
        except Exception as e:
            self.log(f"Error opening folder: {e}")

    def _monitor_progress(self, target_dir, total_files, char_name):
        """Monitor progress by checking target directory every 5 seconds"""
        import time

        while not self.stop_monitoring:
            try:
                # Count files in target directory (excluding remaining folder)
                target_files = []
                if target_dir.exists():
                    for file_path in target_dir.iterdir():
                        if file_path.is_file() and file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']:
                            target_files.append(file_path)

                processed_count = len(target_files)

                # Check for no progress
                if processed_count == self.last_processed_count:
                    self.no_progress_checks += 1
                    self.log(f"No progress detected: {self.no_progress_checks}/8 checks")

                    # Show remaining files button after 8 checks with no progress
                    if self.no_progress_checks >= 8:
                        self._show_remaining_button()
                else:
                    self.no_progress_checks = 0  # Reset counter if progress is made
                    self.last_processed_count = processed_count

                # Calculate progress percentage
                progress_percent = min((processed_count / total_files) * 100, 100) if total_files > 0 else 0

                # Update progress dialog
                if self.progress_dialog:
                    self.progress_dialog.update_stage_progress(progress_percent)
                    self.progress_dialog.update_stage(f"Upscaling Progress: {processed_count}/{total_files} files", progress_percent)

                # Update main progress callback
                self.update_progress(progress_percent)

                # Log progress
                self.log(f"Progress: {processed_count}/{total_files} files ({progress_percent:.1f}%)")

                # Check if complete
                if processed_count >= total_files:
                    self.log("Upscaling complete!")
                    self._on_completion()
                    break

                # Wait 5 seconds before next check
                time.sleep(5)

            except Exception as e:
                self.log(f"Error during progress monitoring: {e}")
                break

    def _on_completion(self):
        """Handle completion of upscaling process"""
        try:
            self.stop_monitoring = True
            self.process_completed = True

            # Update dialog to show completion
            if self.progress_dialog:
                self.progress_dialog.update_stage("✓ Upscaling completed!", 100)
                self.progress_dialog.update_stage_progress(100)

            self.log("Upscaling completed successfully!")

            # Signal completion to waiting thread
            self.completion_event.set()

        except Exception as e:
            self.log(f"Error in completion handler: {e}")

    def _handle_pre_existing_files(self, char_name, source_files):
        """Check for pre-existing upscaled files and manage remaining files"""
        try:
            # Check if there are already processed files in target directory
            existing_target_files = set()
            if self.target_dir.exists():
                for file_path in self.target_dir.iterdir():
                    if file_path.is_file() and file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']:
                        existing_target_files.add(file_path.stem.lower())

            # If there are pre-existing target files, move unprocessed source files to remaining
            if existing_target_files:
                self.log(f"Found {len(existing_target_files)} pre-existing processed files in target directory")

                # Create remaining directory if it doesn't exist
                self.remaining_dir.mkdir(parents=True, exist_ok=True)

                # Move source files that don't have a corresponding target file to remaining
                moved_count = 0
                for source_file in source_files[:]:  # Create a copy to iterate over
                    if source_file.stem.lower() not in existing_target_files:
                        # Move unprocessed source file to remaining directory
                        remaining_path = self.remaining_dir / source_file.name
                        if not remaining_path.exists():
                            source_file.rename(remaining_path)
                            moved_count += 1
                            self.log(f"Moved unprocessed source file to remaining: {source_file.name}")

                if moved_count > 0:
                    self.has_remaining_files = True
                    self.log(f"Moved {moved_count} unprocessed source files to remaining folder")

                # Update total source files count after moving files
                total_source_files = len(source_files)

                # If no source files left to process, mark as completed
                if total_source_files == 0:
                    self.log("No new images to process after moving unprocessed files to remaining.")
                    self.process_completed = True
                    self.completion_event.set()

        except Exception as e:
            self.log(f"Error handling pre-existing files: {e}")

    def _handle_completion_cleanup(self):
        """Handle cleanup actions after completion of upscaling process"""
        try:
            # If there are remaining files, move them back to the source directory
            if self.remaining_dir.exists() and any(self.remaining_dir.glob("*")):
                for file_path in self.remaining_dir.glob("*"):
                    if file_path.is_file():
                        # Move file back to source directory
                        new_location = self.source_dir / file_path.name
                        if not new_location.exists():
                            file_path.rename(new_location)
                            self.log(f"Moved remaining file back to source: {file_path} -> {new_location}")
                        else:
                            self.log(f"Skipped moving remaining file (already exists in source): {file_path}")

            # Clean up: remove the remaining directory if empty
            if self.remaining_dir.exists() and not any(self.remaining_dir.glob("*")):
                self.remaining_dir.rmdir()
                self.log(f"Removed empty remaining directory: {self.remaining_dir}")

            # Final step: Ensure target directory has clean filenames matching source
            self._finalize_target_filenames()

        except Exception as e:
            self.log(f"Error during completion cleanup: {e}")

    def _finalize_target_filenames(self):
        """Final cleanup to ensure target directory has clean filenames matching source directory"""
        try:
            if not self.target_dir.exists() or not self.source_dir.exists():
                return

            self.log("Starting final filename normalization...")

            # Reuse the existing normalization method from base processor
            self._normalize_target_filenames(None, self.source_dir, self.target_dir)

            self.log("Final filename normalization completed")

        except Exception as e:
            self.log(f"Error during final filename normalization: {e}")

    def _skip_upscaling(self):
        """Skip the upscaling process with warning if partially completed"""
        try:
            import tkinter.messagebox as messagebox

            # Count currently processed files
            target_files = []
            if self.target_dir.exists():
                for file_path in self.target_dir.iterdir():
                    if file_path.is_file() and file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']:
                        target_files.append(file_path)

            processed_count = len(target_files)

            # Get total source files count
            source_files = self.get_image_files(self.source_dir, self.current_char_name) if self.source_dir.exists() else []
            total_files = len(source_files)

            # Show warning if only some images were processed
            if processed_count > 0 and processed_count < total_files:
                warning_message = (
                    f"Warning: Only {processed_count} out of {total_files} images have been upscaled.\n\n"
                    "If you skip now, only the processed images will be used in the rest of the stages, "
                    "and the remaining images will be ignored.\n\n"
                    "Do you want to continue with skipping?"
                )

                parent_window = self.progress_dialog.popup if self.progress_dialog else None
                result = messagebox.askyesno(
                    "Partial Upscaling Warning",
                    warning_message,
                    parent=parent_window
                )
                if not result:
                    return

            # Confirm skip action
            parent_window = self.progress_dialog.popup if self.progress_dialog else None
            confirm_result = messagebox.askyesno(
                "Confirm Skip",
                "Are you sure you want to skip the upscaling process?",
                parent=parent_window
            )

            if confirm_result:
                self.process_skipped = True
                self.process_completed = True
                self.stop_monitoring = True

                if self.progress_dialog:
                    self.progress_dialog.update_stage("Upscaling skipped", 100)

                self.log(f"Upscaling skipped. Processed {processed_count} out of {total_files} images.")
                self.completion_event.set()

        except Exception as e:
            self.log(f"Error during skip operation: {e}")

    def _show_remaining_button(self):
        """Show the Move Remaining Files button when no progress is detected"""
        try:
            if hasattr(self, 'remaining_button'):
                self.remaining_button.pack(side="left")
                self.log("No progress detected for 8 checks. Move Remaining Files button is now available.")
        except Exception as e:
            self.log(f"Error showing remaining button: {e}")

    def _move_remaining_files(self):
        """Move unprocessed files to the remaining folder"""
        try:
            import tkinter.messagebox as messagebox

            # Confirm the action
            parent_window = self.progress_dialog.popup if self.progress_dialog else None

            # Get list of source files
            source_files = self.get_image_files(self.source_dir, self.current_char_name) if self.source_dir.exists() else []

            # Get list of already processed files in target directory
            processed_files = set()
            if self.target_dir.exists():
                for file_path in self.target_dir.iterdir():
                    if file_path.is_file() and file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']:
                        processed_files.add(file_path.stem.lower())

            # Ensure remaining directory exists
            self.remaining_dir.mkdir(parents=True, exist_ok=True)

            # Move unprocessed source files to remaining
            moved_count = 0
            for source_file in source_files:
                # Check if this file has been processed (exists in target directory)
                if source_file.stem.lower() not in processed_files:
                    # Check if it's not already in remaining
                    remaining_path = self.remaining_dir / source_file.name
                    if not remaining_path.exists():
                        # Move source file to remaining directory (not copy)
                        source_file.rename(remaining_path)
                        moved_count += 1
                        self.log(f"Moved unprocessed file to remaining: {source_file.name}")

            # Update UI and complete the process
            if moved_count > 0:
                self.has_remaining_files = True

                # Update target folder button text
                if hasattr(self, 'target_folder_button'):
                    self.target_folder_button.config(text="Open Remaining Folder")

                messagebox.showinfo(
                    "Files Moved",
                    f"Moved {moved_count} unprocessed files to the remaining folder.\n\n"
                    "You can process these files later by running the upscaling stage again.",
                    parent=parent_window
                )

                self.log(f"Moved {moved_count} unprocessed files to remaining folder.")
            else:
                messagebox.showinfo("No Files to Move", "No unprocessed files found to move.", parent=parent_window)
                self.log("No unprocessed files found to move to remaining folder.")

            # Mark as completed with current progress
            self.process_completed = True
            self.stop_monitoring = True

            if self.progress_dialog:
                self.progress_dialog.update_stage("Remaining files moved", 100)

            # Hide remaining button
            if hasattr(self, 'remaining_button'):
                self.remaining_button.pack_forget()

            self.completion_event.set()

        except Exception as e:
            self.log(f"Error moving remaining files: {e}")
            import tkinter.messagebox as messagebox
            parent_window = self.progress_dialog.popup if self.progress_dialog else None
            messagebox.showerror("Error", f"Failed to move remaining files: {e}", parent=parent_window)

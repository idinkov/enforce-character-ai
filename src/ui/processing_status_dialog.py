"""
Processing status dialog for handling stage processing operations.
"""
from tkinter import messagebox
import threading
import tkinter as tk
from tkinter import ttk
from typing import Optional, Callable, List
from datetime import datetime, timedelta
import time

from src.config.app_config import config


class ProgressPopup:
    """A popup window that shows progress for long-running operations."""

    def __init__(self, parent, title: str, operation_name: str, character_name: str):
        self.parent = parent
        self.operation_name = operation_name
        self.character_name = character_name
        self.cancelled = False
        self.completed = False
        self.start_time = datetime.now()

        # Stage tracking
        self.stage_start_time = None
        self.current_stage = ""

        # Progress tracking for gradual overall progress
        self.total_stages = 1  # Default to 1 for single stage operations
        self.current_stage_index = 0
        self.current_stage_progress = 0.0

        # Callbacks
        self.on_cancel: Optional[Callable[[], None]] = None
        self.on_complete: Optional[Callable[[bool], None]] = None

        # Create popup window
        self.popup = tk.Toplevel(parent)
        self.popup.title(title)
        self.popup.geometry("640x640")
        self.popup.resizable(True, True)

        # Center the popup
        self._center_popup()

        # Make popup modal but keep main window responsive
        self.popup.transient(parent)
        # Don't grab_set to keep main window responsive

        # Handle window close
        self.popup.protocol("WM_DELETE_WINDOW", self._on_close)

        self._create_widgets()

    def set_total_stages(self, total: int):
        """Set the total number of stages for gradual progress calculation."""
        self.total_stages = max(1, total)

    def set_current_stage_index(self, index: int):
        """Set the current stage index (0-based)."""
        self.current_stage_index = max(0, min(index, self.total_stages - 1))
        self._update_gradual_overall_progress()

    def _calculate_gradual_overall_progress(self) -> float:
        """Calculate gradual overall progress based on current stage and its progress."""
        if self.total_stages <= 1:
            return self.current_stage_progress

        # Each stage contributes equally to the overall progress
        stage_weight = 100.0 / self.total_stages

        # Progress from completed stages
        completed_stages_progress = self.current_stage_index * stage_weight

        # Progress from current stage
        current_stage_contribution = (self.current_stage_progress / 100.0) * stage_weight

        # Total gradual progress
        total_progress = completed_stages_progress + current_stage_contribution

        return min(100.0, max(0.0, total_progress))

    def _update_gradual_overall_progress(self):
        """Update the overall progress bar with gradual progress."""
        gradual_progress = self._calculate_gradual_overall_progress()
        self.overall_progress_var.set(gradual_progress)
        self.overall_percentage_label.config(text=f"{gradual_progress:.1f}%")
        self.popup.update_idletasks()

    def _center_popup(self):
        """Center the popup window on the parent."""
        self.popup.update_idletasks()

        # Get parent window position and size
        parent_x = self.parent.winfo_rootx()
        parent_y = self.parent.winfo_rooty()
        parent_width = self.parent.winfo_width()
        parent_height = self.parent.winfo_height()

        # Calculate center position
        popup_width = 640
        popup_height = 640  # Reduced height
        x = parent_x + (parent_width - popup_width) // 2
        y = parent_y + (parent_height - popup_height) // 2

        self.popup.geometry(f"{popup_width}x{popup_height}+{x}+{y}")

    def _create_widgets(self):
        """Create all widgets for the popup."""
        # Header frame
        header_frame = ttk.Frame(self.popup)
        header_frame.pack(fill="x", padx=10, pady=10)

        # Operation title
        title_label = ttk.Label(header_frame, text=self.operation_name, font=("Arial", 14, "bold"))
        title_label.pack()

        # Character name
        char_label = ttk.Label(header_frame, text=f"Character: {self.character_name}", font=("Arial", 11))
        char_label.pack(pady=(5, 0))

        # Time info frame
        time_frame = ttk.Frame(header_frame)
        time_frame.pack(fill="x", pady=(5, 0))

        self.elapsed_time_label = ttk.Label(time_frame, text="Elapsed: 00:00:00", font=("Arial", 9))
        self.elapsed_time_label.pack(side="left")

        self.eta_label = ttk.Label(time_frame, text="ETA: --:--:--", font=("Arial", 9))
        self.eta_label.pack(side="right")

        # Overall progress frame
        overall_frame = ttk.LabelFrame(self.popup, text="Overall Progress")
        overall_frame.pack(fill="x", padx=10, pady=10)

        # Overall progress bar with percentage
        overall_progress_frame = ttk.Frame(overall_frame)
        overall_progress_frame.pack(fill="x", padx=10, pady=5)

        self.overall_progress_var = tk.DoubleVar()
        self.overall_progress_bar = ttk.Progressbar(
            overall_progress_frame,
            variable=self.overall_progress_var,
            maximum=100,
            length=400
        )
        self.overall_progress_bar.pack(side="left", fill="x", expand=True)

        self.overall_percentage_label = ttk.Label(overall_progress_frame, text="0.0%", font=("Arial", 10, "bold"))
        self.overall_percentage_label.pack(side="right", padx=(10, 0))

        # Current stage frame
        self.stage_frame = ttk.LabelFrame(self.popup, text="Current Stage")
        self.stage_frame.pack(fill="x", padx=10, pady=10)

        # Current stage info
        stage_info_frame = ttk.Frame(self.stage_frame)
        stage_info_frame.pack(fill="x", padx=10, pady=5)

        self.stage_label = ttk.Label(stage_info_frame, text="Initializing...", font=("Arial", 10))
        self.stage_label.pack(side="left")

        self.stage_eta_label = ttk.Label(stage_info_frame, text="--:--", font=("Arial", 9))
        self.stage_eta_label.pack(side="right")

        # Stage progress bar with percentage
        stage_progress_frame = ttk.Frame(self.stage_frame)
        stage_progress_frame.pack(fill="x", padx=10, pady=5)

        self.stage_progress_var = tk.DoubleVar()
        self.stage_progress_bar = ttk.Progressbar(
            stage_progress_frame,
            variable=self.stage_progress_var,
            maximum=100,
            length=400
        )
        self.stage_progress_bar.pack(side="left", fill="x", expand=True)

        self.stage_percentage_label = ttk.Label(stage_progress_frame, text="0.0%", font=("Arial", 9))
        self.stage_percentage_label.pack(side="right", padx=(10, 0))

        # Log frame (will be packed after any processor-specific controls)
        self.log_frame = ttk.LabelFrame(self.popup, text="Processing Log")
        self.log_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Log text with scrollbar
        log_scroll_frame = ttk.Frame(self.log_frame)
        log_scroll_frame.pack(fill="both", expand=True, padx=5, pady=5)

        self.log_text = tk.Text(log_scroll_frame, height=6, wrap="word", font=("Consolas", 9))
        log_scrollbar = ttk.Scrollbar(log_scroll_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)

        self.log_text.pack(side="left", fill="both", expand=True)
        log_scrollbar.pack(side="right", fill="y")

        # Button frame
        self.button_frame = ttk.Frame(self.popup)
        self.button_frame.pack(fill="x", padx=10, pady=10)

        # Cancel/Close button
        self.cancel_button = ttk.Button(self.button_frame, text="Cancel", command=self._on_cancel)
        self.cancel_button.pack(side="right", padx=5)

        # Keep window on top option
        self.stay_on_top_var = tk.BooleanVar(value=True)
        stay_on_top_cb = ttk.Checkbutton(
            self.button_frame,
            text="Keep on top",
            variable=self.stay_on_top_var,
            command=self._toggle_stay_on_top
        )
        stay_on_top_cb.pack(side="left")

        # Set initial stay on top
        self.popup.attributes('-topmost', True)

        # Start time update thread
        self._start_time_update_thread()

    def _start_time_update_thread(self):
        """Start a thread to update elapsed time and ETAs."""
        def update_time():
            while not self.cancelled and not self.completed:
                try:
                    self._update_time_displays()
                    time.sleep(1)
                except:
                    break

        thread = threading.Thread(target=update_time, daemon=True)
        thread.start()

    def _update_time_displays(self):
        """Update elapsed time and ETA displays."""
        if self.cancelled or self.completed:
            return

        try:
            elapsed = datetime.now() - self.start_time
            elapsed_str = str(elapsed).split('.')[0] # Remove microseconds

            self.elapsed_time_label.config(text=f"Elapsed: {elapsed_str}")

            # Calculate ETA based on overall progress
            overall_progress = self.overall_progress_var.get()
            if overall_progress > 0:
                total_estimated = elapsed.total_seconds() * 100 / overall_progress
                remaining = total_estimated - elapsed.total_seconds()
                if remaining > 0:
                    eta = timedelta(seconds=int(remaining))
                    self.eta_label.config(text=f"ETA: {str(eta)}")
                else:
                    self.eta_label.config(text="ETA: --:--:--")

            # Update stage ETA
            if self.stage_start_time and self.stage_progress_var.get() > 0:
                stage_elapsed = datetime.now() - self.stage_start_time
                stage_progress = self.stage_progress_var.get()
                if stage_progress > 0:
                    stage_total = stage_elapsed.total_seconds() * 100 / stage_progress
                    stage_remaining = stage_total - stage_elapsed.total_seconds()
                    if stage_remaining > 0:
                        stage_eta = timedelta(seconds=int(stage_remaining))
                        self.stage_eta_label.config(text=str(stage_eta).split('.')[0])

            self.popup.update_idletasks()
        except:
            pass

    def _toggle_stay_on_top(self):
        """Toggle the stay on top behavior."""
        self.popup.attributes('-topmost', self.stay_on_top_var.get())

    def update_stage(self, stage_name: str, stage_progress: float = 0):
        """Update the current stage information."""
        if self.cancelled or self.completed:
            return

        if stage_name != self.current_stage:
            self.current_stage = stage_name
            self.stage_start_time = datetime.now()

        self.stage_label.config(text=stage_name)
        self.update_stage_progress(stage_progress)
        self.popup.update_idletasks()

    def update_overall_progress(self, progress: float):
        """Update the overall progress (0-100)."""
        if self.cancelled or self.completed:
            return

        self.overall_progress_var.set(progress)
        self.overall_percentage_label.config(text=f"{progress:.1f}%")
        self.popup.update_idletasks()

    def update_stage_progress(self, progress: float):
        """Update the current stage progress (0-100)."""
        if self.cancelled or self.completed:
            return

        self.stage_progress_var.set(progress)
        self.stage_percentage_label.config(text=f"{progress:.1f}%")
        self.popup.update_idletasks()

        # Update the current stage progress for gradual overall progress
        self.current_stage_progress = progress
        self._update_gradual_overall_progress()

    def log(self, message: str):
        """Add a message to the log."""
        if self.cancelled or self.completed:
            return

        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"

        # Insert at end and scroll to bottom
        self.log_text.insert(tk.END, log_message)
        self.log_text.see(tk.END)
        self.popup.update_idletasks()

    def log_error(self, message: str):
        """Add an error message to the log."""
        if self.cancelled or self.completed:
            return

        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] ERROR: {message}\n"

        # Insert at end in red color
        self.log_text.insert(tk.END, log_message)
        self.log_text.tag_add("error", f"end-{len(log_message)}c", "end-1c")
        self.log_text.tag_config("error", foreground="red")
        self.log_text.see(tk.END)
        self.popup.update_idletasks()

    def complete(self, success: bool = True):
        """Mark the operation as complete."""
        if self.cancelled:
            return

        self.completed = True
        end_time = datetime.now()
        total_duration = end_time - self.start_time

        if success:
            self.stage_label.config(text="✓ Completed successfully!")
            self.update_overall_progress(100)
            self.update_stage_progress(100)

            # Show completion summary
            summary = self._generate_completion_summary(total_duration)
            self.log("=" * 50)
            self.log("PROCESSING COMPLETE")
            self.log("=" * 50)
            for line in summary:
                self.log(line)
            self.log("=" * 50)
        else:
            self.stage_label.config(text="✗ Operation failed!")
            self.log_error("Operation failed!")

        # Change button to Close
        self.cancel_button.config(text="Close")

        # Notify callback
        if self.on_complete:
            self.on_complete(success)

    def _generate_completion_summary(self, duration: timedelta) -> List[str]:
        """Generate a summary of what was accomplished."""
        summary = []
        summary.append(f"Character: {self.character_name}")
        summary.append(f"Total processing time: {str(duration).split('.')[0]}")

        return summary

    def _on_cancel(self):
        """Handle cancel/close button click."""
        if not self.completed:
            # Ask for confirmation if operation is still running
            self.cancelled = True
            self.stage_label.config(text="Cancelling...")
            self.log("Operation cancelled by user")

            if self.on_cancel:
                self.on_cancel()

        self._close_popup()

    def _on_close(self):
        """Handle window close event."""
        self._on_cancel()

    def _close_popup(self):
        """Close the popup window."""
        try:
            self.popup.destroy()
        except:
            pass

    def is_cancelled(self) -> bool:
        """Check if the operation was cancelled."""
        return self.cancelled

    def is_completed(self) -> bool:
        """Check if the operation is completed."""
        return self.completed

    def close_after_delay(self, delay_ms: int = 1000):
        """Close the popup after a specified delay in milliseconds."""
        def delayed_close() -> None:
            try:
                if not self.cancelled and self.popup.winfo_exists():
                    self.popup.destroy()
            except:
                pass

        # Schedule the close operation
        self.popup.after(delay_ms, delayed_close)

class ProcessingStatusDialog:
    """Handler for processing status dialogs and operations."""

    def __init__(self, parent_frame, current_character_getter, current_stage_getter,
                 progress_tracker,
                 update_progress_display_callback=None, refresh_images_callback=None,
                 main_app_reference=None):
        """
        Initialize the processing status dialog handler.

        Args:
            parent_frame: The parent tkinter frame
            current_character_getter: Function that returns the current character name
            current_stage_getter: Function that returns the current stage
            progress_tracker: The progress tracker instance
            update_progress_display_callback: Callback to update progress display
            refresh_images_callback: Callback to refresh images
            main_app_reference: Direct reference to the main app instance
        """
        self.frame = parent_frame
        self.get_current_character = current_character_getter
        self.get_current_stage = current_stage_getter
        self.progress_tracker = progress_tracker
        self.update_progress_display_callback = update_progress_display_callback
        self.refresh_images_callback = refresh_images_callback
        self.main_app = main_app_reference

    def process_current_stage(self):
        """Process current stage with progress popup."""
        current_character = self.get_current_character()
        if not current_character:
            messagebox.showwarning("Warning", "Please select a character first")
            return

        # Get stage information for display
        current_stage = self.get_current_stage()

        # Get the correct processing information from config
        stage_info = config.STAGE_PROCESSING.get(current_stage)
        if stage_info and stage_info[0]:  # Check if there's a valid processor
            processor_key, stage_display_name, target_stage = stage_info
            stage_name = stage_display_name
        else:
            # Fallback to basic stage info
            for stage_id, stage_num, stage_display_name in config.STAGES:
                if stage_id == current_stage:
                    stage_name = f"Stage {stage_num}: {stage_display_name}"
                    processor_key = None
                    break
            else:
                stage_name = f"Stage: {current_stage}"
                processor_key = None

        # Create progress popup
        progress_popup = ProgressPopup(
            self.frame,
            "Processing Stage",
            stage_name,
            current_character
        )

        # Setup cancel callback
        def on_cancel():
            # Here you could implement cancellation logic if your processors support it
            progress_popup.log("Cancellation requested...")

        def on_complete(success):
            if success:
                # Refresh the display after successful processing
                if self.update_progress_display_callback:
                    self.update_progress_display_callback()
                if self.refresh_images_callback:
                    self.refresh_images_callback()

            # Close the progress popup after a brief delay to show the completion message
            progress_popup.close_after_delay(1000)  # Close after 1 second

        progress_popup.on_cancel = on_cancel
        progress_popup.on_complete = on_complete

        # Run processing in a separate thread
        def run_processing():
            try:
                progress_popup.log(f"Starting {stage_name} for {current_character}")
                progress_popup.update_stage("Initializing...", 0)
                progress_popup.update_overall_progress(10)

                if processor_key:
                    # Call the actual processing with the correct processor key
                    progress_popup.update_stage(f"Processing: {stage_name}", 0)
                    progress_popup.update_overall_progress(20)
                    progress_popup.log(f"Running processor: {processor_key}")

                    success = self.main_app._process_single_stage_for_character_with_popup(
                        current_character, processor_key, progress_popup
                    )

                    if success:
                        progress_popup.update_stage_progress(100)
                        progress_popup.log(f"Completed {stage_display_name}")
                    else:
                        progress_popup.log_error(f"Failed at {stage_display_name}")
                        progress_popup.complete(False)
                        return
                else:
                    progress_popup.log(f"No processor available for stage {current_stage}")
                    progress_popup.update_overall_progress(50)

                progress_popup.update_stage("Finalizing...", 100)
                progress_popup.update_overall_progress(100)
                progress_popup.complete(True)

            except Exception as e:
                progress_popup.log_error(f"Processing failed: {str(e)}")
                progress_popup.complete(False)
                print(f"Error processing stage: {e}")

        threading.Thread(target=run_processing, daemon=True).start()

    def process_all_stages(self):
        """Process all stages with progress popup."""
        current_character = self.get_current_character()
        if not current_character:
            messagebox.showwarning("Warning", "Please select a character first")
            return

        # Create progress popup
        progress_popup = ProgressPopup(
            self.frame,
            "Processing All Stages",
            "Complete Pipeline",
            current_character
        )

        # Setup cancel callback
        def on_cancel():
            progress_popup.log("Cancellation requested...")

        def on_complete(success):
            if success:
                # Refresh the display after successful processing
                if self.update_progress_display_callback:
                    self.update_progress_display_callback()
                if self.refresh_images_callback:
                    self.refresh_images_callback()

            # Close the progress popup after a brief delay to show the completion message
            progress_popup.close_after_delay(1000)  # Close after 1 second

        progress_popup.on_cancel = on_cancel
        progress_popup.on_complete = on_complete

        # Run processing in a separate thread
        def run_processing():
            try:
                # Get the correct processor keys and names from config
                processors = []
                for stage_id, stage_info in config.STAGE_PROCESSING.items():
                    if stage_info[0]:  # If there's a valid processor
                        processor_key, stage_display_name, target_stage = stage_info
                        processors.append((processor_key, stage_display_name))

                total_stages = len(processors)

                # Setup gradual progress tracking
                progress_popup.set_total_stages(total_stages)

                progress_popup.log(f"Starting batch processing for {current_character}")
                progress_popup.update_stage("Preparing...", 0)

                for i, (processor_key, stage_display) in enumerate(processors):
                    if progress_popup.is_cancelled():
                        progress_popup.log("Processing cancelled by user")
                        return

                    # Set current stage index for gradual progress calculation
                    progress_popup.set_current_stage_index(i)

                    progress_popup.log(f"Starting: {stage_display}")
                    progress_popup.update_stage(stage_display, 0)

                    try:
                        # Update stage progress to show processing has started
                        progress_popup.update_stage_progress(25)
                        progress_popup.log(f"Running processor: {processor_key}")

                        success = self.main_app._process_single_stage_for_character_with_popup(
                            current_character, processor_key, progress_popup
                        )

                        progress_popup.update_stage_progress(100)
                        progress_popup.log(f"Completed {stage_display}")

                    except Exception as e:
                        progress_popup.log_error(f"Failed at {stage_display}: {str(e)}")
                        progress_popup.complete(False)
                        return

                # Final completion - set to last stage index and 100% progress
                progress_popup.set_current_stage_index(total_stages - 1)
                progress_popup.update_stage_progress(100)
                progress_popup.log("All stages completed successfully!")
                progress_popup.complete(True)

            except Exception as e:
                progress_popup.log_error(f"Batch processing failed: {str(e)}")
                progress_popup.complete(False)
                print(f"Error processing all stages: {e}")

        threading.Thread(target=run_processing, daemon=True).start()

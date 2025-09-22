"""
Training Status Window for monitoring OneTrainer progress.
"""
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import re
from pathlib import Path
from typing import Optional
import subprocess
import os
from datetime import timedelta


class TrainingStatusWindow:
    """Window for monitoring OneTrainer training progress."""

    def __init__(self, parent, character_name: str, training_outputs_dir: Path, base_model=None, output_model_name=None):
        """Initialize the training status window.

        Args:
            parent: Parent tkinter window
            character_name: Name of character being trained
            training_outputs_dir: Directory containing training outputs
            base_model: The base model being trained on
            output_model_name: The output model name
        """
        self.parent = parent
        self.character_name = character_name
        self.training_outputs_dir = training_outputs_dir
        self.base_model = base_model
        self.output_model_name = output_model_name
        self.log_file: Optional[Path] = None
        self.training_process: Optional[subprocess.Popen] = None
        self.monitoring = False
        self.window = None

        # Training status data
        self.current_epoch = 0
        self.total_epochs = 0
        self.current_step = 0
        self.total_steps = 0
        self.current_loss = 0.0
        self.smooth_loss = 0.0
        self.training_stage = "Initializing..."
        self.last_update_time = time.time()

        # Timing data for ETA calculations
        self.epoch_start_times = {}  # epoch_number -> start_time
        self.step_start_times = {}   # step_number -> start_time
        self.training_start_time = None
        self.epoch_eta = "Calculating..."
        self.step_eta = "Calculating..."
        self.last_step_time = time.time()
        self.last_epoch_time = time.time()
        self.epoch_times = []  # Store recent epoch durations for averaging
        self.step_times = []   # Store recent step durations for averaging
        self.steps_per_epoch = 0

        self._create_window()
        self._find_log_file()
        self._start_monitoring()

    def _create_window(self):
        """Create the training status window."""
        self.window = tk.Toplevel(self.parent)
        self.window.title(f"Training Status - {self.character_name}")
        self.window.geometry("835x387")
        self.window.resizable(True, True)

        # Configure window icon and properties
        self.window.transient(self.parent)
        self.window.grab_set()

        # Main frame
        main_frame = ttk.Frame(self.window, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")

        # Configure grid weights
        self.window.columnconfigure(0, weight=1)
        self.window.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

        # Title
        title_label = ttk.Label(main_frame, text=f"Training Model: {self.character_name}",
                               font=('Arial', 14, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 5), sticky=tk.W)

        # Base model and output model name display
        base_model_text = f"Base Model: {self.base_model if self.base_model else 'Unknown'}"
        output_model_text = f"Output Model: {self.output_model_name if self.output_model_name else 'Unknown'}"
        base_model_label = ttk.Label(main_frame, text=base_model_text, font=('Arial', 10))
        base_model_label.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(0, 2))
        output_model_label = ttk.Label(main_frame, text=output_model_text, font=('Arial', 10))
        output_model_label.grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=(0, 10))

        # Status section
        status_frame = ttk.LabelFrame(main_frame, text="Training Status", padding="10")
        status_frame.grid(row=3, column=0, columnspan=2, sticky="we", pady=10)
        status_frame.columnconfigure(1, weight=1)

        # Current stage
        ttk.Label(status_frame, text="Stage:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.stage_label = ttk.Label(status_frame, text="Initializing...", foreground="blue")
        self.stage_label.grid(row=0, column=1, sticky=tk.W, pady=2)

        # Epoch progress
        ttk.Label(status_frame, text="Epoch:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.epoch_label = ttk.Label(status_frame, text="0 / 0")
        self.epoch_label.grid(row=1, column=1, sticky=tk.W, pady=2)

        # Epoch ETA
        ttk.Label(status_frame, text="Epoch ETA:").grid(row=1, column=2, sticky=tk.W, pady=2, padx=(20, 0))
        self.epoch_eta_label = ttk.Label(status_frame, text="Calculating...", foreground="gray")
        self.epoch_eta_label.grid(row=1, column=3, sticky=tk.W, pady=2)

        self.epoch_progress = ttk.Progressbar(status_frame, mode='determinate')
        self.epoch_progress.grid(row=2, column=0, columnspan=4, sticky="we", pady=5)

        # Step progress
        ttk.Label(status_frame, text="Step:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.step_label = ttk.Label(status_frame, text="0 / 0")
        self.step_label.grid(row=3, column=1, sticky=tk.W, pady=2)

        # Step ETA
        ttk.Label(status_frame, text="Step ETA:").grid(row=3, column=2, sticky=tk.W, pady=2, padx=(20, 0))
        self.step_eta_label = ttk.Label(status_frame, text="Calculating...", foreground="gray")
        self.step_eta_label.grid(row=3, column=3, sticky=tk.W, pady=2)

        self.step_progress = ttk.Progressbar(status_frame, mode='determinate')
        self.step_progress.grid(row=4, column=0, columnspan=4, sticky="we", pady=5)

        # Move buttons to the bottom of the window
        # Buttons
        button_frame = ttk.Frame(main_frame)
        # Place button_frame at the bottom (row=4), sticky to east/west, with padding
        button_frame.grid(row=5, column=0, columnspan=2, pady=(20, 0), sticky="we")
        main_frame.rowconfigure(5, weight=1)  # Allow bottom row to expand if needed

        self.stop_button = ttk.Button(button_frame, text="Stop Training",
                                     command=self._stop_training, style='Warning.TButton')
        self.stop_button.pack(side="left", padx=10)

        self.open_log_button = ttk.Button(button_frame, text="Open Log File",
                                        command=self._open_log_file)
        self.open_log_button.pack(side="left", padx=10)

        # Add Open Models Dir button
        self.open_models_dir_button = ttk.Button(button_frame, text="Open Models Dir",
                                                command=self._open_models_dir)
        self.open_models_dir_button.pack(side="left", padx=10)

        # Add 'Close when done' checkbox to the right of Open Models Dir button
        self.close_when_done_var = tk.BooleanVar(value=True)
        self.close_when_done_checkbox = ttk.Checkbutton(
            button_frame, text="Close when done", variable=self.close_when_done_var)
        self.close_when_done_checkbox.pack(side="left", padx=10)

        self.close_button = ttk.Button(button_frame, text="Close",
                                     command=self._close_window)
        self.close_button.pack(side="right")

        # Handle window close
        self.window.protocol("WM_DELETE_WINDOW", self._close_window)

    def _find_log_file(self):
        """Find the latest training log file."""
        try:
            # First, try to find log files in the new character models directory structure
            characters_dir = self.training_outputs_dir.parent / "characters" / self.character_name / "models"

            if characters_dir.exists():
                self._append_log(f"Searching in character models directory: {characters_dir}\n")

                # Look for training_* directories in the character's models folder
                training_dirs = []
                for item in characters_dir.iterdir():
                    if item.is_dir() and item.name.startswith("training_"):
                        training_dirs.append(item)
                        self._append_log(f"Found training directory: {item.name}\n")

                if training_dirs:
                    # Get the most recent training directory (by creation time)
                    latest_training_dir = max(training_dirs, key=lambda x: x.stat().st_mtime)
                    self._append_log(f"Using most recent training directory: {latest_training_dir.name}\n")

                    # Look for training.log in that directory
                    potential_log = latest_training_dir / "training.log"
                    self._append_log(f"Looking for log file at: {potential_log}\n")

                    if potential_log.exists():
                        self.log_file = potential_log
                        self._append_log(f"Found log file in character models: {self.log_file}\n")

                        # Read existing log content immediately
                        try:
                            with open(self.log_file, 'r', encoding='utf-8', errors='ignore') as f:
                                existing_content = f.read()

                            if existing_content.strip():
                                self._append_log("=== Existing Log Content ===\n")
                                self._append_log(existing_content)
                                self._append_log("\n=== Live Updates ===\n")
                                # Parse existing content for current status
                                self._parse_log_content(existing_content)

                        except Exception as e:
                            self._append_log(f"Error reading existing log content: {e}\n")

                        return  # Found log file in new structure, exit early
                    else:
                        # List files in the training directory for debugging
                        files_in_dir = [item.name for item in latest_training_dir.iterdir()]
                        self._append_log(f"Files in training directory: {files_in_dir}\n")
                else:
                    self._append_log(f"No training directories found in character models folder\n")
            else:
                self._append_log(f"Character models directory does not exist: {characters_dir}\n")

            # Fallback: search in the old training_outputs directory structure
            self._append_log("Falling back to training_outputs directory search...\n")

            # Find the most recent training directory for this character
            if not self.training_outputs_dir.exists():
                self._append_log("Training outputs directory does not exist.\n")
                return

            # Look for directories matching the character name pattern
            character_dirs = []
            character_name_with_underscore = self.character_name.replace(' ', '_')
            character_name_with_space = self.character_name

            self._append_log(f"Searching for directories matching: '{character_name_with_underscore}' or '{character_name_with_space}'\n")

            for item in self.training_outputs_dir.iterdir():
                if item.is_dir():
                    # Check if directory name starts with character name (with underscore or space)
                    if (item.name.startswith(character_name_with_underscore) or
                        item.name.startswith(character_name_with_space)):
                        character_dirs.append(item)
                        self._append_log(f"Found matching directory: {item.name}\n")

            if not character_dirs:
                self._append_log(f"No training directories found for character: {self.character_name}\n")
                # List all directories for debugging
                all_dirs = [item.name for item in self.training_outputs_dir.iterdir() if item.is_dir()]
                self._append_log(f"Available directories: {all_dirs}\n")
                return

            # Get the most recent directory (by creation time)
            latest_dir = max(character_dirs, key=lambda x: x.stat().st_mtime)
            self._append_log(f"Using most recent directory: {latest_dir.name}\n")

            # Look for training.log in that directory
            potential_log = latest_dir / "training.log"
            self._append_log(f"Looking for log file at: {potential_log}\n")

            if potential_log.exists():
                self.log_file = potential_log
                self._append_log(f"Found log file in training_outputs: {self.log_file}\n")

                # Read existing log content immediately
                try:
                    with open(self.log_file, 'r', encoding='utf-8', errors='ignore') as f:
                        existing_content = f.read()

                    if existing_content.strip():
                        self._append_log("=== Existing Log Content ===\n")
                        self._append_log(existing_content)
                        self._append_log("\n=== Live Updates ===\n")
                        # Parse existing content for current status
                        self._parse_log_content(existing_content)

                except Exception as e:
                    self._append_log(f"Error reading existing log content: {e}\n")

            else:
                self._append_log(f"Log file not found in: {latest_dir}\n")
                # List files in the directory for debugging
                files_in_dir = [item.name for item in latest_dir.iterdir()]
                self._append_log(f"Files in directory: {files_in_dir}\n")

        except Exception as e:
            self._append_log(f"Error finding log file: {e}\n")

    def _start_monitoring(self):
        """Start monitoring the training log."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_log, daemon=True)
        self.monitor_thread.start()

    def _monitor_log(self):
        """Monitor the training log file for updates and detect training completion."""
        last_size = 0
        if self.log_file and self.log_file.exists():
            last_size = self.log_file.stat().st_size

        model_file_written = False
        training_prompt = None
        # Get the training prompt from training_prompt.txt
        character_dir = self.training_outputs_dir.parent / "characters" / self.character_name
        training_prompt_file = character_dir / "training_prompt.txt"
        if training_prompt_file.exists():
            try:
                with open(training_prompt_file, 'r', encoding='utf-8') as f:
                    training_prompt = f.read().strip()
            except Exception:
                training_prompt = None

        # Use the known output model path
        model_file_path = None
        if self.output_model_name:
            model_file_path = Path(self.output_model_name)

        while self.monitoring:
            try:
                if self.log_file and self.log_file.exists():
                    current_size = self.log_file.stat().st_size

                    if current_size > last_size:
                        with open(self.log_file, 'r', encoding='utf-8', errors='ignore') as f:
                            f.seek(last_size)
                            new_content = f.read()
                        if new_content.strip():
                            self._parse_log_content(new_content)
                            if self.window and self.window.winfo_exists():
                                self.window.after(0, lambda content=new_content: self._append_log(content))
                        last_size = current_size

                    # Check for model file if not already written
                    if not model_file_written and model_file_path and model_file_path.exists():
                        txt_path = model_file_path.with_suffix('.txt')
                        if not txt_path.exists() and training_prompt:
                            try:
                                with open(txt_path, 'w', encoding='utf-8') as f:
                                    f.write(training_prompt)
                                model_file_written = True
                                if self.window and self.window.winfo_exists():
                                    self.window.after(0, lambda: self._on_training_complete())
                            except Exception as e:
                                if self.window and self.window.winfo_exists():
                                    self.window.after(0, lambda: self._append_log(f'Error writing model info txt: {e}\n'))
                        elif txt_path.exists():
                            model_file_written = True
                            if self.window and self.window.winfo_exists():
                                self.window.after(0, lambda: self._on_training_complete())

                    current_time = time.time()
                    if current_time - self.last_update_time > 300:  # 5 minutes without updates
                        if self.window and self.window.winfo_exists():
                            self.window.after(0, lambda: self._set_stage("Training may have completed or stopped"))
                else:
                    self._find_log_file()

                # Check if training process has finished
                if self.training_process and self.training_process.poll() is not None:
                    # Training finished
                    self._set_stage("Training complete")
                    self.stop_button.config(state='disabled')
                    if self.close_when_done_var.get():
                        if self.window and self.window.winfo_exists():
                            self.window.after(0, self._close_window)
                    self.monitoring = False
                    break

                time.sleep(2)
            except Exception as e:
                if self.window and self.window.winfo_exists():
                    error_msg = f"Monitor error: {e}\n"
                    self.window.after(0, lambda msg=error_msg: self._append_log(msg))
                time.sleep(5)

    def _parse_log_content(self, content: str):
        """Parse log content to extract training information."""
        try:
            lines = content.split('\n')

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Update last activity time
                self.last_update_time = time.time()
                current_time = time.time()

                # Parse epoch information
                epoch_match = re.search(r'epoch:\s*(\d+)%.*?(\d+)/(\d+)', line)
                if epoch_match:
                    epoch_percent = int(epoch_match.group(1))
                    current_epoch = int(epoch_match.group(2))
                    total_epochs = int(epoch_match.group(3))

                    # Track epoch timing
                    if current_epoch != self.current_epoch and current_epoch > 0:
                        # New epoch started
                        if self.current_epoch > 0:
                            # Calculate time for previous epoch
                            epoch_duration = current_time - self.last_epoch_time
                            self.epoch_times.append(epoch_duration)
                            # Keep only last 5 epochs for averaging
                            if len(self.epoch_times) > 5:
                                self.epoch_times.pop(0)

                        self.last_epoch_time = current_time

                    self.current_epoch = current_epoch
                    self.total_epochs = total_epochs

                    if self.window and self.window.winfo_exists():
                        self.window.after(0, self._update_epoch_progress)

                # Parse step information
                step_match = re.search(r'step:\s*(\d+)%.*?(\d+)/(\d+).*?loss=([\d.]+).*?smooth loss=([\d.]+)', line)
                if step_match:
                    step_percent = int(step_match.group(1))
                    current_step = int(step_match.group(2))
                    total_steps = int(step_match.group(3))
                    current_loss = float(step_match.group(4))
                    smooth_loss = float(step_match.group(5))

                    # Track step timing
                    if current_step != self.current_step and current_step > 0:
                        # New step completed
                        if self.current_step > 0:
                            # Calculate time for previous step
                            step_duration = current_time - self.last_step_time
                            self.step_times.append(step_duration)
                            # Keep only last 20 steps for averaging
                            if len(self.step_times) > 20:
                                self.step_times.pop(0)

                    self.last_step_time = current_time

                    self.current_step = current_step
                    self.total_steps = total_steps
                    self.current_loss = current_loss
                    self.smooth_loss = smooth_loss

                    # Calculate steps per epoch if not set
                    if self.steps_per_epoch == 0 and self.total_steps > 0 and self.total_epochs > 0:
                        self.steps_per_epoch = self.total_steps // self.total_epochs

                    if self.window and self.window.winfo_exists():
                        self.window.after(0, self._update_step_progress)

                # Parse training stages
                if 'Loading pipeline components' in line:
                    self._set_stage("Loading model components...")
                elif 'caching:' in line:
                    cache_match = re.search(r'caching:\s*(\d+)%', line)
                    if cache_match:
                        cache_percent = cache_match.group(1)
                        self._set_stage(f"Caching data: {cache_percent}%")
                elif 'enumerating sample paths' in line:
                    self._set_stage("Preparing dataset...")
                elif 'step:' in line and 'loss=' in line:
                    self._set_stage(f"Training in progress - Epoch {self.current_epoch}")
                    # Set training start time if not set
                    if self.training_start_time is None:
                        self.training_start_time = current_time
                elif 'Fetching' in line and 'files:' in line:
                    self._set_stage("Downloading model files...")

        except Exception as e:
            print(f"Error parsing log content: {e}")

    def _update_epoch_progress(self):
        """Update epoch progress in the UI."""
        try:
            if self.total_epochs > 0:
                progress = (self.current_epoch / self.total_epochs) * 100
                self.epoch_progress['value'] = progress
                self.epoch_label.config(text=f"{self.current_epoch} / {self.total_epochs}")

                # Calculate ETA based on average epoch time
                eta_text = "Calculating..."
                if len(self.epoch_times) > 0 and self.current_epoch > 0:
                    # Use average of recent epoch times
                    avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
                    remaining_epochs = self.total_epochs - self.current_epoch
                    eta_seconds = remaining_epochs * avg_epoch_time
                    eta_text = str(timedelta(seconds=int(eta_seconds)))
                elif self.training_start_time and self.current_epoch > 0:
                    # Fallback: use total elapsed time divided by completed epochs
                    elapsed_total = time.time() - self.training_start_time
                    avg_epoch_time = elapsed_total / self.current_epoch
                    remaining_epochs = self.total_epochs - self.current_epoch
                    eta_seconds = remaining_epochs * avg_epoch_time
                    eta_text = str(timedelta(seconds=int(eta_seconds)))

                self.epoch_eta_label.config(text=eta_text)

        except Exception as e:
            print(f"Error updating epoch progress: {e}")

    def _update_step_progress(self):
        """Update step progress in the UI."""
        try:
            if self.total_steps > 0:
                progress = (self.current_step / self.total_steps) * 100
                self.step_progress['value'] = progress
                self.step_label.config(text=f"{self.current_step} / {self.total_steps}")

                # Calculate ETA based on average step time
                eta_text = "Calculating..."
                if len(self.step_times) > 0 and self.current_step > 0:
                    # Use average of recent step times
                    avg_step_time = sum(self.step_times) / len(self.step_times)
                    remaining_steps = self.total_steps - self.current_step
                    eta_seconds = remaining_steps * avg_step_time
                    eta_text = str(timedelta(seconds=int(eta_seconds)))
                elif self.training_start_time and self.current_step > 0:
                    # Fallback: use total elapsed time divided by completed steps
                    elapsed_total = time.time() - self.training_start_time
                    avg_step_time = elapsed_total / self.current_step
                    remaining_steps = self.total_steps - self.current_step
                    eta_seconds = remaining_steps * avg_step_time
                    eta_text = str(timedelta(seconds=int(eta_seconds)))

                self.step_eta_label.config(text=eta_text)

        except Exception as e:
            print(f"Error updating step progress: {e}")

    def _set_stage(self, stage: str):
        """Set the current training stage."""
        self.training_stage = stage
        if self.window and self.window.winfo_exists():
            self.window.after(0, lambda: self.stage_label.config(text=stage))

    def _append_log(self, text: str):
        pass

    def _stop_training(self):
        """Stop the training process immediately (no confirmation)."""
        try:
            if self.training_process and self.training_process.poll() is None:
                self.training_process.terminate()
                time.sleep(2)
                if self.training_process.poll() is None:
                    self.training_process.kill()
                self._append_log("\n[TRAINING STOPPED BY USER]\n")
                self._set_stage("Training stopped by user")
                self.stop_button.config(state='disabled')
            else:
                self.stop_button.config(state='disabled')
        except Exception as e:
            messagebox.showerror("Error", f"Failed to stop training: {e}")

    def _close_window(self):
        """Close the training status window."""
        try:
            # Stop monitoring
            self.monitoring = False

            # Close window
            if self.window:
                self.window.destroy()

        except Exception as e:
            print(f"Error closing window: {e}")

    def set_training_process(self, process: subprocess.Popen):
        """Set the training process for monitoring and control."""
        self.training_process = process

    def show(self):
        """Show the training status window."""
        if self.window:
            self.window.deiconify()
            self.window.lift()
            self.window.focus()

    def _open_log_file(self):
        """Open the log file in the default system editor."""
        try:
            if self.log_file and self.log_file.exists():
                # Close the log file if already open
                if os.name == 'posix':  # macOS or Linux
                    subprocess.run(['pkill', '-f', str(self.log_file)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                # Open the log file in the default editor
                if os.name == 'nt':  # Windows
                    os.startfile(str(self.log_file))
                elif os.name == 'posix':  # macOS or Linux
                    subprocess.run(['open', str(self.log_file)], check=True)
                else:
                    messagebox.showerror("Error", "Unsupported operating system for opening log file.")

            else:
                messagebox.showwarning("Warning", "Log file not found.")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to open log file: {e}")

    def _open_models_dir(self):
        """Open the character's models directory in the file explorer."""
        try:
            models_dir = self.training_outputs_dir.parent / "characters" / self.character_name / "models"
            if models_dir.exists():
                os.startfile(str(models_dir))  # Windows only
            else:
                messagebox.showerror("Error", f"Models directory does not exist: {models_dir}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open models directory: {e}")

    def _on_training_complete(self):
        """Handle actions after training is complete: delete training folder and close window."""
        self._set_stage('Training complete. Model and prompt file saved.')
        # Delete the training_.* folder
        import shutil
        try:
            # Find the training_.* folder in the parent directory
            models_dir = self.training_outputs_dir.parent / "characters" / self.character_name / "models"
            for item in models_dir.iterdir():
                if item.is_dir() and re.match(r"training_.*", item.name):
                    shutil.rmtree(item)
        except Exception as e:
            self._append_log(f"Error deleting training folder: {e}\n")
        # Autoclose the window after a short delay
        if self.window and self.window.winfo_exists():
            self.window.after(1500, self._close_window)


def show_training_status(parent, character_name: str, training_outputs_dir: Path,
                        training_process: Optional[subprocess.Popen] = None, base_model=None, output_model_name=None) -> TrainingStatusWindow:
    """Create and show a training status window.

    Args:
        parent: Parent tkinter window
        character_name: Name of character being trained
        training_outputs_dir: Directory containing training outputs
        training_process: Optional training process for monitoring
        base_model: The base model being trained on
        output_model_name: The output model name

    Returns:
        TrainingStatusWindow instance
    """
    status_window = TrainingStatusWindow(parent, character_name, training_outputs_dir, base_model, output_model_name)

    if training_process:
        status_window.set_training_process(training_process)

    status_window.show()
    return status_window

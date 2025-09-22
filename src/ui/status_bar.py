import tkinter as tk
from tkinter import ttk
import threading
import psutil

try:
    import pynvml
    pynvml.nvmlInit()
    nvml_available = True
    # Import the functions we need
    nvmlInit = pynvml.nvmlInit
    nvmlDeviceGetHandleByIndex = pynvml.nvmlDeviceGetHandleByIndex
    nvmlDeviceGetMemoryInfo = pynvml.nvmlDeviceGetMemoryInfo
    nvmlDeviceGetUtilizationRates = pynvml.nvmlDeviceGetUtilizationRates
except Exception:
    nvml_available = False

from ..services.gpu_service import get_gpu_service
from ..services.training_queue import get_training_queue_manager
from ..ui.queue_manager_window import show_queue_manager

class StatusBar(ttk.Frame):
    def __init__(self, master, provider_manager=None, characters_path=None, **kwargs):
        super().__init__(master, **kwargs)
        self.provider_manager = provider_manager
        self.characters_path = characters_path or "characters"
        self.gpu_service = get_gpu_service()
        self.training_queue_manager = get_training_queue_manager()
        self.queue_manager_window = None

        self.status_var = tk.StringVar(value="Ready.")
        self.cpu_var = tk.StringVar(value="CPU: --%")
        self.ram_var = tk.StringVar(value="RAM: --%")
        self.vram_var = tk.StringVar(value="VRAM: --%")
        self.gpu_var = tk.StringVar(value="GPU: --%")
        self.auto_check_status_var = tk.StringVar(value="Auto-check: OFF")
        self.queue_status_var = tk.StringVar(value="Queue: 0")

        # Left side - status message
        self.label_status = ttk.Label(self, textvariable=self.status_var, anchor="w")
        self.label_status.pack(side="left", padx=(5, 0), fill="x", expand=True)

        # Center-left - GPU selection
        self._create_gpu_selection_controls()

        # Center-left - Training queue controls
        self._create_queue_controls()

        # Center - Provider auto-check controls
        if self.provider_manager:
            self._create_auto_check_controls()

        # Right side - system stats
        self.label_gpu = ttk.Label(self, textvariable=self.gpu_var, anchor="e")
        self.label_gpu.pack(side="right", padx=(5, 5))
        self.label_vram = ttk.Label(self, textvariable=self.vram_var, anchor="e")
        self.label_vram.pack(side="right", padx=(5, 0))
        self.label_ram = ttk.Label(self, textvariable=self.ram_var, anchor="e")
        self.label_ram.pack(side="right", padx=(5, 0))
        self.label_cpu = ttk.Label(self, textvariable=self.cpu_var, anchor="e")
        self.label_cpu.pack(side="right", padx=(5, 0))

        self._stop_event = threading.Event()

        # Register for GPU selection changes
        self.gpu_service.add_selection_callback(self._on_gpu_selection_changed)

        # Register for training queue changes
        self.training_queue_manager.on_queue_changed = self._on_queue_changed
        self.training_queue_manager.on_job_status_changed = self._on_job_status_changed

        self._update_stats()
        self._update_queue_status()

    def _create_gpu_selection_controls(self):
        """Create GPU selection controls in the status bar."""
        gpu_frame = ttk.Frame(self)
        gpu_frame.pack(side="left", padx=(10, 10))

        # GPU selection label
        gpu_label = ttk.Label(gpu_frame, text="GPU:", font=("Arial", 8))
        gpu_label.pack(side="left", padx=(0, 5))

        # GPU selection combobox
        self.gpu_combo = ttk.Combobox(gpu_frame, width=35, state="readonly", font=("Arial", 8))
        self.gpu_combo.pack(side="left", padx=(0, 5))
        self.gpu_combo.bind('<<ComboboxSelected>>', self._on_gpu_combo_selected)

        # Refresh button
        self.gpu_refresh_button = ttk.Button(gpu_frame, text="↻", width=3,
                                           command=self._refresh_gpu_list,
                                           style="Small.TButton")
        self.gpu_refresh_button.pack(side="left")

        # Populate GPU list
        self._update_gpu_combo()

    def _create_queue_controls(self):
        """Create training queue controls in the status bar."""
        queue_frame = ttk.Frame(self)
        queue_frame.pack(side="left", padx=(10, 10))

        # Queue status label
        queue_label = ttk.Label(queue_frame, text="Queue:", font=("Arial", 8))
        queue_label.pack(side="left", padx=(0, 5))

        # Queue status variable
        self.label_queue_status = ttk.Label(queue_frame, textvariable=self.queue_status_var, anchor="w", font=("Arial", 8))
        self.label_queue_status.pack(side="left", padx=(0, 5))

        # Open queue manager button
        self.queue_manager_button = ttk.Button(queue_frame, text="Manage Queue",
                                              command=self._open_queue_manager,
                                              width=12)
        self.queue_manager_button.pack(side="left", padx=(0, 5))

    def _create_auto_check_controls(self):
        """Create provider auto-check controls in the status bar."""
        auto_check_frame = ttk.Frame(self)
        auto_check_frame.pack(side="right", padx=(10, 10))

        # Status label
        self.auto_check_label = ttk.Label(auto_check_frame, textvariable=self.auto_check_status_var,
                                         font=("Arial", 8))
        self.auto_check_label.pack(side="left", padx=(0, 5))

        # Toggle button
        self.auto_check_button = ttk.Button(auto_check_frame, text="Start",
                                           command=self._toggle_auto_check_service,
                                           width=6)
        self.auto_check_button.pack(side="left")

        # Initialize status
        self._update_auto_check_status()

    def _update_gpu_combo(self):
        """Update the GPU combobox with available devices."""
        try:
            devices = self.gpu_service.get_available_devices()
            device_names = [device['name'] for device in devices]

            self.gpu_combo['values'] = device_names

            # Set current selection
            current_device = self.gpu_service.get_selected_device()
            current_name = current_device['name']

            if current_name in device_names:
                self.gpu_combo.set(current_name)
            elif device_names:
                self.gpu_combo.set(device_names[0])

        except Exception as e:
            print(f"Error updating GPU combo: {e}")

    def _on_gpu_combo_selected(self, event):
        """Handle GPU selection from combobox."""
        try:
            selected_name = self.gpu_combo.get()
            devices = self.gpu_service.get_available_devices()

            # Find the device index for the selected name
            for device in devices:
                if device['name'] == selected_name:
                    success = self.gpu_service.select_device(device['index'])
                    if success:
                        self.set_status(f"Selected device: {selected_name}")
                    else:
                        self.set_status(f"Failed to select device: {selected_name}")
                    break

        except Exception as e:
            self.set_status(f"Error selecting GPU: {e}")

    def _refresh_gpu_list(self):
        """Refresh the list of available GPUs."""
        try:
            self.gpu_service.refresh_devices()
            self._update_gpu_combo()
            self.set_status("GPU list refreshed")
        except Exception as e:
            self.set_status(f"Error refreshing GPU list: {e}")

    def _on_gpu_selection_changed(self, device_info):
        """Callback for when GPU selection changes."""
        try:
            # Update combobox selection
            self._update_gpu_combo()

            # Notify other components that might need to reinitialize with new GPU
            if hasattr(self.master, 'app'):
                app = self.master.app
                if hasattr(app, '_notify_gpu_selection_changed'):
                    app._notify_gpu_selection_changed(device_info)

        except Exception as e:
            print(f"Error handling GPU selection change: {e}")

    def _update_queue_status(self):
        """Update the display of the training queue status."""
        try:
            total_jobs = self.training_queue_manager.get_total_jobs()
            queued_jobs = self.training_queue_manager.get_queue_size()
            current_job = self.training_queue_manager.get_current_job()

            if current_job:
                status_text = f"{queued_jobs} queued, 1 running"
            elif total_jobs > 0:
                status_text = f"{total_jobs} total"
            else:
                status_text = "0 jobs"

            self.queue_status_var.set(status_text)

            print(f"Queue status update: total={total_jobs}, queued={queued_jobs}, current_job={current_job.character_name if current_job else None}")

        except Exception as e:
            print(f"Error updating queue status: {e}")
            self.queue_status_var.set("Queue: Error")

    def _open_queue_manager(self):
        """Open the queue manager window."""
        try:
            if self.queue_manager_window is None or not hasattr(self.queue_manager_window, 'window') or not self.queue_manager_window.window.winfo_exists():
                self.queue_manager_window = show_queue_manager(self.master)
            else:
                # Window exists, just bring it to front
                self.queue_manager_window.show()
        except Exception as e:
            print(f"Error opening queue manager: {e}")
            self.set_status(f"Error opening queue manager: {e}")

    def _on_queue_changed(self):
        """Handle changes in the training queue."""
        if self.winfo_exists():
            self.after(0, self._update_queue_status)

    def _on_job_status_changed(self, job):
        """Handle changes in individual job status within the queue."""
        if self.winfo_exists():
            self.after(0, self._update_queue_status)

    def set_provider_manager(self, provider_manager):
        """Set the provider manager after initialization."""
        self.provider_manager = provider_manager
        if hasattr(self, 'auto_check_label'):  # Controls already created
            self._update_auto_check_status()
        else:
            self._create_auto_check_controls()

    def set_status(self, text):
        self.status_var.set(text)

    def _toggle_auto_check_service(self):
        """Toggle the auto-check service on/off."""
        if not self.provider_manager:
            return

        try:
            # Check current status and toggle
            if self.auto_check_button.config('text')[-1] == "Start":
                success = self.provider_manager.start_auto_check_service(self.characters_path)
                if success:
                    self.auto_check_button.config(text="Stop")
                    self.auto_check_status_var.set("Auto-check: ON")
                    self.set_status("Provider auto-check service started")
                else:
                    self.set_status("Failed to start auto-check service")
            else:
                success = self.provider_manager.stop_auto_check_service()
                if success:
                    self.auto_check_button.config(text="Start")
                    self.auto_check_status_var.set("Auto-check: OFF")
                    self.set_status("Provider auto-check service stopped")
                else:
                    self.set_status("Failed to stop auto-check service")
        except Exception as e:
            self.set_status(f"Auto-check service error: {str(e)}")

    def _update_auto_check_status(self):
        """Update the auto-check service status display."""
        if self.provider_manager:
            try:
                # Check if service is running (this would need to be implemented in provider_manager)
                is_running = getattr(self.provider_manager, 'is_auto_check_running', lambda: False)()
                if is_running:
                    self.auto_check_status_var.set("Auto-check: ON")
                    self.auto_check_button.config(text="Stop")
                else:
                    self.auto_check_status_var.set("Auto-check: OFF")
                    self.auto_check_button.config(text="Start")
            except Exception:
                self.auto_check_status_var.set("Auto-check: OFF")
                self.auto_check_button.config(text="Start")

    def _update_stats(self):
        # CPU and RAM
        self.cpu_var.set(f"CPU: {psutil.cpu_percent(interval=None):.0f}%")
        self.ram_var.set(f"RAM: {psutil.virtual_memory().percent:.0f}%")

        # Update GPU service stats
        self.gpu_service.update_gpu_stats()

        # VRAM and GPU (for currently selected GPU)
        selected_gpu = self.gpu_service.get_selected_gpu_info()
        if selected_gpu:
            self.vram_var.set(f"VRAM: {selected_gpu.get_memory_usage_percent():.0f}%")
            self.gpu_var.set(f"GPU: {selected_gpu.utilization}%")
        elif nvml_available:
            # Fallback to first GPU if available
            try:
                handle = nvmlDeviceGetHandleByIndex(0)
                meminfo = nvmlDeviceGetMemoryInfo(handle)
                percent = (meminfo.used / meminfo.total) * 100
                self.vram_var.set(f"VRAM: {percent:.0f}%")
                util = nvmlDeviceGetUtilizationRates(handle)
                self.gpu_var.set(f"GPU: {util.gpu}%")
            except Exception:
                self.vram_var.set("VRAM: N/A")
                self.gpu_var.set("GPU: N/A")
        else:
            self.vram_var.set("VRAM: N/A")
            self.gpu_var.set("GPU: N/A")

        if not self._stop_event.is_set():
            self.after(1000, self._update_stats)

    def stop(self):
        """Stop the status bar and cleanup."""
        self._stop_event.set()
        # Unregister from GPU service callbacks
        try:
            self.gpu_service.remove_selection_callback(self._on_gpu_selection_changed)
        except Exception:
            pass

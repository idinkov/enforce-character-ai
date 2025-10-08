"""
Training Queue Manager UI Window.
"""
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional
import threading
from datetime import datetime

from src.services.training_queue import get_training_queue_manager, TrainingJob, TrainingJobStatus
from src.ui.training_status_window import show_training_status
from pathlib import Path


class QueueManagerWindow:
    """Window for managing the training queue."""

    def __init__(self, parent):
        self.parent = parent
        self.window: Optional[tk.Toplevel] = None
        self.queue_manager = get_training_queue_manager()

        # UI elements
        self.tree = None
        self.refresh_timer = None

        # Track selected job
        self.selected_job_id = None

        self._create_window()
        self._setup_callbacks()
        self._refresh_queue_display()

        # Auto-refresh every 2 seconds
        self._start_auto_refresh()

    def _create_window(self):
        """Create the queue manager window."""
        self.window = tk.Toplevel(self.parent)
        self.window.title("Training Queue Manager")
        self.window.geometry("900x550")
        self.window.resizable(True, True)

        # Make window modal
        self.window.transient(self.parent)
        self.window.grab_set()

        # Center the window
        self.window.update_idletasks()
        x = (self.window.winfo_screenwidth() // 2) - (900 // 2)
        y = (self.window.winfo_screenheight() // 2) - (550 // 2)
        self.window.geometry(f"900x550+{x}+{y}")

        # Main frame
        main_frame = ttk.Frame(self.window, padding="10")
        main_frame.pack(fill="both", expand=True)

        # Title
        title_label = ttk.Label(main_frame, text="Training Queue Manager",
                               font=("Arial", 14, "bold"))
        title_label.pack(pady=(0, 10))

        # Queue statistics
        self._create_stats_frame(main_frame)

        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill="both", expand=True, pady=(0, 10))

        # Active Jobs tab
        self.active_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.active_frame, text="Active Queue")
        self._create_queue_list(self.active_frame, show_completed=False)

        # Completed Jobs tab
        self.completed_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.completed_frame, text="Completed Jobs")
        self._create_completed_list(self.completed_frame)

        # Control buttons
        self._create_control_buttons(main_frame)

        # Handle window close
        self.window.protocol("WM_DELETE_WINDOW", self._on_window_close)

    def _create_stats_frame(self, parent):
        """Create the statistics frame."""
        stats_frame = ttk.LabelFrame(parent, text="Queue Statistics", padding="10")
        stats_frame.pack(fill="x", pady=(0, 10))

        # Create labels for statistics
        self.stats_labels = {}

        stats_info = [
            ("total", "Total Jobs:"),
            ("queued", "Queued:"),
            ("running", "Running:"),
            ("completed", "Completed:"),
            ("failed", "Failed:")
        ]

        for i, (key, label_text) in enumerate(stats_info):
            label = ttk.Label(stats_frame, text=label_text)
            label.grid(row=0, column=i*2, sticky="w", padx=(0, 5))

            value_label = ttk.Label(stats_frame, text="0", font=("Arial", 10, "bold"))
            value_label.grid(row=0, column=i*2+1, sticky="w", padx=(0, 20))

            self.stats_labels[key] = value_label

    def _create_queue_list(self, parent, show_completed=True):
        """Create the queue list with custom job cards and progress bars."""
        list_frame = ttk.LabelFrame(parent, text="Training Jobs", padding="10")
        list_frame.pack(fill="both", expand=True, pady=(0, 10))

        # Create scrollable canvas
        canvas_frame = ttk.Frame(list_frame)
        canvas_frame.pack(fill="both", expand=True)

        self.jobs_canvas = tk.Canvas(canvas_frame, bg="white", highlightthickness=0)
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.jobs_canvas.yview)

        self.jobs_container = ttk.Frame(self.jobs_canvas)

        self.jobs_canvas.configure(yscrollcommand=v_scrollbar.set)

        # Pack scrollbar and canvas
        v_scrollbar.pack(side="right", fill="y")
        self.jobs_canvas.pack(side="left", fill="both", expand=True)

        # Create window in canvas
        self.canvas_window = self.jobs_canvas.create_window((0, 0), window=self.jobs_container, anchor="nw")

        # Bind canvas resize
        self.jobs_container.bind("<Configure>", self._on_canvas_configure)
        self.jobs_canvas.bind("<Configure>", self._on_canvas_resize)

        # Dictionary to store job widgets
        self.job_widgets = {}
        self.tree_item_to_job_id = {}  # Keep for compatibility

        # Create a dummy tree for context menu compatibility
        self.tree = None

        # Context menu
        self._create_context_menu()

        self.is_active_tab = not show_completed


    def _on_canvas_configure(self, event):
        """Update canvas scroll region when container size changes."""
        if hasattr(self, 'jobs_canvas'):
            self.jobs_canvas.configure(scrollregion=self.jobs_canvas.bbox("all"))

    def _on_canvas_resize(self, event):
        """Resize the container frame to match canvas width."""
        if hasattr(self, 'canvas_window'):
            self.jobs_canvas.itemconfig(self.canvas_window, width=event.width)

    def _on_completed_canvas_configure(self, event):
        """Update completed canvas scroll region when container size changes."""
        if hasattr(self, 'completed_jobs_canvas'):
            self.completed_jobs_canvas.configure(scrollregion=self.completed_jobs_canvas.bbox("all"))

    def _on_completed_canvas_resize(self, event):
        """Resize the completed container frame to match canvas width."""
        if hasattr(self, 'completed_canvas_window'):
            self.completed_jobs_canvas.itemconfig(self.completed_canvas_window, width=event.width)

    def _create_completed_list(self, parent):
        """Create the completed jobs list with custom job cards."""
        list_frame = ttk.LabelFrame(parent, text="Completed Jobs", padding="10")
        list_frame.pack(fill="both", expand=True)

        # Create scrollable canvas
        canvas_frame = ttk.Frame(list_frame)
        canvas_frame.pack(fill="both", expand=True)

        self.completed_jobs_canvas = tk.Canvas(canvas_frame, bg="white", highlightthickness=0)
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.completed_jobs_canvas.yview)

        self.completed_jobs_container = ttk.Frame(self.completed_jobs_canvas)

        self.completed_jobs_canvas.configure(yscrollcommand=v_scrollbar.set)

        # Pack scrollbar and canvas
        v_scrollbar.pack(side="right", fill="y")
        self.completed_jobs_canvas.pack(side="left", fill="both", expand=True)

        # Create window in canvas
        self.completed_canvas_window = self.completed_jobs_canvas.create_window((0, 0), window=self.completed_jobs_container, anchor="nw")

        # Bind canvas resize
        self.completed_jobs_container.bind("<Configure>", self._on_completed_canvas_configure)
        self.completed_jobs_canvas.bind("<Configure>", self._on_completed_canvas_resize)

        # Dictionary to store job widgets
        self.completed_job_widgets = {}
        self.completed_tree_item_to_job_id = {}  # Keep for compatibility

        # Create a dummy tree for context menu compatibility
        self.completed_tree = None

        # Context menu for completed tree
        self._create_completed_context_menu()


    def _create_context_menu(self):
        """Create context menu for job cards."""
        self.context_menu = tk.Menu(self.window, tearoff=0)
        self.context_menu.add_command(label="View Status", command=self._view_job_status)
        self.context_menu.add_command(label="Stop Job", command=self._stop_selected_job)
        self.context_menu.add_command(label="Remove Job", command=self._remove_selected_job)
        self.context_menu.add_separator()
        self.context_menu.add_command(label="Move to Top", command=self._move_job_to_top)


    def _create_completed_context_menu(self):
        """Create context menu for completed job cards."""
        self.completed_context_menu = tk.Menu(self.window, tearoff=0)
        self.completed_context_menu.add_command(label="View Details", command=self._view_job_status)
        self.completed_context_menu.add_command(label="Remove Job", command=self._remove_selected_job)


    def _create_control_buttons(self, parent):
        """Create control buttons."""
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill="x", pady=(0, 10))

        # Left side buttons
        left_buttons = ttk.Frame(button_frame)
        left_buttons.pack(side="left")

        self.pause_button = ttk.Button(left_buttons, text="Stop Queue",
                                      command=self._toggle_queue_pause)
        self.pause_button.pack(side="left", padx=(0, 10))

        ttk.Button(left_buttons, text="Clear Completed",
                  command=self._clear_completed_jobs).pack(side="left", padx=(0, 10))

        ttk.Button(left_buttons, text="Clear All",
                  command=self._clear_all_jobs).pack(side="left", padx=(0, 10))

        # Right side buttons
        right_buttons = ttk.Frame(button_frame)
        right_buttons.pack(side="right")

        ttk.Button(right_buttons, text="Refresh",
                  command=self._refresh_queue_display).pack(side="left", padx=(0, 10))

        ttk.Button(right_buttons, text="Close",
                  command=self._on_window_close).pack(side="left")

    def _setup_callbacks(self):
        """Setup callbacks for queue manager events."""
        self.queue_manager.on_queue_changed = self._on_queue_changed
        self.queue_manager.on_job_status_changed = self._on_job_status_changed

    def _start_auto_refresh(self):
        """Start automatic refresh of the queue display."""
        if self.window and self.window.winfo_exists():
            self._refresh_queue_display()
            self.refresh_timer = self.window.after(2000, self._start_auto_refresh)

    def _refresh_queue_display(self):
        """Refresh the queue display."""
        try:
            # Clear existing job widgets
            for widget in self.jobs_container.winfo_children():
                widget.destroy()
            for widget in self.completed_jobs_container.winfo_children():
                widget.destroy()

            self.job_widgets = {}
            self.completed_job_widgets = {}
            self.tree_item_to_job_id = {}
            self.completed_tree_item_to_job_id = {}

            # Get all jobs
            jobs = self.queue_manager.get_jobs()

            # Separate active and completed jobs
            active_jobs = []
            completed_jobs = []

            for job in jobs:
                if job.status in [TrainingJobStatus.COMPLETED, TrainingJobStatus.FAILED, TrainingJobStatus.CANCELLED]:
                    completed_jobs.append(job)
                else:
                    active_jobs.append(job)

            # Add jobs to respective containers
            for idx, job in enumerate(active_jobs):
                self._create_job_card(job, self.jobs_container, idx, is_completed=False)

            for idx, job in enumerate(completed_jobs):
                self._create_job_card(job, self.completed_jobs_container, idx, is_completed=True)

            # Update statistics
            self._update_statistics(jobs)

            # Update scroll regions
            self.jobs_canvas.update_idletasks()
            self.jobs_canvas.configure(scrollregion=self.jobs_canvas.bbox("all"))
            self.completed_jobs_canvas.update_idletasks()
            self.completed_jobs_canvas.configure(scrollregion=self.completed_jobs_canvas.bbox("all"))

        except Exception as e:
            print(f"Error refreshing queue display: {e}")

    def _create_job_card(self, job: TrainingJob, container, idx, is_completed=False):
        """Create a job card with progress bars."""
        # Determine background color based on status
        if job.status == TrainingJobStatus.RUNNING:
            bg_color = "#e6f3ff"
        elif job.status == TrainingJobStatus.COMPLETED:
            bg_color = "#e6ffe6"
        elif job.status == TrainingJobStatus.FAILED:
            bg_color = "#ffe6e6"
        elif job.status == TrainingJobStatus.CANCELLED:
            bg_color = "#f0f0f0"
        else:
            bg_color = "#ffffff"

        # Create card frame
        card = tk.Frame(container, bg=bg_color, relief="solid", borderwidth=1, padx=10, pady=8)
        card.pack(fill="x", padx=5, pady=3)

        # Store reference for context menu
        card.job_id = job.id
        if is_completed:
            self.completed_job_widgets[job.id] = card
            self.completed_tree_item_to_job_id[job.id] = job.id
        else:
            self.job_widgets[job.id] = card
            self.tree_item_to_job_id[job.id] = job.id

        # Bind click events
        card.bind("<Button-1>", lambda e: self._on_card_click(job.id, is_completed))
        card.bind("<Button-3>", lambda e: self._on_card_right_click(e, job.id, is_completed))

        # Top row: Character name, status, and times
        top_row = tk.Frame(card, bg=bg_color)
        top_row.pack(fill="x", pady=(0, 5))

        # Character name (bold)
        name_label = tk.Label(top_row, text=job.character_name, font=("Arial", 10, "bold"),
                             bg=bg_color, anchor="w")
        name_label.pack(side="left")

        # Status badge
        status_color = {
            TrainingJobStatus.RUNNING: "#0078d4",
            TrainingJobStatus.COMPLETED: "#107c10",
            TrainingJobStatus.FAILED: "#d13438",
            TrainingJobStatus.CANCELLED: "#605e5c",
            TrainingJobStatus.QUEUED: "#8764b8",
            TrainingJobStatus.STOPPING: "#ca5010"
        }.get(job.status, "#605e5c")

        status_label = tk.Label(top_row, text=job.status.value, bg=status_color, fg="white",
                               font=("Arial", 8, "bold"), padx=8, pady=2)
        status_label.pack(side="left", padx=(10, 0))

        # Time info on the right
        time_frame = tk.Frame(top_row, bg=bg_color)
        time_frame.pack(side="right")

        if job.status in [TrainingJobStatus.COMPLETED, TrainingJobStatus.FAILED, TrainingJobStatus.CANCELLED]:
            if job.started_at and job.completed_at:
                duration = job.completed_at - job.started_at
                duration_str = self._format_duration(duration)
                time_label = tk.Label(time_frame, text=f"Duration: {duration_str}",
                                     font=("Arial", 8), bg=bg_color, fg="#605e5c")
                time_label.pack(side="right")
        elif job.status == TrainingJobStatus.RUNNING and job.started_at:
            from datetime import datetime
            running_time = datetime.now() - job.started_at
            time_str = self._format_duration(running_time)
            time_label = tk.Label(time_frame, text=f"Running: {time_str}",
                                 font=("Arial", 8), bg=bg_color, fg="#605e5c")
            time_label.pack(side="right")

        # Progress section (only for running jobs)
        if job.status == TrainingJobStatus.RUNNING:
            progress = job.progress

            # Epoch progress bar
            current_epoch = progress.get('current_epoch', 0)
            total_epochs = progress.get('total_epochs', 0)
            epoch_percent = progress.get('epoch_percent', 0)

            if total_epochs > 0:
                epoch_frame = tk.Frame(card, bg=bg_color)
                epoch_frame.pack(fill="x", pady=2)

                # Epoch label
                epoch_text = f"Epoch: {current_epoch}/{total_epochs}"
                epoch_label = tk.Label(epoch_frame, text=epoch_text, font=("Arial", 8),
                                      bg=bg_color, anchor="w", width=15)
                epoch_label.pack(side="left")

                # Epoch progress bar
                epoch_progress_frame = tk.Frame(epoch_frame, bg="#d0d0d0", height=22, relief="sunken", borderwidth=1)
                epoch_progress_frame.pack(side="left", fill="x", expand=True, padx=(5, 5))

                # Calculate bar width
                if epoch_percent > 0:
                    bar_width = epoch_percent / 100.0
                    epoch_bar = tk.Frame(epoch_progress_frame, bg="#0078d4", height=20)
                    epoch_bar.place(relx=0, rely=0, relwidth=bar_width, relheight=1)

                    # Percentage text on the bar
                    percent_text = tk.Label(epoch_progress_frame, text=f"{epoch_percent}%",
                                          font=("Arial", 8, "bold"), bg="#0078d4", fg="white")
                    percent_text.place(relx=bar_width/2, rely=0.5, anchor="center")

                # ETA
                epoch_eta = progress.get('epoch_eta_seconds', 0)
                if epoch_eta > 0:
                    eta_str = self._format_eta(epoch_eta)
                    eta_label = tk.Label(epoch_frame, text=f"ETA: {eta_str}",
                                        font=("Arial", 8), bg=bg_color, width=12)
                    eta_label.pack(side="right")

            # Step progress bar
            current_step = progress.get('current_step', 0)
            total_steps = progress.get('total_steps', 0)
            step_percent = progress.get('step_percent', 0)

            if total_steps > 0:
                step_frame = tk.Frame(card, bg=bg_color)
                step_frame.pack(fill="x", pady=2)

                # Step label
                step_text = f"Step: {current_step}/{total_steps}"
                step_label = tk.Label(step_frame, text=step_text, font=("Arial", 8),
                                     bg=bg_color, anchor="w", width=15)
                step_label.pack(side="left")

                # Step progress bar
                step_progress_frame = tk.Frame(step_frame, bg="#d0d0d0", height=22, relief="sunken", borderwidth=1)
                step_progress_frame.pack(side="left", fill="x", expand=True, padx=(5, 5))

                # Calculate bar width
                if step_percent > 0:
                    bar_width = step_percent / 100.0
                    step_bar = tk.Frame(step_progress_frame, bg="#107c10", height=20)
                    step_bar.place(relx=0, rely=0, relwidth=bar_width, relheight=1)

                    # Percentage text on the bar
                    percent_text = tk.Label(step_progress_frame, text=f"{step_percent}%",
                                          font=("Arial", 8, "bold"), bg="#107c10", fg="white")
                    percent_text.place(relx=bar_width/2, rely=0.5, anchor="center")

                # Loss info
                loss = progress.get('smooth_loss', progress.get('loss', 0))
                if loss > 0:
                    loss_label = tk.Label(step_frame, text=f"Loss: {loss:.4f}",
                                         font=("Arial", 8), bg=bg_color, width=12)
                    loss_label.pack(side="right")

            # Stage info if no progress bars yet
            if total_epochs == 0 and total_steps == 0:
                stage = progress.get('stage', 'Training...')
                stage_label = tk.Label(card, text=f"⟳ {stage}", font=("Arial", 9),
                                      bg=bg_color, fg="#605e5c")
                stage_label.pack(anchor="w", pady=2)

        elif job.status == TrainingJobStatus.QUEUED:
            status_text = tk.Label(card, text="⏸ Waiting in queue...", font=("Arial", 9),
                                  bg=bg_color, fg="#605e5c")
            status_text.pack(anchor="w", pady=2)

        elif job.status == TrainingJobStatus.COMPLETED:
            completed_text = tk.Label(card, text="✓ Training completed successfully",
                                     font=("Arial", 9), bg=bg_color, fg="#107c10")
            completed_text.pack(anchor="w", pady=2)

        elif job.status == TrainingJobStatus.FAILED:
            error_msg = job.error_message or "Training failed"
            failed_text = tk.Label(card, text=f"✗ {error_msg}", font=("Arial", 9),
                                  bg=bg_color, fg="#d13438")
            failed_text.pack(anchor="w", pady=2)

        elif job.status == TrainingJobStatus.CANCELLED:
            cancelled_text = tk.Label(card, text="⊘ Training cancelled", font=("Arial", 9),
                                     bg=bg_color, fg="#605e5c")
            cancelled_text.pack(anchor="w", pady=2)

        # Bottom row: Model info
        model_text = job.base_model
        if len(model_text) > 60:
            model_text = model_text[:57] + "..."

        model_label = tk.Label(card, text=f"Model: {model_text}", font=("Arial", 8),
                              bg=bg_color, fg="#605e5c", anchor="w")
        model_label.pack(anchor="w", pady=(5, 0))

        # Bind all child widgets to the same click events
        for widget in card.winfo_children():
            widget.bind("<Button-1>", lambda e: self._on_card_click(job.id, is_completed))
            widget.bind("<Button-3>", lambda e: self._on_card_right_click(e, job.id, is_completed))
            for child in widget.winfo_children():
                child.bind("<Button-1>", lambda e: self._on_card_click(job.id, is_completed))
                child.bind("<Button-3>", lambda e: self._on_card_right_click(e, job.id, is_completed))

    def _format_eta(self, seconds: int) -> str:
        """Format ETA in seconds to human readable format."""
        if seconds < 60:
            return f"{seconds}s"
        elif seconds < 3600:
            minutes = seconds // 60
            return f"{minutes}m"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours}h{minutes}m"

    def _on_card_click(self, job_id, is_completed):
        """Handle job card click."""
        self.selected_job_id = job_id

        # Update visual selection
        if is_completed:
            for jid, widget in self.completed_job_widgets.items():
                if jid == job_id:
                    widget.configure(relief="raised", borderwidth=2)
                else:
                    widget.configure(relief="solid", borderwidth=1)
        else:
            for jid, widget in self.job_widgets.items():
                if jid == job_id:
                    widget.configure(relief="raised", borderwidth=2)
                else:
                    widget.configure(relief="solid", borderwidth=1)

    def _on_card_right_click(self, event, job_id, is_completed):
        """Handle job card right-click."""
        self.selected_job_id = job_id

        # Update visual selection
        self._on_card_click(job_id, is_completed)

        # Show context menu
        try:
            if is_completed:
                self.completed_context_menu.tk_popup(event.x_root, event.y_root)
            else:
                self.context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            if is_completed:
                self.completed_context_menu.grab_release()
            else:
                self.context_menu.grab_release()

    def _on_job_selected(self, event):
        """Handle job selection - kept for compatibility."""
        pass

    def _on_completed_job_selected(self, event):
        """Handle completed job selection - kept for compatibility."""
        pass

    def _show_context_menu(self, event):
        """Show context menu - kept for compatibility."""
        pass

    def _show_completed_context_menu(self, event):
        """Show completed context menu - kept for compatibility."""
        pass

    def _cleanup_progress_bars(self, tree_widget):
        """Clean up progress bars when refreshing the tree."""
        # No longer needed with actual progress tracking
        pass


    def _format_duration(self, duration) -> str:
        """Format a timedelta as a human-readable string."""
        total_seconds = int(duration.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60

        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"

    def _show_tooltip(self, event, text):
        """Show a tooltip at the given event position."""
        if hasattr(self, '_tooltip'):
            self._hide_tooltip()

        self._tooltip = tk.Toplevel()
        self._tooltip.wm_overrideredirect(True)
        self._tooltip.wm_geometry(f"+{event.x_root + 10}+{event.y_root + 10}")

        label = tk.Label(self._tooltip, text=text, background="#ffffe0",
                        relief="solid", borderwidth=1, font=("Arial", 9),
                        justify="left")
        label.pack()

    def _hide_tooltip(self):
        """Hide the current tooltip."""
        if hasattr(self, '_tooltip'):
            self._tooltip.destroy()
            delattr(self, '_tooltip')

    def _update_statistics(self, jobs):
        """Update the statistics display."""
        stats = {
            "total": len(jobs),
            "queued": len([j for j in jobs if j.status == TrainingJobStatus.QUEUED]),
            "running": len([j for j in jobs if j.status == TrainingJobStatus.RUNNING]),
            "completed": len([j for j in jobs if j.status == TrainingJobStatus.COMPLETED]),
            "failed": len([j for j in jobs if j.status == TrainingJobStatus.FAILED])
        }

        for key, value in stats.items():
            if key in self.stats_labels:
                self.stats_labels[key].config(text=str(value))

    def _on_queue_changed(self):
        """Handle queue changes."""
        if self.window and self.window.winfo_exists():
            self.window.after(0, self._refresh_queue_display)

    def _on_job_status_changed(self, job: TrainingJob):
        """Handle job status changes."""
        if self.window and self.window.winfo_exists():
            self.window.after(0, self._refresh_queue_display)


    def _view_job_status(self):
        """View detailed status of selected job."""
        if not self.selected_job_id:
            return

        job = self.queue_manager.get_job(self.selected_job_id)
        if not job:
            return

        if job.status == TrainingJobStatus.RUNNING:
            # Show training status window
            try:
                training_outputs_dir = Path(job.output_model_path).parent.parent.parent / "training_outputs"
                show_training_status(
                    self.parent,
                    job.character_name,
                    training_outputs_dir,
                    job.process,
                    job.base_model,
                    job.output_model_path
                )
            except Exception as e:
                messagebox.showerror("Error", f"Failed to open status window: {e}")
        else:
            # Show job details in a message box
            details = f"Character: {job.character_name}\n"
            details += f"Status: {job.status.value}\n"
            details += f"Created: {job.created_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
            if job.started_at:
                details += f"Started: {job.started_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
            if job.completed_at:
                details += f"Completed: {job.completed_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
            if job.error_message:
                details += f"Error: {job.error_message}\n"
            details += f"Model: {job.base_model}\n"
            details += f"Output: {job.output_model_path}"

            messagebox.showinfo("Job Details", details)

    def _stop_selected_job(self):
        """Stop the selected job."""
        if not self.selected_job_id:
            messagebox.showwarning("Warning", "Please select a job to stop.")
            return

        job = self.queue_manager.get_job(self.selected_job_id)
        if not job:
            return

        if job.status != TrainingJobStatus.RUNNING:
            messagebox.showwarning("Warning", "Only running jobs can be stopped.")
            return

        # Confirm stop
        result = messagebox.askyesno(
            "Confirm Stop",
            f"Are you sure you want to stop training for '{job.character_name}'?\n\n"
            "This will terminate the training process immediately."
        )

        if result:
            success = self.queue_manager.stop_job(self.selected_job_id)
            if success:
                messagebox.showinfo("Success", "Job stop requested.")
            else:
                messagebox.showerror("Error", "Failed to stop job.")

    def _remove_selected_job(self):
        """Remove the selected job from queue."""
        if not self.selected_job_id:
            messagebox.showwarning("Warning", "Please select a job to remove.")
            return

        job = self.queue_manager.get_job(self.selected_job_id)
        if not job:
            return

        if job.status == TrainingJobStatus.RUNNING:
            messagebox.showwarning("Warning", "Cannot remove running job. Stop it first.")
            return

        # Confirm removal
        result = messagebox.askyesno(
            "Confirm Removal",
            f"Are you sure you want to remove the job for '{job.character_name}'?"
        )

        if result:
            success = self.queue_manager.remove_job(self.selected_job_id)
            if success:
                self.selected_job_id = None
            else:
                messagebox.showerror("Error", "Failed to remove job.")

    def _move_job_to_top(self):
        """Move selected job to top of queue."""
        if not self.selected_job_id:
            messagebox.showwarning("Warning", "Please select a job to move.")
            return

        job = self.queue_manager.get_job(self.selected_job_id)
        if not job:
            return

        if job.status != TrainingJobStatus.QUEUED:
            messagebox.showwarning("Warning", "Only queued jobs can be moved.")
            return

        # Move to top of queue order
        if self.selected_job_id in self.queue_manager.job_order:
            self.queue_manager.job_order.remove(self.selected_job_id)
            self.queue_manager.job_order.insert(0, self.selected_job_id)
            self.queue_manager._notify_queue_changed()

    def _toggle_queue_pause(self):
        """Toggle queue pause/resume."""
        if self.queue_manager.is_running:
            self.queue_manager.stop_worker()
            self.pause_button.config(text="Resume Queue")
        else:
            self.queue_manager.start_worker()
            self.pause_button.config(text="Stop Queue")

    def _clear_completed_jobs(self):
        """Clear all completed and failed jobs."""
        completed_statuses = [TrainingJobStatus.COMPLETED, TrainingJobStatus.FAILED, TrainingJobStatus.CANCELLED]
        jobs_to_remove = [job.id for job in self.queue_manager.get_jobs()
                         if job.status in completed_statuses]

        if not jobs_to_remove:
            messagebox.showinfo("Info", "No completed jobs to remove.")
            return

        result = messagebox.askyesno(
            "Confirm Clear",
            f"Remove {len(jobs_to_remove)} completed/failed jobs from the queue?"
        )

        if result:
            for job_id in jobs_to_remove:
                self.queue_manager.remove_job(job_id)

    def _clear_all_jobs(self):
        """Clear all jobs from queue."""
        total_jobs = self.queue_manager.get_total_jobs()
        if total_jobs == 0:
            messagebox.showinfo("Info", "Queue is already empty.")
            return

        # Check if there are running jobs
        running_jobs = [job for job in self.queue_manager.get_jobs()
                       if job.status == TrainingJobStatus.RUNNING]

        if running_jobs:
            result = messagebox.askyesno(
                "Confirm Clear All",
                f"This will stop {len(running_jobs)} running job(s) and remove all {total_jobs} jobs.\n\n"
                "Are you sure you want to continue?"
            )
        else:
            result = messagebox.askyesno(
                "Confirm Clear All",
                f"Remove all {total_jobs} jobs from the queue?"
            )

        if result:
            # Stop all running jobs first
            for job in running_jobs:
                self.queue_manager.stop_job(job.id)

            # Remove all jobs
            jobs_to_remove = list(self.queue_manager.jobs.keys())
            for job_id in jobs_to_remove:
                self.queue_manager.remove_job(job_id)

    def _on_window_close(self):
        """Handle window close event."""
        try:
            # Stop auto-refresh
            if self.refresh_timer:
                self.window.after_cancel(self.refresh_timer)

            # Remove callbacks
            self.queue_manager.on_queue_changed = None
            self.queue_manager.on_job_status_changed = None

            # Close window
            self.window.destroy()

        except Exception as e:
            print(f"Error closing queue manager window: {e}")

    def show(self):
        """Show the queue manager window."""
        if self.window:
            self.window.deiconify()
            self.window.lift()
            self.window.focus()


def show_queue_manager(parent) -> QueueManagerWindow:
    """Create and show the queue manager window."""
    return QueueManagerWindow(parent)

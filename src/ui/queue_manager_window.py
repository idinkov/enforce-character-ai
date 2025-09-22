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
        """Create the queue list with treeview."""
        list_frame = ttk.LabelFrame(parent, text="Training Jobs", padding="10")
        list_frame.pack(fill="both", expand=True, pady=(0, 10))

        # Create treeview with scrollbars
        tree_frame = ttk.Frame(list_frame)
        tree_frame.pack(fill="both", expand=True)

        # Configure columns - add hidden job_id column
        columns = ("Character", "Status", "Progress", "Created", "Started", "Model")
        self.tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=12)

        # Configure column headings and widths
        column_configs = {
            "Character": (120, "w"),
            "Status": (100, "center"),
            "Progress": (150, "center"),
            "Created": (120, "center"),
            "Started": (120, "center"),
            "Model": (200, "w")
        }

        for col in columns:
            width, anchor = column_configs[col]
            self.tree.heading(col, text=col)
            self.tree.column(col, width=width, anchor=anchor)

        # Scrollbars
        v_scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        h_scrollbar = ttk.Scrollbar(tree_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

        # Pack treeview and scrollbars
        self.tree.pack(side="left", fill="both", expand=True)
        v_scrollbar.pack(side="right", fill="y")
        h_scrollbar.pack(side="bottom", fill="x")

        # Bind selection event
        self.tree.bind("<<TreeviewSelect>>", self._on_job_selected)

        # Context menu
        self._create_context_menu()

        # Dictionary to store job_id for each tree item
        self.tree_item_to_job_id = {}

        # Hide completed jobs in Active Queue tab
        if not show_completed:
            for item in self.tree.get_children():
                job_id = self.tree_item_to_job_id.get(item)
                if job_id:
                    job = self.queue_manager.get_job(job_id)
                    if job and job.status in [TrainingJobStatus.COMPLETED, TrainingJobStatus.FAILED, TrainingJobStatus.CANCELLED]:
                        self.tree.detach(item)

    def _create_completed_list(self, parent):
        """Create the completed jobs list."""
        list_frame = ttk.LabelFrame(parent, text="Completed Jobs", padding="10")
        list_frame.pack(fill="both", expand=True)

        # Create treeview with scrollbars
        tree_frame = ttk.Frame(list_frame)
        tree_frame.pack(fill="both", expand=True)

        # Configure columns - different headers for completed jobs
        columns = ("Character", "Status", "Progress", "Created", "Completed", "Model")
        self.completed_tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=12)

        # Configure column headings and widths
        column_configs = {
            "Character": (120, "w"),
            "Status": (100, "center"),
            "Progress": (150, "center"),
            "Created": (120, "center"),
            "Completed": (120, "center"),  # Changed from "Started" to "Completed"
            "Model": (200, "w")
        }

        for col in columns:
            width, anchor = column_configs[col]
            self.completed_tree.heading(col, text=col)
            self.completed_tree.column(col, width=width, anchor=anchor)

        # Scrollbars
        v_scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=self.completed_tree.yview)
        h_scrollbar = ttk.Scrollbar(tree_frame, orient="horizontal", command=self.completed_tree.xview)
        self.completed_tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

        # Pack treeview and scrollbars
        self.completed_tree.pack(side="left", fill="both", expand=True)
        v_scrollbar.pack(side="right", fill="y")
        h_scrollbar.pack(side="bottom", fill="x")

        # Bind selection event for completed tree
        self.completed_tree.bind("<<TreeviewSelect>>", self._on_completed_job_selected)

        # Context menu for completed tree
        self._create_completed_context_menu()

        # Dictionary to store job_id for each tree item
        self.completed_tree_item_to_job_id = {}

    def _create_context_menu(self):
        """Create context menu for the treeview."""
        self.context_menu = tk.Menu(self.window, tearoff=0)
        self.context_menu.add_command(label="View Status", command=self._view_job_status)
        self.context_menu.add_command(label="Stop Job", command=self._stop_selected_job)
        self.context_menu.add_command(label="Remove Job", command=self._remove_selected_job)
        self.context_menu.add_separator()
        self.context_menu.add_command(label="Move to Top", command=self._move_job_to_top)

        # Bind right-click to show context menu
        self.tree.bind("<Button-3>", self._show_context_menu)

    def _create_completed_context_menu(self):
        """Create context menu for the completed jobs treeview."""
        self.completed_context_menu = tk.Menu(self.window, tearoff=0)
        self.completed_context_menu.add_command(label="View Details", command=self._view_job_status)
        self.completed_context_menu.add_command(label="Remove Job", command=self._remove_selected_job)

        # Bind right-click to show context menu for completed tree
        self.completed_tree.bind("<Button-3>", self._show_completed_context_menu)

    def _create_control_buttons(self, parent):
        """Create control buttons."""
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill="x", pady=(0, 10))

        # Left side buttons
        left_buttons = ttk.Frame(button_frame)
        left_buttons.pack(side="left")

        self.pause_button = ttk.Button(left_buttons, text="Pause Queue",
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
            # Clean up existing progress bars
            self._cleanup_progress_bars(self.tree)
            self._cleanup_progress_bars(self.completed_tree)

            # Clear existing items in both trees
            for item in self.tree.get_children():
                self.tree.delete(item)
            for item in self.completed_tree.get_children():
                self.completed_tree.delete(item)

            # Clear job mapping
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

            # Add jobs to respective trees
            for job in active_jobs:
                self._add_job_to_tree(job, self.tree, self.tree_item_to_job_id)

            for job in completed_jobs:
                self._add_job_to_tree(job, self.completed_tree, self.completed_tree_item_to_job_id)

            # Update statistics
            self._update_statistics(jobs)

        except Exception as e:
            print(f"Error refreshing queue display: {e}")

    def _add_job_to_tree(self, job: TrainingJob, tree_widget, item_mapping):
        """Add a job to the specified treeview."""
        # Format timestamps
        created_str = job.created_at.strftime("%H:%M:%S") if job.created_at else ""
        started_str = job.started_at.strftime("%H:%M:%S") if job.started_at else ""

        # Create progress widget or text based on status
        if job.status == TrainingJobStatus.COMPLETED:
            # Calculate training duration
            if job.started_at and job.completed_at:
                duration = job.completed_at - job.started_at
                duration_str = self._format_duration(duration)
                progress_str = f"✓ {duration_str}"
                completed_str = job.completed_at.strftime("%H:%M:%S")
            else:
                progress_str = "✓ Complete"
                completed_str = job.completed_at.strftime("%H:%M:%S") if job.completed_at else ""
        elif job.status == TrainingJobStatus.FAILED:
            progress_str = "✗ Failed"
            completed_str = job.completed_at.strftime("%H:%M:%S") if job.completed_at else ""
        elif job.status == TrainingJobStatus.CANCELLED:
            progress_str = "⊘ Cancelled"
            completed_str = job.completed_at.strftime("%H:%M:%S") if job.completed_at else ""
        elif job.status == TrainingJobStatus.RUNNING:
            progress_str = "⟳ Training..."
            completed_str = ""
        elif job.status == TrainingJobStatus.QUEUED:
            progress_str = "⏸ Waiting..."
            completed_str = ""
        elif job.status == TrainingJobStatus.STOPPING:
            progress_str = "⏹ Stopping..."
            completed_str = ""
        else:
            progress_str = job.status.value
            completed_str = ""

        # Format model name (truncate if too long)
        model_str = job.base_model
        if len(model_str) > 30:
            model_str = model_str[:27] + "..."

        # Use completed time for completed jobs, started time for others
        if job.status in [TrainingJobStatus.COMPLETED, TrainingJobStatus.FAILED, TrainingJobStatus.CANCELLED]:
            time_column = completed_str
        else:
            time_column = started_str

        # Determine row colors based on status
        tags = []
        if job.status == TrainingJobStatus.RUNNING:
            tags.append("running")
        elif job.status == TrainingJobStatus.COMPLETED:
            tags.append("completed")
        elif job.status == TrainingJobStatus.FAILED:
            tags.append("failed")
        elif job.status == TrainingJobStatus.CANCELLED:
            tags.append("cancelled")

        # Insert into treeview
        item = tree_widget.insert("", "end",
                                  values=(job.character_name, job.status.value, progress_str,
                                         created_str, time_column, model_str),
                                  tags=tags)

        # Store job ID with the item
        item_mapping[item] = job.id

        # Configure row colors
        tree_widget.tag_configure("running", background="#e6f3ff")
        tree_widget.tag_configure("completed", background="#e6ffe6")
        tree_widget.tag_configure("failed", background="#ffe6e6")
        tree_widget.tag_configure("cancelled", background="#f0f0f0")

        # Create progress bar for running jobs
        if job.status == TrainingJobStatus.RUNNING:
            self._create_progress_bar_for_job(tree_widget, item, job)

        # Create tooltip for detailed progress information
        self._create_progress_tooltip(tree_widget, item, job)

    def _create_progress_bar_for_job(self, tree_widget, item, job: TrainingJob):
        """Create a progress bar for a running training job."""
        try:
            # For treeview, we can't embed widgets directly like in Canvas
            # Instead, we'll update the progress text to show animated indicators
            # and use a more sophisticated text-based progress indication

            # Schedule periodic updates for running jobs
            if not hasattr(tree_widget, '_animated_jobs'):
                tree_widget._animated_jobs = set()

            tree_widget._animated_jobs.add(job.id)

            # Start animation for this job
            self._animate_progress_text(tree_widget, item, job)

        except Exception as e:
            print(f"Error creating progress bar: {e}")

    def _animate_progress_text(self, tree_widget, item, job: TrainingJob):
        """Animate the progress text for running jobs."""
        try:
            # Check if the item still exists in the tree
            if not tree_widget.exists(item):
                # Item was deleted during refresh, stop animation for this item
                if hasattr(tree_widget, '_animated_jobs'):
                    tree_widget._animated_jobs.discard(job.id)
                return

            if not hasattr(tree_widget, '_animated_jobs') or job.id not in tree_widget._animated_jobs:
                return

            if job.status != TrainingJobStatus.RUNNING:
                # Job is no longer running, stop animation
                if hasattr(tree_widget, '_animated_jobs'):
                    tree_widget._animated_jobs.discard(job.id)
                return

            # Get current animation frame
            if not hasattr(job, '_animation_frame'):
                job._animation_frame = 0

            # Create animated progress indicators
            progress_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
            char = progress_chars[job._animation_frame % len(progress_chars)]

            # Calculate running time
            if job.started_at:
                from datetime import datetime
                running_time = datetime.now() - job.started_at
                time_str = self._format_duration(running_time)
                progress_text = f"{char} Training... ({time_str})"
            else:
                progress_text = f"{char} Training..."

            # Update the progress column for this item - double check item still exists
            try:
                current_values = list(tree_widget.item(item, 'values'))
                if len(current_values) >= 3:
                    current_values[2] = progress_text  # Progress column is index 2
                    tree_widget.item(item, values=current_values)
            except tk.TclError:
                # Item no longer exists, stop animation
                if hasattr(tree_widget, '_animated_jobs'):
                    tree_widget._animated_jobs.discard(job.id)
                return

            # Increment animation frame
            job._animation_frame += 1

            # Schedule next animation frame
            if self.window and self.window.winfo_exists():
                self.window.after(200, lambda: self._animate_progress_text(tree_widget, item, job))

        except Exception as e:
            # Clean up animation tracking for this job on any error
            if hasattr(tree_widget, '_animated_jobs'):
                tree_widget._animated_jobs.discard(job.id)
            print(f"Error animating progress text: {e}")

    def _cleanup_progress_bars(self, tree_widget):
        """Clean up progress bars when refreshing the tree."""
        if hasattr(tree_widget, '_animated_jobs'):
            tree_widget._animated_jobs.clear()

    def _create_progress_tooltip(self, tree_widget, item, job: TrainingJob):
        """Create a tooltip showing detailed progress information."""
        def show_tooltip(event):
            # Check if mouse is over the progress column
            region = tree_widget.identify_region(event.x, event.y)
            if region == "cell":
                column = tree_widget.identify_column(event.x)  # Only pass x coordinate
                if column == "#3":  # Progress column (0-indexed, but identify_column returns 1-indexed)
                    tooltip_text = self._get_detailed_progress_text(job)
                    self._show_tooltip(event, tooltip_text)

        def hide_tooltip(event):
            self._hide_tooltip()

        tree_widget.bind("<Motion>", show_tooltip, add="+")
        tree_widget.bind("<Leave>", hide_tooltip, add="+")

    def _get_detailed_progress_text(self, job: TrainingJob) -> str:
        """Get detailed progress text for tooltip."""
        details = []

        details.append(f"Character: {job.character_name}")
        details.append(f"Status: {job.status.value}")

        if job.created_at:
            details.append(f"Created: {job.created_at.strftime('%Y-%m-%d %H:%M:%S')}")

        if job.started_at:
            details.append(f"Started: {job.started_at.strftime('%Y-%m-%d %H:%M:%S')}")

        if job.completed_at:
            details.append(f"Completed: {job.completed_at.strftime('%Y-%m-%d %H:%M:%S')}")

            # Calculate duration if we have start and end times
            if job.started_at:
                duration = job.completed_at - job.started_at
                details.append(f"Duration: {self._format_duration(duration)}")

        if job.status == TrainingJobStatus.RUNNING and job.started_at:
            # Calculate current duration
            from datetime import datetime
            current_duration = datetime.now() - job.started_at
            details.append(f"Running for: {self._format_duration(current_duration)}")

        if job.error_message:
            details.append(f"Error: {job.error_message}")

        details.append(f"Model: {job.base_model}")

        return "\n".join(details)

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

    def _on_job_selected(self, event):
        """Handle job selection in treeview."""
        selection = self.tree.selection()
        if selection:
            item = selection[0]
            # Get job ID stored with the item
            self.selected_job_id = self.tree_item_to_job_id.get(item)
        else:
            self.selected_job_id = None

    def _on_completed_job_selected(self, event):
        """Handle job selection in completed jobs treeview."""
        selection = self.completed_tree.selection()
        if selection:
            item = selection[0]
            # Get job ID stored with the item
            self.selected_job_id = self.completed_tree_item_to_job_id.get(item)
        else:
            self.selected_job_id = None

    def _show_context_menu(self, event):
        """Show context menu at cursor position."""
        # Select item under cursor
        item = self.tree.identify_row(event.y)
        if item:
            self.tree.selection_set(item)
            self.selected_job_id = self.tree_item_to_job_id.get(item)

            # Show context menu
            try:
                self.context_menu.tk_popup(event.x_root, event.y_root)
            finally:
                self.context_menu.grab_release()

    def _show_completed_context_menu(self, event):
        """Show context menu for completed jobs at cursor position."""
        # Select item under cursor
        item = self.completed_tree.identify_row(event.y)
        if item:
            self.completed_tree.selection_set(item)
            self.selected_job_id = self.completed_tree_item_to_job_id.get(item)

            # Show context menu
            try:
                self.completed_context_menu.tk_popup(event.x_root, event.y_root)
            finally:
                self.completed_context_menu.grab_release()

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
            self.pause_button.config(text="Pause Queue")

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

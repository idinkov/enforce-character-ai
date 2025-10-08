"""
Training Queue Manager for handling multiple training jobs.
"""
import threading
import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from pathlib import Path
import subprocess
from datetime import datetime

from src.utils.onetrainer_util import (
    start_character_training,
    create_character_training_config
)


class TrainingJobStatus(Enum):
    QUEUED = "Queued"
    RUNNING = "Running"
    COMPLETED = "Completed"
    FAILED = "Failed"
    CANCELLED = "Cancelled"
    STOPPING = "Stopping"


@dataclass
class TrainingJob:
    """Represents a single training job in the queue."""
    id: str
    character_name: str
    stage_7_path: Path
    training_config: Dict[str, Any]
    selected_model_id: str
    output_model_path: str
    base_model: str
    status: TrainingJobStatus = TrainingJobStatus.QUEUED
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    process: Optional[subprocess.Popen] = None
    progress: Dict[str, Any] = field(default_factory=dict)

    # Callbacks for status updates
    on_status_changed: Optional[Callable[['TrainingJob'], None]] = None
    on_progress_update: Optional[Callable[['TrainingJob'], None]] = None


class TrainingQueueManager:
    """Manages a queue of training jobs and executes them sequentially."""

    def __init__(self):
        self.jobs: Dict[str, TrainingJob] = {}
        self.job_order: List[str] = []  # Maintains order of jobs
        self.current_job_id: Optional[str] = None
        self.is_running = False
        self.worker_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

        # Callbacks for queue events
        self.on_queue_changed: Optional[Callable[[], None]] = None
        self.on_job_status_changed: Optional[Callable[[TrainingJob], None]] = None

        # Lock for thread safety
        self._lock = threading.Lock()

    def add_job(self, character_name: str, stage_7_path: Path, training_config: Dict[str, Any],
                selected_model_id: str, output_model_path: str, base_model: str,
                character_data: Optional[Dict] = None) -> str:
        """Add a new training job to the queue.

        Returns:
            Job ID of the added job
        """
        with self._lock:
            # Generate unique job ID
            job_id = f"job_{int(time.time() * 1000)}_{character_name.replace(' ', '_')}"

            # Create training job
            job = TrainingJob(
                id=job_id,
                character_name=character_name,
                stage_7_path=stage_7_path,
                training_config=training_config,
                selected_model_id=selected_model_id,
                output_model_path=output_model_path,
                base_model=base_model
            )

            # Set up job callbacks
            job.on_status_changed = self._on_job_status_changed
            job.on_progress_update = self._on_job_progress_update

            # Add to queue
            self.jobs[job_id] = job
            self.job_order.append(job_id)

            print(f"Added training job to queue: {character_name} (ID: {job_id})")

            # Notify listeners
            self._notify_queue_changed()

            # Start worker thread if not running
            if not self.is_running:
                self.start_worker()

            return job_id

    def remove_job(self, job_id: str) -> bool:
        """Remove a job from the queue.

        Args:
            job_id: ID of the job to remove

        Returns:
            True if job was removed, False if job not found or cannot be removed
        """
        with self._lock:
            if job_id not in self.jobs:
                return False

            job = self.jobs[job_id]

            # Cannot remove running job directly - must stop it first
            if job.status == TrainingJobStatus.RUNNING:
                return False

            # Remove from queue
            del self.jobs[job_id]
            if job_id in self.job_order:
                self.job_order.remove(job_id)

            print(f"Removed job from queue: {job.character_name} (ID: {job_id})")

            # Notify listeners
            self._notify_queue_changed()

            return True

    def stop_job(self, job_id: str) -> bool:
        """Stop a running job.

        Args:
            job_id: ID of the job to stop

        Returns:
            True if job was stopped, False if job not found or not running
        """
        with self._lock:
            if job_id not in self.jobs:
                return False

            job = self.jobs[job_id]

            if job.status != TrainingJobStatus.RUNNING:
                return False

            # Mark as stopping
            job.status = TrainingJobStatus.STOPPING
            self._notify_job_status_changed(job)

            # Stop the process
            if job.process:
                try:
                    job.process.terminate()
                    # Give it time to terminate gracefully
                    threading.Timer(5.0, lambda: self._force_kill_process(job)).start()
                except Exception as e:
                    print(f"Error stopping job process: {e}")

            return True

    def _force_kill_process(self, job: TrainingJob):
        """Force kill a process if it doesn't terminate gracefully."""
        try:
            if job.process and job.process.poll() is None:
                job.process.kill()
                print(f"Force killed training process for job: {job.character_name}")
        except Exception as e:
            print(f"Error force killing process: {e}")

    def get_jobs(self) -> List[TrainingJob]:
        """Get all jobs in order."""
        with self._lock:
            return [self.jobs[job_id] for job_id in self.job_order if job_id in self.jobs]

    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        """Get a specific job by ID."""
        with self._lock:
            return self.jobs.get(job_id)

    def get_current_job(self) -> Optional[TrainingJob]:
        """Get the currently running job."""
        with self._lock:
            if self.current_job_id:
                return self.jobs.get(self.current_job_id)
            return None

    def get_queue_size(self) -> int:
        """Get the number of jobs in the queue."""
        with self._lock:
            return len([j for j in self.jobs.values() if j.status == TrainingJobStatus.QUEUED])

    def get_total_jobs(self) -> int:
        """Get the total number of jobs."""
        with self._lock:
            return len(self.jobs)

    def start_worker(self):
        """Start the worker thread to process jobs."""
        if self.is_running:
            return

        self.is_running = True
        self.stop_event.clear()
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        print("Training queue worker started")

    def stop_worker(self):
        """Stop the worker thread."""
        self.is_running = False
        self.stop_event.set()

        # Stop current job if running
        if self.current_job_id:
            self.stop_job(self.current_job_id)

        print("Training queue worker stopped")

    def _worker_loop(self):
        """Main worker loop that processes jobs."""
        while self.is_running and not self.stop_event.is_set():
            try:
                # Get next job to process
                next_job = self._get_next_job()

                if next_job:
                    self._process_job(next_job)
                else:
                    # No jobs to process, wait a bit
                    time.sleep(1)

            except Exception as e:
                print(f"Error in training queue worker: {e}")
                time.sleep(5)  # Wait before retrying

    def _get_next_job(self) -> Optional[TrainingJob]:
        """Get the next job to process."""
        with self._lock:
            for job_id in self.job_order:
                if job_id in self.jobs:
                    job = self.jobs[job_id]
                    if job.status == TrainingJobStatus.QUEUED:
                        return job
            return None

    def _process_job(self, job: TrainingJob):
        """Process a single training job."""
        try:
            with self._lock:
                self.current_job_id = job.id
                job.status = TrainingJobStatus.RUNNING
                job.started_at = datetime.now()

            print(f"Starting training job: {job.character_name} (ID: {job.id})")
            self._notify_job_status_changed(job)

            # Start the training process
            success, error_message, process = start_character_training(
                job.character_name,
                job.stage_7_path,
                job.training_config,
                selected_model_id=job.selected_model_id
            )

            if success and process:
                job.process = process

                # Monitor the process
                self._monitor_job_process(job)

            else:
                # Training failed to start
                with self._lock:
                    job.status = TrainingJobStatus.FAILED
                    job.error_message = error_message or "Failed to start training"
                    job.completed_at = datetime.now()

                print(f"Training job failed to start: {job.character_name} - {job.error_message}")
                self._notify_job_status_changed(job)

        except Exception as e:
            with self._lock:
                job.status = TrainingJobStatus.FAILED
                job.error_message = str(e)
                job.completed_at = datetime.now()

            print(f"Error processing training job: {e}")
            self._notify_job_status_changed(job)

        finally:
            with self._lock:
                self.current_job_id = None

    def _monitor_job_process(self, job: TrainingJob):
        """Monitor a training job process until completion."""
        try:
            # Find and monitor the log file for progress updates
            log_monitoring_thread = threading.Thread(
                target=self._monitor_job_log,
                args=(job,),
                daemon=True
            )
            log_monitoring_thread.start()

            while job.process and job.process.poll() is None:
                # Check if job was marked for stopping
                if job.status == TrainingJobStatus.STOPPING:
                    job.status = TrainingJobStatus.CANCELLED
                    job.completed_at = datetime.now()
                    self._notify_job_status_changed(job)
                    return

                time.sleep(2)

            # Process finished
            exit_code = job.process.poll() if job.process else -1

            with self._lock:
                if job.status == TrainingJobStatus.STOPPING:
                    job.status = TrainingJobStatus.CANCELLED
                elif exit_code == 0:
                    job.status = TrainingJobStatus.COMPLETED
                    # Generate training prompt file when training completes successfully
                    self._generate_training_prompt_file(job)
                    # Clean up training directory after successful completion
                    self._cleanup_training_directory(job)
                else:
                    job.status = TrainingJobStatus.FAILED
                    job.error_message = f"Process exited with code {exit_code}"

                job.completed_at = datetime.now()

            print(f"Training job finished: {job.character_name} - {job.status.value}")
            self._notify_job_status_changed(job)

        except Exception as e:
            with self._lock:
                job.status = TrainingJobStatus.FAILED
                job.error_message = str(e)
                job.completed_at = datetime.now()

            print(f"Error monitoring job process: {e}")
            self._notify_job_status_changed(job)

    def _monitor_job_log(self, job: TrainingJob):
        """Monitor the training log file for progress updates."""
        import re

        # Find the log file
        log_file = self._find_training_log(job)
        if not log_file:
            print(f"Could not find log file for job {job.character_name}")
            return

        print(f"Monitoring log file for {job.character_name}: {log_file}")

        last_size = 0
        if log_file.exists():
            last_size = log_file.stat().st_size

        while job.status == TrainingJobStatus.RUNNING:
            try:
                if log_file.exists():
                    current_size = log_file.stat().st_size

                    if current_size > last_size:
                        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                            f.seek(last_size)
                            new_content = f.read()

                        if new_content.strip():
                            self._parse_training_log(job, new_content)

                        last_size = current_size

                time.sleep(2)
            except Exception as e:
                print(f"Error monitoring log for {job.character_name}: {e}")
                time.sleep(5)

    def _find_training_log(self, job: TrainingJob) -> Optional[Path]:
        """Find the training log file for a job."""
        # Try character models directory first
        character_dir = Path("characters") / job.character_name / "models"

        if character_dir.exists():
            # Look for training_* directories
            training_dirs = [d for d in character_dir.iterdir()
                           if d.is_dir() and d.name.startswith("training_")]

            if training_dirs:
                # Get most recent
                latest_dir = max(training_dirs, key=lambda x: x.stat().st_mtime)
                log_file = latest_dir / "training.log"
                if log_file.exists():
                    return log_file

        return None

    def _parse_training_log(self, job: TrainingJob, content: str):
        """Parse training log content to extract progress information."""
        import re

        lines = content.split('\n')
        updated = False

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Parse epoch information: epoch: 45% 9/20
            epoch_match = re.search(r'epoch:\s*(\d+)%.*?(\d+)/(\d+)', line)
            if epoch_match:
                job.progress['epoch_percent'] = int(epoch_match.group(1))
                job.progress['current_epoch'] = int(epoch_match.group(2))
                job.progress['total_epochs'] = int(epoch_match.group(3))
                updated = True

            # Parse step information: step: 91% 1820/2000 | loss=0.0891 | smooth loss=0.0945
            step_match = re.search(r'step:\s*(\d+)%.*?(\d+)/(\d+).*?loss=([\d.]+).*?smooth loss=([\d.]+)', line)
            if step_match:
                job.progress['step_percent'] = int(step_match.group(1))
                job.progress['current_step'] = int(step_match.group(2))
                job.progress['total_steps'] = int(step_match.group(3))
                job.progress['loss'] = float(step_match.group(4))
                job.progress['smooth_loss'] = float(step_match.group(5))
                updated = True

            # Parse caching progress: caching: 67%
            cache_match = re.search(r'caching:\s*(\d+)%', line)
            if cache_match:
                job.progress['stage'] = f"Caching: {cache_match.group(1)}%"
                updated = True
            elif 'Loading pipeline components' in line:
                job.progress['stage'] = "Loading model..."
                updated = True
            elif 'enumerating sample paths' in line:
                job.progress['stage'] = "Preparing dataset..."
                updated = True
            elif 'step:' in line and 'loss=' in line:
                epoch = job.progress.get('current_epoch', 0)
                job.progress['stage'] = f"Training - Epoch {epoch}"
                updated = True

        # Calculate ETA based on progress
        if updated and job.started_at:
            self._calculate_eta(job)
            # Notify progress update
            if job.on_progress_update:
                job.on_progress_update(job)

    def _calculate_eta(self, job: TrainingJob):
        """Calculate estimated time to completion based on current progress."""
        current_time = datetime.now()
        elapsed = (current_time - job.started_at).total_seconds()

        # Calculate based on step progress if available
        current_step = job.progress.get('current_step', 0)
        total_steps = job.progress.get('total_steps', 0)

        if current_step > 0 and total_steps > 0:
            # Time per step
            time_per_step = elapsed / current_step
            remaining_steps = total_steps - current_step
            eta_seconds = remaining_steps * time_per_step
            job.progress['eta_seconds'] = int(eta_seconds)

            # Also calculate epoch ETA
            current_epoch = job.progress.get('current_epoch', 0)
            total_epochs = job.progress.get('total_epochs', 0)
            if current_epoch > 0 and total_epochs > 0:
                time_per_epoch = elapsed / current_epoch
                remaining_epochs = total_epochs - current_epoch
                epoch_eta_seconds = remaining_epochs * time_per_epoch
                job.progress['epoch_eta_seconds'] = int(epoch_eta_seconds)

    def _cleanup_training_directory(self, job: TrainingJob):
        """Clean up the training directory after successful training completion."""
        try:
            import shutil

            # Find the training directory
            character_dir = Path("characters") / job.character_name / "models"

            if not character_dir.exists():
                print(f"Character models directory not found: {character_dir}")
                return

            # Look for training_* directories
            training_dirs = [d for d in character_dir.iterdir()
                           if d.is_dir() and d.name.startswith("training_")]

            if not training_dirs:
                print(f"No training directories found for {job.character_name}")
                return

            # Get the most recent training directory (should be the one we just completed)
            latest_training_dir = max(training_dirs, key=lambda x: x.stat().st_mtime)

            # Verify it's safe to delete (check if it has expected structure)
            expected_subdirs = ['workspace', 'cache']
            has_expected_structure = any((latest_training_dir / subdir).exists() for subdir in expected_subdirs)

            if has_expected_structure:
                print(f"Cleaning up training directory: {latest_training_dir}")
                shutil.rmtree(latest_training_dir)
                print(f"Successfully deleted training directory: {latest_training_dir}")
            else:
                print(f"Training directory doesn't have expected structure, skipping cleanup: {latest_training_dir}")

        except Exception as e:
            print(f"Error cleaning up training directory for {job.character_name}: {e}")

    def _generate_training_prompt_file(self, job: TrainingJob):
        """Generate the training prompt .txt file alongside the trained model."""
        try:
            # Get the training prompt from the character's training_prompt.txt file
            character_dir = Path("characters") / job.character_name
            training_prompt_file = character_dir / "training_prompt.txt"

            training_prompt = None
            if training_prompt_file.exists():
                try:
                    with open(training_prompt_file, 'r', encoding='utf-8') as f:
                        training_prompt = f.read().strip()
                except Exception as e:
                    print(f"Error reading training prompt file: {e}")

            # Fallback to character name if no training prompt found
            if not training_prompt:
                training_prompt = job.character_name

            # Create the .txt file alongside the model file
            model_file_path = Path(job.output_model_path)
            if model_file_path.exists():
                txt_path = model_file_path.with_suffix('.txt')
                if not txt_path.exists():
                    try:
                        with open(txt_path, 'w', encoding='utf-8') as f:
                            f.write(training_prompt)
                        print(f"Created training prompt file: {txt_path}")
                        print(f"Training prompt content: {training_prompt}")
                    except Exception as e:
                        print(f"Error writing training prompt file: {e}")
                else:
                    print(f"Training prompt file already exists: {txt_path}")
            else:
                print(f"Warning: Model file not found at {model_file_path}")

        except Exception as e:
            print(f"Error generating training prompt file for {job.character_name}: {e}")

    def _on_job_status_changed(self, job: TrainingJob):
        """Handle job status changes."""
        self._notify_job_status_changed(job)

    def _on_job_progress_update(self, job: TrainingJob):
        """Handle job progress updates."""
        # Notify the global callback if set
        if self.on_job_status_changed:
            try:
                self.on_job_status_changed(job)
            except Exception as e:
                print(f"Error in progress update callback: {e}")

    def _notify_queue_changed(self):
        """Notify listeners that the queue has changed."""
        if self.on_queue_changed:
            try:
                self.on_queue_changed()
            except Exception as e:
                print(f"Error in queue changed callback: {e}")

    def _notify_job_status_changed(self, job: TrainingJob):
        """Notify listeners that a job status has changed."""
        if self.on_job_status_changed:
            try:
                self.on_job_status_changed(job)
            except Exception as e:
                print(f"Error in job status changed callback: {e}")


# Global training queue manager instance
_training_queue_manager = None


def get_training_queue_manager() -> TrainingQueueManager:
    """Get the global training queue manager instance."""
    global _training_queue_manager
    if _training_queue_manager is None:
        _training_queue_manager = TrainingQueueManager()
    return _training_queue_manager

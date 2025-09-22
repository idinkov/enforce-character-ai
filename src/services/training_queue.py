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
        # This could be implemented to parse log files and update progress
        pass

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

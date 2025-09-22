"""
Splash screen component for application startup.
"""
import tkinter as tk
from tkinter import ttk
import threading
import subprocess
import sys
import os
from pathlib import Path
import time
from ..models import get_model_manager


class SplashScreen:
    """Splash screen with progress bar for application startup."""

    def __init__(self, app_title, version, character_repo=None, image_service=None):
        self.app_title = app_title
        self.version = version
        self.splash = None
        self.progress_var = None
        self.status_var = None
        self.cancel_flag = False

        # Character services for pre-loading
        self.character_repo = character_repo
        self.image_service = image_service
        self.character_info_cache = {}  # Cache for character info

        # Define startup tasks with their weights (for progress calculation)
        self.startup_tasks = [
            ("Checking dependencies...", 8, self._check_dependencies),
            ("Installing/updating packages...", 20, self._install_requirements),
            ("Initializing OneTrainer...", 25, self._initialize_onetrainer),
            ("Downloading models...", 20, self._download_models),
            ("Loading characters...", 15, self._initialize_characters),
            ("Initializing services...", 10, self._initialize_services),
            ("Loading UI...", 2, self._load_ui)
        ]

        self.total_weight = sum(task[1] for task in self.startup_tasks)
        self.current_progress = 0

    def show(self):
        """Display the splash screen."""
        self.splash = tk.Tk()
        self.splash.title("")
        self.splash.overrideredirect(True)  # Remove window decorations

        # Set the application icon (even though overrideredirect removes decorations)
        try:
            icon_path = Path(__file__).parent.parent.parent / "favicon.ico"
            if icon_path.exists():
                self.splash.iconbitmap(str(icon_path))
        except Exception as e:
            print(f"Warning: Could not set splash screen icon: {e}")

        # Set window size and center it
        width, height = 500, 300
        screen_width = self.splash.winfo_screenwidth()
        screen_height = self.splash.winfo_screenheight()
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        self.splash.geometry(f"{width}x{height}+{x}+{y}")

        # Create main frame
        main_frame = tk.Frame(self.splash, bg='#2c3e50', relief='raised', bd=2)
        main_frame.pack(fill="both", expand=True)

        # Title
        title_label = tk.Label(
            main_frame,
            text=self.app_title,
            font=('Arial', 24, 'bold'),
            fg='white',
            bg='#2c3e50'
        )
        title_label.pack(pady=(30, 10))

        # Version
        version_label = tk.Label(
            main_frame,
            text=f"Version {self.version}",
            font=('Arial', 12),
            fg='#bdc3c7',
            bg='#2c3e50'
        )
        version_label.pack(pady=(0, 20))

        # Status label
        self.status_var = tk.StringVar(value="Initializing...")
        status_label = tk.Label(
            main_frame,
            textvariable=self.status_var,
            font=('Arial', 10),
            fg='#ecf0f1',
            bg='#2c3e50'
        )
        status_label.pack(pady=(0, 10))

        # Progress bar
        self.progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(
            main_frame,
            variable=self.progress_var,
            maximum=100,
            length=400,
            mode='determinate'
        )
        progress_bar.pack(pady=10)

        # Progress percentage label
        self.progress_text_var = tk.StringVar(value="0%")
        progress_text_label = tk.Label(
            main_frame,
            textvariable=self.progress_text_var,
            font=('Arial', 9),
            fg='#95a5a6',
            bg='#2c3e50'
        )
        progress_text_label.pack(pady=(5, 20))

        # Cancel button
        cancel_button = tk.Button(
            main_frame,
            text="Cancel",
            command=self._cancel_startup,
            bg='#e74c3c',
            fg='white',
            font=('Arial', 9),
            relief='flat',
            padx=20
        )
        cancel_button.pack(pady=(0, 20))

        # Update display
        self.splash.update()

    def update_progress(self, percentage, status_text):
        """Update progress bar and status text."""
        if self.splash and not self.cancel_flag:
            self.progress_var.set(percentage)
            self.progress_text_var.set(f"{percentage:.1f}%")
            self.status_var.set(status_text)
            self.splash.update()

    def _cancel_startup(self):
        """Cancel the startup process."""
        self.cancel_flag = True
        self.close()

    def close(self):
        """Close the splash screen."""
        if self.splash:
            self.splash.destroy()
            self.splash = None

    def run_startup_sequence(self, on_complete_callback=None, on_error_callback=None):
        """Run the startup sequence in a separate thread."""
        def startup_thread():
            try:
                for i, (task_name, weight, task_func) in enumerate(self.startup_tasks):
                    if self.cancel_flag:
                        return

                    self.update_progress(self.current_progress, task_name)

                    # Execute the task
                    success = task_func()

                    if not success and not self.cancel_flag:
                        if on_error_callback:
                            on_error_callback(f"Failed at: {task_name}")
                        return

                    # Update progress
                    self.current_progress += weight / self.total_weight * 100
                    self.update_progress(self.current_progress, task_name)

                # Completion
                if not self.cancel_flag:
                    self.update_progress(100, "Startup complete!")
                    time.sleep(0.5)  # Brief pause to show completion

                    if on_complete_callback:
                        on_complete_callback()

            except Exception as e:
                if on_error_callback and not self.cancel_flag:
                    on_error_callback(f"Startup error: {str(e)}")

        # Start the startup sequence in a separate thread
        thread = threading.Thread(target=startup_thread, daemon=True)
        thread.start()

    def _check_dependencies(self):
        """Check if critical dependencies are available."""
        try:
            # Check if pip is available
            subprocess.run([sys.executable, "-m", "pip", "--version"],
                         check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError:
            return False

    def _install_requirements(self):
        """Install or update requirements from requirements.txt."""
        try:
            requirements_path = Path(__file__).parent.parent.parent / "requirements.txt"
            if requirements_path.exists():
                self.update_progress(self.current_progress, "Installing packages...")

                # Get the current environment's pip executable
                pip_executable = [sys.executable, "-m", "pip"]

                # Check if we're in a virtual environment
                if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
                    env_info = f"(virtual env: {sys.prefix})"
                else:
                    env_info = f"(system env: {sys.prefix})"

                self.update_progress(self.current_progress, f"Installing packages in {env_info}...")

                # Run pip install with verbose output
                cmd = pip_executable + ["install", "-r", str(requirements_path), "--verbose"]

                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    cwd=str(requirements_path.parent),
                    bufsize=1  # Line buffered
                )

                # Monitor the process and show real-time output
                install_log = []
                current_package = ""

                while True:
                    if self.cancel_flag:
                        process.terminate()
                        return False

                    line = process.stdout.readline()
                    if line == '' and process.poll() is not None:
                        break

                    if line:
                        line = line.strip()
                        install_log.append(line)

                        # Extract package name from pip output
                        if "Collecting" in line:
                            try:
                                current_package = line.split("Collecting ")[1].split(">=")[0].split("==")[0].split("[")[0]
                                self.update_progress(
                                    self.current_progress,
                                    f"Collecting {current_package}..."
                                )
                            except:
                                pass
                        elif "Installing collected packages:" in line:
                            self.update_progress(
                                self.current_progress,
                                "Installing collected packages..."
                            )
                        elif "Successfully installed" in line:
                            packages = line.replace("Successfully installed ", "").split()
                            if packages:
                                self.update_progress(
                                    self.current_progress,
                                    f"Successfully installed: {', '.join(packages[:3])}{'...' if len(packages) > 3 else ''}"
                                )
                        elif "Requirement already satisfied:" in line:
                            try:
                                package = line.split("Requirement already satisfied: ")[1].split(" in")[0]
                                self.update_progress(
                                    self.current_progress,
                                    f"Already satisfied: {package}"
                                )
                            except:
                                pass
                        elif "Building wheel for" in line:
                            try:
                                package = line.split("Building wheel for ")[1].split(" (")[0]
                                self.update_progress(
                                    self.current_progress,
                                    f"Building wheel for {package}..."
                                )
                            except:
                                pass
                        elif "Installing build dependencies" in line:
                            self.update_progress(
                                self.current_progress,
                                "Installing build dependencies..."
                            )
                        elif "Getting requirements to build wheel" in line:
                            self.update_progress(
                                self.current_progress,
                                "Getting build requirements..."
                            )
                        elif "ERROR:" in line or "FAILED:" in line:
                            self.update_progress(
                                self.current_progress,
                                f"Error: {line[:50]}..."
                            )
                            print(f"Pip install error: {line}")

                    # Small delay to prevent UI freezing
                    time.sleep(0.01)

                return_code = process.returncode

                if return_code == 0:
                    self.update_progress(
                        self.current_progress,
                        "Package installation completed successfully!"
                    )
                    print("Package installation completed successfully")
                else:
                    self.update_progress(
                        self.current_progress,
                        f"Package installation failed (exit code: {return_code})"
                    )
                    print(f"Package installation failed with exit code: {return_code}")
                    # Print last few lines of log for debugging
                    if install_log:
                        print("Last few lines of pip install log:")
                        for line in install_log[-10:]:
                            print(f"  {line}")

                return return_code == 0
            else:
                self.update_progress(self.current_progress, "No requirements.txt found, skipping...")
                print("No requirements.txt found, skipping package installation")
                return True

        except Exception as e:
            error_msg = f"Error installing requirements: {e}"
            print(error_msg)
            self.update_progress(self.current_progress, f"Installation error: {str(e)[:50]}...")
            return False

    def _download_models(self):
        """Download required models using the model manager."""
        try:
            # Get the model manager instance
            model_manager = get_model_manager()

            # Check which models are missing
            missing_models = model_manager.get_missing_models()

            if not missing_models:
                self.update_progress(
                    self.current_progress,
                    "All models are already available!"
                )
                return True

            # Progress callback for individual model downloads
            def progress_callback(downloaded_bytes, total_bytes, model_name):
                if self.cancel_flag:
                    return

                if total_bytes > 0:
                    download_percent = (downloaded_bytes / total_bytes) * 100
                    self.update_progress(
                        self.current_progress,
                        f"Downloading {model_name}: {download_percent:.1f}%"
                    )

            # Download all required models
            success = model_manager.download_all_required_models(progress_callback)

            if success:
                self.update_progress(
                    self.current_progress,
                    "All models downloaded successfully!"
                )
            else:
                self.update_progress(
                    self.current_progress,
                    "Some models failed to download, but continuing..."
                )

            return True  # Don't fail startup for model download issues

        except Exception as e:
            print(f"Error downloading models: {e}")
            self.update_progress(
                self.current_progress,
                "Model download error, but continuing..."
            )
            return True  # Don't fail startup for model download issues

    def _initialize_services(self):
        """Initialize application services."""
        try:
            self.update_progress(self.current_progress, "Initializing services...")
            # This is a placeholder - actual service initialization will happen in main app
            time.sleep(1)  # Simulate service initialization time
            return True
        except Exception:
            return False

    def _load_ui(self):
        """Load the user interface."""
        try:
            self.update_progress(self.current_progress, "Loading user interface...")
            # This is a placeholder - actual UI loading will happen in main app
            time.sleep(0.5)  # Simulate UI loading time
            return True
        except Exception:
            return False

    def _initialize_onetrainer(self):
        """Initialize OneTrainer by downloading and setting up in a separate environment."""
        try:
            print("=== Starting OneTrainer Initialization ===")
            # Define paths
            project_root = Path(__file__).parent.parent.parent
            repositories_dir = project_root / "repositories"
            onetrainer_dir = repositories_dir / "OneTrainer"
            venv_dir = onetrainer_dir / "venv"

            print(f"Project root: {project_root}")
            print(f"Repositories dir: {repositories_dir}")
            print(f"OneTrainer dir: {onetrainer_dir}")
            print(f"Venv dir: {venv_dir}")

            self.update_progress(self.current_progress, "Checking OneTrainer installation...")

            # Check if OneTrainer is already installed and functional
            print("Checking existing OneTrainer installation...")
            if self._check_onetrainer_installation(onetrainer_dir, venv_dir):
                print("OneTrainer already installed and functional!")
                self.update_progress(self.current_progress, "OneTrainer already installed and ready!")
                return True

            print("OneTrainer not found or not functional, proceeding with installation...")

            # Create repositories directory if it doesn't exist
            print(f"Creating repositories directory: {repositories_dir}")
            repositories_dir.mkdir(exist_ok=True)
            print(f"Repositories directory exists: {repositories_dir.exists()}")

            # Step 1: Clone OneTrainer repository if not exists
            if not onetrainer_dir.exists():
                print("OneTrainer directory doesn't exist, cloning repository...")
                self.update_progress(self.current_progress, "Cloning OneTrainer repository...")
                if not self._clone_onetrainer(repositories_dir):
                    print("ERROR: Failed to clone OneTrainer repository")
                    return False
                print("OneTrainer repository cloned successfully")
            else:
                print("OneTrainer directory already exists, skipping clone")

            # Step 2: Check Python version compatibility
            print("Checking Python version compatibility...")
            self.update_progress(self.current_progress, "Checking Python version...")
            python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            print(f"Current Python version: {python_version}")
            if not self._check_python_version():
                print(f"ERROR: Python version {python_version} is incompatible (need 3.10-3.12)")
                self.update_progress(self.current_progress, "Python version incompatible (need 3.10-3.12)")
                return False
            print("Python version is compatible")

            # Step 3: Create virtual environment if not exists
            if not venv_dir.exists():
                print("Virtual environment doesn't exist, creating...")
                self.update_progress(self.current_progress, "Creating Python virtual environment...")
                if not self._create_onetrainer_venv(onetrainer_dir, venv_dir):
                    print("ERROR: Failed to create virtual environment")
                    return False
                print("Virtual environment created successfully")
            else:
                print("Virtual environment already exists, skipping creation")

            # Step 4: Install OneTrainer requirements
            print("Installing OneTrainer requirements...")
            self.update_progress(self.current_progress, "Installing OneTrainer dependencies...")
            if not self._install_onetrainer_requirements(onetrainer_dir, venv_dir):
                print("ERROR: Failed to install OneTrainer requirements")
                return False
            print("OneTrainer requirements installed successfully")

            # Step 5: Verify installation
            print("Verifying OneTrainer installation...")
            self.update_progress(self.current_progress, "Verifying OneTrainer installation...")
            if self._check_onetrainer_installation(onetrainer_dir, venv_dir):
                print("OneTrainer installation verified successfully!")
                self.update_progress(self.current_progress, "OneTrainer initialized successfully!")
                return True
            else:
                print("ERROR: OneTrainer verification failed")
                self.update_progress(self.current_progress, "OneTrainer verification failed")
                return False

        except Exception as e:
            print(f"EXCEPTION in OneTrainer initialization: {e}")
            import traceback
            traceback.print_exc()
            self.update_progress(self.current_progress, f"OneTrainer initialization error: {str(e)[:50]}...")
            return False

    def _check_onetrainer_installation(self, onetrainer_dir, venv_dir):
        """Check if OneTrainer is properly installed and functional."""
        try:
            print(f"=== Checking OneTrainer Installation ===")
            print(f"OneTrainer dir: {onetrainer_dir}")
            print(f"Venv dir: {venv_dir}")

            # Check if directory and venv exist
            print(f"OneTrainer dir exists: {onetrainer_dir.exists()}")
            print(f"Venv dir exists: {venv_dir.exists()}")

            if not onetrainer_dir.exists() or not venv_dir.exists():
                print("ERROR: OneTrainer or venv directory doesn't exist")
                return False

            # Check if main OneTrainer files exist - using correct file structure
            main_files = ['requirements.txt', 'modules', 'start-ui.sh', 'pyproject.toml']
            for file in main_files:
                file_path = onetrainer_dir / file
                exists = file_path.exists()
                print(f"File {file} exists: {exists}")
                if not exists:
                    print(f"ERROR: Missing required file: {file}")
                    return False

            # Check if virtual environment has the required packages
            if os.name == 'nt':  # Windows
                python_exe = venv_dir / "Scripts" / "python.exe"
                pip_exe = venv_dir / "Scripts" / "pip.exe"
            else:  # Unix-like
                python_exe = venv_dir / "bin" / "python"
                pip_exe = venv_dir / "bin" / "pip"

            print(f"Python exe path: {python_exe}")
            print(f"Python exe exists: {python_exe.exists()}")

            if not python_exe.exists():
                print("ERROR: Python executable not found in venv")
                return False

            # Try to run a simple check to see if basic packages are installed
            try:
                print("Testing package imports...")
                result = subprocess.run(
                    [str(python_exe), "-c", "import torch; import PIL; print('OK')"],
                    capture_output=True,
                    text=True,
                    timeout=15
                )
                print(f"Import test return code: {result.returncode}")
                print(f"Import test stdout: {result.stdout}")
                print(f"Import test stderr: {result.stderr}")

                success = result.returncode == 0 and "OK" in result.stdout
                print(f"Import test success: {success}")
                return success
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
                print(f"ERROR: Import test failed with exception: {e}")
                return False

        except Exception as e:
            print(f"EXCEPTION in OneTrainer installation check: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _clone_onetrainer(self, repositories_dir):
        """Clone the OneTrainer repository."""
        try:
            # Check if git is available
            try:
                subprocess.run(["git", "--version"], check=True, capture_output=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                self.update_progress(self.current_progress, "Git not found - cannot clone OneTrainer")
                print("Git is not available. Please install Git to proceed with OneTrainer setup.")
                return False

            # Clone the repository
            self.update_progress(self.current_progress, "Downloading OneTrainer from GitHub...")

            cmd = [
                "git", "clone",
                "https://github.com/Nerogar/OneTrainer.git",
                str(repositories_dir / "OneTrainer")
            ]

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                cwd=str(repositories_dir)
            )

            while True:
                if self.cancel_flag:
                    process.terminate()
                    return False

                line = process.stdout.readline()
                if line == '' and process.poll() is not None:
                    break

                if line:
                    line = line.strip()
                    if "Receiving objects:" in line or "Resolving deltas:" in line:
                        # Try to extract progress information
                        if "%" in line:
                            try:
                                percent_part = line.split("%")[0]
                                percent = percent_part.split("(")[-1].strip()
                                self.update_progress(
                                    self.current_progress,
                                    f"Downloading OneTrainer: {percent}%"
                                )
                            except:
                                pass
                    elif "Cloning into" in line:
                        self.update_progress(self.current_progress, "Starting OneTrainer download...")

                time.sleep(0.01)

            return_code = process.returncode
            if return_code == 0:
                self.update_progress(self.current_progress, "OneTrainer repository downloaded successfully!")
                return True
            else:
                self.update_progress(self.current_progress, f"Failed to clone OneTrainer (exit code: {return_code})")
                return False

        except Exception as e:
            print(f"Error cloning OneTrainer: {e}")
            self.update_progress(self.current_progress, f"Clone error: {str(e)[:50]}...")
            return False

    def _check_python_version(self):
        """Check if Python version is compatible with OneTrainer (3.10-3.12)."""
        try:
            version = sys.version_info
            return version.major == 3 and 10 <= version.minor < 13
        except Exception:
            return False

    def _create_onetrainer_venv(self, onetrainer_dir, venv_dir):
        """Create a virtual environment for OneTrainer."""
        try:
            self.update_progress(self.current_progress, "Creating virtual environment for OneTrainer...")

            # Create virtual environment
            cmd = [sys.executable, "-m", "venv", str(venv_dir)]

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                cwd=str(onetrainer_dir)
            )

            while True:
                if self.cancel_flag:
                    process.terminate()
                    return False

                line = process.stdout.readline()
                if line == '' and process.poll() is not None:
                    break

                time.sleep(0.01)

            return_code = process.returncode
            if return_code == 0:
                self.update_progress(self.current_progress, "Virtual environment created successfully!")
                return True
            else:
                self.update_progress(self.current_progress, f"Failed to create venv (exit code: {return_code})")
                return False

        except Exception as e:
            print(f"Error creating OneTrainer venv: {e}")
            self.update_progress(self.current_progress, f"Venv creation error: {str(e)[:50]}...")
            return False

    def _install_onetrainer_requirements(self, onetrainer_dir, venv_dir):
        """Install OneTrainer requirements in the virtual environment."""
        try:
            # Get the pip executable from the virtual environment
            if os.name == 'nt':  # Windows
                pip_exe = venv_dir / "Scripts" / "pip.exe"
                python_exe = venv_dir / "Scripts" / "python.exe"
            else:  # Unix-like
                pip_exe = venv_dir / "bin" / "pip"
                python_exe = venv_dir / "bin" / "python"

            if not pip_exe.exists():
                self.update_progress(self.current_progress, "Pip not found in virtual environment")
                return False

            # Upgrade pip first
            self.update_progress(self.current_progress, "Upgrading pip in OneTrainer environment...")
            try:
                subprocess.run([str(python_exe), "-m", "pip", "install", "--upgrade", "pip"],
                             check=True, capture_output=True, timeout=60)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                print(f"Warning: Failed to upgrade pip: {e}")
                # Continue anyway

            # Install main requirements
            requirements_file = onetrainer_dir / "requirements.txt"
            if requirements_file.exists():
                self.update_progress(self.current_progress, "Installing OneTrainer requirements...")

                cmd = [str(pip_exe), "install", "-r", str(requirements_file)]

                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    cwd=str(onetrainer_dir)
                )

                while True:
                    if self.cancel_flag:
                        process.terminate()
                        return False

                    line = process.stdout.readline()
                    if line == '' and process.poll() is not None:
                        break

                    if line:
                        line = line.strip()

                        # Extract package name from pip output
                        if "Collecting" in line:
                            try:
                                package = line.split("Collecting ")[1].split(">=")[0].split("==")[0].split("[")[0]
                                self.update_progress(
                                    self.current_progress,
                                    f"Collecting {package} for OneTrainer..."
                                )
                            except:
                                pass
                        elif "Installing collected packages:" in line:
                            self.update_progress(
                                self.current_progress,
                                "Installing OneTrainer packages..."
                            )
                        elif "Successfully installed" in line:
                            self.update_progress(
                                self.current_progress,
                                "OneTrainer packages installed successfully!"
                            )
                        elif "Building wheel for" in line:
                            try:
                                package = line.split("Building wheel for ")[1].split(" (")[0]
                                self.update_progress(
                                    self.current_progress,
                                    f"Building {package} for OneTrainer..."
                                )
                            except:
                                pass

                    time.sleep(0.01)

                return_code = process.returncode
                if return_code != 0:
                    self.update_progress(self.current_progress, "OneTrainer requirements installation failed")
                    return False

            # Try to install CUDA requirements if available (optional)
            cuda_requirements = onetrainer_dir / "requirements-cuda.txt"
            if cuda_requirements.exists():
                self.update_progress(self.current_progress, "Installing CUDA support for OneTrainer...")
                try:
                    # Use a shorter timeout and handle failure gracefully
                    result = subprocess.run([str(pip_exe), "install", "-r", str(cuda_requirements)],
                                          capture_output=True, text=True, timeout=180)
                    if result.returncode == 0:
                        self.update_progress(self.current_progress, "CUDA support installed successfully!")
                    else:
                        print(f"CUDA installation failed: {result.stderr}")
                        self.update_progress(self.current_progress, "CUDA support installation failed - continuing with CPU-only...")
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                    print(f"CUDA installation error: {e}")
                    self.update_progress(self.current_progress, "CUDA support installation failed - continuing with CPU-only...")

            # Always return True - don't fail the entire setup if CUDA fails
            return True

        except Exception as e:
            print(f"Error installing OneTrainer requirements: {e}")
            self.update_progress(self.current_progress, f"Requirements install error: {str(e)[:50]}...")
            return False

    def _initialize_characters(self):
        """Pre-load character information during startup."""
        try:
            if not self.character_repo or not self.image_service:
                self.update_progress(self.current_progress, "Character services not available, skipping...")
                return True

            self.update_progress(self.current_progress, "Loading character list...")

            # Get all character names
            chars = self.character_repo.get_all_character_names()
            total_chars = len(chars)

            if total_chars == 0:
                self.update_progress(self.current_progress, "No characters found")
                return True

            self.update_progress(self.current_progress, f"Pre-loading {total_chars} characters...")

            # Pre-load character information in batches to show progress
            for i, char_name in enumerate(chars):
                if self.cancel_flag:
                    return False

                try:
                    # Load character info and cache it
                    photo_count = self.image_service.count_stage_images(char_name, "1_raw")
                    current_stage = self.image_service.get_current_stage(char_name)
                    is_completed = self.image_service.is_character_completed(char_name)

                    # Cache the results for instant access by character tab
                    self.character_info_cache[char_name] = {
                        'photo_count': photo_count,
                        'current_stage': current_stage,
                        'is_completed': is_completed
                    }

                    # Update progress
                    progress_percent = (i + 1) / total_chars * 100
                    self.update_progress(
                        self.current_progress,
                        f"Loaded {char_name} ({i + 1}/{total_chars})"
                    )

                    # Small delay to prevent overwhelming the system
                    time.sleep(0.01)

                except Exception as e:
                    print(f"Error loading character {char_name}: {e}")
                    # Continue with other characters
                    continue

            self.update_progress(self.current_progress, f"Pre-loaded {len(self.character_info_cache)} characters!")
            print(f"Character initialization complete: {len(self.character_info_cache)} characters cached")
            return True

        except Exception as e:
            print(f"Error initializing characters: {e}")
            self.update_progress(self.current_progress, f"Character loading error: {str(e)[:50]}...")
            return True  # Don't fail startup for character loading issues

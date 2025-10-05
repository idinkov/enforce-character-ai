"""
Package installation utilities for managing PyTorch and other dependencies.
"""
import subprocess
import sys
import time
from pathlib import Path


class PackageInstaller:
    """Handles package installation with progress tracking."""

    def __init__(self, progress_callback=None, cancel_check_callback=None):
        """
        Initialize the package installer.

        Args:
            progress_callback: Function(percentage, message) to report progress
            cancel_check_callback: Function() -> bool to check if operation should be cancelled
        """
        self.progress_callback = progress_callback
        self.cancel_check_callback = cancel_check_callback
        self.pip_executable = [sys.executable, "-m", "pip"]

    def _update_progress(self, percentage, message):
        """Update progress if callback is provided."""
        if self.progress_callback:
            self.progress_callback(percentage, message)

    def _should_cancel(self):
        """Check if operation should be cancelled."""
        if self.cancel_check_callback:
            return self.cancel_check_callback()
        return False

    def check_pytorch_installation(self):
        """
        Check PyTorch installation status.

        Returns:
            tuple: (is_installed, version, has_cuda, needs_reinstall)
        """
        try:
            # Try to import torch in a subprocess to avoid caching issues
            result = subprocess.run(
                [sys.executable, "-c",
                 "import torch; print(torch.__version__); print(torch.cuda.is_available())"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) >= 2:
                    version = lines[0].strip()
                    cuda_available = lines[1].strip().lower() == 'true'

                    # Check if it's version 2.8.0
                    is_correct_version = version.startswith("2.8.0")

                    # Need reinstall if correct version but no CUDA
                    needs_reinstall = is_correct_version and not cuda_available

                    return True, version, cuda_available, needs_reinstall

            return False, None, False, False

        except Exception as e:
            print(f"Error checking PyTorch installation: {e}")
            return False, None, False, False

    def install_pytorch(self, version="2.8.0", cuda_version="cu128"):
        """
        Install PyTorch with CUDA support.

        Args:
            version: PyTorch version to install
            cuda_version: CUDA version (e.g., 'cu128' for CUDA 12.8)

        Returns:
            tuple: (success: bool, needs_restart: bool)
                  - success: True if installation succeeded or already installed
                  - needs_restart: True if PyTorch was installed/upgraded and app needs restart
        """
        try:
            self._update_progress(0, "Checking PyTorch installation...")

            # Check current installation
            is_installed, current_version, has_cuda, needs_reinstall = self.check_pytorch_installation()

            if is_installed and current_version:
                if current_version.startswith(version):
                    if has_cuda:
                        self._update_progress(100, f"PyTorch {current_version} with CUDA already installed!")
                        print(f"PyTorch {current_version} with CUDA support already installed, skipping...")
                        return (True, False)  # Already installed, no restart needed
                    else:
                        # CPU version found, need to reinstall GPU version
                        self._update_progress(0, f"PyTorch {current_version} (CPU-only) found, installing GPU version...")
                        print(f"PyTorch {current_version} is CPU-only, reinstalling with CUDA support...")
                        if not self._uninstall_pytorch():
                            print("Warning: Failed to uninstall CPU version, attempting to install anyway...")
                else:
                    self._update_progress(0, f"PyTorch {current_version} found, upgrading to {version} with CUDA...")
                    print(f"PyTorch {current_version} found, upgrading to {version} with CUDA support...")
            else:
                self._update_progress(0, "PyTorch not found, installing...")
                print("PyTorch not installed, proceeding with installation...")

            # Install PyTorch with CUDA support
            install_success = self._install_pytorch_with_cuda(version, cuda_version, needs_reinstall)

            # If installation succeeded, we need to restart to reinitialize PyTorch
            return (install_success, install_success)

        except Exception as e:
            error_msg = f"Error installing PyTorch: {e}"
            print(error_msg)
            self._update_progress(0, f"PyTorch installation error: {str(e)[:50]}...")
            return (False, False)

    def _uninstall_pytorch(self):
        """Uninstall existing PyTorch packages."""
        try:
            self._update_progress(0, "Uninstalling CPU-only PyTorch version...")
            print("Uninstalling CPU-only PyTorch packages...")

            uninstall_cmd = self.pip_executable + ["uninstall", "-y", "torch", "torchvision", "torchaudio"]
            result = subprocess.run(uninstall_cmd, capture_output=True, text=True, timeout=60)

            if result.returncode == 0:
                print("CPU-only PyTorch uninstalled successfully")
                self._update_progress(0, "CPU version uninstalled, installing GPU version...")
                return True
            else:
                print(f"Warning: Uninstall returned code {result.returncode}")
                return False

        except Exception as e:
            print(f"Warning during uninstall: {e}")
            return False

    def _install_pytorch_with_cuda(self, version, cuda_version, force_reinstall=False):
        """Install PyTorch with CUDA support."""
        try:
            self._update_progress(0, f"Installing PyTorch {version} with CUDA {cuda_version}...")

            # Build install command
            cmd = self.pip_executable + [
                "install",
                f"torch=={version}",
                "torchvision",
                "torchaudio",
                "--index-url",
                f"https://download.pytorch.org/whl/test/{cuda_version}"
            ]

            if force_reinstall:
                cmd.append("--force-reinstall")

            print(f"Running PyTorch installation command: {' '.join(cmd)}")

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )

            # Monitor the process and show real-time output
            install_log = []

            while True:
                if self._should_cancel():
                    process.terminate()
                    return False

                line = process.stdout.readline()
                if line == '' and process.poll() is not None:
                    break

                if line:
                    line = line.strip()
                    install_log.append(line)
                    self._process_pip_output_line(line, "PyTorch")

                time.sleep(0.01)

            return_code = process.returncode

            if return_code == 0:
                self._update_progress(100, "PyTorch installation completed successfully!")
                print("PyTorch installation completed successfully")
                return True
            else:
                self._update_progress(0, f"PyTorch installation failed (exit code: {return_code})")
                print(f"PyTorch installation failed with exit code: {return_code}")
                if install_log:
                    print("Last few lines of PyTorch install log:")
                    for line in install_log[-10:]:
                        print(f"  {line}")
                return False

        except Exception as e:
            print(f"Error during PyTorch installation: {e}")
            return False

    def install_requirements(self, requirements_file):
        """
        Install packages from requirements.txt file.

        Args:
            requirements_file: Path to requirements.txt

        Returns:
            bool: True if installation succeeded
        """
        try:
            requirements_path = Path(requirements_file)

            if not requirements_path.exists():
                self._update_progress(0, "No requirements.txt found, skipping...")
                print("No requirements.txt found, skipping package installation")
                return True

            self._update_progress(0, "Installing packages from requirements.txt...")

            cmd = self.pip_executable + ["install", "-r", str(requirements_path), "--verbose"]

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                cwd=str(requirements_path.parent),
                bufsize=1
            )

            install_log = []

            while True:
                if self._should_cancel():
                    process.terminate()
                    return False

                line = process.stdout.readline()
                if line == '' and process.poll() is not None:
                    break

                if line:
                    line = line.strip()
                    install_log.append(line)
                    self._process_pip_output_line(line, "requirements")

                time.sleep(0.01)

            return_code = process.returncode

            if return_code == 0:
                self._update_progress(100, "Package installation completed successfully!")
                print("Package installation completed successfully")
                return True
            else:
                self._update_progress(0, f"Package installation failed (exit code: {return_code})")
                print(f"Package installation failed with exit code: {return_code}")
                if install_log:
                    print("Last few lines of pip install log:")
                    for line in install_log[-10:]:
                        print(f"  {line}")
                return False

        except Exception as e:
            error_msg = f"Error installing requirements: {e}"
            print(error_msg)
            self._update_progress(0, f"Installation error: {str(e)[:50]}...")
            return False

    def _process_pip_output_line(self, line, context="package"):
        """Process a line of pip output and update progress accordingly."""
        try:
            if "Collecting" in line:
                package = line.split("Collecting ")[1].split(">=")[0].split("==")[0].split("[")[0]
                self._update_progress(0, f"Collecting {package}...")
            elif "Downloading" in line and ("MB" in line or "GB" in line):
                self._update_progress(0, f"Downloading {context} package...")
            elif "Installing collected packages:" in line:
                self._update_progress(0, f"Installing {context} packages...")
            elif "Successfully installed" in line:
                packages = line.replace("Successfully installed ", "").split()
                if packages:
                    self._update_progress(0, f"Successfully installed {context} packages!")
            elif "Requirement already satisfied:" in line:
                package = line.split("Requirement already satisfied: ")[1].split(" in")[0]
                self._update_progress(0, f"Already satisfied: {package}")
            elif "Building wheel for" in line:
                package = line.split("Building wheel for ")[1].split(" (")[0]
                self._update_progress(0, f"Building wheel for {package}...")
            elif "Installing build dependencies" in line:
                self._update_progress(0, "Installing build dependencies...")
            elif "Getting requirements to build wheel" in line:
                self._update_progress(0, "Getting build requirements...")
            elif "ERROR:" in line or "FAILED:" in line:
                self._update_progress(0, f"Error: {line[:50]}...")
                print(f"Pip install error: {line}")
        except Exception:
            # Ignore parsing errors
            pass


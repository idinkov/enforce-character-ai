"""
OneTrainer utility module for managing OneTrainer installation and launching.
"""
import subprocess
import os
import json
import yaml
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union
import time

# Import ModelManager for getting base model paths
from ..models.model_manager import ModelManager


class OneTrainerManager:
    """Manages OneTrainer installation, verification, and launching."""

    def __init__(self, project_root: Optional[Path] = None):
        """Initialize OneTrainer manager.

        Args:
            project_root: Root directory of the project. If None, auto-detects.
        """
        if project_root is None:
            # Auto-detect project root (assumes this file is in src/utils/)
            self.project_root = Path(__file__).parent.parent.parent
        else:
            self.project_root = project_root

        self.repositories_dir = self.project_root / "repositories"
        self.onetrainer_dir = self.repositories_dir / "OneTrainer"
        self.venv_dir = self.onetrainer_dir / "venv"

        # Platform-specific paths
        if os.name == 'nt':  # Windows
            self.python_exe = self.venv_dir / "Scripts" / "python.exe"
            self.pip_exe = self.venv_dir / "Scripts" / "pip.exe"
            self.start_script = self.onetrainer_dir / "start-ui.bat"
        else:  # Unix-like
            self.python_exe = self.venv_dir / "bin" / "python"
            self.pip_exe = self.venv_dir / "bin" / "pip"
            self.start_script = self.onetrainer_dir / "start-ui.sh"

    def get_installation_status(self) -> Dict[str, Union[bool, str, None]]:
        """Get detailed OneTrainer installation status.

        Returns:
            Dictionary with installation status information.
        """
        status: Dict[str, Union[bool, str, None]] = {
            'is_installed': False,
            'is_functional': False,
            'onetrainer_dir_exists': self.onetrainer_dir.exists(),
            'venv_exists': self.venv_dir.exists(),
            'python_exe_exists': self.python_exe.exists(),
            'required_files_exist': False,
            'packages_functional': False,
            'error_message': None
        }

        try:
            # Check if OneTrainer directory exists
            if not status['onetrainer_dir_exists']:
                status['error_message'] = f"OneTrainer directory not found: {self.onetrainer_dir}"
                return status

            # Check if virtual environment exists
            if not status['venv_exists']:
                status['error_message'] = f"Virtual environment not found: {self.venv_dir}"
                return status

            # Check if Python executable exists
            if not status['python_exe_exists']:
                status['error_message'] = f"Python executable not found: {self.python_exe}"
                return status

            # Check required OneTrainer files
            required_files = ['requirements.txt', 'modules', 'start-ui.sh', 'pyproject.toml']
            missing_files = []
            for file in required_files:
                if not (self.onetrainer_dir / file).exists():
                    missing_files.append(file)

            if missing_files:
                status['error_message'] = f"Missing required files: {', '.join(missing_files)}"
                return status

            status['required_files_exist'] = True
            status['is_installed'] = True

            # Test package functionality
            try:
                result = subprocess.run(
                    [str(self.python_exe), "-c", "import torch; import PIL; print('OK')"],
                    capture_output=True,
                    text=True,
                    timeout=15
                )

                if result.returncode == 0 and "OK" in result.stdout:
                    status['packages_functional'] = True
                    status['is_functional'] = True
                else:
                    status['error_message'] = f"Package import test failed: {result.stderr}"

            except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
                status['error_message'] = f"Package test failed: {str(e)}"

        except Exception as e:
            status['error_message'] = f"Error checking installation: {str(e)}"

        return status

    def launch(self, show_console: bool = True) -> Tuple[bool, Optional[str], Optional[subprocess.Popen]]:
        """Launch OneTrainer application.

        Args:
            show_console: Whether to show console window (Windows only).

        Returns:
            Tuple of (success, error_message, process)
        """
        try:
            # Verify installation first
            status = self.get_installation_status()
            if not status['is_functional']:
                return False, status.get('error_message', 'OneTrainer is not functional'), None

            # Determine launch method
            launch_cmd = None
            launch_cwd = str(self.onetrainer_dir)
            creation_flags = 0

            if os.name == 'nt':  # Windows
                if show_console:
                    creation_flags = subprocess.CREATE_NEW_CONSOLE

                # Prefer batch script if it exists
                if self.start_script.exists():
                    launch_cmd = [str(self.start_script)]
                else:
                    # Fall back to direct Python launch
                    launch_cmd = [str(self.python_exe), "-m", "modules.ui.main"]
            else:  # Unix-like
                # Prefer shell script if it exists
                if self.start_script.exists():
                    launch_cmd = ["bash", str(self.start_script)]
                else:
                    # Fall back to direct Python launch
                    launch_cmd = [str(self.python_exe), "-m", "modules.ui.main"]

            # Launch the process
            if os.name == 'nt' and creation_flags:
                process = subprocess.Popen(
                    launch_cmd,
                    cwd=launch_cwd,
                    creationflags=creation_flags
                )
            else:
                process = subprocess.Popen(
                    launch_cmd,
                    cwd=launch_cwd
                )

            # Give it a moment to start
            time.sleep(1)

            # Check if process is still running (basic validation)
            if process.poll() is None:
                return True, None, process
            else:
                return False, f"OneTrainer process exited immediately with code {process.returncode}", None

        except Exception as e:
            return False, f"Failed to launch OneTrainer: {str(e)}", None


    def get_available_training_models(self) -> Dict[str, Dict[str, Any]]:
        """Get available trainable models for dropdown selection.

        Returns:
            Dictionary with model_id as key and model info as value for UI dropdown.
        """
        try:
            # Initialize ModelManager to get trainable models
            model_manager = ModelManager()
            available_models = model_manager.get_available_trainable_models()

            # Format for UI dropdown
            model_options = {}
            for model_id, model_info in available_models.items():
                model_options[model_id] = {
                    'id': model_id,
                    'name': model_info.name,
                    'description': model_info.description,
                    'version': model_info.version,
                    'framework': model_info.framework,
                    'model_type': model_info.model_type,
                    'train_suffix': model_info.train_suffix,
                    'file_size_mb': model_info.file_size_mb,
                    'exists': model_info.exists,
                    'display_name': f"{model_info.name} ({model_info.version})"
                }

            return model_options

        except Exception as e:
            print(f"Error getting available training models: {str(e)}")
            return {}

    def get_all_training_models(self) -> Dict[str, Dict[str, Any]]:
        """Get all trainable models (both available and not downloaded) for dropdown selection.

        Returns:
            Dictionary with model_id as key and model info as value for UI dropdown.
        """
        try:
            # Initialize ModelManager to get all trainable models
            model_manager = ModelManager()

            # Get all trainable models (not just available ones)
            all_models = {
                model_id: model_info for model_id, model_info in model_manager.models.items()
                if model_info.is_trainable
            }

            # Format for UI dropdown
            model_options = {}
            for model_id, model_info in all_models.items():
                status = "Available" if model_info.exists else "Not Downloaded"
                model_options[model_id] = {
                    'id': model_id,
                    'name': model_info.name,
                    'description': model_info.description,
                    'version': model_info.version,
                    'framework': model_info.framework,
                    'model_type': model_info.model_type,
                    'train_suffix': model_info.train_suffix,
                    'file_size_mb': model_info.file_size_mb,
                    'exists': model_info.exists,
                    'status': status,
                    'display_name': f"{model_info.name} ({model_info.version}) - {status}"
                }

            return model_options

        except Exception as e:
            print(f"Error getting all training models: {str(e)}")
            return {}

    def get_default_training_model(self) -> Optional[str]:
        """Get the default training model ID.

        Returns:
            Default model ID or None if no models available.
        """
        try:
            # First try to get available models
            available_models = self.get_available_training_models()
            if available_models:
                # Prefer XL model if available
                for model_id in available_models:
                    if 'xl' in model_id.lower() or 'XL' in available_models[model_id].get('train_suffix', ''):
                        return model_id
                # Return first available model if no XL found
                return next(iter(available_models))

            # If no available models, get all models and return the first one
            all_models = self.get_all_training_models()
            if all_models:
                # Prefer XL model if exists
                for model_id in all_models:
                    if 'xl' in model_id.lower() or 'XL' in all_models[model_id].get('train_suffix', ''):
                        return model_id
                # Return first model if no XL found
                return next(iter(all_models))

            return None

        except Exception as e:
            print(f"Error getting default training model: {str(e)}")
            return 'sd_xl_base_1_0_0_9vae'  # Fallback to known model

    def start_training(self,
                      character_name: str,
                      dataset_path: Path,
                      training_config: Optional[Dict[str, Any]] = None,
                      config_template_path: Optional[Path] = None,
                      selected_model_id: Optional[str] = None) -> Tuple[bool, Optional[str], Optional[subprocess.Popen]]:
        """Start training a model for a character using OneTrainer.

        Args:
            character_name: Name of the character being trained.
            dataset_path: Path to the dataset (stage 7 images).
            training_config: Optional training configuration overrides.
            config_template_path: Path to OneTrainer config template. If None, uses default based on selected model.
            selected_model_id: ID of the model to use for training. If None, defaults to 'sd_xl_base_1_0_0_9vae'.

        Returns:
            Tuple of (success, error_message, process)
        """
        try:
            # Verify installation first
            status = self.get_installation_status()
            if not status['is_functional']:
                return False, status.get('error_message', 'OneTrainer is not functional'), None

            # Check if dataset path exists and has images
            if not dataset_path.exists():
                return False, f"Dataset path does not exist: {dataset_path}", None

            # Count images in dataset
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
            image_files = [f for f in dataset_path.iterdir()
                          if f.is_file() and f.suffix.lower() in image_extensions]

            if len(image_files) == 0:
                return False, f"No images found in dataset path: {dataset_path}", None

            print(f"Found {len(image_files)} images for training character: {character_name}")

            # Initialize ModelManager to get trainable models and validate selection
            model_manager = ModelManager()

            # Default to XL model if no selection provided
            if selected_model_id is None:
                selected_model_id = 'sd_xl_base_1_0_0_9vae'

            # Validate that the selected model is trainable and available
            selected_model = model_manager.get_model_for_training(selected_model_id)
            if not selected_model:
                available_models = model_manager.get_available_trainable_models()
                if not available_models:
                    return False, "No trainable models are available. Please download required models first.", None

                # If selected model is not available, suggest available alternatives
                available_names = [f"{mid} ({model.name})" for mid, model in available_models.items()]
                return False, f"Selected model '{selected_model_id}' is not available for training. Available models: {', '.join(available_names)}", None

            print(f"Using training model: {selected_model.name} ({selected_model_id})")

            # Use model-specific config template if none provided
            if config_template_path is None:
                config_template_path = self.project_root / "src/models" / "config_train_xl_1024.json"

            if not config_template_path.exists():
                return False, f"Config template not found: {config_template_path}", None

            # Create output directory - save to character's models folder
            character_models_dir = self.project_root / "characters" / character_name / "models"
            character_models_dir.mkdir(parents=True, exist_ok=True)

            # Create a timestamp-based subdirectory for this training session
            training_session_dir = character_models_dir / f"training_{int(time.time())}"
            training_session_dir.mkdir(parents=True, exist_ok=True)

            # Load and modify the config template
            import json
            with open(config_template_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            # Get base model path from the selected model
            base_model_path = selected_model.full_path
            if base_model_path.exists():
                config['base_model_name'] = str(base_model_path.resolve())
                print(f"Using base model: {base_model_path.resolve()}")
            else:
                return False, f"Selected model file not found: {base_model_path}", None

            # Load character data to get training prompt
            character_data = self.load_character_data(character_name)

            # Create training prompt file in character's folder
            character_dir = self.project_root / "characters" / character_name
            training_prompt_file = character_dir / "training_prompt.txt"

            # Get training prompt from character data or use character name as fallback
            training_prompt = character_data.get('training_prompt', '').strip()
            if not training_prompt:
                training_prompt = character_name

            # Create the training prompt file
            character_dir.mkdir(parents=True, exist_ok=True)
            with open(training_prompt_file, 'w', encoding='utf-8') as f:
                f.write(training_prompt)

            print(f"Created training prompt file: {training_prompt_file}")
            print(f"Training prompt: {training_prompt}")

            # Update config with character-specific settings
            if config.get('concepts') and len(config['concepts']) > 0:
                # Update the first concept with our character data
                concept = config['concepts'][0]
                concept['name'] = character_name
                concept['path'] = str(dataset_path)
                concept['enabled'] = True

                # Update prompt path to use the character's training_prompt.txt file
                if 'text' in concept and isinstance(concept['text'], dict):
                    concept['text']['prompt_path'] = str(training_prompt_file)
                elif 'text' in concept:
                    # If text is not a dict, make it one
                    concept['text'] = {'prompt_path': str(training_prompt_file)}
                else:
                    # If text doesn't exist, create it
                    concept['text'] = {'prompt_path': str(training_prompt_file)}

                # Calculate balancing value based on actual image count vs ideal count of 166
                try:
                    # Count actual images in the dataset
                    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
                    actual_image_count = sum(1 for f in dataset_path.iterdir()
                                           if f.is_file() and f.suffix.lower() in image_extensions)

                    # Ideal image count is 166 (where balancing = 1.0)
                    ideal_image_count = 166

                    # Set balancing based on training_config['autobalance']
                    autobalance = None
                    if training_config and 'autobalance' in training_config:
                        autobalance = training_config['autobalance']
                    if autobalance == 'auto' or autobalance is True:
                        # Calculate balancing value: if we have fewer images, increase balancing to repeat them
                        # if we have more images, decrease balancing to use fewer repetitions
                        if actual_image_count > 0:
                            balancing_value = ideal_image_count / actual_image_count
                            concept['balancing'] = round(balancing_value, 3)

                            print(f"Image count: {actual_image_count}, Ideal: {ideal_image_count}")
                            print(f"Calculated balancing value: {concept['balancing']}")
                        else:
                            print("Warning: No images found in dataset, using default balancing value of 1.0")
                            concept['balancing'] = 1.0
                    else:
                        concept['balancing'] = 1.0

                except Exception as e:
                    print(f"Warning: Could not calculate balancing value: {e}, using default 1.0")
                    concept['balancing'] = 1.0

                # Disable other concepts to focus on our character
                for i in range(1, len(config['concepts'])):
                    config['concepts'][i]['enabled'] = False

            # Update output destination - save model to character's models folder
            model_suffix = selected_model.train_suffix if selected_model.train_suffix else "XL"
            output_model_name = f"{character_name.replace(' ', '_')}_{model_suffix}.safetensors"
            config['output_model_destination'] = str(character_models_dir / output_model_name)

            # Update workspace and cache directories
            config['workspace_dir'] = str(training_session_dir / "workspace")
            config['cache_dir'] = str(training_session_dir / "cache")

            # Apply any custom training config overrides
            if training_config:
                # Map common training config keys to OneTrainer config structure
                if 'epochs' in training_config:
                    # OneTrainer uses different epoch configuration, this would need to be set in the training loop
                    pass
                if 'learning_rate' in training_config:
                    # This would be in the optimizer settings
                    pass
                # Additional mappings can be added as needed

            # Save the modified config
            character_config_path = training_session_dir / f"{character_name}_config.json"
            with open(character_config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)

            # Find the train.py script
            train_script = self.onetrainer_dir / "train.py"
            if not train_script.exists():
                # Try alternative locations
                train_script = self.onetrainer_dir / "scripts" / "train.py"
                if not train_script.exists():
                    return False, f"train.py not found in OneTrainer directory: {self.onetrainer_dir}", None

            # Build training command using config file
            launch_cmd = [
                str(self.python_exe),
                str(train_script),
                "--config-path", str(character_config_path)
            ]

            # Create a training log file
            log_file = training_session_dir / "training.log"

            print(f"Starting training for {character_name}")
            print(f"Command: {' '.join(launch_cmd)}")
            print(f"Config file: {character_config_path}")
            print(f"Output directory: {training_session_dir}")
            print(f"Log file: {log_file}")

            # Launch training process
            with open(log_file, 'w') as log:
                if os.name == 'nt':  # Windows
                    process = subprocess.Popen(
                        launch_cmd,
                        cwd=str(self.onetrainer_dir),
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NO_WINDOW
                    )
                else:  # Unix-like
                    process = subprocess.Popen(
                        launch_cmd,
                        cwd=str(self.onetrainer_dir),
                        stdout=log,
                        stderr=subprocess.STDOUT
                    )

            # Give it a moment to start
            time.sleep(2)

            # Check if process is still running
            if process.poll() is None:
                return True, None, process
            else:
                return False, f"Training process exited immediately with code {process.returncode}", None

        except Exception as e:
            return False, f"Failed to start training: {str(e)}", None

    def create_training_config(self,
                             character_name: str,
                             character_data: Optional[Dict[str, Any]] = None,
                             autobalance: bool = True) -> Dict[str, Any]:
        """Create a training configuration based on character data.

        Args:
            character_name: Name of the character.
            character_data: Optional character information to customize training.
            autobalance: Whether to use autobalance (True for 'auto', False for 1.0).

        Returns:
            Training configuration dictionary.
        """
        # Base configuration
        config = {
            'base_model': 'runwayml/stable-diffusion-v1-5',
            'epochs': 20,
            'batch_size': 1,
            'learning_rate': 1e-4,
            'method': 'lora',
            'resolution': 512,
            'use_ema': True,
            'scheduler': 'cosine',
            'save_every': 5,
            'sample_every': 5,
            'warmup_steps': 100,
            'gradient_accumulation_steps': 1,
            'mixed_precision': 'fp16',
            'enable_xformers': True,
            'cache_latents': True,
        }

        # Customize based on character data
        if character_data:
            # Use training prompt if available
            if 'training_prompt' in character_data and character_data['training_prompt']:
                config['prompt'] = character_data['training_prompt']
            else:
                # Generate a basic prompt from character info
                prompt_parts = [character_name]
                if 'description' in character_data and character_data['description']:
                    prompt_parts.append(character_data['description'])
                config['prompt'] = ', '.join(prompt_parts)

            # Adjust resolution based on character type or preferences
            # Could be extended with more sophisticated logic

        # Set autobalance based on the parameter
        if autobalance:
            config['autobalance'] = 'auto'
        else:
            config['autobalance'] = 1.0

        return config

    def load_character_data(self, character_name: str) -> Dict[str, Any]:
        """Load character data from character.yaml file.

        Args:
            character_name: Name of the character.

        Returns:
            Dictionary containing character data.
        """
        character_dir = self.project_root / "characters" / character_name
        character_file = character_dir / "character.yaml"

        character_data = {}

        if character_file.exists():
            try:
                with open(character_file, 'r', encoding='utf-8') as f:
                    character_data = yaml.safe_load(f) or {}
            except Exception as e:
                print(f"Warning: Could not load character data for {character_name}: {e}")

        return character_data

# Convenience functions for easy use
def get_onetrainer_manager(project_root: Optional[Path] = None) -> OneTrainerManager:
    """Get OneTrainer manager instance."""
    return OneTrainerManager(project_root)


# Training convenience functions
def start_character_training(character_name: str,
                            dataset_path: Path,
                            training_config: Optional[Dict[str, Any]] = None,
                            config_template_path: Optional[Path] = None,
                            selected_model_id: Optional[str] = None,
                            project_root: Optional[Path] = None) -> Tuple[bool, Optional[str], Optional[subprocess.Popen]]:
    """Start training for a character.

    Args:
        character_name: Name of the character to train.
        dataset_path: Path to stage 7 images.
        training_config: Optional training configuration.
        config_template_path: Path to OneTrainer config template. If None, uses default XL config.
        selected_model_id: ID of the model to use for training. If None, uses default.
        project_root: Optional project root path.

    Returns:
        Tuple of (success, error_message, process)
    """
    return get_onetrainer_manager(project_root).start_training(
        character_name, dataset_path, training_config, config_template_path, selected_model_id
    )


def create_character_training_config(character_name: str,
                                    character_data: Optional[Dict[str, Any]] = None,
                                    autobalance: bool = True,
                                    project_root: Optional[Path] = None) -> Dict[str, Any]:
    """Create training configuration for a character.

    Args:
        character_name: Name of the character.
        character_data: Optional character data.
        autobalance: Whether to use autobalance (True for 'auto', False for 1.0).
        project_root: Optional project root path.

    Returns:
        Training configuration dictionary.
    """
    return get_onetrainer_manager(project_root).create_training_config(character_name, character_data, autobalance=autobalance)


# Model selection convenience functions
def get_available_training_models(project_root: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """Get available trainable models for dropdown selection.

    Args:
        project_root: Optional project root path.

    Returns:
        Dictionary with model_id as key and model info as value for UI dropdown.
    """
    return get_onetrainer_manager(project_root).get_available_training_models()


def get_all_training_models(project_root: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """Get all trainable models (both available and not downloaded) for dropdown selection.

    Args:
        project_root: Optional project root path.

    Returns:
        Dictionary with model_id as key and model info as value for UI dropdown.
    """
    return get_onetrainer_manager(project_root).get_all_training_models()


def get_default_training_model(project_root: Optional[Path] = None) -> Optional[str]:
    """Get the default training model ID.

    Args:
        project_root: Optional project root path.

    Returns:
        Default model ID or None if no models available.
    """
    return get_onetrainer_manager(project_root).get_default_training_model()

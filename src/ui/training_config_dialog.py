"""
Training configuration dialog and related training functions.
"""
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Dict, Any
from pathlib import Path
import threading

from src.utils.onetrainer_util import (
    get_onetrainer_manager,
    start_character_training,
    create_character_training_config,
    get_all_training_models,
    get_default_training_model
)
from src.ui.training_status_window import show_training_status
from src.models.model_manager import get_model_manager
from src.services.training_queue import get_training_queue_manager


def train_model(parent_frame, current_character, image_service, character_repo):
    """Train the model using OneTrainer with stage 7 images."""
    if not current_character:
        messagebox.showwarning("Warning", "Please select a character first")
        return

    # Get the stage 7 image path
    try:
        stage_7_path = image_service.characters_path / current_character / "images" / "7_final_dataset"
        if not stage_7_path.exists():
            messagebox.showerror("Error", f"Stage 7 directory not found: {stage_7_path}")
            return

        # Check if there are images in stage 7
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = [f for f in stage_7_path.iterdir()
                      if f.is_file() and f.suffix.lower() in image_extensions]

        if len(image_files) == 0:
            messagebox.showerror("Error", "No images found in stage 7. Please complete image processing first.")
            return

        # Load character data for training configuration
        character_data = None
        try:
            character = character_repo.load_character(current_character)
            if character:
                character_data = {
                    'name': character.name,
                    'description': character.description,
                    'training_prompt': character.training_prompt,
                    'personality': character.personality
                }
        except Exception as e:
            print(f"Warning: Could not load character data: {e}")

        # Show training configuration dialog FIRST (before OneTrainer check)
        config_result = show_training_config_dialog(parent_frame, current_character, len(image_files))
        if not config_result['confirmed']:
            return

        # Check OneTrainer installation AFTER dialog is confirmed
        ot_manager = get_onetrainer_manager()
        status = ot_manager.get_installation_status()

        if not status['is_functional']:
            messagebox.showerror(
                "OneTrainer Not Ready",
                f"OneTrainer is not properly installed or functional.\n\n"
                f"Error: {status.get('error_message', 'Unknown error')}\n\n"
                "Please restart the application to install OneTrainer."
            )
            return

        # Get selected model ID from the dialog result
        selected_model_id = None
        if 'model_id_map' in config_result and 'selected_model_var' in config_result:
            selected_display = config_result['selected_model_var'].get()
            selected_model_id = config_result['model_id_map'].get(selected_display)
            print(f"Selected model: {selected_display} (ID: {selected_model_id})")

        # Create training configuration
        training_config = create_character_training_config(
            current_character,
            character_data,
            config_result.get('autobalance', True)
        )

        # Prepare output model path
        model_manager = get_model_manager()
        model_info = model_manager.get_model_info(selected_model_id) if model_manager else None
        train_suffix = None
        if model_info:
            train_suffix = getattr(model_info, 'train_suffix', None)
            if not train_suffix and isinstance(model_info, dict):
                train_suffix = model_info.get('train_suffix')
        if not train_suffix:
            train_suffix = selected_model_id if selected_model_id else 'lora'

        char_name_safe = current_character.replace(' ', '_')
        output_model_filename = f"{char_name_safe}_{train_suffix}.safetensors"
        output_model_dir = image_service.characters_path / current_character / "models"
        output_model_dir.mkdir(exist_ok=True)
        output_model_path = str(output_model_dir / output_model_filename)

        # Add job to training queue instead of starting immediately
        queue_manager = get_training_queue_manager()
        job_id = queue_manager.add_job(
            character_name=current_character,
            stage_7_path=stage_7_path,
            training_config=training_config,
            selected_model_id=selected_model_id,
            output_model_path=output_model_path,
            base_model=selected_model_id or 'Unknown',
            character_data=character_data
        )

        # Automatically show Training Queue Manager window instead of messagebox
        from src.ui.queue_manager_window import show_queue_manager
        show_queue_manager(parent_frame)

        # Update status bar if accessible
        try:
            queue_size = queue_manager.get_queue_size()
            if hasattr(parent_frame, 'master') and hasattr(parent_frame.master, 'app'):
                app = parent_frame.master.app
                if hasattr(app, 'set_status_bar_text'):
                    if queue_size > 0:
                        app.set_status_bar_text(f"Training job added to queue. {queue_size} job(s) waiting.")
                    else:
                        app.set_status_bar_text(f"Training started for {current_character}")
        except Exception as e:
            print(f"Could not update status bar: {e}")

    except Exception as e:
        print(f"Error in train model: {e}")
        messagebox.showerror(
            "Training Error",
            f"An error occurred while preparing training:\n{str(e)}"
        )


def show_training_status_window(parent_frame, current_character, training_outputs_dir: Path,
                               training_process=None, base_model=None, output_model_name=None):
    """Show the training status monitoring window."""
    try:
        # Create and show the training status window
        status_window = show_training_status(
            parent_frame,
            current_character,
            training_outputs_dir,
            training_process,
            base_model,
            output_model_name
        )

        print(f"Training status window opened for {current_character}")

    except Exception as e:
        print(f"Error showing training status window: {e}")
        messagebox.showerror(
            "Status Window Error",
            f"Could not open training status window:\n{str(e)}\n\n"
            f"Training is still running in the background."
        )


def show_training_config_dialog(parent_frame, current_character, image_count: int) -> Dict[str, Any]:
    """Show a dialog to configure training parameters.

    Args:
        parent_frame: The parent frame for the dialog
        current_character: Name of the current character
        image_count: Number of images available for training.

    Returns:
        Dictionary with 'confirmed' key and model selection data.
    """
    dialog = tk.Toplevel(parent_frame)
    dialog.title("Training Configuration")
    dialog.geometry("400x420")
    dialog.grab_set()
    dialog.resizable(False, False)

    # Center the dialog relative to the main window
    dialog.transient(parent_frame)
    dialog.update_idletasks()

    # Get the main window (root)
    main_window = parent_frame.winfo_toplevel()

    # Get main window position and size
    main_x = main_window.winfo_x()
    main_y = main_window.winfo_y()
    main_width = main_window.winfo_width()
    main_height = main_window.winfo_height()

    # Calculate center position relative to main window
    dialog_width = 400
    dialog_height = 420
    x = main_x + (main_width // 2) - (dialog_width // 2)
    y = main_y + (main_height // 2) - (dialog_height // 2)

    dialog.geometry(f"{dialog_width}x{dialog_height}+{x}+{y}")

    result: Dict[str, Any] = {'confirmed': False}

    # Title
    title_label = ttk.Label(dialog, text=f"Train Model: {current_character}",
                           font=("Arial", 14, "bold"))
    title_label.pack(pady=10)

    # Info frame
    info_frame = ttk.Frame(dialog)
    info_frame.pack(fill="x", padx=20, pady=10)

    ttk.Label(info_frame, text=f"Dataset: {image_count} images from stage 7").pack(anchor="w")
    ttk.Label(info_frame, text="Method: LoRA (Low-Rank Adaptation)").pack(anchor="w")

    # Model selection frame
    model_frame = ttk.Frame(info_frame)
    model_frame.pack(fill="x", pady=5)

    ttk.Label(model_frame, text="Base Model:").pack(side="left")

    # Get available training models for dropdown
    try:
        all_models = get_all_training_models()
        default_model_id = get_default_training_model()

        if all_models:
            # Create model selection variable
            selected_model_var = tk.StringVar()

            # Prepare dropdown options
            model_options = []
            model_id_map = {}

            for model_id, model_info in all_models.items():
                display_name = model_info['display_name']
                model_options.append(display_name)
                model_id_map[display_name] = model_id

            # Set default selection
            if default_model_id and default_model_id in all_models:
                default_display = all_models[default_model_id]['display_name']
                selected_model_var.set(default_display)
            elif model_options:
                selected_model_var.set(model_options[0])

            # Create dropdown
            model_dropdown = ttk.Combobox(model_frame, textvariable=selected_model_var,
                                        values=model_options, state="readonly", width=40)
            model_dropdown.pack(side="left", padx=(5, 0))

            # Store the mapping for later use
            result['model_id_map'] = model_id_map
            result['selected_model_var'] = selected_model_var

        else:
            # No models available - show warning
            ttk.Label(model_frame, text="No trainable models available!",
                     foreground="red").pack(side="left", padx=(5, 0))

    except Exception as e:
        print(f"Error loading training models: {e}")
        ttk.Label(model_frame, text="SD XL Base (fallback)",
                 foreground="gray").pack(side="left", padx=(5, 0))

    # Configuration frame
    config_frame = ttk.LabelFrame(dialog, text="Training Parameters", padding=10)
    config_frame.pack(fill="x", padx=20, pady=10)

    ttk.Label(config_frame, text="• Epochs: 30 (training iterations)").pack(anchor="w")
    ttk.Label(config_frame, text="• Resolution: 1024x1024 pixels").pack(anchor="w")

    # --- Custom additions for image count, balance, and autobalance ---
    # Calculate initial balancing value
    def calculate_balancing(image_count, autobalance_enabled):
        if autobalance_enabled:
            if image_count > 0:
                return round(166 / image_count, 3)
            else:
                return 1.0
        else:
            return 1.0

    autobalance_var = tk.BooleanVar(value=True)
    balancing_value = calculate_balancing(image_count, autobalance_var.get())

    # Images label (above Epochs)
    images_label = ttk.Label(config_frame, text=f"• Images: {image_count}")
    images_label.pack(anchor="w", before=config_frame.winfo_children()[0])

    # Balance and Autobalance (below Epochs)
    balance_frame = ttk.Frame(config_frame)
    balance_frame.pack(anchor="w", pady=(2, 0))
    balance_label = ttk.Label(balance_frame, text=f"• Balance: {balancing_value}")
    balance_label.pack(side="left")
    autobalance_cb = ttk.Checkbutton(
        balance_frame, text="Autobalance", variable=autobalance_var
    )
    autobalance_cb.pack(side="left", padx=(8, 0))

    # Tooltip for autobalance
    def create_tooltip(widget, text):
        tooltip = tk.Toplevel(widget)
        tooltip.withdraw()
        tooltip.overrideredirect(True)
        label = tk.Label(tooltip, text=text, background="#ffffe0", relief="solid", borderwidth=1, font=("Arial", 9))
        label.pack(ipadx=1)
        def enter(event):
            x = widget.winfo_rootx() + 20
            y = widget.winfo_rooty() + 20
            tooltip.geometry(f"+{x}+{y}")
            tooltip.deiconify()
        def leave(event):
            tooltip.withdraw()
        widget.bind('<Enter>', enter)
        widget.bind('<Leave>', leave)
    create_tooltip(autobalance_cb, "When enabled it will autobalance the training set (repeat images or skip images) to maintain 166 images per epoch.")

    # Update balance label when autobalance is toggled
    def on_autobalance_toggle(*args):
        new_val = calculate_balancing(image_count, autobalance_var.get())
        balance_label.config(text=f"• Balance: {new_val}")
    autobalance_var.trace_add('write', on_autobalance_toggle)

    # Warning frame
    warning_frame = ttk.Frame(dialog)
    warning_frame.pack(fill="x", padx=20, pady=10)

    warning_label = ttk.Label(warning_frame,
                             text="⚠️ Training may take 30-60 minutes depending on your hardware.\n"
                                  "Training will run in the background.",
                             foreground="orange", font=("Arial", 9))
    warning_label.pack()

    # Button frame
    button_frame = ttk.Frame(dialog)
    button_frame.pack(fill="x", padx=20, pady=20)

    def on_confirm():
        result['confirmed'] = True
        result['autobalance'] = autobalance_var.get()  # Return autobalance value
        dialog.destroy()

    def on_cancel():
        result['confirmed'] = False
        result['autobalance'] = autobalance_var.get()  # Return autobalance value
        dialog.destroy()

    ttk.Button(button_frame, text="Start Training", command=on_confirm,
              style='Accent.TButton').pack(side="right", padx=(5, 0))
    ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(side="right")

    # Wait for dialog to close
    dialog.wait_window()

    return result

"""
Dialog for creating images using text-to-image generation.
"""
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
from pathlib import Path
from PIL import Image, ImageTk
from ..services.gpu_service import get_gpu_service


def get_available_base_models():
    """Get available base models for SDXL text-to-image generation from ModelManager.

    Returns:
        List of dictionaries with model info: [{'name': str, 'model_id': str, 'description': str}]
    """
    from ..models import get_model_manager

    model_manager = get_model_manager()
    text2img_models = model_manager.list_models(model_type="text2img")

    base_models = []
    for model_id, model_info in text2img_models.items():
        # Create model entry with consistent naming
        base_models.append({
            'name': model_info.name,
            'model_id': model_id,  # Use the model_id from ModelManager instead of huggingface repo
            'description': model_info.description
        })

    # Fallback to hardcoded models if none found in ModelManager
    if not base_models:
        base_models = [
            {
                'name': 'SDXL Base 1.0',
                'model_id': 'stabilityai/stable-diffusion-xl-base-1.0',
                'description': 'Standard SDXL base model - balanced quality and speed'
            }
        ]

    return base_models


def get_available_models(character_path: Path):
    """Get available LoRA models for a character.

    Args:
        character_path: Path to the character directory

    Returns:
        List of dictionaries with model info: [{'name': str, 'path': str, 'trigger_word': str}]
    """
    models = []
    models_dir = character_path / "models"

    if not models_dir.exists():
        return models

    # Look for .safetensors files
    for model_file in models_dir.glob("*.safetensors"):
        model_name = model_file.stem
        trigger_word = ""

        # Try to find trigger word from training_prompt.txt
        training_prompt_file = character_path / "training_prompt.txt"
        if training_prompt_file.exists():
            try:
                trigger_word = training_prompt_file.read_text(encoding='utf-8').strip()
            except:
                trigger_word = model_name.replace("_", " ")
        else:
            # Fallback to model name as trigger word
            trigger_word = model_name.replace("_", " ")

        models.append({
            'name': model_name,
            'path': str(model_file),
            'trigger_word': trigger_word
        })

    return models


def open_txt2img_dialog(parent_frame, current_character: str, image_service, refresh_images_callback):
    """Open a dialog to generate an image from text using SDXL txt2img.

    Args:
        parent_frame: The parent frame for the dialog
        current_character: The name of the current character
        image_service: The image service instance
        refresh_images_callback: Callback to refresh images after generation
    """
    if not current_character:
        messagebox.showwarning("Warning", "Please select a character first")
        return

    # Get GPU service for device information
    gpu_service = get_gpu_service()
    selected_device = gpu_service.get_selected_device()

    # Get available models for the character
    character_path = image_service.characters_path / current_character
    available_models = get_available_models(character_path)

    # Base model selection UI
    available_base_models = get_available_base_models()

    dialog = tk.Toplevel(parent_frame)
    dialog.title("Create Image with SDXL txt2img")
    dialog.geometry("600x700")  # Increased size for image preview
    dialog.grab_set()
    dialog.resizable(True, True)  # Allow resizing for better image viewing
    dialog.transient(parent_frame)
    dialog.update_idletasks()
    x = (dialog.winfo_screenwidth() // 2) - (600 // 2)
    y = (dialog.winfo_screenheight() // 2) - (700 // 2)
    dialog.geometry(f"600x700+{x}+{y}")

    # Create main frame with scrollbar
    main_frame = ttk.Frame(dialog)
    main_frame.pack(fill="both", expand=True)

    # Create canvas and scrollbar for scrollable content
    canvas = tk.Canvas(main_frame)
    scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # GPU info frame
    gpu_frame = ttk.Frame(scrollable_frame)
    gpu_frame.pack(fill="x", padx=10, pady=5)
    ttk.Label(gpu_frame, text="Using Device:", font=("Arial", 10, "bold")).pack(side="left")
    device_name = selected_device['name']
    device_color = "green" if selected_device['type'] == 'gpu' else "blue"
    device_label = tk.Label(gpu_frame, text=device_name, font=("Arial", 10), fg=device_color)
    device_label.pack(side="left", padx=(5, 0))

    # Base model selection frame
    base_model_frame = ttk.LabelFrame(scrollable_frame, text="Base Model Selection", padding=10)
    base_model_frame.pack(fill="x", padx=10, pady=5)

    ttk.Label(base_model_frame, text="Select Base Model:").pack(anchor="w")
    base_model_var = tk.StringVar()
    base_model_combo = ttk.Combobox(base_model_frame, textvariable=base_model_var, state="readonly", width=50)
    base_model_combo['values'] = [model['name'] for model in available_base_models]
    base_model_combo.pack(fill="x", pady=(5, 0))

    # Set default to SDXL Base 1.0
    base_model_combo.current(0)

    # Info label for base model description
    base_model_info = tk.StringVar()
    base_model_info.set(available_base_models[0]['description'])
    base_model_info_label = ttk.Label(base_model_frame, textvariable=base_model_info, font=("Arial", 9), foreground="blue")
    base_model_info_label.pack(anchor="w", pady=(5, 0))

    # Model selection frame
    model_frame = ttk.LabelFrame(scrollable_frame, text="LoRA Model Selection", padding=10)
    model_frame.pack(fill="x", padx=10, pady=5)

    model_var = tk.StringVar()
    selected_model_info = tk.StringVar()

    if available_models:
        ttk.Label(model_frame, text="Select Model:").pack(anchor="w")
        model_combo = ttk.Combobox(model_frame, textvariable=model_var, state="readonly", width=50)
        model_combo['values'] = [model['name'] for model in available_models]
        model_combo.pack(fill="x", pady=(5, 0))

        # Set default to first model
        if available_models:
            model_combo.current(0)
            selected_model_info.set(f"Trigger: {available_models[0]['trigger_word']}")

        # Info label for trigger word
        info_label = ttk.Label(model_frame, textvariable=selected_model_info, font=("Arial", 9), foreground="blue")
        info_label.pack(anchor="w", pady=(5, 0))

        def on_model_change(event=None):
            selected_name = model_var.get()
            for model in available_models:
                if model['name'] == selected_name:
                    selected_model_info.set(f"Trigger: {model['trigger_word']}")
                    break

        model_combo.bind('<<ComboboxSelected>>', on_model_change)
    else:
        ttk.Label(model_frame, text="No LoRA models found for this character", foreground="orange").pack()

    # Prompt frame
    prompt_frame = ttk.Frame(scrollable_frame)
    prompt_frame.pack(fill="x", padx=10, pady=10)
    ttk.Label(prompt_frame, text="Enter prompt for image generation:", font=("Arial", 12)).pack(anchor="w")
    prompt_var = tk.StringVar()
    prompt_entry = ttk.Entry(prompt_frame, textvariable=prompt_var, width=50)
    prompt_entry.pack(fill="x", pady=5)
    prompt_entry.focus_set()

    # Auto-insert trigger word button
    if available_models:
        def insert_trigger_word():
            selected_name = model_var.get()
            for model in available_models:
                if model['name'] == selected_name:
                    current_prompt = prompt_var.get()
                    trigger = model['trigger_word']
                    if trigger and trigger not in current_prompt:
                        new_prompt = f"{trigger}, {current_prompt}" if current_prompt else trigger
                        prompt_var.set(new_prompt)
                    break

        trigger_btn = ttk.Button(prompt_frame, text="Insert Trigger Word", command=insert_trigger_word)
        trigger_btn.pack(pady=(0, 5))

    # Advanced options
    options_frame = ttk.LabelFrame(scrollable_frame, text="Advanced Options", padding=10)
    options_frame.pack(fill="x", padx=10, pady=10)

    # First row: Steps, Guidance, Seed
    ttk.Label(options_frame, text="Steps:").grid(row=0, column=0, sticky=tk.W)
    steps_var = tk.IntVar(value=30)
    ttk.Entry(options_frame, textvariable=steps_var, width=5).grid(row=0, column=1, sticky=tk.W)
    ttk.Label(options_frame, text="Guidance:").grid(row=0, column=2, sticky=tk.W, padx=(10,0))
    guidance_var = tk.DoubleVar(value=7.5)
    ttk.Entry(options_frame, textvariable=guidance_var, width=5).grid(row=0, column=3, sticky=tk.W)

    # Now that variables are defined, set up the base model change handler properly
    def on_base_model_change(event=None):
        selected_name = base_model_var.get()
        for model in available_base_models:
            if model['name'] == selected_name:
                base_model_info.set(model['description'])
                # Update default steps for SDXL Turbo
                if 'turbo' in model['model_id'].lower():
                    steps_var.set(4)
                    guidance_var.set(0.0)  # Turbo models work best with no guidance
                else:
                    steps_var.set(30)
                    guidance_var.set(7.5)
                break

    base_model_combo.bind('<<ComboboxSelected>>', on_base_model_change)

    # Seed section with random option
    ttk.Label(options_frame, text="Seed:").grid(row=0, column=4, sticky=tk.W, padx=(10,0))
    seed_var = tk.StringVar(value="")
    seed_entry = ttk.Entry(options_frame, textvariable=seed_var, width=8)
    seed_entry.grid(row=0, column=5, sticky=tk.W)

    # Random seed checkbox
    random_seed_var = tk.BooleanVar(value=True)  # Default to True
    random_seed_cb = ttk.Checkbutton(
        options_frame,
        text="Random",
        variable=random_seed_var,
        command=lambda: seed_entry.configure(state="disabled" if random_seed_var.get() else "normal")
    )
    random_seed_cb.grid(row=0, column=6, sticky=tk.W, padx=(5,0))

    # Initially disable seed entry since random is selected by default
    seed_entry.configure(state="disabled")

    # Second row: LoRA strength
    ttk.Label(options_frame, text="LoRA Strength:").grid(row=1, column=0, sticky=tk.W, pady=(10,0))
    lora_strength_var = tk.DoubleVar(value=1.0)
    ttk.Entry(options_frame, textvariable=lora_strength_var, width=5).grid(row=1, column=1, sticky=tk.W, pady=(10,0))

    # Progress frame
    progress_frame = ttk.LabelFrame(scrollable_frame, text="Generation Progress", padding=10)
    progress_frame.pack(fill="x", padx=10, pady=10)

    # Progress bar
    progress_var = tk.DoubleVar()
    progress_bar = ttk.Progressbar(progress_frame, variable=progress_var, maximum=100)
    progress_bar.pack(fill="x", pady=(0, 5))

    # Status label
    status_var = tk.StringVar(value="Ready to generate")
    status_label = ttk.Label(progress_frame, textvariable=status_var)
    status_label.pack(anchor="w")

    # Image preview frame
    preview_frame = ttk.LabelFrame(scrollable_frame, text="Generated Image", padding=10)
    preview_frame.pack(fill="both", expand=True, padx=10, pady=10)

    # Image label (will hold the generated image)
    image_label = ttk.Label(preview_frame, text="Generated image will appear here", anchor="center")
    image_label.pack(expand=True, fill="both")

    # Image info label
    image_info_var = tk.StringVar()
    image_info_label = ttk.Label(preview_frame, textvariable=image_info_var, font=("Arial", 9), foreground="gray")
    image_info_label.pack(pady=(5, 0))

    def update_progress(step, total_steps, message):
        """Update progress bar and status"""
        progress = (step / total_steps) * 100
        progress_var.set(progress)
        status_var.set(f"{message} ({step}/{total_steps})")
        dialog.update_idletasks()

    def display_generated_image(image_path):
        """Display the generated image in the preview frame"""
        try:
            # Load and resize image for display
            image = Image.open(image_path)
            original_size = image.size

            # Calculate display size (max 400x400 while maintaining aspect ratio)
            max_size = 400
            ratio = min(max_size / original_size[0], max_size / original_size[1])
            new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))

            display_image = image.resize(new_size, Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(display_image)

            # Update image label
            image_label.configure(image=photo, text="")
            image_label.image = photo  # Keep a reference

            # Update image info
            image_info_var.set(f"Size: {original_size[0]}x{original_size[1]} | Path: {image_path}")

        except Exception as e:
            image_label.configure(text=f"Error displaying image: {str(e)}")
            image_info_var.set("")

    def on_generate():
        prompt = prompt_var.get().strip()
        if not prompt:
            messagebox.showwarning("Warning", "Please enter a prompt.")
            return
        steps = steps_var.get()
        guidance = guidance_var.get()
        lora_strength = lora_strength_var.get()

        # Handle seed generation
        if random_seed_var.get():
            # Generate a random seed suitable for Stable Diffusion (32-bit integer)
            import random
            seed = random.randint(0, 2**32 - 1)
        else:
            try:
                seed = int(seed_var.get()) if seed_var.get().strip() else None
            except ValueError:
                messagebox.showwarning("Warning", "Seed must be an integer.")
                return

        # Get selected base model
        selected_base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"  # Default
        selected_base_name = base_model_var.get()
        for model in available_base_models:
            if model['name'] == selected_base_name:
                selected_base_model_id = model['model_id']
                break

        # Get selected model info
        selected_model = None
        if available_models and model_var.get():
            selected_name = model_var.get()
            for model in available_models:
                if model['name'] == selected_name:
                    selected_model = model
                    break

        # Save to stage 8 creations folder by default
        out_dir = image_service.characters_path / current_character / "images" / "8_creations"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"txt2img_{int(time.time())}.png"

        # Disable generate button during generation
        generate_btn.configure(state="disabled")

        # Reset progress
        progress_var.set(0)
        status_var.set("Initializing...")
        image_label.configure(image="", text="Generating image...")
        image_info_var.set("")

        # Run generation in background
        def run_generation():
            try:
                # Import here to avoid circular imports
                from src.utils.txt2img_sdxl import txt2img_with_lora_progress

                # Create progress callback
                def progress_callback(step, total_steps, message="Generating"):
                    dialog.after(0, lambda: update_progress(step, total_steps, message))

                # Use the new function with progress tracking and selected base model
                result_path = txt2img_with_lora_progress(
                    prompt=prompt,
                    output_path=str(out_path),
                    model_id=selected_base_model_id,  # Pass the selected base model
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    seed=seed,
                    lora_path=selected_model['path'] if selected_model else None,
                    lora_strength=lora_strength if selected_model else 0.0,
                    progress_callback=progress_callback
                )

                # Update UI on completion
                dialog.after(0, lambda: [
                    progress_var.set(100),
                    status_var.set("Generation completed!"),
                    display_generated_image(result_path),
                    generate_btn.configure(state="normal"),
                    refresh_images_callback()
                ])

            except Exception as e:
                # Handle errors - capture error message immediately
                error_message = str(e)
                dialog.after(0, lambda: [
                    progress_var.set(0),
                    status_var.set(f"Error: {error_message}"),
                    image_label.configure(text="Generation failed"),
                    generate_btn.configure(state="normal"),
                    messagebox.showerror("Generation Error", error_message)
                ])

        threading.Thread(target=run_generation, daemon=True).start()

    # Button frame
    button_frame = ttk.Frame(scrollable_frame)
    button_frame.pack(pady=15)

    generate_btn = ttk.Button(button_frame, text="Generate", command=on_generate, style='Accent.TButton')
    generate_btn.pack(side="right", padx=5)

    # Bind mousewheel to canvas for scrolling with safety checks
    def on_mousewheel(event):
        try:
            # Check if canvas still exists and is valid
            if canvas.winfo_exists():
                canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        except tk.TclError:
            # Canvas has been destroyed, ignore the event
            pass

    # Bind the mousewheel event to the dialog instead of globally
    dialog.bind("<MouseWheel>", on_mousewheel)

    # Also bind to canvas and scrollable_frame for better coverage
    canvas.bind("<MouseWheel>", on_mousewheel)
    scrollable_frame.bind("<MouseWheel>", on_mousewheel)

    # Custom close function to properly clean up bindings
    def close_dialog():
        try:
            # Unbind events to prevent accessing destroyed widgets
            dialog.unbind("<MouseWheel>")
            canvas.unbind("<MouseWheel>")
            scrollable_frame.unbind("<MouseWheel>")
        except:
            pass
        dialog.destroy()

    ttk.Button(button_frame, text="Close", command=close_dialog).pack(side="right", padx=5)

    # Set the window close protocol to use our custom close function
    dialog.protocol("WM_DELETE_WINDOW", close_dialog)

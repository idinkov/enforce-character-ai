"""
txt2img_sdxl.py

Single-file utility exposing `txt2img` which performs text-to-image generation with SDXL model
from Hugging Face (default: stabilityai/stable-diffusion-xl-base-1.0).

Function:
    txt2img(
        prompt: str,
        output_path: str = "out.png",
        model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
        hf_token: str | None = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: int | None = None,
        width: int = 1024,
        height: int = 1024,
        save_model_dir: str | None = None
    ) -> str
Returns path to saved output image. Uses the centralized model manager for model handling.

Class-based approach for batch processing:
    Txt2ImgPipeline: A class that loads the model once and can process multiple prompts

Example:
    pipeline = Txt2ImgPipeline()
    pipeline.load_model()
    for prompt in prompts:
        pipeline.process_prompt(prompt, output_path)
    pipeline.unload_model()  # Optional cleanup

Notes:
 - You must have `diffusers`, `transformers`, `accelerate`, `torch`, and `huggingface_hub` installed.
 - Set HUGGINGFACE_HUB_TOKEN in the env or pass hf_token to authenticate private models.
 - This file only contains code. Running the generation requires a suitable GPU + CUDA and the dependencies.
"""

from __future__ import annotations
import os
from typing import Optional
from PIL import Image
import torch
from pathlib import Path
from ..models import get_model_manager
from ..services.gpu_service import get_gpu_service

class Txt2ImgPipeline:
    def __init__(self,
                 model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
                 hf_token: Optional[str] = None,
                 save_model_dir: Optional[str] = None,
                 device: Optional[str] = None):
        self.model_id = model_id
        self.hf_token = hf_token
        self.save_model_dir = save_model_dir
        self.pipe = None
        self.device = device  # Allow external device specification
        self.torch_dtype = None

    def load_model(self):
        if self.pipe is not None:
            return  # Already loaded
        try:
            from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
        except Exception as e:
            raise RuntimeError("This function requires the 'diffusers' package. Install with 'pip install diffusers[torch] transformers accelerate'") from e

        # Use provided device or get from GPU service
        if self.device is None:
            gpu_service = get_gpu_service()
            self.device = gpu_service.get_torch_device()

        self.torch_dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        model_manager = get_model_manager()
        if self.model_id in model_manager.models:
            model_info = model_manager.get_model_info(self.model_id)
            if model_info.exists:
                model_path = model_manager.get_model_path(self.model_id)
                print(f"Loading SDXL model from local file: {model_path}")
                auth_kwargs = {}
                if self.hf_token:
                    auth_kwargs["token"] = self.hf_token
                hf_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
                self.pipe = StableDiffusionXLPipeline.from_pretrained(
                    hf_model_id,
                    torch_dtype=self.torch_dtype,
                    **auth_kwargs
                )
                print(f"Model loaded from Hugging Face: {hf_model_id}")
            else:
                print(f"Model {model_info.name} not found locally, attempting to download...")
                success = model_manager.download_model(self.model_id)
                if success:
                    print(f"Successfully downloaded {model_info.name}")
                else:
                    print(f"Failed to download {model_info.name}, falling back to Hugging Face")
                auth_kwargs = {}
                if self.hf_token:
                    auth_kwargs["token"] = self.hf_token
                hf_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
                self.pipe = StableDiffusionXLPipeline.from_pretrained(
                    hf_model_id,
                    torch_dtype=self.torch_dtype,
                    **auth_kwargs
                )
        else:
            auth_kwargs = {}
            if self.hf_token:
                auth_kwargs["token"] = self.hf_token
            print(f"Loading SDXL model from Hugging Face: {self.model_id}")
            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype,
                **auth_kwargs
            )
        try:
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        except Exception:
            pass
        self.pipe = self.pipe.to(self.device)
        print(f"Model loaded successfully on {self.device}")
        if self.save_model_dir:
            Path(self.save_model_dir).mkdir(parents=True, exist_ok=True)
            self.pipe.save_pretrained(self.save_model_dir)

    def unload_model(self):
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            torch.cuda.empty_cache()

    def process_prompt(self, prompt: str, output_path: str = "out.png", num_inference_steps: int = 50,
                      guidance_scale: float = 7.5, seed: Optional[int] = None, width: int = 1024, height: int = 1024) -> str:
        if self.pipe is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        generator = torch.Generator(self.device)
        if seed is not None:
            generator = generator.manual_seed(seed)
        result = self.pipe(prompt=prompt, num_inference_steps=num_inference_steps,
                           guidance_scale=guidance_scale, width=width, height=height, generator=generator)
        image = result.images[0]
        image.save(output_path)
        return output_path

    def process_prompt_with_progress(self, prompt: str, output_path: str = "out.png", num_inference_steps: int = 50,
                                    guidance_scale: float = 7.5, seed: Optional[int] = None, width: int = 1024, height: int = 1024,
                                    progress_callback=None) -> str:
        if self.pipe is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        generator = torch.Generator(self.device)
        if seed is not None:
            generator = generator.manual_seed(seed)

        # Create a callback wrapper for diffusers progress tracking
        def callback_on_step_end(pipe, step_index, timestep, callback_kwargs):
            if progress_callback:
                progress_callback(step_index, num_inference_steps, f"Generating step {step_index + 1}")
            return callback_kwargs

        result = self.pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=["latents"]
        )
        image = result.images[0]
        image.save(output_path)
        return output_path

def txt2img(prompt: str, output_path: str = "out.png", model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
            hf_token: Optional[str] = None, num_inference_steps: int = 50, guidance_scale: float = 7.5,
            seed: Optional[int] = None, width: int = 1024, height: int = 1024, save_model_dir: Optional[str] = None) -> str:
    pipeline = Txt2ImgPipeline(model_id=model_id, hf_token=hf_token, save_model_dir=save_model_dir)
    pipeline.load_model()
    output = pipeline.process_prompt(prompt, output_path, num_inference_steps, guidance_scale, seed, width, height)
    pipeline.unload_model()
    return output

def txt2img_with_lora(prompt: str, output_path: str = "out.png", model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
                     hf_token: Optional[str] = None, num_inference_steps: int = 50, guidance_scale: float = 7.5,
                     seed: Optional[int] = None, width: int = 1024, height: int = 1024, save_model_dir: Optional[str] = None,
                     lora_path: Optional[str] = None, lora_strength: float = 1.0) -> str:
    """
    Generate image with optional LoRA model support.

    Args:
        prompt: Text prompt for image generation
        output_path: Path to save the generated image
        model_id: Hugging Face model ID for base SDXL model
        hf_token: Hugging Face token for authentication
        num_inference_steps: Number of denoising steps
        guidance_scale: Guidance scale for classifier-free guidance
        seed: Random seed for reproducible generation
        width: Image width in pixels
        height: Image height in pixels
        save_model_dir: Directory to save downloaded models
        lora_path: Path to LoRA model file (.safetensors)
        lora_strength: Strength of LoRA application (0.0 to 1.0)

    Returns:
        Path to the generated image
    """
    pipeline = Txt2ImgPipeline(model_id=model_id, hf_token=hf_token, save_model_dir=save_model_dir)

    # Load the base model
    pipeline.load_model()

    # Load LoRA if provided
    if lora_path and Path(lora_path).exists():
        try:
            print(f"Loading LoRA model: {lora_path} with strength {lora_strength}")

            # Try loading with PEFT backend first (for newer LoRA models)
            try:
                from peft import PeftModel
                pipeline.pipe.load_lora_weights(lora_path, adapter_name="character_lora")
                pipeline.pipe.set_adapters(["character_lora"], adapter_weights=[lora_strength])
                print("LoRA model loaded successfully with PEFT backend")
            except ImportError:
                print("PEFT not available, trying standard diffusers LoRA loading...")
                pipeline.pipe.load_lora_weights(lora_path, adapter_name="character_lora")
                pipeline.pipe.set_adapters(["character_lora"], adapter_weights=[lora_strength])
                print("LoRA model loaded successfully with standard backend")
            except Exception as lora_error:
                print(f"Failed to load with PEFT, trying alternative method: {lora_error}")
                # Try loading as a local LoRA file directly
                pipeline.pipe.load_lora_weights(Path(lora_path).parent, weight_name=Path(lora_path).name, adapter_name="character_lora")
                pipeline.pipe.set_adapters(["character_lora"], adapter_weights=[lora_strength])
                print("LoRA model loaded successfully with alternative method")

        except Exception as e:
            print(f"Warning: Failed to load LoRA model {lora_path}: {e}")
            print("Continuing with base model only...")

    # Generate the image
    output = pipeline.process_prompt(prompt, output_path, num_inference_steps, guidance_scale, seed, width, height)

    # Unload LoRA and cleanup
    if lora_path and Path(lora_path).exists():
        try:
            pipeline.pipe.unload_lora_weights()
        except:
            pass

    pipeline.unload_model()
    return output

def txt2img_with_lora_progress(prompt: str, output_path: str = "out.png", model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
                              hf_token: Optional[str] = None, num_inference_steps: int = 50, guidance_scale: float = 7.5,
                              seed: Optional[int] = None, width: int = 1024, height: int = 1024, save_model_dir: Optional[str] = None,
                              lora_path: Optional[str] = None, lora_strength: float = 1.0,
                              progress_callback=None) -> str:
    """
    Generate image with optional LoRA model support and progress tracking.

    Args:
        prompt: Text prompt for image generation
        output_path: Path to save the generated image
        model_id: Hugging Face model ID for base SDXL model
        hf_token: Hugging Face token for authentication
        num_inference_steps: Number of denoising steps
        guidance_scale: Guidance scale for classifier-free guidance
        seed: Random seed for reproducible generation
        width: Image width in pixels
        height: Image height in pixels
        save_model_dir: Directory to save downloaded models
        lora_path: Path to LoRA model file (.safetensors)
        lora_strength: Strength of LoRA application (0.0 to 1.0)
        progress_callback: Callback function for progress updates (step, total_steps, message)

    Returns:
        Path to the generated image
    """
    pipeline = Txt2ImgPipeline(model_id=model_id, hf_token=hf_token, save_model_dir=save_model_dir)

    # Progress tracking
    if progress_callback:
        progress_callback(0, num_inference_steps + 3, "Loading model...")

    # Load the base model
    pipeline.load_model()

    if progress_callback:
        progress_callback(1, num_inference_steps + 3, "Model loaded")

    # Load LoRA if provided
    if lora_path and Path(lora_path).exists():
        try:
            if progress_callback:
                progress_callback(2, num_inference_steps + 3, "Loading LoRA...")

            print(f"Loading LoRA model: {lora_path} with strength {lora_strength}")

            # Try loading with PEFT backend first (for newer LoRA models)
            try:
                from peft import PeftModel
                pipeline.pipe.load_lora_weights(lora_path, adapter_name="character_lora")
                pipeline.pipe.set_adapters(["character_lora"], adapter_weights=[lora_strength])
                print("LoRA model loaded successfully with PEFT backend")
            except ImportError:
                print("PEFT not available, trying standard diffusers LoRA loading...")
                pipeline.pipe.load_lora_weights(lora_path, adapter_name="character_lora")
                pipeline.pipe.set_adapters(["character_lora"], adapter_weights=[lora_strength])
                print("LoRA model loaded successfully with standard backend")
            except Exception as lora_error:
                print(f"Failed to load with PEFT, trying alternative method: {lora_error}")
                # Try loading as a local LoRA file directly
                pipeline.pipe.load_lora_weights(Path(lora_path).parent, weight_name=Path(lora_path).name, adapter_name="character_lora")
                pipeline.pipe.set_adapters(["character_lora"], adapter_weights=[lora_strength])
                print("LoRA model loaded successfully with alternative method")

        except Exception as e:
            print(f"Warning: Failed to load LoRA model {lora_path}: {e}")
            print("Continuing with base model only...")

    if progress_callback:
        progress_callback(3, num_inference_steps + 3, "Starting generation...")

    # Create progress callback for the pipeline
    def pipeline_progress_callback(step, timestep, latents):
        if progress_callback:
            progress_callback(3 + step, num_inference_steps + 3, f"Generating step {step + 1}")

    # Generate the image with progress tracking
    output = pipeline.process_prompt_with_progress(
        prompt, output_path, num_inference_steps, guidance_scale, seed, width, height,
        progress_callback=pipeline_progress_callback
    )

    # Unload LoRA and cleanup
    if lora_path and Path(lora_path).exists():
        try:
            pipeline.pipe.unload_lora_weights()
        except:
            pass

    pipeline.unload_model()

    if progress_callback:
        progress_callback(num_inference_steps + 3, num_inference_steps + 3, "Complete!")

    return output

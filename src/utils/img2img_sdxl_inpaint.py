"""
img2img_sdxl_inpaint.py

Single-file utility exposing `img2img` which performs inpainting with SDXL inpainting model
from Hugging Face (default: diffusers/stable-diffusion-xl-1.0-inpainting-0.1).

Function:
    img2img(
        input_image_path: str,
        mask_image_path: str,
        prompt: str,
        output_path: str = "out.png",
        model_id: str = "img2img_sdxl_inpaint",
        hf_token: str | None = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 9.0,
        seed: int | None = None,
        save_model_dir: str | None = None
    ) -> str
Returns path to saved output image. Uses the centralized model manager for model handling.

New Class-based approach for batch processing:
    InpaintingPipeline: A class that loads the model once and can process multiple images

    Example:
        pipeline = InpaintingPipeline()
        pipeline.load_model()
        for image in images:
            pipeline.process_image(input_path, mask_path, prompt, output_path)
        pipeline.unload_model()  # Optional cleanup

Notes:
 - The function expects a 1024x1024 input image and a 1024x1024 mask image.
 - The mask must have white (255,255,255) where you WANT inpainting to happen (as you requested).
 - You must have `diffusers`, `transformers`, `accelerate`, `torch`, and `huggingface_hub` installed.
 - Set HUGGINGFACE_HUB_TOKEN in the env or pass hf_token to authenticate private models.
 - This file only contains code. Running the inpainting requires a suitable GPU + CUDA and the dependencies.
"""

from __future__ import annotations
import os
from typing import Optional
from PIL import Image, ImageFilter, ImageOps
import cv2
import numpy as np
import torch
from ..models import get_model_manager


class InpaintingPipeline:
    """Class-based inpainting pipeline that loads the model once and can process multiple images."""

    def __init__(self,
                 model_id: str = "img2img_sdxl_inpaint",
                 hf_token: Optional[str] = None,
                 save_model_dir: Optional[str] = None,
                 device: Optional[str] = None):
        self.model_id = model_id
        self.hf_token = hf_token
        self.save_model_dir = save_model_dir
        self.pipe = None
        self.device = device  # Allow device to be specified
        self.torch_dtype = None

    def load_model(self):
        """Load the SDXL inpainting model. Call this once before processing multiple images."""
        if self.pipe is not None:
            return  # Already loaded

        # Lazy imports
        try:
            from diffusers import StableDiffusionXLInpaintPipeline, DPMSolverMultistepScheduler
        except Exception as e:
            raise RuntimeError("This function requires the 'diffusers' package. Install with 'pip install diffusers[torch] transformers accelerate'") from e

        # Device selection - use specified device or auto-detect
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Set torch dtype based on the actual device (not just checking if "cuda" is in device string)
        if self.device == "cpu":
            self.torch_dtype = torch.float32
        else:
            # For any CUDA device (cuda, cuda:0, cuda:1, etc.)
            self.torch_dtype = torch.float16

        print(f"InpaintingPipeline: Loading model on device: {self.device}")

        # Set the default device for torch operations to our selected device
        if self.device != "cpu":
            # Extract GPU index from device string (e.g., "cuda:0" -> 0)
            if ":" in self.device:
                gpu_index = int(self.device.split(":")[1])
            else:
                gpu_index = 0

            # Set PyTorch's default CUDA device
            torch.cuda.set_device(gpu_index)
            print(f"Set PyTorch default CUDA device to: {gpu_index}")

        # Use model manager to get model path and info
        model_manager = get_model_manager()

        # Check if we should use local model file or Hugging Face model
        if self.model_id in model_manager.models:
            model_info = model_manager.get_model_info(self.model_id)

            # Check if model is available locally
            if model_info.exists:
                model_path = model_manager.get_model_path(self.model_id)
                print(f"Loading SDXL inpainting model from local file: {model_path}")

                # For safetensors files, we need to use the Hugging Face model ID but with local_files_only
                # Since SDXL inpainting models are complex pipelines, we'll use the HF model ID
                auth_kwargs = {}
                if self.hf_token:
                    auth_kwargs["token"] = self.hf_token

                # Load from Hugging Face (the safetensors file is just the UNet weights)
                hf_model_id = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
                self.pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
                    hf_model_id,
                    torch_dtype=self.torch_dtype,
                    **auth_kwargs
                )
                print(f"Model loaded from Hugging Face: {hf_model_id}")
            else:
                # Try to download the model if it's missing
                print(f"Model {model_info.name} not found locally, attempting to download...")
                success = model_manager.download_model(self.model_id)
                if success:
                    print(f"Successfully downloaded {model_info.name}")
                else:
                    print(f"Failed to download {model_info.name}, falling back to Hugging Face")

                # Load from Hugging Face anyway
                auth_kwargs = {}
                if self.hf_token:
                    auth_kwargs["token"] = self.hf_token

                hf_model_id = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
                self.pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
                    hf_model_id,
                    torch_dtype=self.torch_dtype,
                    **auth_kwargs
                )
        else:
            # Fallback to direct Hugging Face loading for unknown model IDs
            auth_kwargs = {}
            if self.hf_token:
                auth_kwargs["token"] = self.hf_token

            print(f"Loading SDXL inpainting model from Hugging Face: {self.model_id}")
            self.pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype,
                **auth_kwargs
            )

        # Attach a faster scheduler (optional)
        try:
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        except Exception:
            # if this fails, continue with default
            pass

        # Move to device
        self.pipe = self.pipe.to(self.device)
        print(f"Model loaded successfully on {self.device}")

        # Optionally save the model files
        if self.save_model_dir:
            os.makedirs(self.save_model_dir, exist_ok=True)
            self.pipe.save_pretrained(self.save_model_dir)

    def unload_model(self):
        """Unload the model to free memory. Call this when done processing all images."""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("Model unloaded and GPU memory cleared")

    def fill_with_blur(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        """
    Fills masked regions with colors from the surrounding image using progressive blurs.
    - image: input image (RGB or RGBA)
    - mask: L or 1 mode mask (white = fill, black = keep)

    Returns: RGB image with masked areas filled.
    """
        # Ensure modes
        if mask.mode != "L":
            mask = mask.convert("L")
        if image.mode != "RGBA":
            image = image.convert("RGBA")

        # Create masked image: keep original where mask is black
        image_masked = Image.new("RGBA", image.size)
        image_masked.paste(image, mask=ImageOps.invert(mask))

        # Composite progressively blurred versions
        image_mod = Image.new("RGBA", image.size)
        for radius, repeats in [(256, 1), (64, 1), (16, 2), (4, 4), (2, 2), (0, 1)]:
            blurred = image_masked.filter(ImageFilter.GaussianBlur(radius))
            for _ in range(repeats):
                image_mod.alpha_composite(blurred)

        return image_mod.convert("RGB")

    def process_image(self,
                     input_image_path: str,
                     mask_image_path: str,
                     prompt: str,
                     output_path: str = "out.png",
                     num_inference_steps: int = 40,
                     guidance_scale: float = 7.0,
                     seed: Optional[int] = None) -> str:
        """Process a single image with the loaded model."""
        if self.pipe is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Ensure we're using the correct device for all operations
        if self.device != "cpu":
            # Extract GPU index and set as current device
            if ":" in self.device:
                gpu_index = int(self.device.split(":")[1])
            else:
                gpu_index = 0
            torch.cuda.set_device(gpu_index)
            print(f"InpaintingPipeline: Processing on device {self.device} (GPU {gpu_index})")

        # Load images
        input_img = Image.open(input_image_path).convert("RGB")
        mask_img = Image.open(mask_image_path).convert("L")

        # Validate size (expect 1024x1024)
        if input_img.size != (1024, 1024):
            raise ValueError(f"Input image must be 1024x1024 — got {input_img.size}")
        if mask_img.size != (1024, 1024):
            raise ValueError(f"Mask image must be 1024x1024 — got {mask_img.size}")

        # Binarize mask
        mask = mask_img.point(lambda p: 255 if p > 128 else 0).convert("L")

        # Expand the white area (inpainting region) by ~3px (adjust size for 2–5px)
        mask = mask.filter(ImageFilter.MaxFilter(size=17))  # size must be odd → 3, 5, 7...

        # Make a blurred version of the input image (acts as "fill")
        blurred = self.fill_with_blur(input_img, mask)

        # Ensure the pipeline is on the correct device
        self.pipe = self.pipe.to(self.device)

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)

        # Run first inpainting pass
        result = self.pipe(
            prompt=prompt,
            negative_prompt="watermark, text, EasyNegative, paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, glans, mutated hands, (poorly drawn hands:1.5), blurry, (bad anatomy:1.21), extra limbs, lowers, bad hands, missing fingers, extra digit ,bad hands, missing fingers, edges, borders",
            image=blurred,
            mask_image=mask,
            num_inference_steps=20,
            guidance_scale=guidance_scale,
            generator=generator,
            strength=1.0,
        )
        out_img = result.images[0]
        prefilled = Image.composite(out_img, input_img, mask_img)

        # Create new generator for second pass
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)

        # Run second inpainting pass
        result = self.pipe(
            prompt=prompt,
            negative_prompt="watermark, text, EasyNegative, paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, glans, mutated hands, (poorly drawn hands:1.5), blurry, (bad anatomy:1.21), extra limbs, lowers, bad hands, missing fingers, extra digit ,bad hands, missing fingers, edges, borders",
            image=prefilled,
            mask_image=mask,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            strength=0.96,
        )

        out_img = result.images[0]
        out_img.save(output_path)
        return output_path
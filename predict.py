# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import time
import torch
import subprocess
from diffusers import CogView4Pipeline
from cog import BasePredictor, Input, Path

MODEL_CACHE = "checkpoints"
MODEL_URL = "https://weights.replicate.delivery/default/THUDM/CogView4-6B/model.tar"

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # Download weights if they don't exist
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)

        # Load CogView4-6B model with bfloat16 precision as recommended
        self.pipe = CogView4Pipeline.from_pretrained(
            MODEL_CACHE, 
            torch_dtype=torch.bfloat16
        )
        
        # Enable optimizations to reduce GPU memory usage and improve speed
        self.pipe.enable_model_cpu_offload()
        self.pipe.vae.enable_slicing()
        self.pipe.vae.enable_tiling()


    def predict(
        self,
        prompt: str = Input(
            description="Text prompt to generate an image from"
        ),
        negative_prompt: str = Input(
            description="Negative prompt to guide image generation away from certain concepts",
            default=None
        ),
        width: int = Input(
            description="Width of the generated image (must be between 512 and 2048, divisible by 32)",
            default=1024,
            ge=512,
            le=2048
        ),
        height: int = Input(
            description="Height of the generated image (must be between 512 and 2048, divisible by 32)",
            default=1024,
            ge=512,
            le=2048
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps",
            default=50,
            ge=1,
            le=100
        ),
        guidance_scale: float = Input(
            description="Guidance scale for classifier-free guidance",
            default=3.5,
            ge=0.0,
            le=20.0
        ),
        seed: int = Input(
            description="Random seed for reproducible image generation",
            default=None
        )
    ) -> Path:
        """Run a single prediction on the model"""
        # Validate dimensions
        if width % 32 != 0 or height % 32 != 0:
            raise ValueError("Width and height must be divisible by 32")
        if width * height > 2**21:
            raise ValueError(f"Resolution {width}x{height} exceeds maximum allowed pixels (2^21)")
        
        # Set seed for reproducibility
        generator = None
        if seed is None:
            seed = int.from_bytes(os.urandom(3), "big")
        generator = torch.Generator().manual_seed(seed)
        print("Using seed: ", seed)

        # Generate image(s)
        images = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            num_images_per_prompt=1,
        ).images
        
        # Save the first generated image
        output_path = Path(f"/tmp/output.png")
        images[0].save(output_path)
        return output_path


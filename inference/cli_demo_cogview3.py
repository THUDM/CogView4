"""
Running the Script:
To run the script, use the following command with appropriate arguments:

```bash
python cli_demo.py --prompt "A beautiful sunset over a mountain" --width 1024 --height 1024
```

Additional options are available to specify the model path, guidance scale, number of inference steps, image generation type, and output paths.
"""

import argparse

import torch
from diffusers import CogView3PlusPipeline


def generate_image(
    prompt, model_path, guidance_scale, num_images_per_prompt, num_inference_steps, width, height, output_path, dtype
):
    # Load the pre-trained model with the specified precision
    pipe = CogView3PlusPipeline.from_pretrained(model_path, torch_dtype=dtype)

    # Enable CPU offloading to free up GPU memory when layers are not actively being used
    pipe.enable_model_cpu_offload()

    # Enable VAE slicing and tiling for memory optimization
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    # Generate the image based on the prompt
    image = pipe(
        prompt=prompt,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_images_per_prompt,
        num_inference_steps=num_inference_steps,
        width=width,
        height=height,
    ).images[0]

    # Save the generated image to the local file system
    image.save(output_path)

    print(f"Image saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate an image using the CogView3-Plus-3B model.")

    # Define arguments for prompt, model path, etc.
    parser.add_argument("--prompt", type=str, required=True, help="The text description for generating the image.")
    parser.add_argument(
        "--model_path", type=str, default="THUDM/CogView3-Plus-3B", help="Path to the pre-trained model."
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=7.0, help="The guidance scale for classifier-free guidance."
    )
    parser.add_argument(
        "--num_images_per_prompt", type=int, default=1, help="Number of images to generate per prompt."
    )
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of denoising steps for inference.")
    parser.add_argument("--width", type=int, default=1024, help="Width of the generated image.")
    parser.add_argument("--height", type=int, default=1024, help="Height of the generated image.")
    parser.add_argument("--output_path", type=str, default="cogview3.png", help="Path to save the generated image.")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Precision type (float16 or bfloat16).")

    # Parse the arguments
    args = parser.parse_args()

    # Convert dtype argument to torch dtype
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    # Call the function to generate the image
    generate_image(
        prompt=args.prompt,
        model_path=args.model_path,
        guidance_scale=args.guidance_scale,
        num_images_per_prompt=args.num_images_per_prompt,
        num_inference_steps=args.num_inference_steps,
        width=args.width,
        height=args.height,
        output_path=args.output_path,
        dtype=dtype,
    )

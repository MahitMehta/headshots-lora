import gc
from pathlib import Path
from diffusers import StableDiffusionXLInpaintPipeline, DPMSolverMultistepScheduler  # type: ignore
import torch
from PIL import Image
import os
from datetime import datetime

from utils.format import InputImageFormatter

# Configuration
model_id = "stabilityai/stable-diffusion-xl-base-1.0"

HG_HEADSHOTS_LORA_ID = "mahitm/mahitm-headshots-v1"
HG_HEADSHOTS_WEIGHT_NAME = "headshot-v1.1.safetensors"
lora_path = HG_HEADSHOTS_LORA_ID  # "headshot.safetensors"
lora_weight_name = HG_HEADSHOTS_WEIGHT_NAME  # os.path.basename(lora_path)

input_image_path = (
    Path(__file__).parent / "images_raw" / "mahit_2.jpg"
)  # "images" / "image_0002.jpg"


prompt = "mahitm-headshot-v1, Professional headshot, male, black suit"
negative_prompt = ""
output_dir = "output/local_inference"
image_width = 1024
image_height = 1024
guidance_scale_val = 7.5
num_inference_steps_val = 30

print("Loading Base Pipeline...")
base = StableDiffusionXLInpaintPipeline.from_pretrained(
    model_id, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cpu")
base.scheduler = DPMSolverMultistepScheduler.from_config(base.scheduler.config)
base.safety_checker = None

# base.unet = torch.compile(base.unet, mode="reduce-overhead", fullgraph=True)
print("Base Pipeline Loaded.")

# Load LoRA
print(f"Loading LoRA weights from: {lora_path}")
base.load_lora_weights(
    lora_path, weight_name=lora_weight_name, adapter_name="headshot_lora"
)
print("LoRA weights loaded.")
start = datetime.now()
base.fuse_lora()
end = datetime.now()
print(f"LoRA weights fused in {end - start} seconds.")
base.unload_lora_weights()

base.enable_model_cpu_offload()
base.enable_attention_slicing()
base.enable_xformers_memory_efficient_attention()
torch.backends.cuda.matmul.allow_tf32 = True
print("Optimizations applied.")

print("Loading Refiner Pipeline...")
refiner = StableDiffusionXLInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
).to("cpu")

refiner.enable_model_cpu_offload()
refiner.enable_attention_slicing()
refiner.enable_xformers_memory_efficient_attention()
torch.backends.cuda.matmul.allow_tf32 = True
print("Refiner Pipeline Loaded.")

# Load Input Image and Mask
print(f"Loading input image from: {input_image_path}")
try:
    input_image = Image.open(input_image_path)
except FileNotFoundError:
    print(
        f"ERROR: Input image not found at {input_image_path}. Please provide a valid path."
    )
    exit()

print("Generating model inputs")

with InputImageFormatter(with_hair_mask=False) as formatter:
    input_image, mask_image, small_mask_image = base.get_model_inputs(
        input_image, width=image_width, height=image_height
    )


os.makedirs(output_dir, exist_ok=True)
print(f"Output directory: {output_dir}")

# seed for reproducibility
# generator = torch.Generator(device=base.device).manual_seed(42)

with torch.inference_mode():
    print("Generating image...")

    latent = base(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=input_image,
        mask_image=mask_image,
        width=image_width,
        height=image_height,
        guidance_scale=guidance_scale_val,
        num_inference_steps=num_inference_steps_val,
        cross_attention_kwargs={"scale": 1.0},
        # generator=generator,
        strength=0.9,
        denoising_end=0.85,
        output_type="latent",
    ).images  # type: ignore

    base.to("cpu")
    del base
    gc.collect()
    torch.cuda.empty_cache()

    final = refiner(
        prompt="Professional headshot, male, black suit",
        image=latent,
        width=image_width,
        height=image_height,
        guidance_scale=guidance_scale_val,
        num_inference_steps=num_inference_steps_val,
        # generator=generator,
        denoising_start=0.85,
        mask_image=small_mask_image,
        strength=0.99,
    ).images[0]  # type: ignore

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = f"{output_dir}/output_{timestamp}.png"
    final.save(output_path)
    print(f"Saved image to {output_path}")

print("Image generation complete.")

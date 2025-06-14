import gc
from pathlib import Path
from diffusers import StableDiffusionXLInpaintPipeline, DPMSolverMultistepScheduler, StableDiffusionXLImg2ImgPipeline  # type: ignore
import torch
from PIL import Image
import os
from datetime import datetime

from utils.format_image import resize_pad_image
from utils.mask_original import generate_mask as generate_original_mask

# Configuration
model_id = "stabilityai/stable-diffusion-xl-base-1.0"

HG_HEADSHOTS_LORA_ID = "mahitm/mahitm-headshots-v1"
HG_HEADSHOTS_WEIGHT_NAME = "headshot-v1.1.safetensors"
lora_path = HG_HEADSHOTS_LORA_ID # "headshot.safetensors"
lora_weight_name = HG_HEADSHOTS_WEIGHT_NAME # os.path.basename(lora_path)

input_image_path = Path(__file__).parent / "images_raw" / "mahit_2.jpg" # "images" / "image_0002.jpg"
tmp_inference_image_filename = "inference_image.jpg"

tmp_dir = Path(__file__).parent / "../output" / "tmp"
os.makedirs(tmp_dir, exist_ok=True)

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

# --- Load LoRA ---
from diffusers.utils.import_utils import is_safetensors_available

if not is_safetensors_available():
    raise ImportError(
        "Please install safetensors to load LoRA weights (pip install safetensors)"
    )

from datetime import datetime
print(f"Loading LoRA weights from: {lora_path}")
base.load_lora_weights(lora_path, weight_name=lora_weight_name, adapter_name="headshot_lora")
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

# refiner.unet = torch.compile(refiner.unet, mode="reduce-overhead", fullgraph=True)
refiner.enable_model_cpu_offload()
refiner.enable_attention_slicing()
refiner.enable_xformers_memory_efficient_attention()
torch.backends.cuda.matmul.allow_tf32 = True
print("Refiner Pipeline Loaded.")

# --- Load Input Image and Mask ---
print(f"Loading input image from: {input_image_path}")
try:
    input_image = resize_pad_image(Image.open(input_image_path))
    input_image.save(os.path.join(tmp_dir, f"cropped_{tmp_inference_image_filename}"))
except FileNotFoundError:
    print(
        f"ERROR: Input image not found at {input_image_path}. Please provide a valid path."
    )
    exit()

print(f"Generating mask for input image...")

# from utils.mask_hair import generate_mask
from utils.mask import generate_mask

generate_mask(
    filenames=[f"cropped_{tmp_inference_image_filename}"],
    input_dir=tmp_dir,
    output_filename_prefix="mask_",
    output_mask_dir=tmp_dir,
)

mask_image = (
    Image.open(os.path.join(tmp_dir, f"mask_cropped_{tmp_inference_image_filename}"))
    .resize((image_width, image_height))
)

generate_mask(
    filenames=[f"cropped_{tmp_inference_image_filename}"],
    input_dir=tmp_dir,
    output_mask_dir=tmp_dir,
    output_filename_prefix="small_mask_",
    inset=0.20
)

small_mask_image = (
    Image.open(os.path.join(tmp_dir, f"small_mask_cropped_{tmp_inference_image_filename}"))
    .resize((image_width, image_height))
)

os.makedirs(output_dir, exist_ok=True)
print(f"Output directory: {output_dir}")

# seed for reproducibility
# generator = torch.Generator(device=base.device).manual_seed(42)

with torch.inference_mode():
    print(f"Generating image...")

    latent = base(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=input_image,
        mask_image=mask_image,
        width=image_width,
        height=image_height,
        guidance_scale=guidance_scale_val,
        num_inference_steps=num_inference_steps_val,
        cross_attention_kwargs={"scale": 1.0 },
        # generator=generator,
        strength=0.9,
        denoising_end=0.85,
        output_type="latent",
    ).images # type: ignore

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
        strength=0.99
    ).images[0] # type: ignore

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = f"{output_dir}/output_{timestamp}.png"
    final.save(output_path)
    print(f"Saved image to {output_path}")

print("Image generation complete.")

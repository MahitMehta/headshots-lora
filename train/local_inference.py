from diffusers import StableDiffusionXLInpaintPipeline, DPMSolverMultistepScheduler  # type: ignore
import torch
from PIL import Image
import os
from datetime import datetime

from utils.format_image import format_image

# --- Configuration ---
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
lora_path = "headshot.safetensors"

base_dir = os.path.dirname(os.path.abspath(__file__))
input_image_path = os.path.join(base_dir, "training_data\\images\\image_0024.jpg")
tmp_inference_image_filename = "inference_image.jpg"

tmp_dir = os.path.join(base_dir, "tmp")
os.makedirs(tmp_dir, exist_ok=True)

prompt = "mahitm-headshot-v1, professional headshot"
negative_prompt = ""
output_dir = "output/inpainting_inference"
num_images_to_generate = 3
image_width = 1024
image_height = 1024
guidance_scale_val = 7.5
num_inference_steps_val = 30

print("Loading pipeline...")
pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
    model_id, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.safety_checker = None
pipe = pipe.to("cuda")
print("Pipeline loaded.")

# --- Load LoRA ---
from diffusers.utils.import_utils import is_safetensors_available

if not is_safetensors_available():
    raise ImportError(
        "Please install safetensors to load LoRA weights (pip install safetensors)"
    )

print(f"Loading LoRA weights from: {lora_path}")
# pipe.load_lora_weights(lora_path, weight_name=os.path.basename(lora_path), adapter_name="headshot_lora")
print("LoRA weights loaded.")

pipe.enable_model_cpu_offload()
pipe.enable_attention_slicing()
pipe.enable_xformers_memory_efficient_attention()
torch.backends.cuda.matmul.allow_tf32 = True
print("Optimizations applied.")

# --- Load Input Image and Mask ---
print(f"Loading input image from: {input_image_path}")
try:
    init_image = format_image(Image.open(input_image_path), size=image_width)
    init_image.save(os.path.join(tmp_dir, f"mask_{tmp_inference_image_filename}"))
    init_image.save(os.path.join(tmp_dir, f"cropped_{tmp_inference_image_filename}"))
except FileNotFoundError:
    print(
        f"ERROR: Input image not found at {input_image_path}. Please provide a valid path."
    )
    exit()

print(f"Generating mask for input image...")

# from mask_hair import process as mask_process
from utils.mask import process as mask_process

mask_process(
    filenames=[f"mask_{tmp_inference_image_filename}"],
    input_dir=tmp_dir,
    output_mask_dir=tmp_dir,
)

mask_image = (
    Image.open(os.path.join(tmp_dir, f"mask_{tmp_inference_image_filename}"))
    .convert("L")
    .resize((image_width, image_height))
)

os.makedirs(output_dir, exist_ok=True)
print(f"Output directory: {output_dir}")

# seed for reproducibility
# generator = torch.Generator(device=pipe.device).manual_seed(42)

print(f"Generating {num_images_to_generate} images...")
with torch.inference_mode():
    for i in range(num_images_to_generate):
        print(f"Generating image {i + 1}/{num_images_to_generate}...")

        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init_image,
            mask_image=mask_image,
            width=image_width,
            height=image_height,
            guidance_scale=guidance_scale_val,
            num_inference_steps=num_inference_steps_val,
            # cross_attention_kwargs={"scale": 1.0},
            # generator=generator,
            strength=0.9,
        ).images[0]  # type: ignore

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_path = f"{output_dir}/output_{timestamp}_{i + 1}.png"
        image.save(output_path)
        print(f"Saved image to {output_path}")

print("Image generation complete.")

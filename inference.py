from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
import torch

model_id = "stabilityai/stable-diffusion-xl-base-1.0"

pipe = StableDiffusionXLPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.safety_checker = None
pipe = pipe.to("cuda")

from diffusers.utils.import_utils import is_safetensors_available

if is_safetensors_available():
    from safetensors.torch import load_file
else:
    raise ImportError("Please install safetensors to load LoRA weights.")

lora_path = "headshot.safetensors"

lora_state_dict = load_file(lora_path)
pipe.load_lora_weights(lora_path, alpha=1)  # alpha is the scaling factor, adjust as needed

pipe.enable_model_cpu_offload()
pipe.enable_attention_slicing()
pipe.enable_xformers_memory_efficient_attention()
torch.backends.cuda.matmul.allow_tf32 = True

prompt = "Professional headshot, woman, long black wavy hair, medium shot. Slightly angled camera, soft lighting, white shirt, and a plain gray background."
output_dir = "output/lora"

import os

os.makedirs(output_dir, exist_ok=True)

from datetime import datetime

width = 1024
height = 1024

with torch.inference_mode():
  for i in range(5):
    image = pipe(prompt, width=width, height=height, guidance_scale=1, num_inference_steps=30).images[0]
    image.save(f"{output_dir}/output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")

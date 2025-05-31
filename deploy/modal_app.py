import time
from io import BytesIO
from pathlib import Path
from PIL import Image

import modal

# should be no greater than host CUDA version
cuda_version = "12.4.0" 
# includes full CUDA toolkit
flavor = "devel"  
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

cuda_dev_image = modal.Image.from_registry(
    f"nvidia/cuda:{tag}", add_python="3.11"
).entrypoint([])

diffusers_commit_sha = "81cf3b2f155f1de322079af28f625349ee21ec6b"

image = (
    cuda_dev_image.apt_install(
        "git",
        "libglib2.0-0",
        "libsm6",
        "libxrender1",
        "libxext6",
        "ffmpeg",
        "libgl1",
    )
    .pip_install(
        "opencv-python==4.11.0.86",
        "mediapipe==0.10.21",
        "pillow==11.2.1",
        "invisible_watermark==0.2.0",
        "transformers==4.44.0",
        "huggingface_hub[hf_transfer]==0.26.2",
        "accelerate==0.33.0",
        "safetensors==0.4.4",
        "sentencepiece==0.2.0",
        "torch==2.5.0",
        f"git+https://github.com/huggingface/diffusers.git@{diffusers_commit_sha}",
        "numpy<2",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HUB_CACHE": "/cache"})
)

image = image.env(
    {
        "TORCHINDUCTOR_CACHE_DIR": "/root/.inductor-cache",
        "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
    }
)

app = modal.App("headshots-app", image=image)

with image.imports():
    import torch
    from diffusers import StableDiffusionXLInpaintPipeline, DPMSolverMultistepScheduler
    from PIL.Image import Image

SCALEDOWN_WINDOW = 5 * 60 # seconds
TIMEOUT = 60 * 60  # 1 hour timeout for compilation

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
NUM_INFERENCE_STEPS = 30
GUIDANCE_SCALE = 7.5
WIDTH = 1024
HEIGHT = 1024

@app.cls(
    gpu="T4",
    scaledown_window=SCALEDOWN_WINDOW,
    timeout=TIMEOUT, 
    volumes={ 
        "/cache": modal.Volume.from_name("hf-hub-cache", create_if_missing=True),
        "/root/.nv": modal.Volume.from_name("nv-cache", create_if_missing=True),
        "/root/.triton": modal.Volume.from_name("triton-cache", create_if_missing=True),
        "/root/.inductor-cache": modal.Volume.from_name(
            "inductor-cache", create_if_missing=True
        ),
    },
)
class Model:
    compile: bool = ( 
        modal.parameter(default=False)
    )

    @modal.enter()
    def enter(self):
        pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )

        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.safety_checker = None
        pipe = pipe.to("cuda") 
        self.pipe = optimize(pipe, compile=self.compile)

    @modal.method()
    def inference(self, prompt: str, input_image) -> bytes:
        print("Generating image with prompt:", prompt)

        out = self.pipe(
            prompt=prompt,
            image=input_image,
            mask_image=input_image,
            width=WIDTH,
            height=HEIGHT,
            guidance_scale=GUIDANCE_SCALE,
            num_inference_steps=NUM_INFERENCE_STEPS,
            output_type="pil",
            cross_attention_kwargs={"scale": 1.0},
            strength=0.9,
        ).images[0]

        byte_stream = BytesIO()
        out.save(byte_stream, format="JPEG")
        return byte_stream.getvalue()
    
@app.local_entrypoint()
def main(
    prompt: str = "A professional headshot of a person wearing a suit, high quality, studio lighting",
    twice: bool = True,
    compile: bool = False,
):
    input_image_path = "/Users/mahitmehta/Desktop/Misc/headshots-lora/training_images/image_0018.jpg"
    input_image = Image.open(input_image_path)

    t0 = time.time()
    image_bytes = Model(compile=compile).inference.remote(prompt, input_image)
    print(f"🎨 first inference latency: {time.time() - t0:.2f} seconds")

    if twice:
        t0 = time.time()
        image_bytes = Model(compile=compile).inference.remote(prompt, input_image)
        print(f"🎨 second inference latency: {time.time() - t0:.2f} seconds")

    output_path = Path("/tmp") / "flux" / "output.jpg"
    output_path.parent.mkdir(exist_ok=True, parents=True)
    print(f"🎨 saving output to {output_path}")
    output_path.write_bytes(image_bytes)

def optimize(pipe, compile=True):
    # fuse QKV projections in VAE
    pipe.vae.fuse_qkv_projections()

    # switch memory layout to Torch's preferred, channels_last
    pipe.vae.to(memory_format=torch.channels_last)

    if not compile:
        return pipe

    # set torch compile flags
    config = torch._inductor.config
    config.disable_progress = False  # show progress bar
    config.conv_1x1_as_mm = True  # treat 1x1 convolutions as matrix muls
    # adjust autotuning algorithm
    config.coordinate_descent_tuning = True
    config.coordinate_descent_check_all_directions = True
    config.epilogue_fusion = False  # do not fuse pointwise ops into matmuls

    # tag the compute-intensive modules, the Transformer and VAE decoder, for compilation
    pipe.transformer = torch.compile(
        pipe.transformer, mode="max-autotune", fullgraph=True
    )
    pipe.vae.decode = torch.compile(
        pipe.vae.decode, mode="max-autotune", fullgraph=True
    )

    print("🔦 finished torch compilation")
    return pipe
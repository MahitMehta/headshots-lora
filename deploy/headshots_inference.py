import time
from pathlib import Path
from PIL import Image
import os

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

inference_image = (
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
        "peft==0.15.2",
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

inference_image = inference_image.env(
    {
        "TORCHINDUCTOR_CACHE_DIR": "/root/.inductor-cache",
        "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
    }
)

app = modal.App("headshots-inference", image=inference_image)

with inference_image.imports():
    import torch
    from diffusers import StableDiffusionXLInpaintPipeline, DPMSolverMultistepScheduler
    from PIL import Image
    import os

SCALEDOWN_WINDOW = 5 * 60  # seconds
TIMEOUT = 60 * 60  # 1 hour timeout for compilation

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
NUM_INFERENCE_STEPS = 30
GUIDANCE_SCALE = 7.5
WIDTH = 1024
HEIGHT = 1024

HG_HEADSHOTS_LORA_ID = "mahitm/mahitm-headshots-v1"
HG_HEADSHOTS_WEIGHT_NAME = "headshot-v1.1.safetensors"


@app.cls(
    gpu="L4",
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
    compile: bool = modal.parameter(default=False)
    with_lora: bool = modal.parameter(default=True)
    fixed_seed: bool = modal.parameter(default=False)

    @modal.enter()
    def enter(self):
        pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        )

        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.safety_checker = None

        if self.with_lora:
            print("Loading LoRA weights")
            pipe.load_lora_weights(
                HG_HEADSHOTS_LORA_ID, weight_name=HG_HEADSHOTS_WEIGHT_NAME
            )
            print("LoRA weights loaded.")

        pipe = pipe.to("cuda")

        self.pipe = optimize(pipe, compile=self.compile)

    @modal.method()
    def inference(
        self, prompt: str, input_image: Image.Image, mask_image: Image.Image
    ) -> Image.Image:
        print("Generating image with prompt:", prompt)

        generator = None
        if self.fixed_seed:
            generator = torch.Generator(device=self.pipe.device).manual_seed(42)

        out = self.pipe(
            prompt=prompt,
            image=input_image,
            mask_image=mask_image,
            width=WIDTH,
            height=HEIGHT,
            guidance_scale=GUIDANCE_SCALE,
            num_inference_steps=NUM_INFERENCE_STEPS,
            output_type="pil",
            cross_attention_kwargs={"scale": 0.8},
            strength=0.9,
            mask_content="latent_noise",
            generator=generator,
        ).images[0]

        return out


inference_caller_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")  # needed for OpenCV
    .pip_install(
        "pillow==11.2.1",
        "supabase==2.15.2",
        "opencv-python-headless[standard]",
        "mediapipe==0.10.21",
    )
)

inference_caller_image = inference_caller_image.add_local_python_source("utils")

with inference_caller_image.imports():
    from utils.format_image import get_image_inputs
    from io import BytesIO
    from utils.types.input import Gender


@app.function(
    image=inference_caller_image,
    secrets=[modal.Secret.from_name("supabase-credentials")],
)
def inference(
    user_id: str,
    request_id: str,
    gender: "Gender",
    input_image: Image.Image,
    with_lora: bool = True,
    fixed_seed: bool = False,
    with_hair_mask: bool = False,
) -> bool:
    """
    Run inference on the model with the given prompt and input image.
    """

    input_image, mask_image = get_image_inputs(input_image, with_hair_mask)

    # construct prompt
    prompt = ["mahitm-headshots-v1.1", "Professional headshot"]
    if gender != "non-binary":
        prompt.append(gender)

    prompt.append("suit")  # default attire

    prompt = ", ".join(prompt)  # comma-separated prompt

    out_image = Model(with_lora=with_lora, fixed_seed=fixed_seed).inference.remote(
        prompt, input_image, mask_image
    )

    # convert to 2:3 aspect ratio
    target_width = 682
    side_padding = (out_image.width - target_width) // 2
    cropped_image = out_image.crop(
        (side_padding, 0, target_width + side_padding, out_image.height)
    )

    byte_stream = BytesIO()
    cropped_image.save(byte_stream, format="JPEG")
    output_bytes = byte_stream.getvalue()

    from supabase import create_client, Client

    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")
    supabase: Client = create_client(supabase_url, supabase_key)

    file_path = f"{user_id}/{request_id}/1.jpg"

    bucket_name = "headshots"
    response = supabase.storage.from_(bucket_name).upload(
        path=file_path, file=output_bytes, file_options={"content-type": "image/jpeg"}
    )

    print("Uploaded output to:", response.fullPath)

    return [file_path]  # return only the path inside the bucket


@app.local_entrypoint()
def main(
    prompt: str = "A professional headshot of a person wearing a suit, high quality, studio lighting",
    twice: bool = False,
    compile: bool = False,
):
    input_image_path = Path(__file__).parent / "../train" / "images" / "image_0001.jpg"
    input_image = Image.open(input_image_path).convert("RGB")

    input_image, mask_image = get_image_inputs(input_image)

    t0 = time.time()
    image = Model(compile=compile).inference.remote(prompt, input_image, mask_image)
    print(f"First inference latency: {time.time() - t0:.2f} seconds")

    if twice:
        t0 = time.time()
        image = Model(compile=compile).inference.remote(prompt, input_image, mask_image)
        print(f"Second inference latency: {time.time() - t0:.2f} seconds")

    output_path = (
        Path(__file__).parent / "../output" / "inpainting_inference" / "test.jpg"
    )
    output_path.parent.mkdir(exist_ok=True, parents=True)

    print(f"Saving output to {output_path}")
    image.save(output_path, format="JPEG")


def optimize(pipe, compile=True):
    # fuse QKV projections in VAE
    pipe.vae.fuse_qkv_projections()

    # switch memory layout to Torch's preferred, channels_last
    pipe.vae.to(memory_format=torch.channels_last)

    if not compile:
        return pipe

    torch.set_float32_matmul_precision("high")

    # tag the compute-intensive modules, the VAE decoder, for compilation
    pipe.vae.decode = torch.compile(
        pipe.vae.decode, mode="max-autotune", fullgraph=True
    )

    print("Finished PyTorch compilation")
    return pipe

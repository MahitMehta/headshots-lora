import time
from io import BytesIO
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

inference_image = inference_image.env(
    {
        "TORCHINDUCTOR_CACHE_DIR": "/root/.inductor-cache",
        "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
    }
)

inference_image = inference_image.add_local_python_source(
    "utils",
)

app = modal.App("headshots-inference", image=inference_image)

with inference_image.imports():
    import torch
    from diffusers import StableDiffusionXLInpaintPipeline, DPMSolverMultistepScheduler
    from PIL import Image
    from utils.format_image import resize_pad_image
    import tempfile
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
    with_hair_mask: bool = modal.parameter(default=False)

    @modal.enter()
    def enter(self):
        pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        )

        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.safety_checker = None

        if self.with_lora:
            print(f"Loading LoRA weights")
            pipe.load_lora_weights(
                HG_HEADSHOTS_LORA_ID, weight_name=HG_HEADSHOTS_WEIGHT_NAME
            )
            print("LoRA weights loaded.")

        pipe = pipe.to("cuda")

        self.pipe = optimize(pipe, compile=self.compile)

    @modal.method()
    def inference(self, prompt: str, input_image) -> bytes:
        print("Generating image with prompt:", prompt)

        input_image = resize_pad_image(input_image)

        tmp_dir = tempfile.gettempdir()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            input_image.save(temp_file)
            temp_file_path = temp_file.name
            print(f"Saved input image to temporary file: {temp_file_path}")

            # modify the input image in place, mask will be stored at the same location
            if self.with_hair_mask:
                from utils.mask_hair import process
            else:
                from utils.mask import process

            process(
                filenames=[os.path.basename(temp_file_path)],
                input_dir=tmp_dir,
                output_mask_dir=tmp_dir,
            )
            print(f"Generated mask for input image @ {tmp_dir}")

        mask_image = Image.open(temp_file_path).convert("L").resize((WIDTH, HEIGHT))

        generator = torch.Generator(device=self.pipe.device)
        if self.fixed_seed:
            generator = generator.manual_seed(42)

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

        byte_stream = BytesIO()
        out.save(byte_stream, format="JPEG")
        return byte_stream.getvalue()


inference_caller_image = modal.Image.debian_slim().pip_install(
    "pillow==11.2.1", "supabase==2.15.2"
)


@app.function(
    image=inference_caller_image,
    secrets=[modal.Secret.from_name("supabase-credentials")],
)
def inference(
    request_id: str,
    prompt: str,
    input_image: Image.Image,
    with_lora: bool = True,
    fixed_seed: bool = False,
    with_hair_mask: bool = False,
) -> None:
    """
    Run inference on the model with the given prompt and input image.
    """
    output_bytes = Model(
        with_lora=with_lora,
        fixed_seed=fixed_seed,
        with_hair_mask=with_hair_mask,
    ).inference.remote(prompt, input_image)

    # Save the output image to a temporary file

    from supabase import create_client, Client

    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")
    supabase: Client = create_client(supabase_url, supabase_key)

    file_path = f"{request_id}/1.jpg"

    bucket_name = "headshots"
    response = supabase.storage.from_(bucket_name).upload(
        path=file_path, file=output_bytes, file_options={"content-type": "image/jpeg"}
    )

    print("Uploaded output to:", response.full_path)


@app.local_entrypoint()
def main(
    prompt: str = "A professional headshot of a person wearing a suit, high quality, studio lighting",
    twice: bool = True,
    compile: bool = False,
):
    input_image_path = Path(__file__).parent / "../train" / "images" / "image_0001.jpg"
    input_image = Image.open(input_image_path).convert("RGB")

    t0 = time.time()
    image_bytes = Model(compile=compile).inference.remote(prompt, input_image)
    print(f"First inference latency: {time.time() - t0:.2f} seconds")

    if twice:
        t0 = time.time()
        image_bytes = Model(compile=compile).inference.remote(prompt, input_image)
        print(f"Second inference latency: {time.time() - t0:.2f} seconds")

    output_path = Path("/tmp") / "flux" / "output.jpg"
    output_path.parent.mkdir(exist_ok=True, parents=True)
    print(f"Saving output to {output_path}")
    output_path.write_bytes(image_bytes)


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

from fastapi import Form, UploadFile, File
from PIL import Image
from io import BytesIO
import uuid

import modal

web_image = modal.Image.debian_slim().pip_install(
    "fastapi[standard]",
    "pillow==11.2.1",
)

app = modal.App("headshots-web", image=web_image)


@app.function(image=web_image)
@modal.fastapi_endpoint(method="POST", label="headshots-v11-inference")
async def web_inference(
    prompt: str = Form(...), 
    file: UploadFile = File(...), 
    with_lora: bool = Form(True),  # Default to True for LoRA usage,
    fixed_seed: bool = Form(False),  # Default to False for fixed seed
    with_hair_mask: bool = Form(False)
):
    contents = await file.read()
    input_image = Image.open(BytesIO(contents))

    headshots_model = modal.Function.from_name("headshots-inference", "inference")

    request_id = str(uuid.uuid4())
    call = headshots_model.spawn(
        request_id, prompt=prompt,
        input_image=input_image,
        with_lora=with_lora,
        fixed_seed=fixed_seed,
        with_hair_mask=with_hair_mask,
    )

    return {"call_id": call.object_id, "request_id": request_id}

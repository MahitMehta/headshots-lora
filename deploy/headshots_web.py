from fastapi import Depends, Form, HTTPException, UploadFile, File, status
from PIL import Image
from io import BytesIO
import uuid
import os
from supabase import create_client, Client

from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import modal

web_image = modal.Image.debian_slim().pip_install(
    "fastapi[standard]",
    "pillow==11.2.1",
    "supabase==2.15.2"
)

app = modal.App("headshots-web", image=web_image)

auth_scheme = HTTPBearer()

@app.function(
    image=web_image,
    secrets=[modal.Secret.from_name("supabase-credentials")])
@modal.fastapi_endpoint(method="POST", label="headshots-v11-inference")
async def web_inference(
    token: HTTPAuthorizationCredentials = Depends(auth_scheme),
    prompt: str = Form(...), 
    file: UploadFile = File(...), 
    with_lora: bool = Form(True),  # Default to True for LoRA usage,
    fixed_seed: bool = Form(False),  # Default to False for fixed seed
    with_hair_mask: bool = Form(False)
):
    supabase_jwt = token.credentials

    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")
    supabase: Client = create_client(supabase_url, supabase_key)

    try:
        user_response = supabase.auth.get_user(supabase_jwt)
    except Exception as e:
        user_response = None

    if not user_response or not user_response.user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Imvalid Bearer Token",
            headers={"WWW-Authenticate": "Bearer"},
        )

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

    return { "call_id": call.object_id, "request_id": request_id}

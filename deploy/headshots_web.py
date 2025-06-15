from io import BytesIO
import os
import uuid

from PIL import Image
from supabase import create_client, Client
from postgrest.exceptions import APIError as PostgrestAPIError
import modal

import fastapi
from fastapi import Depends, Form, HTTPException, UploadFile, File, status, Query
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.middleware.cors import CORSMiddleware

web_image = (
    modal.Image.debian_slim()
    .pip_install(
        "fastapi[standard]",
        "pillow==11.2.1",
        "supabase==2.15.2",
    )
    .add_local_python_source("utils.types", "utils.db")
)

with web_image.imports():
    from utils.types.input import Gender, AspectRatio
    from utils.db.headshots import set_request_status

app = modal.App("headshots-web", image=web_image)
web_app = fastapi.FastAPI()
auth_scheme = HTTPBearer()


@app.function(image=web_image, secrets=[modal.Secret.from_name("supabase-credentials")])
@modal.concurrent(max_inputs=8)
@modal.asgi_app(label="headshots-v11")
def fastapi_app():
    return web_app


supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)  # type: ignore


web_app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://www.mahitm.com",
        "https://mahitm.com",
        # for local development, loose auth is alright because we have Bearer token auth
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@web_app.post("/trigger-inference")
async def trigger_inference(
    token: HTTPAuthorizationCredentials = Depends(auth_scheme),
    gender: Gender = Form(...),
    file: UploadFile = File(...),
    with_lora: bool = Form(True),  # Default to True for LoRA usage,
    fixed_seed: bool = Form(False),  # Default to False for fixed seed
    with_hair_mask: bool = Form(False),
    aspect_ratio: AspectRatio = Form("2:3"),
):
    supabase_jwt = token.credentials

    try:
        user_response = supabase.auth.get_user(supabase_jwt)
    except Exception:
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
        user_id=user_response.user.id,
        request_id=request_id,
        gender=gender,
        input_image=input_image,
        with_lora=with_lora,
        fixed_seed=fixed_seed,
        with_hair_mask=with_hair_mask,
        aspect_ratio=aspect_ratio,
    )

    try:
        (
            supabase.table("headshots")
            .insert(
                {
                    "id": request_id,
                    "user_id": user_response.user.id,
                    "cost": 1,
                    "status": "pending",
                }
            )
            .execute()
        )
    except PostgrestAPIError:
        return fastapi.responses.JSONResponse(
            content={"error", "Failed to create request in database"},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    return {"call_id": call.object_id, "request_id": request_id}


@web_app.get("/result")
async def get_job_result_endpoint(
    call_id: str = Query(...),
    request_id: str = Query(...),
    token: HTTPAuthorizationCredentials = Depends(auth_scheme),
):
    supabase_jwt = token.credentials

    try:
        user_response = supabase.auth.get_user(supabase_jwt)
    except Exception:
        user_response = None

    if not user_response or not user_response.user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Imvalid Bearer Token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    function_call = modal.FunctionCall.from_id(call_id)
    try:
        result, message = function_call.get(timeout=0)
        print("Function call result:", result)

        if not result or len(result) == 0:
            set_request_status(
                supabase,
                request_id=request_id,
                user_id=user_response.user.id,
                status="error",
            )

            return fastapi.responses.JSONResponse(
                content={"status": "error", "message": message},
                status_code=200,
            )

        return fastapi.responses.JSONResponse(
            content={"status": "success", "object_paths": result},
            status_code=200,
        )
    except modal.exception.OutputExpiredError:
        return fastapi.responses.JSONResponse(
            content={"status": "unavailable"}, status_code=404
        )
    except TimeoutError:
        return fastapi.responses.JSONResponse(
            content={"status": "pending"}, status_code=202
        )
    except Exception as e:
        print(f"Error while getting function call result: {e}")

        return fastapi.responses.JSONResponse(
            content={"status": "error"},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

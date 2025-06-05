from io import BytesIO

from PIL import Image
from supabase import Client

from utils.types.db import HeadshotsRequestStatus

REQUESTS_TABLE_NAME = "headshots"
HEADSHOTS_BUCKET_NAME = "headshots"


def set_request_status(
    supabase: Client, status: HeadshotsRequestStatus, request_id: str, user_id: str
) -> None:
    (
        supabase.table(REQUESTS_TABLE_NAME)
        .update(
            {
                "status": status,
            }
        )
        .eq("id", request_id)
        .eq("user_id", user_id)
        .execute()
    )


def _upload_headshot(supabase: Client, file_path: str, image: Image.Image) -> None:
    """
    Upload a single headshot image to the Supabase storage bucket.
    """
    byte_stream = BytesIO()
    image.save(byte_stream, format="JPEG")
    file_bytes = byte_stream.getvalue()

    supabase.storage.from_(HEADSHOTS_BUCKET_NAME).upload(
        path=file_path, file=file_bytes, file_options={"content-type": "image/jpeg"}
    )


def upload_multiple_headshots(
    supabase: Client, user_id: str, request_id: str, headshot_images: list[Image.Image]
) -> list[str]:
    """
    Upload multiple headshot images to the Supabase storage bucket.
    Return a list of file paths for the uploaded images.
    """
    file_paths = []
    for idx, image in enumerate(headshot_images):
        file_path = f"{user_id}/{request_id}/{idx}.jpg"
        _upload_headshot(supabase, file_path, image)

        file_paths.append(file_path)
        print(f"Uploaded headshot {idx} to {file_path}")

    return file_paths

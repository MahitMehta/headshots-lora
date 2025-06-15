import json
import os
from pathlib import Path
import tempfile

from PIL import Image

import vertexai
from vertexai.preview.generative_models import (
    GenerativeModel,
    Part,
    Image as VertexImage,
)
from google.oauth2 import service_account


from utils.format import InputImageFormatter
from utils.types.input import Gender

PROJECT_ID = "mahitm-headshots"
REGION = "us-central1"

service_account_info = json.loads(os.environ["SERVICE_ACCOUNT_JSON"])
credentials = service_account.Credentials.from_service_account_info(
    service_account_info
)

vertexai.init(project=PROJECT_ID, location=REGION, credentials=credentials)
generative_multimodal_model = GenerativeModel("gemini-2.0-flash-lite-001")


# TODO: remove face tilt
def get_inference_description_prompt(gender: Gender) -> str:
    focus_list = [
        "face direction",
    ]

    if gender == "female":
        focus_list.append("hair (long, bangs, ponytail, bun, straight, wavy, curly)")
        focus_list.append("hair color")
    elif gender == "male":
        focus_list.append(
            "hair color (black hair, brown hair, blonde hair, no hair, etc.)"
        )
        focus_list.append(
            "hair style (short hair, long hair, wavy hair, curly hair, or bald)"
        )
    else:
        focus_list.append("hair")

    # focus_list.append("accessories (glasses, earrings, etc.)")

    prompt = f"""Generate a concise description of the human face.
Focus on {", ".join(focus_list)}.
Do not include any other details like clothing, background, or environment.
No markdown or special characters, just plain text. 
All comma-separated, no periods or other punctuation.
"""

    return prompt


def get_inference_description(gender: Gender, input_image: Image.Image) -> list[str]:
    compressed_image_path = tempfile.gettempdir() + "/compressed_image.jpg"
    input_image.resize((512, 512), Image.Resampling.LANCZOS)
    input_image.save(compressed_image_path, format="JPEG")

    image_part = Part.from_image(
        VertexImage.load_from_file(compressed_image_path),
    )
    try:
        prompt = get_inference_description_prompt(gender)
        response = generative_multimodal_model.generate_content(
            [prompt, image_part],
            stream=False,
        )
        if response.usage_metadata:
            print(
                "Inference Description Token Count: ",
                response.usage_metadata.total_token_count,
            )

        if not response.text:
            print("No description generated.")
            return []

        return [token.lower() for token in response.text.strip().split(", ")]
    except Exception as e:
        print(f"Failed to generate inference description: {e}")
        return []


if __name__ == "__main__":
    import time

    input_image_path = Path(__file__).parent / "../../train/images/image_0000.jpg"
    input_image = Image.open(input_image_path)

    with InputImageFormatter() as formatter:
        input_image, _, _ = formatter.get_model_inputs(input_image)

        start = time.time()
        description = get_inference_description("male", input_image)
        print(f"Caption generation took {time.time() - start:.2f} seconds")
        print(description)

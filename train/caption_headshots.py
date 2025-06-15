import os
import json

from google.oauth2 import service_account
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part, Image

MODEL_ID = "mahitm-headshots-v1.1"

PROJECT_ID = "mahitm-headshots"
REGION = "us-central1"

service_account_info = json.loads(os.environ["SERVICE_ACCOUNT_JSON"])
credentials = service_account.Credentials.from_service_account_info(
    service_account_info
)

vertexai.init(project=PROJECT_ID, location=REGION, credentials=credentials)
generative_multimodal_model = GenerativeModel("gemini-2.0-flash-lite-001")

PROMPT = """
Describe the hair in a few words, including color, length, and style. Additionally, describe the facial border, such as beard or mustache, and any other notable features like glasses or accessories (if any).
Keep it short and readable, no markdown or special characters, just plain text. All comma separated, no periods or other punctuation.
The caption should be concise, ideally under 50 words, and suitable for LoRA training.
"""


def generate_caption(img_path):
    image_part = Part.from_image(Image.load_from_file(img_path))

    try:
        response = generative_multimodal_model.generate_content(
            [PROMPT, image_part], stream=False
        )
        print(response.usage_metadata.total_token_count, "tokens used for captioning")
        return f"{MODEL_ID}, {response.text.strip()}"
    except Exception as e:
        print(f"⚠️ Failed to caption {img_path}: {e}")
        return None

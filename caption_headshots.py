import os
import shutil
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part, Image

IMG_DIR = "training_images"
OUTPUT_DIR = "training_captions"
CAPTION_SUFFIX = ".txt"

PROJECT_ID = "mahitm-headshots"
REGION = "us-central1"

MODEL_ID = "mahitm-headshot-v1"

vertexai.init(project=PROJECT_ID, location=REGION)
generative_multimodal_model = GenerativeModel("gemini-2.0-flash-lite-001")

PROMPT = """
Describe this photo as if it's a professional headshot for use in an AI art prompt. Include hair type, gender if obvious, camera angle, lighting, background, and anything that helps guide a LoRA model.

Don't include details about face details, expressions, or emotions. Focus on clothing and overall appearance.

Keep it short and readable, no markdown or special characters, just plain text.
The caption should be concise, ideally under 50 words, and suitable for training an AI model.
"""

def generate_caption(img_path):
    image_part = Part.from_image(Image.load_from_file(img_path))

    try:
        response = generative_multimodal_model.generate_content([PROMPT, image_part], stream=False)
        print(response.usage_metadata.total_token_count, "tokens used for captioning")
        return response.text.strip()
    except Exception as e:
        print(f"⚠️ Failed to caption {img_path}: {e}")
        return None

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for filename in os.listdir(IMG_DIR):
        if filename.lower().endswith(".jpg"):
            base = os.path.splitext(filename)[0]
            caption_path = os.path.join(OUTPUT_DIR, base + CAPTION_SUFFIX)
            copied_img_path = os.path.join(OUTPUT_DIR, filename)
            img_path = os.path.join(IMG_DIR, filename)

            if os.path.exists(caption_path):
                print(f"🟡 Skipping {filename} (already captioned)")
                continue

            print(f"🧠 Captioning {filename}...")
            caption = generate_caption(os.path.join(IMG_DIR, filename))
            if caption:
                #if not os.path.exists(copied_img_path):
                    #shutil.copy2(img_path, copied_img_path)
                    #print(f"📂 Copied {filename} to {OUTPUT_DIR}")

                with open(caption_path, "w") as f:
                    f.write(f"{MODEL_ID}, {caption}")
                print(f"✅ Saved: {caption_path}")
            else:
                print(f"❌ No caption for {filename}")

if __name__ == "__main__":
    main()

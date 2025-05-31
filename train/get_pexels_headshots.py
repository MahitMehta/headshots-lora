from pathlib import Path
import requests
import os
from PIL import Image
from io import BytesIO

from utils.format_image import resize_pad_image

from dotenv import load_dotenv
load_dotenv()

PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")

if not PEXELS_API_KEY:
    raise ValueError("PEXELS_API_KEY is not set. Please set your Pexels API key.")

OUTPUT_DIR = Path(__file__).parent / "pexels_images"
MIN_SIZE = 1024
PER_PAGE = 20
TOTAL_PAGES = 10
QUERY = "professional headshot"

headers = {"Authorization": PEXELS_API_KEY}
os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_and_process():
    img_count = 0
    for page in range(1, TOTAL_PAGES + 1):
        url = f"https://api.pexels.com/v1/search?query={QUERY}&per_page={PER_PAGE}&page={page}"
        res = requests.get(url, headers=headers)
        data = res.json()
        for photo in data.get("photos", []):
            try:
                img_url = photo['src']['original']
                img = Image.open(BytesIO(requests.get(img_url).content)).convert("RGB")
                
                # Skip tiny images
                if img.width < MIN_SIZE or img.height < MIN_SIZE:
                    continue

                # crop to square from the top-center 
                img = resize_pad_image(img)

                # Save image and caption
                fname = f"img_{img_count:04d}"
                img.save(os.path.join(OUTPUT_DIR, f"{fname}.jpg"))

                img_count += 1
                print(f"✅ Saved {fname}.jpg")

            except Exception as e:
                print(f"⚠️ Error: {e}")

if __name__ == "__main__":
    download_and_process()

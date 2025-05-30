import requests
import os
from PIL import Image
from io import BytesIO

from format_image import format_image

PEXELS_API_KEY = "29BplWSE96Jeh7L9az6B7YQL2GebKlS5h9BDmZ9nyEFlqwsxnOcxc2em"
OUTPUT_DIR = "training_images"
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
                img = format_image(img)

                # Save image and caption
                fname = f"img_{img_count:04d}"
                img.save(os.path.join(OUTPUT_DIR, f"{fname}.jpg"))

                img_count += 1
                print(f"✅ Saved {fname}.jpg")

            except Exception as e:
                print(f"⚠️ Error: {e}")

if __name__ == "__main__":
    download_and_process()

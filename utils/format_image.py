from PIL.Image import Image as ImageType
from PIL import Image


def format_image(img: ImageType, size: int = 1024) -> ImageType:
    """Format the image to a square (top-center crop)"""

    min_edge = min(img.size)
    left = (img.width - min_edge) // 2
    top = 0
    img = img.crop((left, top, left + min_edge, top + min_edge))

    img = img.resize((size, size), Image.LANCZOS)  # type: ignore

    return img


if __name__ == "__main__":
    import os

    input_dir = "training_raw"

    output_dir = "training_images"
    os.makedirs(output_dir, exist_ok=True)

    for idx, filename in enumerate(os.listdir(input_dir)):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            img = Image.open(os.path.join(input_dir, filename))
            img = format_image(img).convert("RGB")
            img.save(os.path.join(output_dir, f"image_{idx:04d}.jpg"))
            print(f"Formatted and saved {filename}")

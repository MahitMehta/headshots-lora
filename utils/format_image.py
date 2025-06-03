from pathlib import Path
from PIL.Image import Image as ImageType
from PIL import Image, ImageOps
import os

content_width = 682  # Width for 2:3 aspect ratio
content_height = 1024
final_size = 1024


# grey padding helps the model focus on the subject, not the borders
def resize_pad_image(img: ImageType, padding_color=(127, 127, 127)) -> ImageType:
    """Format the image to a square (top-center crop)"""

    img = img.convert("RGB")
    orig_w, orig_h = img.size
    new_w = int(orig_w * content_height / orig_h)
    resized = img.resize((new_w, content_height), Image.Resampling.LANCZOS)

    # Crop or pad width to 682 (centered)
    if new_w > content_width:
        left = (new_w - content_width) // 2
        right = left + content_width
        content = resized.crop((left, 0, right, content_height))
    elif new_w < content_width:
        delta_w = content_width - new_w
        pad = (delta_w // 2, 0, delta_w - delta_w // 2, 0)
        content = ImageOps.expand(resized, pad, fill=padding_color)
    else:
        content = resized

    final_pad = ((final_size - content_width) // 2, 0)
    square = Image.new("RGB", (final_size, final_size), padding_color)
    square.paste(content, (final_pad[0], final_pad[1]))

    return square


# TODO: handle when input image doesn't have a face
def get_image_inputs(
    input_image: Image.Image, with_hair_mask=False
) -> tuple[Image.Image, Image.Image]:
    # matches pixel orientation with the EXIF data (metadata found commonly in photos taken by cameras or smartphones)
    input_image = ImageOps.exif_transpose(input_image)  # handle EXIF orientation
    input_image = resize_pad_image(input_image)

    import tempfile

    tmp_dir = tempfile.gettempdir()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        input_image.save(temp_file)
        temp_file_path = temp_file.name
        print(f"Saved input image to temporary file: {temp_file_path}")

        # modify the input image in place, mask will be stored at the same location
        if with_hair_mask:
            from utils.mask_hair import process
        else:
            from utils.mask import process

        process(
            filenames=[os.path.basename(temp_file_path)],
            input_dir=tmp_dir,
            output_mask_dir=tmp_dir,
        )
        print(f"Generated mask for input image @ {tmp_dir}")

    mask_image = (
        Image.open(temp_file_path).convert("L").resize((final_size, final_size))
    )
    return input_image, mask_image


if __name__ == "__main__":
    input_dir = Path(__file__).parent / "../train/images_raw"

    output_dir = Path(__file__).parent / "../train/images"
    os.makedirs(output_dir, exist_ok=True)

    for idx, filename in enumerate(os.listdir(input_dir)):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            img = Image.open(os.path.join(input_dir, filename))
            img = resize_pad_image(img).convert("RGB")
            img.save(os.path.join(output_dir, f"image_{idx:04d}.jpg"))
            print(f"Formatted and saved {filename}")

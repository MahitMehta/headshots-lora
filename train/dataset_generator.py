import os
from pathlib import Path
import shutil
from PIL import Image

from train.caption_headshots import generate_caption
from utils.format import InputImageFormatter

raw_image_folder = Path(__file__).parent / "images_raw"
formatted_image_folder = Path(__file__).parent / "images"
mask_folder = Path(__file__).parent / "masks"

caption_file_extension = ".txt"
caption_folder = Path(__file__).parent / "captions"

merged_images_folder = Path(__file__).parent / "formatted_dataset" / "images"
merged_conditioning_folder = Path(__file__).parent / "formatted_dataset" / "masks"


valid_formats = (".jpg", ".jpeg", ".png")


def merge_inputs():
    os.makedirs(merged_images_folder, exist_ok=True)
    os.makedirs(merged_conditioning_folder, exist_ok=True)

    for filename in os.listdir(formatted_image_folder):
        name, _ext = os.path.splitext(filename)

        image_src_path = formatted_image_folder / filename
        caption_src_path = caption_folder / f"{name}.txt"
        mask_src_path = mask_folder / filename

        # check if caption file exists
        if not os.path.isfile(caption_src_path):
            print(f"Warning: Caption file not found for {filename}, skipping.")
            continue

        # check if mask file exists
        if not os.path.isfile(mask_src_path):
            print(f"Warning: Mask file not found for {filename}, skipping.")
            continue

        image_dst_path = os.path.join(merged_images_folder, filename)
        if os.path.isfile(image_src_path):
            shutil.copy2(
                caption_src_path, os.path.join(merged_images_folder, name + ".txt")
            )
            shutil.copy2(image_src_path, image_dst_path)

            shutil.copy2(
                mask_src_path, os.path.join(merged_conditioning_folder, filename)
            )


def format_images(formatter: InputImageFormatter):
    if raw_image_folder.exists():
        print(f"Raw images found in {raw_image_folder}. Formatting images...")
        os.makedirs(formatted_image_folder, exist_ok=True)

        for filename in os.listdir(raw_image_folder):
            if filename.lower().endswith(valid_formats):
                img = Image.open(raw_image_folder / filename)
                formatter.preprocess_input_image(img)
                img.save(formatted_image_folder / filename)
            else:
                print(f"Skipping {filename}, not a valid image file.")
                continue
    else:
        print(
            f"No raw images found in {raw_image_folder}. Using existing formatted images."
        )
        return


def generate_masks(formatter: InputImageFormatter):
    if formatted_image_folder.exists():
        print(f"Generating masks for formatted images in {formatted_image_folder}...")
        os.makedirs(mask_folder, exist_ok=True)

        for filename in os.listdir(formatted_image_folder):
            if filename.lower().endswith(valid_formats):
                img = Image.open(formatted_image_folder / filename)
                mask = formatter.mask(img)

                if mask is None:
                    print(f"Warning: No mask generated for {filename}. Skipping.")
                    continue

                mask.save(mask_folder / filename)
            else:
                print(f"Skipping {filename}, not a valid image file.")
    else:
        print(
            f"No formatted images found in {formatted_image_folder}. Cannot generate masks."
        )
        return


def generate_captions():
    os.makedirs(caption_folder, exist_ok=True)

    for filename in os.listdir(formatted_image_folder):
        if filename.lower().endswith(".jpg"):
            base = os.path.splitext(filename)[0]
            caption_path = caption_folder / f"{base}{caption_file_extension}"

            if os.path.exists(caption_path):
                print(f"Skipping {filename} (already captioned)")
                continue

            print(f"Captioning {filename}...")
            input_image_path = formatted_image_folder / filename
            caption = generate_caption(input_image_path)
            if caption:
                with open(caption_path, "w") as f:
                    f.write(caption)
                print(f"✅ Saved: {caption_path}")
            else:
                print(f"❌ No caption for {filename}")


if __name__ == "__main__":
    with InputImageFormatter() as formatter:
        format_images(formatter)
        generate_masks(formatter)

    generate_captions()
    merge_inputs()

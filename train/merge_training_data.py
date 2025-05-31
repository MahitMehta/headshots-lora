import os
import shutil

image_folder = "training_images"
dst_folder = "training_data/images"
conditioning_folder = "training_data/masks"

caption_folder = "training_captions"
mask_folder = "training_masks"

os.makedirs(dst_folder, exist_ok=True)
os.makedirs(conditioning_folder, exist_ok=True)

for filename in os.listdir(image_folder):
    name, ext = os.path.splitext(filename)

    image_src_path = os.path.join(image_folder, filename)
    caption_src_path = os.path.join(caption_folder, name + ".txt")
    mask_src_path = os.path.join(mask_folder, filename)

    # check if caption file exists
    if not os.path.isfile(caption_src_path):
        print(f"Warning: Caption file not found for {filename}, skipping.")
        continue

    # check if mask file exists
    if not os.path.isfile(mask_src_path):
        print(f"Warning: Mask file not found for {filename}, skipping.")
        continue

    image_dst_path = os.path.join(dst_folder, filename)
    if os.path.isfile(image_src_path):
        shutil.copy2(caption_src_path, os.path.join(dst_folder, name + ".txt"))
        shutil.copy2(image_src_path, image_dst_path)

        shutil.copy2(mask_src_path, os.path.join(conditioning_folder, filename))

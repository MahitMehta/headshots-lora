import cv2
import numpy as np
import os
import mediapipe as mp
from PIL import Image

from utils.format_image import resize_pad_image


def _generate_mask(input_dir, filename, output_dir, output_filename, face_mesh, inset: float = 0.0):
    img_path = os.path.join(input_dir, filename)
    image = cv2.imread(img_path)
    if image is None:
        print(f"Failed to load {filename}")
        return

    height, width, _ = image.shape
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.multi_face_landmarks:
        print(f"No face detected in {filename}")
        return

    # Create white mask
    mask = np.ones((height, width), dtype=np.uint8) * 255

    for face_landmarks in results.multi_face_landmarks:
        # Extract the 2D landmark points (normalized)
        points = [
            (int(lm.x * width), int(lm.y * height)) for lm in face_landmarks.landmark
        ]

        # Create a polygon mask using these points
        hull = cv2.convexHull(np.array(points))

        # Fill polygon with black (face area)
        if inset > 0.0:
            hull_points = hull.squeeze()
            centroid = np.mean(hull_points, axis=0)
            scale = 1.0 - inset

            scaled_hull = ((hull_points - centroid) * scale + centroid).astype(np.int32)
            scaled_hull = scaled_hull.reshape((-1, 1, 2))

            cv2.fillConvexPoly(mask, scaled_hull, 0) # type: ignore 
        else:
            cv2.fillConvexPoly(mask, hull, 0)  # type: ignore

    #  blur edges for smooth transition
    mask = cv2.GaussianBlur(mask, (45, 45), 0)

    mask_output_path = os.path.join(output_dir, output_filename)
    cv2.imwrite(mask_output_path, mask)

    # add black borders to maintain 2:3 aspect ratio
    padded_image = resize_pad_image(
        Image.open(mask_output_path).convert("L"),  # Convert to grayscale
        padding_color=(0, 0, 0),  # Black padding for masks
    )
    padded_image.save(mask_output_path)

    print(f"Mask saved: {mask_output_path}")


def generate_mask(
    filenames: list[str],
    input_dir,
    output_mask_dir,
    output_filename_prefix="",
    inset: float = 0.0,
):
    os.makedirs(output_mask_dir, exist_ok=True)

    print(f"Processing images from: {input_dir}")
    print(f"Saving masks to: {output_mask_dir}")

    mp_face_mesh = mp.solutions.face_mesh  # type: ignore
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

    for filename in filenames:
        output_filename = f"{output_filename_prefix}{filename}"
        _generate_mask(input_dir, filename, output_mask_dir, output_filename, face_mesh, inset)

    face_mesh.close()
    print("Processing complete.")


if __name__ == "__main__":
    from pathlib import Path

    input_dir = Path(__file__).parent / "../train/images"
    output_mask_dir = Path(__file__).parent / "../train/masks"

    filenames = [
        filename
        for filename in os.listdir(input_dir)
        if filename.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    generate_mask(filenames, input_dir, output_mask_dir)
    print("Mask generation complete.")

import cv2
import numpy as np
import os
import mediapipe as mp

# --- Configuration ---
segmentation_threshold = 0.5  # Threshold for selfie segmentation (0.0 to 1.0)
blur_kernel_size = (
    21,
    21,
)  # Kernel size for Gaussian blur, e.g., (21, 21). Use (0,0) or None to disable.
# Landmarks to define the bottom of the jaw/chin for the neck cutoff
# These are indices from the MediaPipe FaceMesh 468 landmarks.
# Points around the chin and lower jaw.
JAW_BOTTOM_LANDMARK_INDICES = [
    175,
    152,
    171,
    148,
    396,
    377,
    400,  # Chin and lower jaw center
    176,
    190,  # Right jaw (viewer's perspective)
    414,
    397,  # Left jaw (viewer's perspective)
]
NECK_CUTOFF_OFFSET_FACTOR = 0  # 0.02 # Percentage of image height to offset the cutoff line downwards from the lowest jaw point


def generate_mask(input_dir, filename, output_dir, selfie_segmentation, face_mesh):
    img_path = os.path.join(input_dir, filename)
    image = cv2.imread(img_path)
    if image is None:
        print(f"Failed to load {filename}")
        return

    height, width, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert for MediaPipe

    # 1. Perform Selfie Segmentation
    segmentation_results = selfie_segmentation.process(rgb_image)

    if segmentation_results.segmentation_mask is None:
        print(
            f"No segmentation mask produced for {filename} by SelfieSegmentation. Skipping."
        )
        return

    # Create a binary mask from selfie segmentation (True for person)
    person_condition = segmentation_results.segmentation_mask > segmentation_threshold

    # Initialize the output mask: white (255) for background/to-be-inpainted
    output_mask = np.ones((height, width), dtype=np.uint8) * 255
    # Set the segmented person area to black (0 - to be kept)
    output_mask[person_condition] = 0

    # 2. Perform Face Mesh Detection to find neck line
    face_mesh_results = face_mesh.process(rgb_image)
    neck_line_y = height  # Default to bottom of image if no face detected

    if face_mesh_results.multi_face_landmarks:
        for face_landmarks in (
            face_mesh_results.multi_face_landmarks
        ):  # Should be only one due to max_num_faces=1
            try:
                # Get y-coordinates of the specified jaw bottom landmarks
                jaw_points_y = [
                    int(face_landmarks.landmark[i].y * height)
                    for i in JAW_BOTTOM_LANDMARK_INDICES
                    if 0 <= i < len(face_landmarks.landmark)  # Check index bounds
                ]

                if jaw_points_y:
                    lowest_jaw_point_y = max(jaw_points_y)
                    # Add a small offset to cut slightly below the jaw
                    neck_line_y = lowest_jaw_point_y + int(
                        NECK_CUTOFF_OFFSET_FACTOR * height
                    )
                    neck_line_y = min(
                        neck_line_y, height
                    )  # Ensure it doesn't go beyond image boundary
                    print(
                        f"  Detected neck line for {filename} at y={neck_line_y} (lowest jaw_y={lowest_jaw_point_y})"
                    )
                else:
                    print(
                        f"  Could not extract specified jaw landmarks for {filename}. Using full person mask."
                    )
                    neck_line_y = height  # Keep full person if specific landmarks fail

            except Exception as e:
                print(
                    f"  Error processing face landmarks for {filename}: {e}. Using full person mask."
                )
                neck_line_y = height  # Keep full person if error

        # Make everything below the determined neck_line_y white (255)
        if (
            neck_line_y < height
        ):  # Only apply cutoff if a valid neck_line was found above image bottom
            output_mask[neck_line_y:height, :] = 255
    else:
        print(
            f"No face detected by FaceMesh in {filename}. Mask will retain full segmented person."
        )
        # If no face landmarks, output_mask remains as per selfie_segmentation (full person)

    # Optional: blur edges for smooth transition
    if blur_kernel_size and blur_kernel_size[0] > 0 and blur_kernel_size[1] > 0:
        output_mask = cv2.GaussianBlur(output_mask, blur_kernel_size, 0)

    mask_output_path = os.path.join(output_dir, filename)
    try:
        cv2.imwrite(mask_output_path, output_mask)
        print(f"Mask saved: {mask_output_path}")
    except Exception as e:
        print(f"Failed to save mask for {filename}: {e}")


def process(
    filenames: list[str],
    input_dir,
    output_mask_dir,
):
    mp_selfie_segmentation = mp.solutions.selfie_segmentation  # type: ignore
    selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(
        model_selection=0
    )  # model_selection=0 for general model

    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh  # type: ignore
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,  # Process each image independently
        max_num_faces=1,
        refine_landmarks=True,  # Get more accurate landmarks for lips, eyes, irises
        min_detection_confidence=0.5,
    )

    print(f"Processing images from: {input_dir}")
    print(f"Saving masks to: {output_mask_dir}")

    os.makedirs(output_mask_dir, exist_ok=True)

    for filename in filenames:
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        generate_mask(
            input_dir, filename, output_mask_dir, selfie_segmentation, face_mesh
        )

    selfie_segmentation.close()
    face_mesh.close()
    print("Processing complete.")


if __name__ == "__main__":
    input_dir = "training_images"
    filenames = [
        filename
        for filename in os.listdir(input_dir)
        if filename.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    process(
        filenames=filenames, input_dir=input_dir, output_mask_dir="training_masks_hair"
    )

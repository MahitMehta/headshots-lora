import cv2
import numpy as np
import os
import mediapipe as mp

input_dir = "training_images"
output_mask_dir = "training_masks"

os.makedirs(output_mask_dir, exist_ok=True)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

for filename in os.listdir(input_dir):
    if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    img_path = os.path.join(input_dir, filename)
    image = cv2.imread(img_path)
    if image is None:
        print(f"Failed to load {filename}")
        continue

    height, width, _ = image.shape
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.multi_face_landmarks:
        print(f"No face detected in {filename}")
        continue

    # Create white mask
    mask = np.ones((height, width), dtype=np.uint8) * 255

    for face_landmarks in results.multi_face_landmarks:
        # Extract the 2D landmark points (normalized)
        points = [(int(lm.x * width), int(lm.y * height)) for lm in face_landmarks.landmark]

        # Create a polygon mask using these points
        hull = cv2.convexHull(np.array(points))
        
        # Fill polygon with black (face area)
        cv2.fillConvexPoly(mask, hull, 0)

    # Optional: blur edges for smooth transition
    mask = cv2.GaussianBlur(mask, (31, 31), 0)

    mask_output_path = os.path.join(output_mask_dir, filename)
    cv2.imwrite(mask_output_path, mask)
    print(f"Mask saved: {mask_output_path}")
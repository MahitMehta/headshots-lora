import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageOps
from PIL.Image import Image as PIL_Image

from utils.types.input import AspectRatio

# Face mask with hair config
segmentation_threshold = 0.5  # Threshold for selfie segmentation (0.0 to 1.0)
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
NECK_CUTOFF_OFFSET_FACTOR = 0  # Percentage of image height to offset the cutoff line downwards from the lowest jaw point


class InputImageFormatter:
    content_width = 682  # Default: 2:3 aspect ratio
    content_height = 1024
    target_dim = 1024
    selfie_segmentation = None

    def __init__(self):
        mp_face_mesh = mp.solutions.face_mesh  # type: ignore
        self.face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

    def _resize_pad_image(
        self, img: PIL_Image, padding_color=(127, 127, 127)
    ) -> PIL_Image:
        """
        Format the image to a square (top-center crop)

        :param padding_color: Default is grey (helps the model focus on the subject, not the borders)
        """

        img = img.convert("RGB")
        orig_w, orig_h = img.size
        new_w = int(orig_w * self.content_height / orig_h)
        resized = img.resize((new_w, self.content_height), Image.Resampling.LANCZOS)

        # Crop or pad width to 682 (centered)
        if new_w > self.content_width:
            left = (new_w - self.content_width) // 2
            right = left + self.content_width
            content = resized.crop((left, 0, right, self.content_height))
        elif new_w < self.content_width:
            delta_w = self.content_width - new_w
            pad = (delta_w // 2, 0, delta_w - delta_w // 2, 0)
            content = ImageOps.expand(resized, pad, fill=padding_color)
        else:
            content = resized

        final_pad = ((self.target_dim - self.content_width) // 2, 0)
        square = Image.new("RGB", (self.target_dim, self.target_dim), padding_color)
        square.paste(content, (final_pad[0], final_pad[1]))

        return square

    def mask_with_hair(self, input_image: PIL_Image) -> PIL_Image | None:
        if self.selfie_segmentation is None:
            raise ValueError(
                "Selfie Segmentation model is not initialized. Set with_hair_mask=True when creating InputImageFormatter."
            )

        image = np.array(input_image)

        height, width, _ = image.shape
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert for MediaPipe

        # Perform Selfie Segmentation
        segmentation_results = self.selfie_segmentation.process(rgb_image)

        if segmentation_results.segmentation_mask is None:
            return None

        # Create a binary mask from selfie segmentation (True for person)
        person_condition = (
            segmentation_results.segmentation_mask > segmentation_threshold
        )

        # Initialize the output mask: white (255) for background/to-be-inpainted
        output_mask = np.ones((height, width), dtype=np.uint8) * 255
        # Set the segmented person area to black (0 - to be kept)
        output_mask[person_condition] = 0

        # 2. Perform Face Mesh Detection to find neck line
        face_mesh_results = self.face_mesh.process(rgb_image)
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
                            f"Detected neck line at y={neck_line_y} (lowest jaw_y={lowest_jaw_point_y})"
                        )
                    else:
                        print(
                            "Could not extract specified jaw landmarks. Using full person mask."
                        )
                        neck_line_y = (
                            height  # Keep full person if specific landmarks fail
                        )

                except Exception as e:
                    print(
                        f"Error processing face landmarks: {e}. Using full person mask."
                    )
                    neck_line_y = height  # Keep full person if error

            # Make everything below the determined neck_line_y white (255)
            if (
                neck_line_y < height
            ):  # Only apply cutoff if a valid neck_line was found above image bottom
                output_mask[neck_line_y:height, :] = 255

            # use bottom of fash mesh hull convex poly to remove any clothing below and outline a sharp jawline
            for face_landmarks in face_mesh_results.multi_face_landmarks:
                # Extract the 2D landmark points (normalized)
                points = [
                    (int(lm.x * width), int(lm.y * height))
                    for lm in face_landmarks.landmark
                ]

                # Create a polygon mask using these points
                hull = cv2.convexHull(np.array(points))
                blank_mask = np.ones((height, width), dtype=np.uint8) * 255
                cv2.fillConvexPoly(blank_mask, hull, 0)  # type: ignore

                for x in range(width):
                    # Find the lowest (largest y value) black pixel in this column
                    y_coords = np.where(blank_mask[:, x] == 0)[0]
                    if len(y_coords) > 0:
                        lowest_y = np.max(y_coords)
                        # Make everything below this point white in the output mask
                        output_mask[lowest_y + 1 : height, x] = 255

        else:
            return None  # No face detected by FaceMesh

        # Blur edges for smooth transition
        output_mask = cv2.GaussianBlur(output_mask, (45, 45), 0)

        # Save the mask as a PIL Image
        mask_image = Image.fromarray(output_mask)

        # add black borders to aspect ratio
        padded_image = self._resize_pad_image(
            mask_image,
            padding_color=(0, 0, 0),  # Black padding for masks
        )

        return padded_image

    def mask(self, input_image: PIL_Image, inset: float = 0.0) -> PIL_Image | None:
        image = np.array(input_image)

        height, width, _ = image.shape
        results = self.face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.multi_face_landmarks:
            return None

        # Create white mask
        mask = np.ones((height, width), dtype=np.uint8) * 255

        for face_landmarks in results.multi_face_landmarks:
            # Extract the 2D landmark points (normalized)
            points = [
                (int(lm.x * width), int(lm.y * height))
                for lm in face_landmarks.landmark
            ]

            # Create a polygon mask using these points
            hull = cv2.convexHull(np.array(points))

            # Fill polygon with black (face area)
            if inset > 0.0:
                hull_points = hull.squeeze()
                centroid = np.mean(hull_points, axis=0)
                scale = 1.0 - inset

                scaled_hull = ((hull_points - centroid) * scale + centroid).astype(
                    np.int32
                )
                scaled_hull = scaled_hull.reshape((-1, 1, 2))

                cv2.fillConvexPoly(mask, scaled_hull, 0)  # type: ignore
            else:
                cv2.fillConvexPoly(mask, hull, 0)  # type: ignore

        #  blur edges for smooth transition
        mask = cv2.GaussianBlur(mask, (45, 45), 0)

        # Save the mask as an PIL Image
        mask_image = Image.fromarray(mask)

        # add black borders to aspect ratio
        padded_image = self._resize_pad_image(
            mask_image,
            padding_color=(0, 0, 0),  # Black padding for masks
        )

        return padded_image

    def set_aspect_ratio(self, aspect_ratio: AspectRatio):
        if aspect_ratio == "2:3":
            self.content_width = 682
            self.content_height = 1024
        elif aspect_ratio == "1:1":
            self.content_width = 1024
            self.content_height = 1024

    def preprocess_input_image(self, input_image: PIL_Image) -> PIL_Image:
        input_image = input_image.convert("RGB")

        # matches pixel orientation with the EXIF data (metadata found commonly in photos taken by cameras or smartphones)
        input_image = ImageOps.exif_transpose(input_image)
        input_image = self._resize_pad_image(input_image)
        return input_image

    def load_selfie_segmentation(self):
        mp_selfie_segmentation = mp.solutions.selfie_segmentation  # type: ignore
        self.selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(
            model_selection=0
        )

    def get_model_inputs(
        self, input_image: PIL_Image, with_hair_mask: bool = False
    ) -> tuple[PIL_Image, PIL_Image | None, PIL_Image | None]:
        input_image = self.preprocess_input_image(input_image)

        if with_hair_mask and self.selfie_segmentation is None:
            self.load_selfie_segmentation()

        if with_hair_mask:
            mask = self.mask_with_hair(input_image)
        else:
            mask = self.mask(input_image, inset=0.0)

        if mask is None:
            return (input_image, None, None)

        inset_mask = self.mask(input_image, inset=0.20)  # 20% inset for small mask
        if inset_mask is None:
            return (input_image, None, None)

        return input_image, mask, inset_mask

    def close(self):
        if self.selfie_segmentation is not None:
            self.selfie_segmentation.close()
        self.face_mesh.close()

    def __enter__(self):
        return self

    def __exit__(self, _type, _value, _traceback):
        self.close()


if __name__ == "__main__":
    from pathlib import Path

    with InputImageFormatter() as formatter:
        input_image = Image.open(Path(__file__).parent / "../output/mahit.jpeg")
        input_image, mask, _ = formatter.get_model_inputs(
            input_image, with_hair_mask=True
        )

        if mask is None:
            print("No mask generated.")
            exit()

        input_image.show("Input Image")
        mask.show("Mask")

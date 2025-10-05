# VideoBuilder class for building mp4 videos from images
from typing import List, Union
from PIL import Image

class VideoBuilder:
    def __init__(self, fps: int, width: int, height: int) -> None:
        """
        Initialize the VideoBuilder.
        Args:
            fps (int): Frames per second for the output video.
            width (int): Width of the video frames.
            height (int): Height of the video frames.
        """
        self.fps = fps
        self.width = width
        self.height = height
        self.frames: List[np.ndarray] = []

    def add_frame(self, image: Union[np.ndarray, Image.Image]) -> None:
        """
        Add an image frame to the video.
        Args:
            image (Union[np.ndarray, PIL.Image.Image]): Image to add (will be resized to (width, height) if needed).
        """
        if isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if image.shape[1] != self.width or image.shape[0] != self.height:
            image = cv2.resize(image, (self.width, self.height))
        self.frames.append(image)

    def save(self, output_path: str) -> None:
        """
        Save the added frames as an mp4 video and clear the frame buffer.
        Args:
            output_path (str): Path to save the mp4 video file.
        """
        if not self.frames:
            print("No frames to save.")
            return
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        for frame in self.frames:
            writer.write(frame)
        writer.release()
        self.frames.clear()
        print(f"Video saved to {output_path} and frame buffer cleared.")
# -*- coding: utf-8 -*-

import cv2
import numpy as np


def project_trajectory_to_image(
    image: np.ndarray,
    trajectory_points: list[list[float]],
    K: np.ndarray
) -> np.ndarray:
    """
    Project 3D trajectory points in the FLU world coordinate system onto the image.

    Args:
        image (np.ndarray): The image to draw the trajectory on (OpenCV BGR format)
        trajectory_points (list[list[float]]): List of trajectory points in the format [[x, y, z, yaw], ...]
        K (np.ndarray): 3x3 camera intrinsic matrix

    Returns:
        image_with_trajectory (np.ndarray): Image with the trajectory drawn
    """
    # Copy the image to avoid modifying the original
    image_with_trajectory = image.copy()
    h, w, _ = image_with_trajectory.shape
    
    # Extract parameters from the intrinsic matrix
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    projected_points = []

    for point in trajectory_points:
        x_w, y_w, z_w, _ = point  # yaw is not used here

        # Step 1: Coordinate transformation (FLU World -> RDF Camera)
        # FLU (X: forward, Y: left, Z: up) -> RDF (X: right, Y: down, Z: forward)
        x_c = -y_w
        y_c = -z_w
        z_c = x_w

        # Step 2: Check if the point is in front of the camera
        if z_c <= 0.1:  # Add a small threshold to avoid division by zero or projecting points behind the camera
            continue

        # Step 3: 3D to 2D projection
        u = int(fx * (x_c / z_c) + cx)
        v = int(fy * (y_c / z_c) + cy)

        # Step 4: Check if the point is within the image bounds
        if 0 <= u < w and 0 <= v < h:
            projected_points.append((u, v))

    # Step 5: Visualize the trajectory
    # Draw trajectory lines (color gradient from far to near to better express depth)
    num_points = len(projected_points)
    for i in range(num_points - 1):
        # Green intensity from 100 to 255
        green_intensity = int(100 + 155 * (i / max(1, num_points - 1)))
        color = (0, green_intensity, 0)
        cv2.line(image_with_trajectory, projected_points[i], projected_points[i+1], color, 2)

    # Draw trajectory points
    for pt in projected_points:
        cv2.circle(image_with_trajectory, pt, 3, (0, 0, 255), -1)  # Red solid circle

    return image_with_trajectory


def get_intrinsics(
    W: int,
    H: int,
    fov_x_deg: float = 84,
    fov_y_deg: float | None = None
) -> np.ndarray:
    """Construct camera intrinsic matrix K"""
    fov_x = np.deg2rad(fov_x_deg)
    if fov_y_deg is None:
        fov_y = 2 * np.arctan((H / W) * np.tan(fov_x / 2))
    else:
        fov_y = np.deg2rad(fov_y_deg)

    fx = W / (2 * np.tan(fov_x / 2))
    fy = H / (2 * np.tan(fov_y / 2))
    cx = W / 2
    cy = H / 2

    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0, 1]])
    return K


# --- Main program ---
if __name__ == '__main__':
    # 1. Define camera and image parameters
    # DJI Mavic 3T thermal camera parameters (estimated from public data)
    ORIGINAL_WIDTH = 640
    ORIGINAL_HEIGHT = 512
    # Pixel pitch 12μm -> sensor width = 640 * 0.012mm
    SENSOR_WIDTH_MM = 7.68
    FOCAL_LENGTH_MM = 9.1

    # Your dataset image size (after center crop)
    CROP_WIDTH = 256
    CROP_HEIGHT = 256

    # 2. Calculate the intrinsic matrix after cropping
    # K_matrix = calculate_intrinsics_for_crop(
    #     ORIGINAL_WIDTH, ORIGINAL_HEIGHT, SENSOR_WIDTH_MM, FOCAL_LENGTH_MM, CROP_WIDTH, CROP_HEIGHT
    # )
    # print("\nCalculated intrinsic matrix K after center crop:")
    K_matrix = get_intrinsics(CROP_WIDTH, CROP_HEIGHT, fov_x_deg=84)
    print(K_matrix)

    # 3. Load your image
    # Note: Replace 'path_to_your_image.jpg' with your actual image path
    # Here we create a black placeholder image for demonstration
    image_path = 'path_to_your_image.jpg'
    ego_view_image = cv2.imread(image_path)

    if ego_view_image is None:
        print(f"\nWarning: Unable to load image '{image_path}', using a black placeholder image.")
        ego_view_image = np.zeros((CROP_HEIGHT, CROP_WIDTH, 3), dtype=np.uint8)
    else:
        # Ensure the image size is correct
        if ego_view_image.shape[0] != CROP_HEIGHT or ego_view_image.shape[1] != CROP_WIDTH:
            print(f"Warning: Your image size is {ego_view_image.shape[:2]}, resizing to ({CROP_HEIGHT}, {CROP_WIDTH}).")
            ego_view_image = cv2.resize(ego_view_image, (CROP_WIDTH, CROP_HEIGHT))

    # 4. Define a sample trajectory (FLU coordinate system: forward, left, up), unit: meter
    # The trajectory first flies forward, then left, then up
    # (x: forward, y: left, z: up)
    trajectory_flu = [
        [0, 0, 0, 0],      # Start point
        [15, 3, 0.2, 0],  # 15m forward, 3m left, slight climb
        [20, 1, 0, 0],     # 20m forward, 1m left, back to level
        [25, -2, 0.2, 0],  # 25m forward, 2m right (y negative), slight climb
        [30, -3, 0.5, 0],  # 30m forward, 3m right, continue climbing
        [35, -1, 1.0, 0],  # 35m forward, 1m right, climb to 1m
        [40, 0, 1.5, 0],   # 40m forward, back to center, climb to 1.5m
    ]

    # 5. Perform projection and visualization
    result_image = project_trajectory_to_image(ego_view_image, trajectory_flu, K_matrix)

    # 6. Show the result
    # cv2.imshow('Drone Trajectory Projection (Cropped Image)', result_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Optional: Save the result image
    cv2.imwrite('trajectory_visualization_cropped.png', result_image)

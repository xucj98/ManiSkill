import cv2
import numpy as np
from typing import Union
from scipy.spatial.transform import Rotation as R_scipy


# visualize object pose
def visualize_pose(
        rgb: np.ndarray,
        pose: np.ndarray,
        intrinsic: np.ndarray,
        axes_len: float = 0.1
) -> np.ndarray:
    """
    Visualizes an object's pose in the camera coordinate system on an RGB image.

    Args:
        rgb (np.ndarray): The original image, shape (H, W, 3), RGB order.
                            Assumed to be in [0, 1] float or [0, 255] uint8.
        pose (np.ndarray): Object's pose in camera coordinates, shape (7,).
                             Represents (tx, ty, tz, qw, qx, qy, qz) where
                             (tx, ty, tz) is position and (qw, qx, qy, qz) is WXYZ quaternion.
        intrinsic (np.ndarray): Camera intrinsic matrix, shape (3, 3).
                                  [[fx, 0,  cx],
                                   [0,  fy, cy],
                                   [0,  0,  1 ]]
        axes_len (float): Length of the coordinate axes to be drawn for the object.

    Returns:
        np.ndarray: Image with the axes drawn, shape (H, W, 3), BGR order (for cv2.imshow).
    """

    # Make a writable copy to draw on
    output_img_rgb = rgb.copy()

    # 3. Parse pose: extract translation and rotation
    position = pose[:3]  # tx, ty, tz
    quaternion_wxyz = pose[3:]  # qw, qx, qy, qz

    # Scipy's Rotation expects quaternion in (x, y, z, w) order
    quaternion_xyzw = np.array([quaternion_wxyz[1], quaternion_wxyz[2], quaternion_wxyz[3], quaternion_wxyz[0]])
    try:
        rotation_matrix = R_scipy.from_quat(quaternion_xyzw).as_matrix()
    except Exception as e:
        print(f"Error converting quaternion: {e}. Quaternion was: {quaternion_xyzw}")
        # Return original image if pose is invalid
        return rgb

    # 4. Define 3D axes points in the object's local coordinate system
    #    Origin, X-end, Y-end, Z-end
    axes_points_object = np.array([
        [0, 0, 0],  # Origin
        [axes_len, 0, 0],  # X-axis endpoint
        [0, axes_len, 0],  # Y-axis endpoint
        [0, 0, axes_len]  # Z-axis endpoint
    ], dtype=np.float32)

    # 5. Transform these points to the camera coordinate system
    #    P_camera = R * P_object + t
    axes_points_camera = (rotation_matrix @ axes_points_object.T).T + position

    # 6. Project 3D points from camera coordinates to 2D image plane
    #    p_image_homogeneous = K @ P_camera
    #    (u, v) = (p_image_homogeneous[0]/p_image_homogeneous[2], p_image_homogeneous[1]/p_image_homogeneous[2])

    projected_points_2d_list = []
    valid_projection_mask = []

    for point_3d_cam in axes_points_camera:
        # Check if point is in front of the camera (Z > 0)
        if point_3d_cam[2] <= 1e-5:  # Add a small epsilon for stability
            valid_projection_mask.append(False)
            projected_points_2d_list.append(np.array([-1, -1], dtype=int))  # Placeholder for invalid points
            continue

        valid_projection_mask.append(True)
        # P_camera is (X, Y, Z)^T
        # K @ P_camera results in (u*Z, v*Z, Z)^T
        uvw = intrinsic @ point_3d_cam.reshape(3, 1)
        u = uvw[0, 0] / uvw[2, 0]
        v = uvw[1, 0] / uvw[2, 0]
        projected_points_2d_list.append(np.array([int(round(u)), int(round(v))], dtype=int))

    projected_points_2d = np.array(projected_points_2d_list)

    # 7. Draw the axes on the image
    #    Colors: X=Red, Y=Green, Z=Blue (BGR format for OpenCV)
    colors = {
        "x": (255, 0, 0),  # Red
        "y": (0, 255, 0),  # Green
        "z": (0, 0, 255)  # Blue
    }
    thickness = 1  # You can make this a parameter

    origin_2d = tuple(projected_points_2d[0])

    # Only draw if origin is validly projected
    if valid_projection_mask[0]:
        # Draw X-axis (Origin to X-endpoint)
        if valid_projection_mask[1]:
            x_axis_2d = tuple(projected_points_2d[1])
            cv2.line(output_img_rgb, origin_2d, x_axis_2d, colors["x"], thickness)

        # Draw Y-axis (Origin to Y-endpoint)
        if valid_projection_mask[2]:
            y_axis_2d = tuple(projected_points_2d[2])
            cv2.line(output_img_rgb, origin_2d, y_axis_2d, colors["y"], thickness)

        # Draw Z-axis (Origin to Z-endpoint)
        if valid_projection_mask[3]:
            z_axis_2d = tuple(projected_points_2d[3])
            cv2.line(output_img_rgb, origin_2d, z_axis_2d, colors["z"], thickness)
    else:
        # print("Warning: Object origin is behind or too close to the camera. Axes not drawn.")
        pass

    return output_img_rgb

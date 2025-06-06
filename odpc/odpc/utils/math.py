import torch
import numpy as np

from mani_skill.utils.geometry.rotation_conversions import (
    quaternion_multiply, quaternion_apply,
)


def pose_inv(pose: torch.Tensor) -> torch.Tensor:
    new_pose = pose.clone()
    new_pose[..., 4:] = -pose[..., 4:]
    new_pose[..., :3] = quaternion_apply(new_pose[..., 3:], -pose[..., :3])
    return new_pose


def _pose_multiply(pose1: torch.Tensor, pose2: torch.Tensor) -> torch.Tensor:
    p1, q1 = pose1[..., :3], pose1[..., 3:]
    p2, q2 = pose2[..., :3], pose2[..., 3:]
    new_q = quaternion_multiply(q1, q2)
    new_p = p1 + quaternion_apply(q1, p2)
    return torch.cat([new_p, new_q], dim=-1)


def pose_multiply(*args: torch.Tensor) -> torch.Tensor:
    if len(args) == 2:
        return _pose_multiply(args[0], args[1])
    return _pose_multiply(args[0], pose_multiply(*args[1:]))


def pose_to_matrix(pose: np.ndarray) -> np.ndarray:
    """Converts a 7D pose [tx, ty, tz, qw, qx, qy, qz] to a 4x4 matrix."""
    position = pose[:3]
    quaternion_wxyz = pose[3:] # Assuming w, x, y, z order

    # Normalize quaternion
    norm = np.linalg.norm(quaternion_wxyz)
    if norm < 1e-8: # Avoid division by zero for zero quaternion
        q = np.array([1.0, 0.0, 0.0, 0.0])
    else:
        q = quaternion_wxyz / norm
    
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]

    # Rotation matrix from quaternion
    R = np.zeros((3, 3), dtype=np.float32)
    R[0, 0] = 1 - 2 * (qy**2 + qz**2)
    R[0, 1] = 2 * (qx * qy - qw * qz)
    R[0, 2] = 2 * (qx * qz + qw * qy)
    R[1, 0] = 2 * (qx * qy + qw * qz)
    R[1, 1] = 1 - 2 * (qx**2 + qz**2)
    R[1, 2] = 2 * (qy * qz - qw * qx)
    R[2, 0] = 2 * (qx * qz - qw * qy)
    R[2, 1] = 2 * (qy * qz + qw * qx)
    R[2, 2] = 1 - 2 * (qx**2 + qy**2)

    # Homogeneous transformation matrix
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = position
    return T


def invert_matrix(T: np.ndarray) -> np.ndarray:
    """Inverts a 4x4 homogeneous transformation matrix."""
    R = T[:3, :3]
    t = T[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    T_inv = np.eye(4, dtype=np.float32)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv
    return T_inv


def project_points_to_pixels(points_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Projects 3D points in camera coordinates to 2D pixel coordinates.
    Args:
        points_cam: (N, 3) array of 3D points [X, Y, Z] in camera frame.
        K: (3, 3) camera intrinsic matrix (OpenCV format: fx,0,cx; 0,fy,cy; 0,0,1).
    Returns:
        pixels: (N, 2) array of 2D pixel coordinates [u, v] (or [x, y]).
                Returns NaN for points behind the camera or if Z is too small.
    """
    if points_cam.ndim == 1:
        points_cam = points_cam.reshape(1, -1) # Ensure (N,3)
    
    N = points_cam.shape[0]
    pixels = np.full((N, 2), np.nan, dtype=np.float32)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    for i in range(N):
        Xc, Yc, Zc = points_cam[i, 0], points_cam[i, 1], points_cam[i, 2]
        if Zc > 1e-3: # Point must be in front of the camera
            u = (fx * Xc / Zc) + cx
            v = (fy * Yc / Zc) + cy
            pixels[i] = [u, v]
    return pixels
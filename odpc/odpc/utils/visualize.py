import cv2
import numpy as np
from typing import Union, Optional
from scipy.spatial.transform import Rotation as R_scipy

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

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

def visualize_video_with_metric(
    video: np.ndarray, 
    metric_data: np.ndarray, 
    is_key_frame: Optional[np.ndarray] = None
):
    """
    创建一个交互式可视化窗口，同步显示视频、指标曲线和关键帧标记。
    通过拖动滑块，可以实时查看任意时刻的视频帧、曲线位置和指标数值。

    参数:
        - video (np.ndarray): 视频帧序列。
            - 形状: (t, h, w, 3)
            - 类型: uint8
        - metric_data (np.ndarray): 每帧对应的指标数据。
            - 形状: (t,)
            - 类型: float
        - is_key_frame (np.ndarray): 关键帧标记。
            - 形状: (t,)
            - 类型: bool
    
    """
    assert video.shape[0] == metric_data.shape[0], \
        f"视频的帧数 ({video.shape[0]}) 与指标数据的长度 ({metric_data.shape[0]}) 不匹配。"
    if is_key_frame is not None:
        assert video.shape[0] == is_key_frame.shape[0], \
            f"视频的帧数 ({video.shape[0]}) 与关键帧标记的长度 ({is_key_frame.shape[0]}) 不匹配。"

    total_frames = video.shape[0]
    x_data = np.arange(total_frames)

    # --- 1. 创建可视化布局 ---
    fig_height = 18
    if is_key_frame is not None:
        fig, (ax_video, ax_indicator, ax_plot) = plt.subplots(
            3, 1, 
            figsize=(32, fig_height), 
            gridspec_kw={'height_ratios': [10, 1, 6]}
        )
        plt.subplots_adjust(bottom=0.15, hspace=0.4)
    else:
        fig, (ax_video, ax_plot) = plt.subplots(
            2, 1, 
            figsize=(32, fig_height), 
            gridspec_kw={'height_ratios': [3, 2]}
        )
        plt.subplots_adjust(bottom=0.15, hspace=0.3)

    # --- 2. 初始化显示 ---
    im_video = ax_video.imshow(video[0])
    ax_video.set_title("Video Frame (Frame: 0)")
    ax_video.axis('off')

    ax_plot.plot(x_data, metric_data, lw=2, color='royalblue')
    ax_plot.set_xlim(0, total_frames - 1)
    ax_plot.set_title("Metric Curve")
    ax_plot.set_xlabel("Frame Number")
    ax_plot.set_ylabel("Metric Value")
    ax_plot.grid(True)
    vline_plot = ax_plot.axvline(x=0, color='red', linestyle='--', lw=2)
    annotation = ax_plot.text(0, metric_data[0], f'Value: {metric_data[0]:.2f}', bbox=dict(boxstyle="round,pad=0.4", fc="lemonchiffon", ec="black", lw=1))

    # --- 3. 绘制关键帧指示器 (如果提供) ---
    if is_key_frame is not None:
        # 创建一个 (1, T, 4) 的RGBA图像用于色带
        indicator_bar = np.zeros((1, total_frames, 4))
        indicator_bar[:, :, 3] = 1.0  # Alpha channel
        indicator_bar[:, is_key_frame, 0] = 1.0  # Red channel for key frames
        indicator_bar[:, ~is_key_frame, 1] = 0.5  # Green channel for normal frames (grayish)
        indicator_bar[:, ~is_key_frame, 2] = 0.5  # Blue channel for normal frames (grayish)

        ax_indicator.imshow(indicator_bar, aspect='auto', interpolation='nearest')
        ax_indicator.set_title("Key-Frame Indicator")
        ax_indicator.set_yticks([])
        ax_indicator.set_xlabel("Frame Number")
        vline_indicator = ax_indicator.axvline(x=0, color='white', linestyle='--', lw=2)

    # --- 4. 创建滑块控件 ---
    ax_slider = plt.axes([0.15, 0.05, 0.75, 0.03])
    slider = Slider(ax=ax_slider, label='Frame', valmin=0, valmax=total_frames - 1, valinit=0, valstep=1)

    # --- 5. 定义更新函数并绑定 ---
    def update(val):
        frame_idx = int(slider.val)
        im_video.set_data(video[frame_idx])
        ax_video.set_title(f"Video Frame (Frame: {frame_idx})")
        
        vline_plot.set_xdata([frame_idx])
        current_value = metric_data[frame_idx]
        annotation.set_position((frame_idx, current_value))
        annotation.set_text(f'Value: {current_value:.2f}')
        
        if is_key_frame is not None:
            vline_indicator.set_xdata([frame_idx])
        
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()
    plt.close(fig)
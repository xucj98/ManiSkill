import cv2
import matplotlib.pyplot as plt
import argparse
from typing import List, Any

import torch
from torch.utils.data.dataloader import DataLoader

import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

from mani_skill.utils import common

from diffusion_policy.data_converison import DataConversion, pose_multiply
from diffusion_policy.odpc_dataset import ODPCDataset
from diffusion_policy.utils import worker_init_fn


# visualize object pose
def visualize(rgb: torch.Tensor, pose: torch.Tensor, intrinsic: torch.Tensor, axes_len: float = 0.1) -> np.ndarray:
    """
    Visualizes an object's pose in the camera coordinate system on an RGB image.

    Args:
        rgb (torch.Tensor): The original image, shape (3, H, W), RGB order.
                            Assumed to be in [0, 1] float or [0, 255] uint8.
        pose (torch.Tensor): Object's pose in camera coordinates, shape (7,).
                             Represents (tx, ty, tz, qw, qx, qy, qz) where
                             (tx, ty, tz) is position and (qw, qx, qy, qz) is WXYZ quaternion.
        intrinsic (torch.Tensor): Camera intrinsic matrix, shape (3, 3).
                                  [[fx, 0,  cx],
                                   [0,  fy, cy],
                                   [0,  0,  1 ]]
        axes_len (float): Length of the coordinate axes to be drawn for the object.

    Returns:
        np.ndarray: Image with the axes drawn, shape (H, W, 3), BGR order (for cv2.imshow).
    """

    # 1. Convert input tensors to NumPy arrays on CPU
    img_tensor_chw = rgb.cpu()
    pose_np = pose.cpu().numpy()
    intrinsic_np = intrinsic.cpu().numpy()

    # 2. Prepare the image:
    #    - Permute to HWC
    #    - Convert to NumPy array
    #    - Normalize to 0-255 uint8 if it's float
    #    - Convert RGB to BGR for OpenCV drawing
    img_np_hwc_rgb = img_tensor_chw.permute(1, 2, 0).numpy()

    if img_np_hwc_rgb.dtype == np.float32 or img_np_hwc_rgb.dtype == np.float64:
        if img_np_hwc_rgb.max() <= 1.0:  # Assuming 0-1 range for float
            img_np_hwc_rgb = (img_np_hwc_rgb * 255).astype(np.uint8)
        else:  # Assuming 0-255 range for float
            img_np_hwc_rgb = img_np_hwc_rgb.astype(np.uint8)
    elif img_np_hwc_rgb.dtype != np.uint8:  # Other types, try to convert
        img_np_hwc_rgb = img_np_hwc_rgb.astype(np.uint8)

    # Ensure 3 channels (e.g., if input was grayscale but specified as 3 channels)
    if img_np_hwc_rgb.ndim == 2:  # Grayscale
        output_img_bgr = cv2.cvtColor(img_np_hwc_rgb, cv2.COLOR_GRAY2BGR)
    elif img_np_hwc_rgb.shape[2] == 1:  # Single channel image
        output_img_bgr = cv2.cvtColor(img_np_hwc_rgb, cv2.COLOR_GRAY2BGR)
    else:  # Assuming 3 channels RGB
        output_img_bgr = cv2.cvtColor(img_np_hwc_rgb, cv2.COLOR_RGB2BGR)

    # Make a writable copy to draw on
    output_img_bgr = output_img_bgr.copy()

    H, W = output_img_bgr.shape[:2]

    # 3. Parse pose: extract translation and rotation
    position = pose_np[:3]  # tx, ty, tz
    quaternion_wxyz = pose_np[3:]  # qw, qx, qy, qz

    # Scipy's Rotation expects quaternion in (x, y, z, w) order
    quaternion_xyzw = np.array([quaternion_wxyz[1], quaternion_wxyz[2], quaternion_wxyz[3], quaternion_wxyz[0]])
    try:
        rotation_matrix = R_scipy.from_quat(quaternion_xyzw).as_matrix()
    except Exception as e:
        print(f"Error converting quaternion: {e}. Quaternion was: {quaternion_xyzw}")
        # Return original image if pose is invalid
        return cv2.cvtColor(img_np_hwc_rgb, cv2.COLOR_RGB2BGR) if img_np_hwc_rgb.shape[2] == 3 else cv2.cvtColor(
            img_np_hwc_rgb, cv2.COLOR_GRAY2BGR)

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
        uvw = intrinsic_np @ point_3d_cam.reshape(3, 1)
        u = uvw[0, 0] / uvw[2, 0]
        v = uvw[1, 0] / uvw[2, 0]
        projected_points_2d_list.append(np.array([int(round(u)), int(round(v))], dtype=int))

    projected_points_2d = np.array(projected_points_2d_list)

    # 7. Draw the axes on the image
    #    Colors: X=Red, Y=Green, Z=Blue (BGR format for OpenCV)
    colors = {
        "x": (0, 0, 255),  # Red
        "y": (0, 255, 0),  # Green
        "z": (255, 0, 0)  # Blue
    }
    thickness = 2  # You can make this a parameter

    origin_2d = tuple(projected_points_2d[0])

    # Only draw if origin is validly projected
    if valid_projection_mask[0]:
        # Draw X-axis (Origin to X-endpoint)
        if valid_projection_mask[1]:
            x_axis_2d = tuple(projected_points_2d[1])
            cv2.line(output_img_bgr, origin_2d, x_axis_2d, colors["x"], thickness)

        # Draw Y-axis (Origin to Y-endpoint)
        if valid_projection_mask[2]:
            y_axis_2d = tuple(projected_points_2d[2])
            cv2.line(output_img_bgr, origin_2d, y_axis_2d, colors["y"], thickness)

        # Draw Z-axis (Origin to Z-endpoint)
        if valid_projection_mask[3]:
            z_axis_2d = tuple(projected_points_2d[3])
            cv2.line(output_img_bgr, origin_2d, z_axis_2d, colors["z"], thickness)
    else:
        print("Warning: Object origin is behind or too close to the camera. Axes not drawn.")

    return output_img_bgr


def plot(pred: torch.Tensor, gt: torch.Tensor, dim_names: List[str] = None):
    """
    Plots predicted data vs ground truth data over time, grouped into two subplots.
    The first subplot shows up to the first 3 dimensions.
    The second subplot shows the remaining dimensions.

    Args:
        pred (torch.Tensor): Predicted data, shape (T, n).
        gt (torch.Tensor): Ground truth data, shape (T, n).
        dim_names (List[str], optional): Names for each dimension.
                                         If None, dimensions will be named "Dim 1", "Dim 2", etc.
                                         Length must match n.
    """
    if pred.shape != gt.shape:
        raise ValueError(f"Shapes of pred {pred.shape} and gt {gt.shape} must match.")
    if pred.ndim != 2 or gt.ndim != 2:
        raise ValueError("Input tensors must be 2-dimensional (T, n).")

    T, n_total_dims = pred.shape

    pred_np = pred.cpu().detach().numpy()
    gt_np = gt.cpu().detach().numpy()
    time_steps = np.arange(T)

    if dim_names is None:
        dim_names = [f"Dimension {i + 1}" for i in range(n_total_dims)]
    elif len(dim_names) != n_total_dims:
        raise ValueError(f"Length of dim_names ({len(dim_names)}) must match n ({n_total_dims}).")

    # Always create two subplots as requested
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True, squeeze=False)
    axes: Any = axes.flatten()

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    # --- Plot 1: First up to 3 dimensions ---
    ax1 = axes[0]
    dims_in_plot1 = min(n_total_dims, 3)
    plot1_has_data = dims_in_plot1 > 0

    title_plot1_names = []
    if plot1_has_data:
        for i in range(dims_in_plot1):
            current_color = colors[i % len(colors)]  # Colors for first group
            ax1.plot(time_steps, gt_np[:, i], linestyle='-', color=current_color, label=f"{dim_names[i]} GT")
            ax1.plot(time_steps, pred_np[:, i], linestyle='--', color=current_color, label=f"{dim_names[i]} Pred")
            title_plot1_names.append(dim_names[i])
        ax1.set_title(f"Dimensions: {', '.join(title_plot1_names)}")
        ax1.legend(loc='best')
    else:
        ax1.set_title("First Group (No Data or n=0)")
        ax1.text(0.5, 0.5, "No data for this plot", ha='center', va='center', transform=ax1.transAxes, fontsize=12,
                 color='gray')
        if ax1.get_legend() is not None:  # Ensure legend is not shown if created implicitly
            ax1.get_legend().set_visible(False)

    ax1.set_ylabel("Value")
    ax1.grid(True, linestyle=':', alpha=0.7)

    # --- Plot 2: Remaining dimensions (dimensions from index 3 onwards) ---
    ax2 = axes[1]
    dims_in_plot2_start_index = 3  # Dimensions for the second plot start from index 3

    if n_total_dims > dims_in_plot2_start_index:
        dims_in_plot2_count = n_total_dims - dims_in_plot2_start_index
        title_plot2_names = []
        for i in range(dims_in_plot2_count):
            actual_dim_index = dims_in_plot2_start_index + i
            # Use colors that continue the cycle from the first plot's dimensions
            current_color = colors[actual_dim_index % len(colors)]
            ax2.plot(time_steps, gt_np[:, actual_dim_index], linestyle='-', color=current_color,
                     label=f"{dim_names[actual_dim_index]} GT")
            ax2.plot(time_steps, pred_np[:, actual_dim_index], linestyle='--', color=current_color,
                     label=f"{dim_names[actual_dim_index]} Pred")
            title_plot2_names.append(dim_names[actual_dim_index])
        ax2.set_title(f"Dimensions: {', '.join(title_plot2_names)}")
        ax2.legend(loc='best')
    else:
        ax2.set_title("Second Group (No further dimensions or n <= 3)")
        ax2.text(0.5, 0.5, "No data for this plot", ha='center', va='center', transform=ax2.transAxes, fontsize=12,
                 color='gray')
        if ax2.get_legend() is not None:  # Ensure legend is not shown
            ax2.get_legend().set_visible(False)

    ax2.set_xlabel("Time Step (T)")
    ax2.set_ylabel("Value")
    ax2.grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout, leave space for suptitle
    fig.suptitle("Prediction vs. Ground Truth Comparison (Grouped)", fontsize=16)
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-path", type=str, required=True)
    parser.add_argument("--num-traj", type=int, default=None)
    parser.add_argument("--obs-horizon", type=int, default=1)
    parser.add_argument("--pred-horizon", type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--visualize-pose', action='store_true')
    parser.add_argument('--plot-control', action='store_true')

    parser.add_argument('--frame', type=str, default="camera_first")
    parser.add_argument('--delta_pred', type=str, default='rel_to_prev')
    parser.add_argument('--rot-rep', type=str, default='axis_angle')
    parser.add_argument('--control-mode', type=str, default='pd_ee_delta_pose')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.visualize_pose:
        dataset = ODPCDataset(
            data_path=args.demo_path,
            obs_horizon=16,
            pred_horizon=16,
            slices_step=16,
            num_traj=args.num_traj,
        )
        dc = DataConversion(delta_pred="abs", frame="camera_cur", rot_rep="quaternion")
        for i in range(len(dataset)):
            data = dataset[i]
            data = common.to_tensor(data, 'cpu')
            pred = dc.raw_to_pred(
                poses_obj=data['poses_peg'],
                poses_camera_world=data['poses_cam0_world'],
            )
            intrinsic = data['intrinsic']
            k = 0
            for rgb, pose in zip(data["observations"]["rgb"][1:, :3], pred[:-1]):
                img = visualize(rgb, pose, intrinsic)
                cv2.imshow("img", img)
                k = cv2.waitKey(30)
                if k == 27 or k == ord('q'):
                    break
            if k == 27 or k == ord('q'):
                break

    dataset = ODPCDataset(
        data_path=args.demo_path,
        obs_horizon=args.obs_horizon,
        pred_horizon=args.pred_horizon,
        num_traj=args.num_traj,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        worker_init_fn=lambda worker_id: worker_init_fn(worker_id, base_seed=args.seed),
        drop_last=False,
        num_workers=args.num_workers,
    )

    dc = DataConversion(delta_pred=args.delta_pred, frame=args.frame, rot_rep=args.rot_rep)

    for iteration, data_batch in enumerate(dataloader):
        data_batch = common.to_tensor(data_batch, device)
        pred = dc.raw_to_pred(
            poses_obj=data_batch['poses_peg'],
            poses_camera_world=data_batch['poses_cam0_world'],
        )
        print(pred.shape, torch.min(pred).item(), torch.max(pred).item(), torch.mean(pred), torch.var(pred))

        pose_cam_peg = pose_multiply(data_batch['poses_cam0_world'], data_batch['poses_peg'])
        control = dc.pred_to_control(
            pred=pred,
            poses_ee=data_batch['poses_ee'],
            poses_base=data_batch['poses_base'],
            poses_camera_world=data_batch['poses_cam0_world'],
            control_mode=args.control_mode,
            poses_obj_cur=pose_cam_peg[..., :1, :],
        )
        if args.plot_control:
            if control.shape[-1] == 6:
                dim_names = ['px', 'py', 'pz', 'wx', 'wy', 'wz']
            else:
                dim_names = ['px', 'py', 'pz', 'ww', 'wx', 'wy', 'wz']
            plot(control[0], data_batch["actions"][0, :, :-1], dim_names=dim_names)

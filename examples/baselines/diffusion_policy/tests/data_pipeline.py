import cv2
import matplotlib.pyplot as plt
import argparse
from typing import List, Any

import torch
from torch.utils.data.dataloader import DataLoader

import numpy as np

from mani_skill.utils import common

from diffusion_policy.data_converison import DataConversion, pose_multiply
from diffusion_policy.odpc_dataset import ODPCDataset
from diffusion_policy.utils import worker_init_fn, visualize_pose


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
                img = visualize_pose(rgb, pose, intrinsic)
                cv2.imshow("img", img[:, :, ::-1])
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

    dc = DataConversion(
        delta_pred=args.delta_pred,
        frame=args.frame,
        rot_rep=args.rot_rep,
        control_mode=args.control_mode,
    )

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
            poses_ee_cur=data_batch['poses_ee'][..., :1, :],
            poses_base=data_batch['poses_base'],
            poses_camera_world=data_batch['poses_cam0_world'],
            poses_frame_obj_cur=pose_cam_peg[..., :1, :],
        )
        if args.plot_control:
            if control.shape[-1] == 6:
                dim_names = ['px', 'py', 'pz', 'wx', 'wy', 'wz']
            else:
                dim_names = ['px', 'py', 'pz', 'wx', 'wy', 'wz']
            plot(control[0], data_batch["actions"][0, :, :-1], dim_names=dim_names)

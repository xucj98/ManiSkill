import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.linalg import inv
import matplotlib.pyplot as plt

from diffusion_policy.math_utils import extrinsic_to_t_cam_world, pose_to_matrix

if __name__ == "__main__":
    dt = 0.05
    traj_id = 3

    file = h5py.File('./demo5.state_dict+rgb.pd_ee_delta_pose.physx_cpu.h5', 'r')

    poses_world_peg = file[f'traj_{traj_id}/obs/extra/peg_pose'][:]
    poses_world_base = file[f'traj_{traj_id}/obs/extra/base_pose'][:]
    poses_world_ee = file[f'traj_{traj_id}/obs/extra/tcp_pose'][:]
    extrinsic = file[f'traj_{traj_id}/obs/sensor_param/base_camera/extrinsic_cv'][:]
    intrinsic = file[f'traj_{traj_id}/obs/sensor_param/base_camera/intrinsic_cv'][:]
    actions_gt = file[f'traj_{traj_id}/actions']

    num_frames = poses_world_peg.shape[0]

    T_cam_peg = []
    T_cam_base = []
    T_base_ee = []

    for i in range(num_frames):
        t_cam_world = extrinsic_to_t_cam_world(extrinsic[i])

        t_world_peg = pose_to_matrix(poses_world_peg[i])
        t_cam_peg = t_cam_world @ t_world_peg
        T_cam_peg.append(t_cam_peg)

        t_world_base = pose_to_matrix(poses_world_base[i])
        t_cam_base = t_cam_world @ t_world_base
        T_cam_base.append(t_cam_base)

        t_world_ee = pose_to_matrix(poses_world_ee[i])
        t_base_ee = inv(t_world_base) @ t_world_ee
        T_base_ee.append(t_base_ee)

    # DP pred
    delta_T_cam_peg = []
    for i in range(num_frames - 1):
        delta_T_cam_peg.append(T_cam_peg[i + 1] @ inv(T_cam_peg[i]))

    # action pred
    actions_pred = []
    for i in range(num_frames - 1):
        t_base_ee_cur = T_base_ee[i]
        t_base_ee_next = inv(T_cam_base[i + 1]) @ delta_T_cam_peg[i] @ T_cam_base[i] @ t_base_ee_cur

        t = (t_base_ee_next[:3, 3] - t_base_ee_cur[:3, 3]) / dt
        r = Rotation.from_matrix(t_base_ee_next[:3, :3] @ t_base_ee_cur[:3, :3].T)
        euler = r.as_euler('xyz') / dt

        # 我也不知道为什么 mani_skill 的动作输入是 -euler
        actions_pred.append(np.concatenate((t, -euler)))

    actions_pred = np.stack(actions_pred)
    
    # plot
    time_axis_calc = np.arange(num_frames - 1) * dt
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # 线速度
    axs[0].plot(time_axis_calc, actions_pred[:, 0], 'r--', label='pred vx')
    axs[0].plot(time_axis_calc, actions_gt[:, 0], 'r-', label='gt vx')
    axs[0].plot(time_axis_calc, actions_pred[:, 1], 'g--', label='pred vy')
    axs[0].plot(time_axis_calc, actions_gt[:, 1], 'g-', label='gt vy')
    axs[0].plot(time_axis_calc, actions_pred[:, 2], 'b--', label='pred vz')
    axs[0].plot(time_axis_calc, actions_gt[:, 2], 'b-', label='gt vz')
    axs[0].set_title(f'Linear Velocity Comparison (dt={dt:.3f}s)')
    axs[0].set_ylabel('Velocity (m/s)')
    axs[0].legend()
    axs[0].grid(True)

    # 角速度 (World Frame)
    axs[1].plot(time_axis_calc, actions_pred[:, 3], 'r--', label='pred wx (world)')
    axs[1].plot(time_axis_calc, actions_gt[:, 3], 'r-', label='gt wx (world)')
    axs[1].plot(time_axis_calc, actions_pred[:, 4], 'g--', label='pred wy (world)')
    axs[1].plot(time_axis_calc, actions_gt[:, 4], 'g-', label='gt wy (world)')
    axs[1].plot(time_axis_calc, actions_pred[:, 5], 'b--', label='pred wz (world)')
    axs[1].plot(time_axis_calc, actions_gt[:, 5], 'b-', label='gt wz (world)')
    axs[1].set_title(f'Angular Velocity Comparison (World Frame, dt={dt:.3f}s)')
    axs[1].set_ylabel('Angular Velocity (rad/s)')
    axs[1].set_xlabel('Time (s)')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

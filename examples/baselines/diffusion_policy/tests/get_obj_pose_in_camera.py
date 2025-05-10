import numpy as np
import h5py
import cv2
from diffusion_policy.math_utils import (
    pose_to_matrix, extrinsic_to_t_cam_world,
)

# --- 辅助函数和常量 (与之前类似) ---
def project_points_to_image(points_3d_camera, intrinsic):
    """将相机坐标系下的3D点投影到2D图像坐标。"""
    if points_3d_camera.ndim == 1:
        points_3d_camera = points_3d_camera.reshape(1, -1)
    num_points = points_3d_camera.shape[0]
    points_2d_image = np.zeros((num_points, 2), dtype=np.float32)
    valid_mask = points_3d_camera[:, 2] > 1e-5
    if np.any(valid_mask):
        projected_hom = (intrinsic @ points_3d_camera[valid_mask].T).T
        div_val = projected_hom[:, 2]
        div_val[div_val == 0] = 1e-6
        points_2d_image[valid_mask, 0] = projected_hom[:, 0] / div_val
        points_2d_image[valid_mask, 1] = projected_hom[:, 1] / div_val
    return points_2d_image, valid_mask


def draw_3d_bounding_box(image, corners_2d, valid_mask, edges, color=(0, 255, 0), thickness=2):
    """在图像上根据投影到2D的顶点绘制3D包围框。"""
    drawn_image = image.copy()
    for edge in edges:
        p1_idx, p2_idx = edge
        if valid_mask[p1_idx] and valid_mask[p2_idx]:
            p1 = tuple(corners_2d[p1_idx].astype(int))
            p2 = tuple(corners_2d[p2_idx].astype(int))
            cv2.line(drawn_image, p1, p2, color, thickness)
    return drawn_image


_corners_local_normalized = np.array([
    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
], dtype=np.float32)

_box_edges = [
    (0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7)
]

def visualize_peg_on_image(rgb_image, t_cam_peg, peg_half_size, intrinsic,
                           box_color=(0, 0, 255), box_thickness=1):
    """
    在RGB图像上可视化peg的3D包围框。

    参数:
        rgb_image (np.ndarray): HxWx3 的RGB图像。
        T_camera_from_peg (np.ndarray): 4x4 的齐次变换矩阵，表示从peg局部坐标系到相机坐标系的变换。
        peg_half_extents (np.ndarray): 3维向量 [hx,hy,hz]，peg的半尺寸。
        K_intrinsic (np.ndarray): 3x3 的相机内参矩阵。
        box_color (tuple): 包围框颜色 (B,G,R)。
        box_thickness (int): 包围框线条粗细。

    返回:
        output_image (np.ndarray): 绘制了包围框的图像。
    """
    if t_cam_peg is None:  # 如果位姿无效
        output_image = rgb_image.copy()
        cv2.putText(output_image, "Invalid Peg Pose", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return output_image

    # 1. 计算peg在其局部坐标系下的8个顶点
    corners_local = _corners_local_normalized * peg_half_size

    # 2. 将局部顶点变换到相机坐标系
    #    P_camera = T_camera_from_peg * P_local_homogeneous
    #    首先将局部顶点转换为齐次坐标 (添加w=1)
    corners_local_homogeneous = np.hstack((corners_local, np.ones((corners_local.shape[0], 1))))  # (8, 4)

    # corners_camera_homogeneous = (T_camera_from_peg @ corners_local_homogeneous.T).T
    corners_camera_homogeneous = np.dot(t_cam_peg, corners_local_homogeneous.T).T  # (8,4)

    # 转换回非齐次坐标 (取前三维)
    corners_camera = corners_camera_homogeneous[:, :3]  # (8, 3)

    # 3. 将相机坐标系下的顶点投影到图像（像素）坐标系
    corners_2d_image, valid_projection_mask = project_points_to_image(corners_camera, intrinsic)

    # 4. 在图像上绘制包围框
    output_image = draw_3d_bounding_box(rgb_image, corners_2d_image, valid_projection_mask, _box_edges,
                                        color=box_color, thickness=box_thickness)

    return output_image


if __name__ == "__main__":
    hf = h5py.File('./demo5.state_dict+rgb.pd_ee_delta_pose.physx_cpu.h5', 'r')

    poses_world_peg = hf['traj_4/obs/extra/peg_pose'][:]
    extrinsic = hf['traj_4/obs/sensor_param/base_camera/extrinsic_cv'][:]
    images_raw = hf['traj_4/obs/sensor_data/base_camera/rgb'][:]

    peg_half_size = hf['traj_4/obs/extra/peg_half_size'][0]
    intrinsic = hf['traj_4/obs/sensor_param/base_camera/intrinsic_cv'][0]

    num_frames = poses_world_peg.shape[0]

    for i in range(num_frames):
        print(f"正在处理帧 {i + 1}/{num_frames}")

        # 1. 获取peg在相机坐标系下的4x4位姿矩阵
        t_world_peg = pose_to_matrix(poses_world_peg[i])
        t_cam_world = extrinsic_to_t_cam_world(extrinsic[i])
        t_cam_peg = t_cam_world @ t_world_peg
        # 2. 在图像上可视化peg
        img = images_raw[i].copy()

        output_image_with_peg = visualize_peg_on_image(
            img,
            t_cam_peg,
            peg_half_size,
            intrinsic,
            box_color=(50, 200, 50),  # 绿色
            box_thickness=2
        )

        cv2.imshow(f'test', output_image_with_peg[:, :, ::-1])
        key = cv2.waitKey(30)  # 使用 waitKey(30) 实现约30fps的播放，用0则逐帧按键
        if key == 27:
            print("用户退出。")
            break
        if key == ord('s'):
            save_path = f'frame_{i:04d}_peg_vis_Tmatrix.png'
            cv2.imwrite(save_path, output_image_with_peg)
            print(f"已保存图像: {save_path}")

    cv2.destroyAllWindows()
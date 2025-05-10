import numpy as np
from scipy.linalg import logm, inv
from scipy.spatial.transform import Rotation


def pose_to_matrix(pose, wxyz=True):
    pos = pose[:3]
    quat = pose[3:7]
    if wxyz:
        quat = quat[[1, 2, 3, 0]]  # wxyz -> xyzw
    try:
        # 使用scipy.spatial.transform.Rotation进行转换
        # 注意: 在批量函数中，如果很多帧失败，频繁的try-except可能影响性能。
        # 但对于正确性是必要的。
        rotation = Rotation.from_quat(quat).as_matrix()
    except ValueError:
        print(f"警告: 四元数无效 {quat}")
        return np.full((4, 4), np.nan)

    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = pos
    return matrix


def extrinsic_to_t_cam_world(extrinsic_3x4):
    """
    将3x4的外参 [R_cw | t_cw] 转换为4x4的 T_cam_world。
    假设 extrinsic_3x4 的 R 是从世界到相机，t 是世界原点在相机下的坐标。
    """
    t_cam_world = np.eye(4)
    t_cam_world[:3, :4] = extrinsic_3x4
    return t_cam_world
